//! GPU mesh preprocessing.
//!
//! This is an optional pass that uses a compute shader to reduce the amount of
//! data that has to be transferred from the CPU to the GPU. When enabled,
//! instead of transferring [`MeshUniform`]s to the GPU, we transfer the smaller
//! [`MeshInputUniform`]s instead and use the GPU to calculate the remaining
//! derived fields in [`MeshUniform`].

pub(crate) mod render;
mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{prelude::resource_exists, schedule::IntoScheduleConfigs};
use bevy_render::{
    batching::gpu_preprocessing::{BatchedInstanceBuffers, GpuPreprocessingSupport},
    render_graph::RenderGraphApp,
    render_resource::{Shader, SpecializedComputePipelines},
    Render, RenderApp, RenderSet,
};
use render::{
    BuildIndirectParametersPipeline, ClearIndirectParametersMetadataNode, EarlyGpuPreprocessNode,
    EarlyPrepassBuildIndirectParametersNode, LateGpuPreprocessNode,
    LatePrepassBuildIndirectParametersNode, MainBuildIndirectParametersNode, PreprocessPipeline,
    PreprocessPipelines, ResetIndirectBatchSetsPipeline,
};
use systems::{
    prepare_preprocess_bind_groups, prepare_preprocess_pipelines, write_mesh_culling_data_buffer,
};

use crate::mesh_pipeline::{
    graph::NodeRender3d,
    render::{MeshInputUniform, MeshUniform},
};

/// The handle to the `mesh_preprocess.wgsl` compute shader.
pub const MESH_PREPROCESS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("c8579292-cf92-43b5-9c5a-ec5bd4e44d12");
/// The handle to the `mesh_preprocess_types.wgsl` compute shader.
pub const MESH_PREPROCESS_TYPES_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("06f797ef-a106-4098-9a2e-20a73aa182e2");
/// The handle to the `reset_indirect_batch_sets.wgsl` compute shader.
pub const RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("045fb176-58e2-4e76-b241-7688d761bb23");
/// The handle to the `build_indirect_params.wgsl` compute shader.
pub const BUILD_INDIRECT_PARAMS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("133b01f0-3eaf-4590-9ee9-f0cf91a00b71");

/// The GPU workgroup size.
const WORKGROUP_SIZE: usize = 64;

/// A plugin that builds mesh uniforms on GPU.
///
/// This will only be added if the platform supports compute shaders (e.g. not
/// on WebGL 2).
pub struct GpuMeshPreprocessPlugin {
    /// Whether we're building [`MeshUniform`]s on GPU.
    ///
    /// This requires compute shader support and so will be forcibly disabled if
    /// the platform doesn't support those.
    pub use_gpu_instance_buffer_builder: bool,
}

impl Plugin for GpuMeshPreprocessPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            MESH_PREPROCESS_SHADER_HANDLE,
            "shaders/mesh_preprocess.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            RESET_INDIRECT_BATCH_SETS_SHADER_HANDLE,
            "shaders/reset_indirect_batch_sets.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            BUILD_INDIRECT_PARAMS_SHADER_HANDLE,
            "shaders/build_indirect_params.wgsl",
            Shader::from_wgsl
        );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        // This plugin does nothing if GPU instance buffer building isn't in
        // use.
        let gpu_preprocessing_support = render_app.world().resource::<GpuPreprocessingSupport>();
        if !self.use_gpu_instance_buffer_builder || !gpu_preprocessing_support.is_available() {
            return;
        }

        render_app
            .init_resource::<PreprocessPipelines>()
            .init_resource::<SpecializedComputePipelines<PreprocessPipeline>>()
            .init_resource::<SpecializedComputePipelines<ResetIndirectBatchSetsPipeline>>()
            .init_resource::<SpecializedComputePipelines<BuildIndirectParametersPipeline>>()
            .add_systems(
                Render,
                (
                    prepare_preprocess_pipelines.in_set(RenderSet::Prepare),
                    prepare_preprocess_bind_groups
                        .run_if(resource_exists::<BatchedInstanceBuffers<
                            MeshUniform,
                            MeshInputUniform
                        >>)
                        .in_set(RenderSet::PrepareBindGroups),
                    write_mesh_culling_data_buffer.in_set(RenderSet::PrepareResourcesFlush),
                ),
            )
            .add_render_graph_node::<ClearIndirectParametersMetadataNode>(
                Core3d,
                NodeRender3d::ClearIndirectParametersMetadata
            )
            .add_render_graph_node::<EarlyGpuPreprocessNode>(Core3d, NodeRender3d::EarlyGpuPreprocess)
            .add_render_graph_node::<LateGpuPreprocessNode>(Core3d, NodeRender3d::LateGpuPreprocess)
            .add_render_graph_node::<EarlyPrepassBuildIndirectParametersNode>(
                Core3d,
                NodeRender3d::EarlyPrepassBuildIndirectParameters,
            )
            .add_render_graph_node::<LatePrepassBuildIndirectParametersNode>(
                Core3d,
                NodeRender3d::LatePrepassBuildIndirectParameters,
            )
            .add_render_graph_node::<MainBuildIndirectParametersNode>(
                Core3d,
                NodeRender3d::MainBuildIndirectParameters,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    NodeRender3d::ClearIndirectParametersMetadata,
                    NodeRender3d::EarlyGpuPreprocess,
                    NodeRender3d::EarlyPrepassBuildIndirectParameters,
                    Node3d::EarlyPrepass,
                    Node3d::EarlyDeferredPrepass,
                    Node3d::EarlyDownsampleDepth,
                    NodeRender3d::LateGpuPreprocess,
                    NodeRender3d::LatePrepassBuildIndirectParameters,
                    Node3d::LatePrepass,
                    Node3d::LateDeferredPrepass,
                    NodeRender3d::MainBuildIndirectParameters,
                    Node3d::StartMainPass,
                ),
            ).add_render_graph_edges(
                Core3d,
                (
                    NodeRender3d::EarlyPrepassBuildIndirectParameters,
                    NodeRender3d::EarlyShadowPass,
                    Node3d::EarlyDownsampleDepth,
                )
            ).add_render_graph_edges(
                Core3d,
                (
                    NodeRender3d::LatePrepassBuildIndirectParameters,
                    NodeRender3d::LateShadowPass,
                    NodeRender3d::MainBuildIndirectParameters,
                )
            );
    }
}
