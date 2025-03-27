//! Provides functionality for rendering 3d meshes

pub mod commands;
pub mod graph;
pub mod render;
pub mod render_method;
pub(crate) mod specialization;
mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_core_pipeline::{
    core_3d::{AlphaMask3d, Opaque3d, Transmissive3d, Transparent3d},
    deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
    oit::prepare_oit_buffers,
};
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_render::{
    batching::{
        gpu_preprocessing::{self, GpuPreprocessingSupport},
        no_gpu_preprocessing,
    },
    extract_component::ExtractComponentPlugin,
    extract_resource::ExtractResourcePlugin,
    render_phase::{BinnedRenderPhasePlugin, SortedRenderPhasePlugin},
    render_resource::{GpuArrayBuffer, Shader, ShaderDefVal},
    renderer::RenderDevice,
    ExtractSchedule, Render, RenderApp, RenderDebugFlags, RenderSet,
};

use crate::{
    light::plugin::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS},
    shadow::{phase_item::Shadow, ShadowFilteringMethod},
};

use render::{
    instance::{RenderMeshInstanceGpuQueues, RenderMeshInstances},
    pipeline::{MeshPipeline, MeshPipelineViewLayouts},
    MeshCullingDataBuffer, MeshInputUniform, MeshUniform, MeshesToReextractNextFrame,
    RenderMeshMaterialIds, ViewKeyCache,
};
use render_method::DefaultOpaqueRendererMethod;
use specialization::ViewSpecializationTicks;
use systems::{
    check_views_need_specialization, collect_meshes_for_gpu_building,
    extract_meshes_for_cpu_building, extract_meshes_for_gpu_building, prepare_mesh_bind_groups,
    prepare_mesh_view_bind_groups, set_mesh_motion_vector_flags,
};

const FORWARD_IO_HANDLE: Handle<Shader> = weak_handle!("38111de1-6e35-4dbb-877b-7b6f9334baf6");
const MESH_VIEW_TYPES_HANDLE: Handle<Shader> = weak_handle!("979493db-4ae1-4003-b5c6-fcbb88b152a2");
const MESH_VIEW_BINDINGS_HANDLE: Handle<Shader> =
    weak_handle!("c6fe674b-4c21-4d4b-867a-352848da5337");
const MESH_TYPES_HANDLE: Handle<Shader> = weak_handle!("a4a3fc2e-a57e-4083-a8ab-2840176927f2");
const MESH_BINDINGS_HANDLE: Handle<Shader> = weak_handle!("84e7f9e6-e566-4a61-914e-c568f5dabf49");
const MESH_FUNCTIONS_HANDLE: Handle<Shader> = weak_handle!("c46aa0f0-6c0c-4b3a-80bf-d8213c771f12");
const MESH_SHADER_HANDLE: Handle<Shader> = weak_handle!("1a7bbae8-4b4f-48a7-b53b-e6822e56f321");
const OCCLUSION_CULLING_HANDLE: Handle<Shader> =
    weak_handle!("eaea07d9-7516-482c-aa42-6f8e9927e1f0");

/// How many textures are allowed in the view bind group layout (`@group(0)`) before
/// broader compatibility with WebGL and WebGPU is at risk, due to the minimum guaranteed
/// values for `MAX_TEXTURE_IMAGE_UNITS` (in WebGL) and `maxSampledTexturesPerShaderStage` (in WebGPU),
/// currently both at 16.
///
/// We use 10 here because it still leaves us, in a worst case scenario, with 6 textures for the other bind groups.
///
/// See: <https://gpuweb.github.io/gpuweb/#limits>
#[cfg(debug_assertions)]
pub const MESH_PIPELINE_VIEW_LAYOUT_SAFE_MAX_TEXTURES: usize = 10;

/// Provides support for rendering 3D meshes.
pub struct MeshRenderPlugin {
    /// Whether we're building [`MeshUniform`]s on GPU.
    ///
    /// This requires compute shader support and so will be forcibly disabled if
    /// the platform doesn't support those.
    pub use_gpu_instance_buffer_builder: bool,
    /// Debugging flags that can optionally be set when constructing the renderer.
    pub debug_flags: RenderDebugFlags,
}

impl MeshRenderPlugin {
    /// Creates a new [`MeshRenderPlugin`] with the given debug flags.
    pub fn new(debug_flags: RenderDebugFlags) -> MeshRenderPlugin {
        MeshRenderPlugin {
            use_gpu_instance_buffer_builder: false,
            debug_flags,
        }
    }
}

impl Plugin for MeshRenderPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            FORWARD_IO_HANDLE,
            "shaders/forward_io.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_VIEW_TYPES_HANDLE,
            "shaders/mesh_view_types.wgsl",
            Shader::from_wgsl_with_defs,
            vec![
                ShaderDefVal::UInt(
                    "MAX_DIRECTIONAL_LIGHTS".into(),
                    MAX_DIRECTIONAL_LIGHTS as u32
                ),
                ShaderDefVal::UInt(
                    "MAX_CASCADES_PER_LIGHT".into(),
                    MAX_CASCADES_PER_LIGHT as u32,
                )
            ]
        );
        load_internal_asset!(
            app,
            MESH_VIEW_BINDINGS_HANDLE,
            "shaders/mesh_view_bindings.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_TYPES_HANDLE,
            "shaders/mesh_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_FUNCTIONS_HANDLE,
            "shaders/mesh_functions.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            MESH_SHADER_HANDLE,
            "shaders/mesh.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            OCCLUSION_CULLING_HANDLE,
            "shaders/occlusion_culling.wgsl",
            Shader::from_wgsl
        );

        if app.get_sub_app(RenderApp).is_none() {
            return;
        }

        app.register_type::<ShadowFilteringMethod>();

        app.add_plugins((
            ExtractComponentPlugin::<ShadowFilteringMethod>::default(),
            BinnedRenderPhasePlugin::<Opaque3d, MeshPipeline>::new(self.debug_flags),
            BinnedRenderPhasePlugin::<AlphaMask3d, MeshPipeline>::new(self.debug_flags),
            BinnedRenderPhasePlugin::<Shadow, MeshPipeline>::new(self.debug_flags),
            BinnedRenderPhasePlugin::<Opaque3dDeferred, MeshPipeline>::new(self.debug_flags),
            BinnedRenderPhasePlugin::<AlphaMask3dDeferred, MeshPipeline>::new(self.debug_flags),
            SortedRenderPhasePlugin::<Transmissive3d, MeshPipeline>::new(self.debug_flags),
            SortedRenderPhasePlugin::<Transparent3d, MeshPipeline>::new(self.debug_flags),
        ));

        app.register_type::<DefaultOpaqueRendererMethod>()
            .init_resource::<DefaultOpaqueRendererMethod>()
            .add_plugins(ExtractResourcePlugin::<DefaultOpaqueRendererMethod>::default());

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<MeshCullingDataBuffer>()
                .init_resource::<RenderMeshMaterialIds>()
                .configure_sets(
                    ExtractSchedule,
                    ExtractMeshesSet.after(bevy_render::view::extract_visibility_ranges),
                )
                .configure_sets(
                    Render,
                    GpuMeshBuildingSet
                        .in_set(RenderSet::PrepareMeshes)
                        // This must be before
                        // `set_mesh_motion_vector_flags` so it doesn't
                        // overwrite those flags.
                        .before(set_mesh_motion_vector_flags),
                )
                .add_systems(
                    ExtractSchedule,
                    gpu_preprocessing::clear_batched_gpu_instance_buffers::<MeshPipeline>
                        .before(ExtractMeshesSet),
                )
                .add_systems(
                    Render,
                    (
                        set_mesh_motion_vector_flags.in_set(RenderSet::PrepareMeshes),
                        prepare_mesh_bind_groups.in_set(RenderSet::PrepareBindGroups),
                        prepare_mesh_view_bind_groups
                            .in_set(RenderSet::PrepareBindGroups)
                            .after(prepare_oit_buffers),
                        no_gpu_preprocessing::clear_batched_cpu_instance_buffers::<MeshPipeline>
                            .in_set(RenderSet::Cleanup)
                            .after(RenderSet::Render),
                    ),
                );
        }
    }

    fn finish(&self, app: &mut App) {
        let mut mesh_bindings_shader_defs = Vec::with_capacity(1);

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app
                .init_resource::<ViewKeyCache>()
                .init_resource::<ViewSpecializationTicks>()
                .init_resource::<GpuPreprocessingSupport>()
                .add_systems(
                    Render,
                    check_views_need_specialization.in_set(RenderSet::PrepareAssets),
                );

            let gpu_preprocessing_support =
                render_app.world().resource::<GpuPreprocessingSupport>();
            let use_gpu_instance_buffer_builder =
                self.use_gpu_instance_buffer_builder && gpu_preprocessing_support.is_available();

            let render_mesh_instances = RenderMeshInstances::new(use_gpu_instance_buffer_builder);
            render_app.insert_resource(render_mesh_instances);

            if use_gpu_instance_buffer_builder {
                render_app
                    .init_resource::<gpu_preprocessing::BatchedInstanceBuffers<
                        MeshUniform,
                        MeshInputUniform
                    >>()
                    .init_resource::<RenderMeshInstanceGpuQueues>()
                    .init_resource::<MeshesToReextractNextFrame>()
                    .add_systems(
                        ExtractSchedule,
                        extract_meshes_for_gpu_building.in_set(ExtractMeshesSet),
                    )
                    .add_systems(
                        Render,
                        (
                            gpu_preprocessing::write_batched_instance_buffers::<MeshPipeline>
                                .in_set(RenderSet::PrepareResourcesFlush),
                            gpu_preprocessing::delete_old_work_item_buffers::<MeshPipeline>
                                .in_set(RenderSet::PrepareResources),
                            collect_meshes_for_gpu_building.in_set(GpuMeshBuildingSet)
                        ),
                    );
            } else {
                let render_device = render_app.world().resource::<RenderDevice>();
                let cpu_batched_instance_buffer =
                    no_gpu_preprocessing::BatchedInstanceBuffer::<MeshUniform>::new(render_device);
                render_app
                    .insert_resource(cpu_batched_instance_buffer)
                    .add_systems(
                        ExtractSchedule,
                        extract_meshes_for_cpu_building.in_set(ExtractMeshesSet),
                    )
                    .add_systems(
                        Render,
                        no_gpu_preprocessing::write_batched_instance_buffer::<MeshPipeline>
                            .in_set(RenderSet::PrepareResourcesFlush),
                    );
            };

            let render_device = render_app.world().resource::<RenderDevice>();
            if let Some(per_object_buffer_batch_size) =
                GpuArrayBuffer::<MeshUniform>::batch_size(render_device)
            {
                mesh_bindings_shader_defs.push(ShaderDefVal::UInt(
                    "PER_OBJECT_BUFFER_BATCH_SIZE".into(),
                    per_object_buffer_batch_size,
                ));
            }

            render_app
                .init_resource::<MeshPipelineViewLayouts>()
                .init_resource::<MeshPipeline>();
        }

        // Load the mesh_bindings shader module here as it depends on runtime information about
        // whether storage buffers are supported, or the maximum uniform buffer binding size.
        load_internal_asset!(
            app,
            MESH_BINDINGS_HANDLE,
            "shaders/mesh_bindings.wgsl",
            Shader::from_wgsl_with_defs,
            mesh_bindings_shader_defs
        );
    }
}

/// A [`SystemSet`] that encompasses both [`extract_meshes_for_cpu_building`]
/// and [`extract_meshes_for_gpu_building`].
#[derive(SystemSet, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ExtractMeshesSet;

/// A [`SystemSet`] that encompass [`collect_meshes_for_gpu_building`]
#[derive(SystemSet, Clone, PartialEq, Eq, Debug, Hash)]
pub struct GpuMeshBuildingSet;
