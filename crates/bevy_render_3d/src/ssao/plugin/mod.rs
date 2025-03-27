mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::{Shader, SpecializedComputePipelines, TextureFormat, TextureUsages},
    renderer::{RenderAdapter, RenderDevice},
    sync_component::SyncComponentPlugin,
    ExtractSchedule, Render, RenderApp, RenderSet,
};

use tracing::warn;

use crate::{mesh_pipeline::pipeline::graph::NodeRender3d, ssao::ScreenSpaceAmbientOcclusion};

use systems::{
    extract_ssao_settings, prepare_ssao_bind_groups, prepare_ssao_pipelines, prepare_ssao_textures,
};

use super::ssao::{SsaoNode, SsaoPipelines};

pub(super) const PREPROCESS_DEPTH_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("b7f2cc3d-c935-4f5c-9ae2-43d6b0d5659a");
pub(super) const SSAO_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("9ea355d7-37a2-4cc4-b4d1-5d8ab47b07f5");
pub(super) const SPATIAL_DENOISE_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("0f2764a0-b343-471b-b7ce-ef5d636f4fc3");
const SSAO_UTILS_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("da53c78d-f318-473e-bdff-b388bc50ada2");

/// Plugin for screen space ambient occlusion.
pub struct ScreenSpaceAmbientOcclusionPlugin;

impl Plugin for ScreenSpaceAmbientOcclusionPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            PREPROCESS_DEPTH_SHADER_HANDLE,
            "shaders/preprocess_depth.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SSAO_SHADER_HANDLE,
            "shaders/ssao.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SPATIAL_DENOISE_SHADER_HANDLE,
            "shaders/spatial_denoise.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            SSAO_UTILS_SHADER_HANDLE,
            "shaders/ssao_utils.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ScreenSpaceAmbientOcclusion>();

        app.add_plugins(SyncComponentPlugin::<ScreenSpaceAmbientOcclusion>::default());
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        if !render_app
            .world()
            .resource::<RenderAdapter>()
            .get_texture_format_features(TextureFormat::R16Float)
            .allowed_usages
            .contains(TextureUsages::STORAGE_BINDING)
        {
            warn!("ScreenSpaceAmbientOcclusionPlugin not loaded. GPU lacks support: TextureFormat::R16Float does not support TextureUsages::STORAGE_BINDING.");
            return;
        }

        if render_app
            .world()
            .resource::<RenderDevice>()
            .limits()
            .max_storage_textures_per_shader_stage
            < 5
        {
            warn!("ScreenSpaceAmbientOcclusionPlugin not loaded. GPU lacks support: Limits::max_storage_textures_per_shader_stage is less than 5.");
            return;
        }

        render_app
            .init_resource::<SsaoPipelines>()
            .init_resource::<SpecializedComputePipelines<SsaoPipelines>>()
            .add_systems(ExtractSchedule, extract_ssao_settings)
            .add_systems(
                Render,
                (
                    prepare_ssao_pipelines.in_set(RenderSet::Prepare),
                    prepare_ssao_textures.in_set(RenderSet::PrepareResources),
                    prepare_ssao_bind_groups.in_set(RenderSet::PrepareBindGroups),
                ),
            )
            .add_render_graph_node::<ViewNodeRunner<SsaoNode>>(
                Core3d,
                NodeRender3d::ScreenSpaceAmbientOcclusion,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    // END_PRE_PASSES -> SCREEN_SPACE_AMBIENT_OCCLUSION -> MAIN_PASS
                    Node3d::EndPrepasses,
                    NodeRender3d::ScreenSpaceAmbientOcclusion,
                    Node3d::StartMainPass,
                ),
            );
    }
}
