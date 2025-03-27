mod systems;

use bevy_app::{App, Plugin};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_component::ExtractComponentPlugin,
    render_graph::{RenderGraph, RenderGraphApp, ViewNodeRunner},
    render_resource::{Shader, SpecializedRenderPipelines},
    Render, RenderApp, RenderSet,
};

use systems::{prepare_ssr_pipelines, prepare_ssr_settings};

use crate::{
    mesh_pipeline::graph::NodeRender3d,
    ssr::{
        render::{ScreenSpaceReflectionsBuffer, ScreenSpaceReflectionsNode},
        ScreenSpaceReflections,
    },
};

use super::render::ScreenSpaceReflectionsPipeline;

pub(super) const SSR_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("0b559df2-0d61-4f53-bf62-aea16cf32787");
pub(super) const RAYMARCH_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("798cc6fc-6072-4b6c-ab4f-83905fa4a19e");

/// Enables screen-space reflections for a camera.
///
/// Screen-space reflections are currently only supported with deferred rendering.
pub struct ScreenSpaceReflectionsPlugin;

impl Plugin for ScreenSpaceReflectionsPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(app, SSR_SHADER_HANDLE, "ssr.wgsl", Shader::from_wgsl);
        load_internal_asset!(
            app,
            RAYMARCH_SHADER_HANDLE,
            "raymarch.wgsl",
            Shader::from_wgsl
        );

        app.register_type::<ScreenSpaceReflections>()
            .add_plugins(ExtractComponentPlugin::<ScreenSpaceReflections>::default());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ScreenSpaceReflectionsBuffer>()
            .add_systems(Render, prepare_ssr_pipelines.in_set(RenderSet::Prepare))
            .add_systems(
                Render,
                prepare_ssr_settings.in_set(RenderSet::PrepareResources),
            )
            .add_render_graph_node::<ViewNodeRunner<ScreenSpaceReflectionsNode>>(
                Core3d,
                NodeRender3d::ScreenSpaceReflections,
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<ScreenSpaceReflectionsPipeline>()
            .init_resource::<SpecializedRenderPipelines<ScreenSpaceReflectionsPipeline>>();

        // only reference the default deferred lighting pass
        // if it has been added
        let has_default_deferred_lighting_pass = render_app
            .world_mut()
            .resource_mut::<RenderGraph>()
            .sub_graph(Core3d)
            .get_node_state(NodeRender3d::DeferredLightingPass)
            .is_ok();

        if has_default_deferred_lighting_pass {
            render_app.add_render_graph_edges(
                Core3d,
                (
                    NodeRender3d::DeferredLightingPass,
                    NodeRender3d::ScreenSpaceReflections,
                    Node3d::MainOpaquePass,
                ),
            );
        } else {
            render_app.add_render_graph_edges(
                Core3d,
                (NodeRender3d::ScreenSpaceReflections, Node3d::MainOpaquePass),
            );
        }
    }
}
