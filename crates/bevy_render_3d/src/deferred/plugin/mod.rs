mod systems;

use bevy_app::{App, Plugin, PostUpdate};
use bevy_asset::{load_internal_asset, weak_handle, Handle};
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::schedule::IntoScheduleConfigs;
use bevy_render::{
    extract_component::{ExtractComponentPlugin, UniformComponentPlugin},
    render_graph::{RenderGraphApp, ViewNodeRunner},
    render_resource::{Shader, SpecializedRenderPipelines},
    Render, RenderApp, RenderSet,
};
use systems::{insert_deferred_lighting_pass_id_component, prepare_deferred_lighting_pipelines};

use crate::{
    deferred::{
        render::{DeferredLightingLayout, DeferredOpaquePass3dLightingNode},
        DeferredLightingDepthId,
    },
    mesh_pipeline::graph::NodeRender3d,
};

pub(super) const DEFERRED_LIGHTING_SHADER_HANDLE: Handle<Shader> =
    weak_handle!("f4295279-8890-4748-b654-ca4d2183df1c");
const DEFERRED_TYPES_HANDLE: Handle<Shader> = weak_handle!("43060da7-a717-4240-80a8-dbddd92bd25d");
const DEFERRED_FUNCTIONS_HANDLE: Handle<Shader> =
    weak_handle!("9dc46746-c51d-45e3-a321-6a50c3963420");

pub struct DeferredLightingPlugin;

impl Plugin for DeferredLightingPlugin {
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            DEFERRED_LIGHTING_SHADER_HANDLE,
            "deferred_lighting.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            DEFERRED_TYPES_HANDLE,
            "deferred_types.wgsl",
            Shader::from_wgsl
        );
        load_internal_asset!(
            app,
            DEFERRED_FUNCTIONS_HANDLE,
            "deferred_functions.wgsl",
            Shader::from_wgsl
        );

        app.add_plugins((
            ExtractComponentPlugin::<DeferredLightingDepthId>::default(),
            UniformComponentPlugin::<DeferredLightingDepthId>::default(),
        ))
        .add_systems(PostUpdate, insert_deferred_lighting_pass_id_component);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<DeferredLightingLayout>>()
            .add_systems(
                Render,
                (prepare_deferred_lighting_pipelines.in_set(RenderSet::Prepare),),
            )
            .add_render_graph_node::<ViewNodeRunner<DeferredOpaquePass3dLightingNode>>(
                Core3d,
                NodeRender3d::DeferredLightingPass,
            )
            .add_render_graph_edges(
                Core3d,
                (
                    Node3d::StartMainPass,
                    NodeRender3d::DeferredLightingPass,
                    Node3d::MainOpaquePass,
                ),
            );
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<DeferredLightingLayout>();
    }
}
