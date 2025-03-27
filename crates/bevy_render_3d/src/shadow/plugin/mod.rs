mod systems;

use core::{hash::Hash, marker::PhantomData};

use bevy_app::Plugin;
use bevy_core_pipeline::core_3d::graph::{Core3d, Node3d};
use bevy_ecs::{schedule::IntoScheduleConfigs, world::FromWorld};
use bevy_render::{
    render_graph::RenderGraph,
    render_phase::{AddRenderCommand, DrawFunctions},
    Render, RenderApp, RenderSet,
};

use crate::{
    light::plugin::LightSystems,
    material::{plugin::MaterialRenderSystems, Material},
    mesh_pipeline::graph::NodeRender3d,
    prepass::commands::DrawPrepass,
};

use super::{
    phase_item::Shadow,
    render::{
        EarlyShadowPassNode, LateShadowPassNode, ShadowSamplers,
        SpecializedShadowMaterialPipelineCache,
    },
};

use systems::{queue_shadows, specialize_shadows};

pub struct ShadowPlugin<M: Material>(PhantomData<M>);

impl<M: Material> Default for ShadowPlugin<M> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<M: Material> Plugin for ShadowPlugin<M>
where
    M::Data: Clone + Eq + Hash,
{
    fn build(&self, app: &mut bevy_app::App) {
        if !app.is_plugin_added::<BaseShadowPlugin>() {
            app.add_plugins(BaseShadowPlugin);
        }

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.add_render_command::<Shadow, DrawPrepass<M>>();

            render_app
                .init_resource::<SpecializedShadowMaterialPipelineCache<M>>()
                .add_systems(
                    Render,
                    (
                        // specialize_shadows::<M> also needs to run after prepare_assets::<PreparedMaterial<M>>,
                        // which is fine since ManageViews is after PrepareAssets
                        specialize_shadows::<M>
                            .in_set(RenderSet::ManageViews)
                            .after(LightSystems::Prepare),
                        queue_shadows::<M>.in_set(MaterialRenderSystems::QueueMeshes),
                    ),
                );
        }
    }
}

struct BaseShadowPlugin;

impl Plugin for BaseShadowPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.init_resource::<DrawFunctions<Shadow>>();

            let early_shadow_pass_node = EarlyShadowPassNode::from_world(render_app.world_mut());
            let late_shadow_pass_node = LateShadowPassNode::from_world(render_app.world_mut());
            let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
            let draw_3d_graph = graph.get_sub_graph_mut(Core3d).unwrap();
            draw_3d_graph.add_node(NodeRender3d::EarlyShadowPass, early_shadow_pass_node);
            draw_3d_graph.add_node(NodeRender3d::LateShadowPass, late_shadow_pass_node);
            draw_3d_graph.add_node_edges((
                NodeRender3d::EarlyShadowPass,
                NodeRender3d::LateShadowPass,
                Node3d::StartMainPass,
            ));
        }
    }

    fn finish(&self, app: &mut bevy_app::App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.init_resource::<ShadowSamplers>();
    }
}
