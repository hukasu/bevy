mod systems;

use std::marker::PhantomData;

use bevy_app::{Plugin, PostUpdate};
use bevy_ecs::schedule::{IntoScheduleConfigs, SystemSet};
use bevy_render::{
    camera::{sort_cameras, CameraUpdateSystem},
    extract_component::ExtractComponentPlugin,
    extract_resource::ExtractResourcePlugin,
    sync_component::SyncComponentPlugin,
    view::VisibilitySystems,
    ExtractSchedule, Render, RenderApp, RenderSet,
};
use bevy_transform::TransformSystem;

use crate::{
    cluster::plugin::{ClusterSystems, ClusterableObjectPlugin},
    material::{plugin::MaterialSystems, Material},
};

use super::{
    render::{LightKeyCache, LightMeta, LightSpecializationTicks},
    AmbientLight, CascadeShadowConfig, Cascades, CascadesVisibleEntities, CubemapVisibleEntities,
    DirectionalLight, DirectionalLightShadowMap, NotShadowCaster, NotShadowReceiver, PointLight,
    PointLightShadowMap, SpotLight, VisibleMeshEntities,
};

use systems::{
    check_dir_light_mesh_visibility, check_light_entities_needing_specialization,
    check_point_light_mesh_visibility, check_views_lights_need_specialization,
    clear_directional_light_cascades, extract_lights, prepare_lights,
    update_directional_light_frusta, update_point_light_frusta, update_spot_light_frusta,
};

// NOTE: When running bevy on Adreno GPU chipsets in WebGL, any value above 1 will result in a crash
// when loading the wgsl "pbr_functions.wgsl" in the function apply_fog.
#[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
pub const MAX_DIRECTIONAL_LIGHTS: usize = 1;
#[cfg(any(
    not(feature = "webgl"),
    not(target_arch = "wasm32"),
    feature = "webgpu"
))]
pub const MAX_DIRECTIONAL_LIGHTS: usize = 10;
#[cfg(any(
    not(feature = "webgl"),
    not(target_arch = "wasm32"),
    feature = "webgpu"
))]
pub const MAX_CASCADES_PER_LIGHT: usize = 4;
#[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
pub const MAX_CASCADES_PER_LIGHT: usize = 1;

pub struct LightPlugin<M: Material> {
    pub shadows_enabled: bool,
    data: PhantomData<M>,
}

impl<M: Material> LightPlugin<M> {
    pub fn new(shadows_enabled: bool) -> Self {
        Self {
            shadows_enabled,
            data: PhantomData,
        }
    }
}

impl<M: Material> Plugin for LightPlugin<M> {
    fn build(&self, app: &mut bevy_app::App) {
        if !app.is_plugin_added::<BaseLightPlugin>() {
            app.add_plugins(BaseLightPlugin {
                shadows_enabled: self.shadows_enabled,
            });
        }

        if self.shadows_enabled {
            app.add_systems(
                PostUpdate,
                check_light_entities_needing_specialization::<M>
                    .after(MaterialSystems::SpecializationCheck),
            );
        }
    }
}

struct BaseLightPlugin {
    pub shadows_enabled: bool,
}

impl Plugin for BaseLightPlugin {
    fn build(&self, app: &mut bevy_app::App) {
        app.register_type::<AmbientLight>()
            .register_type::<CascadeShadowConfig>()
            .register_type::<Cascades>()
            .register_type::<DirectionalLight>()
            .register_type::<DirectionalLightShadowMap>()
            .register_type::<NotShadowCaster>()
            .register_type::<NotShadowReceiver>()
            .register_type::<PointLight>()
            .register_type::<PointLightShadowMap>()
            .register_type::<SpotLight>()
            .init_resource::<AmbientLight>()
            .init_resource::<DirectionalLightShadowMap>()
            .init_resource::<PointLightShadowMap>()
            .register_type::<CascadesVisibleEntities>()
            .register_type::<CubemapVisibleEntities>()
            .register_type::<VisibleMeshEntities>();

        app.add_plugins((
            SyncComponentPlugin::<DirectionalLight>::default(),
            SyncComponentPlugin::<PointLight>::default(),
            SyncComponentPlugin::<SpotLight>::default(),
            ExtractResourcePlugin::<AmbientLight>::default(),
            ExtractComponentPlugin::<AmbientLight>::default(),
        ))
        .add_plugins((
            ClusterableObjectPlugin::<0, PointLight>::default(),
            ClusterableObjectPlugin::<1, SpotLight>::default(),
        ));

        app.configure_sets(
            PostUpdate,
            SimulationLightSystems::UpdateDirectionalLightCascades
                .ambiguous_with(SimulationLightSystems::UpdateDirectionalLightCascades),
        )
        .configure_sets(
            PostUpdate,
            SimulationLightSystems::CheckLightVisibility
                .ambiguous_with(SimulationLightSystems::CheckLightVisibility),
        );

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render_app.configure_sets(
                Render,
                LightSystems::Prepare
                    .in_set(RenderSet::ManageViews)
                    .after(sort_cameras),
            );

            if self.shadows_enabled {
                render_app
                    .init_resource::<LightKeyCache>()
                    .init_resource::<LightSpecializationTicks>()
                    .add_systems(
                        Render,
                        check_views_lights_need_specialization.in_set(RenderSet::PrepareAssets),
                    );
            }

            render_app
                .add_systems(ExtractSchedule, extract_lights)
                .add_systems(Render, prepare_lights.in_set(LightSystems::Prepare))
                .add_systems(
                    PostUpdate,
                    (
                        clear_directional_light_cascades
                            .in_set(SimulationLightSystems::UpdateDirectionalLightCascades)
                            .after(TransformSystem::TransformPropagate)
                            .after(CameraUpdateSystem),
                        update_directional_light_frusta
                            .in_set(SimulationLightSystems::UpdateLightFrusta)
                            // This must run after CheckVisibility because it relies on `ViewVisibility`
                            .after(VisibilitySystems::CheckVisibility)
                            .after(TransformSystem::TransformPropagate)
                            .after(SimulationLightSystems::UpdateDirectionalLightCascades)
                            // We assume that no entity will be both a directional light and a spot light,
                            // so these systems will run independently of one another.
                            // FIXME: Add an archetype invariant for this https://github.com/bevyengine/bevy/issues/1481.
                            .ambiguous_with(update_spot_light_frusta),
                        update_point_light_frusta
                            .in_set(SimulationLightSystems::UpdateLightFrusta)
                            .after(TransformSystem::TransformPropagate)
                            .after(ClusterSystems::AssignLightsToClusters),
                        update_spot_light_frusta
                            .in_set(SimulationLightSystems::UpdateLightFrusta)
                            .after(TransformSystem::TransformPropagate)
                            .after(ClusterSystems::AssignLightsToClusters),
                        (
                            check_dir_light_mesh_visibility,
                            check_point_light_mesh_visibility,
                        )
                            .in_set(SimulationLightSystems::CheckLightVisibility)
                            .after(VisibilitySystems::CalculateBounds)
                            .after(TransformSystem::TransformPropagate)
                            .after(SimulationLightSystems::UpdateLightFrusta)
                            // NOTE: This MUST be scheduled AFTER the core renderer visibility check
                            // because that resets entity `ViewVisibility` for the first view
                            // which would override any results from this otherwise
                            .after(VisibilitySystems::CheckVisibility)
                            .before(VisibilitySystems::MarkNewlyHiddenEntitiesInvisible),
                    ),
                )
                .init_resource::<LightMeta>();
        }
    }
}

/// System sets used to run light-related systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum LightSystems {
    Prepare,
}

/// System sets used to run light-related systems.
#[derive(Debug, Hash, PartialEq, Eq, Clone, SystemSet)]
pub enum SimulationLightSystems {
    /// System order ambiguities between systems in this set are ignored:
    /// each [`build_directional_light_cascades`] system is independent of the others,
    /// and should operate on distinct sets of entities.
    UpdateDirectionalLightCascades,
    UpdateLightFrusta,
    /// System order ambiguities between systems in this set are ignored:
    /// the order of systems within this set is irrelevant, as the various visibility-checking systems
    /// assumes that their operations are irreversible during the frame.
    CheckLightVisibility,
}
