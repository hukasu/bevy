mod ambient_light;
mod directional_light;
pub mod light_consts;
pub mod plugin;
mod point_light;
pub(crate) mod render;
mod shadow;
mod spot_light;

use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::Entity,
    query::{Or, With},
    reflect::ReflectComponent,
};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

pub use {
    ambient_light::AmbientLight,
    directional_light::DirectionalLight,
    point_light::{CubemapVisibleEntities, PointLight},
    shadow::*,
    spot_light::SpotLight,
};

/// A convenient alias for `Or<(With<PointLight>, With<SpotLight>,
/// With<DirectionalLight>)>`, for use with [`bevy_render::view::VisibleEntities`].
pub type WithLight = Or<(With<PointLight>, With<SpotLight>, With<DirectionalLight>)>;

/// Collection of mesh entities visible for 3D lighting.
///
/// This component contains all mesh entities visible from the current light view.
/// The collection is updated automatically by [`crate::SimulationLightSystems`].
#[derive(Component, Clone, Debug, Default, Reflect, Deref, DerefMut)]
#[reflect(Component, Debug, Default, Clone)]
pub struct VisibleMeshEntities {
    #[reflect(ignore, clone)]
    pub entities: Vec<Entity>,
}

/// The [`VisibilityClass`] used for all lights (point, directional, and spot).
pub(crate) struct LightVisibilityClass;

/// A component that holds the shadow cascade views for all shadow cascades
/// associated with a camera.
///
/// Note: Despite the name, this component actually holds the shadow cascade
/// views, not the lights themselves.
#[derive(Component)]
pub struct ViewLightEntities {
    /// The shadow cascade views for all shadow cascades associated with a
    /// camera.
    ///
    /// Note: Despite the name, this component actually holds the shadow cascade
    /// views, not the lights themselves.
    pub lights: Vec<Entity>,
}

#[derive(Component)]
pub struct ViewLightsUniformOffset {
    pub offset: u32,
}
