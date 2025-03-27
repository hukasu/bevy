mod cascade;
mod map;

use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

pub use {cascade::*, map::*};

/// Add this component to make a [`Mesh3d`] not cast shadows.
#[derive(Debug, Component, Reflect, Default)]
#[reflect(Component, Default, Debug)]
pub struct NotShadowCaster;

/// Add this component to make a [`Mesh3d`] not receive shadows.
///
/// **Note:** If you're using diffuse transmission, setting [`NotShadowReceiver`] will
/// cause both “regular” shadows as well as diffusely transmitted shadows to be disabled,
/// even when [`TransmittedShadowReceiver`] is being used.
#[derive(Debug, Component, Reflect, Default)]
#[reflect(Component, Default, Debug)]
pub struct NotShadowReceiver;

/// Add this component to make a [`Mesh3d`] using a PBR material with [`diffuse_transmission`](crate::pbr_material::StandardMaterial::diffuse_transmission)`> 0.0`
/// receive shadows on its diffuse transmission lobe. (i.e. its “backside”)
///
/// Not enabled by default, as it requires carefully setting up [`thickness`](crate::pbr_material::StandardMaterial::thickness)
/// (and potentially even baking a thickness texture!) to match the geometry of the mesh, in order to avoid self-shadow artifacts.
///
/// **Note:** Using [`NotShadowReceiver`] overrides this component.
#[derive(Debug, Component, Reflect, Default)]
#[reflect(Component, Default, Debug)]
pub struct TransmittedShadowReceiver;
