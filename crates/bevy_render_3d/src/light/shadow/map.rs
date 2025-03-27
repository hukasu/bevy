use bevy_ecs::{reflect::ReflectResource, resource::Resource};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

/// Controls the resolution of [`DirectionalLight`] shadow maps.
#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource, Debug, Default, Clone)]
pub struct DirectionalLightShadowMap {
    pub size: usize,
}

impl Default for DirectionalLightShadowMap {
    fn default() -> Self {
        Self { size: 2048 }
    }
}

#[derive(Resource, Clone, Debug, Reflect)]
#[reflect(Resource, Debug, Default, Clone)]
pub struct PointLightShadowMap {
    pub size: usize,
}

impl Default for PointLightShadowMap {
    fn default() -> Self {
        Self { size: 1024 }
    }
}
