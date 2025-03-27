pub mod plugin;
pub(crate) mod render;

use bevy_ecs::component::Component;
use bevy_render::{extract_component::ExtractComponent, render_resource::ShaderType};

pub const DEFAULT_DEFERRED_LIGHTING_PASS_ID: u8 = 1;

/// Component with a `depth_id` for specifying which corresponding materials should
/// be rendered by this specific PBR deferred lighting pass.
///
/// Will be automatically added to entities with the [`DeferredPrepass`] component
/// that don't already have a [`PbrDeferredLightingDepthId`].
#[derive(Component, Clone, Copy, ExtractComponent, ShaderType)]
pub struct DeferredLightingDepthId {
    depth_id: u32,

    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_0: f32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_1: f32,
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    _webgl2_padding_2: f32,
}

impl DeferredLightingDepthId {
    pub fn new(value: u8) -> DeferredLightingDepthId {
        DeferredLightingDepthId {
            depth_id: value as u32,

            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_0: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_1: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_2: 0.0,
        }
    }

    pub fn set(&mut self, value: u8) {
        self.depth_id = value as u32;
    }

    pub fn get(&self) -> u8 {
        self.depth_id as u8
    }
}

impl Default for DeferredLightingDepthId {
    fn default() -> Self {
        DeferredLightingDepthId {
            depth_id: DEFAULT_DEFERRED_LIGHTING_PASS_ID as u32,

            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_0: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_1: 0.0,
            #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
            _webgl2_padding_2: 0.0,
        }
    }
}
