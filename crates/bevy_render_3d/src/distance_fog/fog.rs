use bevy_ecs::{component::Component, resource::Resource};
use bevy_math::{Vec3, Vec4};
use bevy_render::render_resource::{DynamicUniformBuffer, ShaderType};

/// Metadata for fog
#[derive(Default, Resource)]
pub struct FogMeta {
    pub gpu_fogs: DynamicUniformBuffer<GpuFog>,
}

/// Inserted on each `Entity` with an `ExtractedView` to keep track of its offset
/// in the `gpu_fogs` `DynamicUniformBuffer` within `FogMeta`
#[derive(Component)]
pub struct ViewFogUniformOffset {
    pub offset: u32,
}

/// The GPU-side representation of the fog configuration that's sent as a uniform to the shader
#[derive(Copy, Clone, ShaderType, Default, Debug)]
pub struct GpuFog {
    /// Fog color
    pub base_color: Vec4,
    /// The color used for the fog where the view direction aligns with directional lights
    pub directional_light_color: Vec4,
    /// Allocated differently depending on fog mode.
    /// See `mesh_view_types.wgsl` for a detailed explanation
    pub be: Vec3,
    /// The exponent applied to the directional light alignment calculation
    pub directional_light_exponent: f32,
    /// Allocated differently depending on fog mode.
    /// See `mesh_view_types.wgsl` for a detailed explanation
    pub bi: Vec3,
    /// Unsigned int representation of the active fog falloff mode
    pub mode: u32,
}
