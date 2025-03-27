use bevy_color::LinearRgba;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::{Component, Tick},
    entity::{hash_map::EntityHashMap, Entity},
    reflect::ReflectComponent,
    resource::Resource,
};
use bevy_math::{Mat4, UVec4, Vec3, Vec4};
use bevy_platform_support::collections::HashMap;
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    primitives::Frustum,
    render_resource::{DynamicUniformBuffer, ShaderType},
    sync_world::MainEntity,
    view::{RenderLayers, RetainedViewEntity},
};
use bevy_transform::components::GlobalTransform;

use crate::{
    light::{Cascade, CascadeShadowConfig},
    mesh_pipeline::render::pipeline::MeshPipelineKey,
};

use super::plugin::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS};

// NOTE: These must match the bit flags in bevy_pbr/src/render/mesh_view_types.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    pub struct PointLightFlags: u32 {
        const SHADOWS_ENABLED                   = 1 << 0;
        const SPOT_LIGHT_Y_NEGATIVE             = 1 << 1;
        const VOLUMETRIC                        = 1 << 2;
        const AFFECTS_LIGHTMAPPED_MESH_DIFFUSE  = 1 << 3;
        const NONE                              = 0;
        const UNINITIALIZED                     = 0xFFFF;
    }
}

// NOTE: These must match the bit flags in bevy_pbr/src/render/mesh_view_types.wgsl!
bitflags::bitflags! {
    #[repr(transparent)]
    pub struct DirectionalLightFlags: u32 {
        const SHADOWS_ENABLED                   = 1 << 0;
        const VOLUMETRIC                        = 1 << 1;
        const AFFECTS_LIGHTMAPPED_MESH_DIFFUSE  = 1 << 2;
        const NONE                              = 0;
        const UNINITIALIZED                     = 0xFFFF;
    }
}

#[derive(Component)]
pub struct ExtractedPointLight {
    pub color: LinearRgba,
    /// luminous intensity in lumens per steradian
    pub intensity: f32,
    pub range: f32,
    pub radius: f32,
    pub transform: GlobalTransform,
    pub shadows_enabled: bool,
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
    pub shadow_map_near_z: f32,
    pub spot_light_angles: Option<(f32, f32)>,
    pub volumetric: bool,
    pub soft_shadows_enabled: bool,
    /// whether this point light contributes diffuse light to lightmapped meshes
    pub affects_lightmapped_mesh_diffuse: bool,
}

#[derive(Component, Debug)]
pub struct ExtractedDirectionalLight {
    pub color: LinearRgba,
    pub illuminance: f32,
    pub transform: GlobalTransform,
    pub shadows_enabled: bool,
    pub volumetric: bool,
    /// whether this directional light contributes diffuse light to lightmapped
    /// meshes
    pub affects_lightmapped_mesh_diffuse: bool,
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
    pub cascade_shadow_config: CascadeShadowConfig,
    pub cascades: EntityHashMap<Vec<Cascade>>,
    pub frusta: EntityHashMap<Vec<Frustum>>,
    pub render_layers: RenderLayers,
    pub soft_shadow_size: Option<f32>,
    /// True if this light is using two-phase occlusion culling.
    pub occlusion_culling: bool,
}

#[derive(Copy, Clone, Debug, ShaderType)]
pub struct GpuLights {
    pub directional_lights: [GpuDirectionalLight; MAX_DIRECTIONAL_LIGHTS],
    pub ambient_color: Vec4,
    // xyz are x/y/z cluster dimensions and w is the number of clusters
    pub cluster_dimensions: UVec4,
    // xy are vec2<f32>(cluster_dimensions.xy) / vec2<f32>(view.width, view.height)
    // z is cluster_dimensions.z / log(far / near)
    // w is cluster_dimensions.z * log(near) / log(far / near)
    pub cluster_factors: Vec4,
    pub n_directional_lights: u32,
    // offset from spot light's light index to spot light's shadow map index
    pub spot_light_shadowmap_offset: i32,
    pub ambient_light_affects_lightmapped_meshes: u32,
}

#[derive(Copy, Clone, ShaderType, Default, Debug)]
pub struct GpuDirectionalLight {
    pub cascades: [GpuDirectionalCascade; MAX_CASCADES_PER_LIGHT],
    pub color: Vec4,
    pub dir_to_light: Vec3,
    pub flags: u32,
    pub soft_shadow_size: f32,
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
    pub num_cascades: u32,
    pub cascades_overlap_proportion: f32,
    pub depth_texture_base_index: u32,
    pub skip: u32,
}

#[derive(Copy, Clone, ShaderType, Default, Debug)]
pub struct GpuDirectionalCascade {
    pub clip_from_world: Mat4,
    pub texel_size: f32,
    pub far_bound: f32,
}

#[derive(Component, Default, Deref, DerefMut)]
/// Component automatically attached to a light entity to track light-view entities
/// for each view.
pub struct LightViewEntities(pub EntityHashMap<Vec<Entity>>);

#[derive(Resource, Deref, DerefMut, Default, Debug, Clone)]
pub struct LightKeyCache(HashMap<RetainedViewEntity, MeshPipelineKey>);

#[derive(Resource, Deref, DerefMut, Default, Debug, Clone)]
pub struct LightSpecializationTicks(HashMap<RetainedViewEntity, Tick>);

#[derive(Component)]
pub enum LightEntity {
    Directional {
        light_entity: Entity,
        cascade_index: usize,
    },
    Point {
        light_entity: Entity,
        face_index: usize,
    },
    Spot {
        light_entity: Entity,
    },
}

#[derive(Resource, Default)]
pub struct LightMeta {
    pub view_gpu_lights: DynamicUniformBuffer<GpuLights>,
}

#[derive(Component, Clone, Debug, Default, Reflect, Deref, DerefMut)]
#[reflect(Component, Debug, Default, Clone)]
pub struct RenderVisibleMeshEntities {
    #[reflect(ignore, clone)]
    pub entities: Vec<(Entity, MainEntity)>,
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct RenderCascadesVisibleEntities {
    /// Map of view entity to the visible entities for each cascade frustum.
    #[reflect(ignore, clone)]
    pub entities: EntityHashMap<Vec<RenderVisibleMeshEntities>>,
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct RenderCubemapVisibleEntities {
    #[reflect(ignore, clone)]
    pub(crate) data: [RenderVisibleMeshEntities; 6],
}

impl RenderCubemapVisibleEntities {
    pub fn get(&self, i: usize) -> &RenderVisibleMeshEntities {
        &self.data[i]
    }

    pub fn get_mut(&mut self, i: usize) -> &mut RenderVisibleMeshEntities {
        &mut self.data[i]
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &RenderVisibleMeshEntities> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl DoubleEndedIterator<Item = &mut RenderVisibleMeshEntities> {
        self.data.iter_mut()
    }
}
