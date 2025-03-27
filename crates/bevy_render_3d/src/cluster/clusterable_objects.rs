use core::{cmp::Reverse, num::NonZero};

use bevy_ecs::{
    component::Component,
    entity::{hash_map::EntityHashMap, Entity},
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_math::{UVec4, Vec4};
use bevy_platform_support::{collections::HashSet, prelude::Box};
use bevy_render::{
    primitives::Sphere,
    render_resource::{
        BindingResource, BufferBindingType, ShaderSize, ShaderType, StorageBuffer, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
    view::RenderLayers,
};
use bevy_transform::components::GlobalTransform;

use super::{
    cluster::ViewClusterBindings, plugin::ClusterAssignableCullMethod,
    CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT, MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS,
};

#[derive(Clone, Component, Debug, Default)]
pub struct VisibleClusterableObjects {
    pub entities: Vec<Entity>,
    pub counts: ClusterableObjectCounts,
}

impl VisibleClusterableObjects {
    #[inline]
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Entity> {
        self.entities.iter()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }
}

#[derive(Resource, Default)]
pub struct GlobalVisibleClusterableObjects {
    pub(crate) entities: HashSet<Entity>,
}

impl GlobalVisibleClusterableObjects {
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &Entity> {
        self.entities.iter()
    }

    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.entities.contains(&entity)
    }
}

#[derive(Component)]
pub struct ExtractedClusterableObjects {
    pub data: Vec<ExtractedClusterableObjectElement>,
}
pub enum ExtractedClusterableObjectElement {
    ClusterHeader(ClusterableObjectCounts),
    ClusterableObjectEntity(Entity),
}

pub enum GpuClusterableObjects {
    Uniform(UniformBuffer<GpuClusterableObjectsUniform>),
    Storage(StorageBuffer<GpuClusterableObjectsStorage>),
}

impl GpuClusterableObjects {
    fn new(buffer_binding_type: BufferBindingType) -> Self {
        match buffer_binding_type {
            BufferBindingType::Storage { .. } => Self::storage(),
            BufferBindingType::Uniform => Self::uniform(),
        }
    }

    fn uniform() -> Self {
        Self::Uniform(UniformBuffer::default())
    }

    fn storage() -> Self {
        Self::Storage(StorageBuffer::default())
    }

    pub(crate) fn set(&mut self, mut clusterable_objects: Vec<GpuClusterableObject>) {
        match self {
            GpuClusterableObjects::Uniform(buffer) => {
                let len = clusterable_objects
                    .len()
                    .min(MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS);
                let src = &clusterable_objects[..len];
                let dst = &mut buffer.get_mut().data[..len];
                dst.copy_from_slice(src);
            }
            GpuClusterableObjects::Storage(buffer) => {
                buffer.get_mut().data.clear();
                buffer.get_mut().data.append(&mut clusterable_objects);
            }
        }
    }

    pub(crate) fn write_buffer(
        &mut self,
        render_device: &RenderDevice,
        render_queue: &RenderQueue,
    ) {
        match self {
            GpuClusterableObjects::Uniform(buffer) => {
                buffer.write_buffer(render_device, render_queue);
            }
            GpuClusterableObjects::Storage(buffer) => {
                buffer.write_buffer(render_device, render_queue);
            }
        }
    }

    pub fn binding(&self) -> Option<BindingResource> {
        match self {
            GpuClusterableObjects::Uniform(buffer) => buffer.binding(),
            GpuClusterableObjects::Storage(buffer) => buffer.binding(),
        }
    }

    pub fn min_size(buffer_binding_type: BufferBindingType) -> NonZero<u64> {
        match buffer_binding_type {
            BufferBindingType::Storage { .. } => GpuClusterableObjectsStorage::min_size(),
            BufferBindingType::Uniform => GpuClusterableObjectsUniform::min_size(),
        }
    }
}

#[derive(Copy, Clone, ShaderType, Default, Debug)]
pub struct GpuClusterableObject {
    // For point lights: the lower-right 2x2 values of the projection matrix [2][2] [2][3] [3][2] [3][3]
    // For spot lights: 2 components of the direction (x,z), spot_scale and spot_offset
    pub(crate) light_custom_data: Vec4,
    pub(crate) color_inverse_square_range: Vec4,
    pub(crate) position_radius: Vec4,
    pub(crate) flags: u32,
    pub(crate) shadow_depth_bias: f32,
    pub(crate) shadow_normal_bias: f32,
    pub(crate) spot_light_tan_angle: f32,
    pub(crate) soft_shadow_size: f32,
    pub(crate) shadow_map_near_z: f32,
    pub(crate) pad_a: f32,
    pub(crate) pad_b: f32,
}

// Make sure that the clusterable object buffer doesn't overflow the maximum
// size of a UBO on WebGL 2.
const _: () =
    assert!(size_of::<GpuClusterableObject>() * MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS <= 16384);

#[derive(ShaderType)]
pub struct GpuClusterableObjectsUniform {
    data: Box<[GpuClusterableObject; MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS]>,
}

impl Default for GpuClusterableObjectsUniform {
    fn default() -> Self {
        Self {
            data: Box::new(
                [GpuClusterableObject::default(); MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS],
            ),
        }
    }
}

#[derive(ShaderType, Default)]
pub struct GpuClusterableObjectIndexListsStorage {
    #[size(runtime)]
    pub data: Vec<u32>,
}

#[derive(ShaderType)]
pub struct GpuClusterableObjectIndexListsUniform {
    pub data: Box<[UVec4; ViewClusterBindings::MAX_UNIFORM_ITEMS]>,
}

// NOTE: Assert at compile time that GpuClusterableObjectIndexListsUniform
// fits within the maximum uniform buffer binding size
const _: () = assert!(GpuClusterableObjectIndexListsUniform::SHADER_SIZE.get() <= 16384);

impl Default for GpuClusterableObjectIndexListsUniform {
    fn default() -> Self {
        Self {
            data: Box::new([UVec4::ZERO; ViewClusterBindings::MAX_UNIFORM_ITEMS]),
        }
    }
}

#[derive(ShaderType, Default)]
pub struct GpuClusterableObjectsStorage {
    #[size(runtime)]
    data: Vec<GpuClusterableObject>,
}

/// Data required for assigning objects to clusters.
pub struct ClusterableObjectAssignmentData {
    pub entity: Entity,
    // TODO: We currently ignore the scale on the transform. This is confusing.
    // Replace with an `Isometry3d`.
    pub transform: GlobalTransform,
    pub range: f32,
    pub object_key: ClusterableObjectKey,
    pub render_layers: RenderLayers,
    pub cull_method: Option<Box<dyn ClusterAssignableCullMethod>>,
}

impl ClusterableObjectAssignmentData {
    pub fn sphere(&self) -> Sphere {
        Sphere {
            center: self.transform.translation_vec3a(),
            radius: self.range,
        }
    }
}

/// Data needed to assign objects to clusters that's specific to the type of
/// clusterable object.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterableObjectKey {
    pub order: u8,
    pub shadows_enabled: bool,
    pub volumetric: bool,
}

impl PartialOrd for ClusterableObjectKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ClusterableObjectKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (
            self.order,
            Reverse(self.shadows_enabled),
            Reverse(self.volumetric),
        )
            .cmp(&(
                other.order,
                Reverse(other.shadows_enabled),
                Reverse(other.volumetric),
            ))
    }
}

#[derive(Resource)]
pub struct GlobalClusterableObjectMeta {
    pub gpu_clusterable_objects: GpuClusterableObjects,
    pub entity_to_index: EntityHashMap<usize>,
}

impl GlobalClusterableObjectMeta {
    pub fn new(buffer_binding_type: BufferBindingType) -> Self {
        Self {
            gpu_clusterable_objects: GpuClusterableObjects::new(buffer_binding_type),
            entity_to_index: EntityHashMap::default(),
        }
    }
}

impl FromWorld for GlobalClusterableObjectMeta {
    fn from_world(world: &mut World) -> Self {
        Self::new(
            world
                .resource::<RenderDevice>()
                .get_supported_read_only_binding_type(CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT),
        )
    }
}

/// Stores the number of each type of clusterable object in a single cluster.
///
/// Note that `reflection_probes` and `irradiance_volumes` won't be clustered if
/// fewer than 3 SSBOs are available, which usually means on WebGL 2.
#[derive(Clone, Copy, Default, Debug)]
pub struct ClusterableObjectCounts {
    /// The number of point lights in the cluster.
    pub point_lights: u32,
    /// The number of spot lights in the cluster.
    pub spot_lights: u32,
    /// The number of reflection probes in the cluster.
    pub reflection_probes: u32,
    /// The number of irradiance volumes in the cluster.
    pub irradiance_volumes: u32,
    /// The number of decals in the cluster.
    pub decals: u32,
}
