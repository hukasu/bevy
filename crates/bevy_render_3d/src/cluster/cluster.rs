use core::num::NonZero;

use bevy_ecs::component::Component;
use bevy_math::{uvec4, UVec2, UVec3, UVec4, Vec3Swizzles};
use bevy_render::{
    render_resource::{
        BindingResource, BufferBindingType, ShaderType, StorageBuffer, UniformBuffer,
    },
    renderer::{RenderDevice, RenderQueue},
};
use tracing::warn;

use super::clusterable_objects::{
    ClusterableObjectCounts, GpuClusterableObjectIndexListsStorage,
    GpuClusterableObjectIndexListsUniform, VisibleClusterableObjects,
};

#[derive(Component, Debug, Default)]
pub struct Clusters {
    /// Tile size
    pub(crate) tile_size: UVec2,
    /// Number of clusters in `X` / `Y` / `Z` in the view frustum
    pub(crate) dimensions: UVec3,
    /// Distance to the far plane of the first depth slice. The first depth slice is special
    /// and explicitly-configured to avoid having unnecessarily many slices close to the camera.
    pub(crate) near: f32,
    pub(crate) far: f32,
    pub(crate) clusterable_objects: Vec<VisibleClusterableObjects>,
}

impl Clusters {
    pub fn update(&mut self, screen_size: UVec2, requested_dimensions: UVec3) {
        debug_assert!(
            requested_dimensions.x > 0 && requested_dimensions.y > 0 && requested_dimensions.z > 0
        );

        let tile_size = (screen_size.as_vec2() / requested_dimensions.xy().as_vec2())
            .ceil()
            .as_uvec2()
            .max(UVec2::ONE);
        self.tile_size = tile_size;
        self.dimensions = (screen_size.as_vec2() / tile_size.as_vec2())
            .ceil()
            .as_uvec2()
            .extend(requested_dimensions.z)
            .max(UVec3::ONE);

        // NOTE: Maximum 4096 clusters due to uniform buffer size constraints
        debug_assert!(self.dimensions.x * self.dimensions.y * self.dimensions.z <= 4096);
    }

    pub fn clear(&mut self) {
        self.tile_size = UVec2::ONE;
        self.dimensions = UVec3::ZERO;
        self.near = 0.0;
        self.far = 0.0;
        self.clusterable_objects.clear();
    }
}

#[derive(Component)]
pub struct ExtractedClusterConfig {
    /// Special near value for cluster calculations
    pub(crate) near: f32,
    pub(crate) far: f32,
    /// Number of clusters in `X` / `Y` / `Z` in the view frustum
    pub(crate) dimensions: UVec3,
}

// this must match CLUSTER_COUNT_SIZE in pbr.wgsl
// and must be large enough to contain MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS
const CLUSTER_COUNT_SIZE: u32 = 9;
const CLUSTER_OFFSET_MASK: u32 = (1 << (32 - (CLUSTER_COUNT_SIZE * 2))) - 1;
const CLUSTER_COUNT_MASK: u32 = (1 << CLUSTER_COUNT_SIZE) - 1;

#[derive(Component)]
pub struct ViewClusterBindings {
    n_indices: usize,
    n_offsets: usize,
    buffers: ViewClusterBuffers,
}

impl ViewClusterBindings {
    pub const MAX_OFFSETS: usize = 16384 / 4;
    pub const MAX_UNIFORM_ITEMS: usize = Self::MAX_OFFSETS / 4;
    pub const MAX_INDICES: usize = 16384;

    pub fn new(buffer_binding_type: BufferBindingType) -> Self {
        Self {
            n_indices: 0,
            n_offsets: 0,
            buffers: ViewClusterBuffers::new(buffer_binding_type),
        }
    }

    pub fn clear(&mut self) {
        match &mut self.buffers {
            ViewClusterBuffers::Uniform {
                clusterable_object_index_lists,
                cluster_offsets_and_counts,
            } => {
                *clusterable_object_index_lists.get_mut().data =
                    [UVec4::ZERO; Self::MAX_UNIFORM_ITEMS];
                *cluster_offsets_and_counts.get_mut().data = [UVec4::ZERO; Self::MAX_UNIFORM_ITEMS];
            }
            ViewClusterBuffers::Storage {
                clusterable_object_index_lists,
                cluster_offsets_and_counts,
                ..
            } => {
                clusterable_object_index_lists.get_mut().data.clear();
                cluster_offsets_and_counts.get_mut().data.clear();
            }
        }
    }

    pub fn push_offset_and_counts(&mut self, offset: usize, counts: &ClusterableObjectCounts) {
        match &mut self.buffers {
            ViewClusterBuffers::Uniform {
                cluster_offsets_and_counts,
                ..
            } => {
                let array_index = self.n_offsets >> 2; // >> 2 is equivalent to / 4
                if array_index >= Self::MAX_UNIFORM_ITEMS {
                    warn!("cluster offset and count out of bounds!");
                    return;
                }
                let component = self.n_offsets & ((1 << 2) - 1);
                let packed =
                    pack_offset_and_counts(offset, counts.point_lights, counts.spot_lights);

                cluster_offsets_and_counts.get_mut().data[array_index][component] = packed;
            }
            ViewClusterBuffers::Storage {
                cluster_offsets_and_counts,
                ..
            } => {
                cluster_offsets_and_counts.get_mut().data.push([
                    uvec4(
                        offset as u32,
                        counts.point_lights,
                        counts.spot_lights,
                        counts.reflection_probes,
                    ),
                    uvec4(counts.irradiance_volumes, counts.decals, 0, 0),
                ]);
            }
        }

        self.n_offsets += 1;
    }

    pub fn n_indices(&self) -> usize {
        self.n_indices
    }

    pub fn push_index(&mut self, index: usize) {
        match &mut self.buffers {
            ViewClusterBuffers::Uniform {
                clusterable_object_index_lists,
                ..
            } => {
                let array_index = self.n_indices >> 4; // >> 4 is equivalent to / 16
                let component = (self.n_indices >> 2) & ((1 << 2) - 1);
                let sub_index = self.n_indices & ((1 << 2) - 1);
                let index = index as u32;

                clusterable_object_index_lists.get_mut().data[array_index][component] |=
                    index << (8 * sub_index);
            }
            ViewClusterBuffers::Storage {
                clusterable_object_index_lists,
                ..
            } => {
                clusterable_object_index_lists
                    .get_mut()
                    .data
                    .push(index as u32);
            }
        }

        self.n_indices += 1;
    }

    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        match &mut self.buffers {
            ViewClusterBuffers::Uniform {
                clusterable_object_index_lists,
                cluster_offsets_and_counts,
            } => {
                clusterable_object_index_lists.write_buffer(render_device, render_queue);
                cluster_offsets_and_counts.write_buffer(render_device, render_queue);
            }
            ViewClusterBuffers::Storage {
                clusterable_object_index_lists,
                cluster_offsets_and_counts,
            } => {
                clusterable_object_index_lists.write_buffer(render_device, render_queue);
                cluster_offsets_and_counts.write_buffer(render_device, render_queue);
            }
        }
    }

    pub fn clusterable_object_index_lists_binding(&self) -> Option<BindingResource> {
        match &self.buffers {
            ViewClusterBuffers::Uniform {
                clusterable_object_index_lists,
                ..
            } => clusterable_object_index_lists.binding(),
            ViewClusterBuffers::Storage {
                clusterable_object_index_lists,
                ..
            } => clusterable_object_index_lists.binding(),
        }
    }

    pub fn offsets_and_counts_binding(&self) -> Option<BindingResource> {
        match &self.buffers {
            ViewClusterBuffers::Uniform {
                cluster_offsets_and_counts,
                ..
            } => cluster_offsets_and_counts.binding(),
            ViewClusterBuffers::Storage {
                cluster_offsets_and_counts,
                ..
            } => cluster_offsets_and_counts.binding(),
        }
    }

    pub fn min_size_clusterable_object_index_lists(
        buffer_binding_type: BufferBindingType,
    ) -> NonZero<u64> {
        match buffer_binding_type {
            BufferBindingType::Storage { .. } => GpuClusterableObjectIndexListsStorage::min_size(),
            BufferBindingType::Uniform => GpuClusterableObjectIndexListsUniform::min_size(),
        }
    }

    pub fn min_size_cluster_offsets_and_counts(
        buffer_binding_type: BufferBindingType,
    ) -> NonZero<u64> {
        match buffer_binding_type {
            BufferBindingType::Storage { .. } => GpuClusterOffsetsAndCountsStorage::min_size(),
            BufferBindingType::Uniform => GpuClusterOffsetsAndCountsUniform::min_size(),
        }
    }
}

enum ViewClusterBuffers {
    Uniform {
        // NOTE: UVec4 is because all arrays in Std140 layout have 16-byte alignment
        clusterable_object_index_lists: UniformBuffer<GpuClusterableObjectIndexListsUniform>,
        // NOTE: UVec4 is because all arrays in Std140 layout have 16-byte alignment
        cluster_offsets_and_counts: UniformBuffer<GpuClusterOffsetsAndCountsUniform>,
    },
    Storage {
        clusterable_object_index_lists: StorageBuffer<GpuClusterableObjectIndexListsStorage>,
        cluster_offsets_and_counts: StorageBuffer<GpuClusterOffsetsAndCountsStorage>,
    },
}

impl ViewClusterBuffers {
    fn new(buffer_binding_type: BufferBindingType) -> Self {
        match buffer_binding_type {
            BufferBindingType::Storage { .. } => Self::storage(),
            BufferBindingType::Uniform => Self::uniform(),
        }
    }

    fn uniform() -> Self {
        ViewClusterBuffers::Uniform {
            clusterable_object_index_lists: UniformBuffer::default(),
            cluster_offsets_and_counts: UniformBuffer::default(),
        }
    }

    fn storage() -> Self {
        ViewClusterBuffers::Storage {
            clusterable_object_index_lists: StorageBuffer::default(),
            cluster_offsets_and_counts: StorageBuffer::default(),
        }
    }
}

#[derive(ShaderType, Default)]
pub struct GpuClusterOffsetsAndCountsStorage {
    /// The starting offset, followed by the number of point lights, spot
    /// lights, reflection probes, and irradiance volumes in each cluster, in
    /// that order. The remaining fields are filled with zeroes.
    #[size(runtime)]
    pub data: Vec<[UVec4; 2]>,
}

#[derive(ShaderType)]
pub struct GpuClusterOffsetsAndCountsUniform {
    pub data: Box<[UVec4; ViewClusterBindings::MAX_UNIFORM_ITEMS]>,
}

impl Default for GpuClusterOffsetsAndCountsUniform {
    fn default() -> Self {
        Self {
            data: Box::new([UVec4::ZERO; ViewClusterBindings::MAX_UNIFORM_ITEMS]),
        }
    }
}

// Compresses the offset and counts of point and spot lights so that they fit in
// a UBO.
//
// This function is only used if storage buffers are unavailable on this
// platform: typically, on WebGL 2.
//
// NOTE: With uniform buffer max binding size as 16384 bytes
// that means we can fit 204 clusterable objects in one uniform
// buffer, which means the count can be at most 204 so it
// needs 9 bits.
// The array of indices can also use u8 and that means the
// offset in to the array of indices needs to be able to address
// 16384 values. log2(16384) = 14 bits.
// We use 32 bits to store the offset and counts so
// we pack the offset into the upper 14 bits of a u32,
// the point light count into bits 9-17, and the spot light count into bits 0-8.
//  [ 31     ..     18 | 17      ..      9 | 8       ..     0 ]
//  [      offset      | point light count | spot light count ]
//
// NOTE: This assumes CPU and GPU endianness are the same which is true
// for all common and tested x86/ARM CPUs and AMD/NVIDIA/Intel/Apple/etc GPUs
//
// NOTE: On platforms that use this function, we don't cluster light probes, so
// the number of light probes is irrelevant.
fn pack_offset_and_counts(offset: usize, point_count: u32, spot_count: u32) -> u32 {
    ((offset as u32 & CLUSTER_OFFSET_MASK) << (CLUSTER_COUNT_SIZE * 2))
        | ((point_count & CLUSTER_COUNT_MASK) << CLUSTER_COUNT_SIZE)
        | (spot_count & CLUSTER_COUNT_MASK)
}
