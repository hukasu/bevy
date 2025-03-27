use std::sync::OnceLock;

use bevy_ecs::{
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_math::Mat4;
use bevy_render::{
    render_resource::{Buffer, BufferDescriptor, BufferUsages},
    renderer::RenderDevice,
    sync_world::{MainEntity, MainEntityHashMap},
};
use offset_allocator::{Allocation, Allocator};
use smallvec::SmallVec;

use crate::skin::JOINTS_PER_ALLOCATION_UNIT;

use super::{MAX_JOINTS, MAX_TOTAL_JOINTS};

/// The GPU buffers containing joint matrices for all skinned meshes.
///
/// This is double-buffered: we store the joint matrices of each mesh for the
/// previous frame in addition to those of each mesh for the current frame. This
/// is for motion vector calculation. Every frame, we swap buffers and overwrite
/// the joint matrix buffer from two frames ago with the data for the current
/// frame.
///
/// Notes on implementation: see comment on top of the `extract_skins` system.
#[derive(Resource)]
pub struct SkinUniforms {
    /// The CPU-side buffer that stores the joint matrices for skinned meshes in
    /// the current frame.
    pub current_staging_buffer: Vec<Mat4>,
    /// The GPU-side buffer that stores the joint matrices for skinned meshes in
    /// the current frame.
    pub current_buffer: Buffer,
    /// The GPU-side buffer that stores the joint matrices for skinned meshes in
    /// the previous frame.
    pub prev_buffer: Buffer,
    /// The offset allocator that manages the placement of the joints within the
    /// [`Self::current_buffer`].
    pub(super) allocator: Allocator,
    /// Allocation information that we keep about each skin.
    pub(super) skin_uniform_info: MainEntityHashMap<SkinUniformInfo>,
    /// Maps each joint entity to the skins it's associated with.
    ///
    /// We use this in conjunction with change detection to only update the
    /// skins that need updating each frame.
    ///
    /// Note that conceptually this is a hash map of sets, but we use a
    /// [`SmallVec`] to avoid allocations for the vast majority of the cases in
    /// which each bone belongs to exactly one skin.
    pub(super) joint_to_skins: MainEntityHashMap<SmallVec<[MainEntity; 1]>>,
    /// The total number of joints in the scene.
    ///
    /// We use this as part of our heuristic to decide whether to use
    /// fine-grained change detection.
    pub(super) total_joints: usize,
}

impl FromWorld for SkinUniforms {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let buffer_usages = (if skins_use_uniform_buffers(device) {
            BufferUsages::UNIFORM
        } else {
            BufferUsages::STORAGE
        }) | BufferUsages::COPY_DST;

        // Create the current and previous buffer with the minimum sizes.
        //
        // These will be swapped every frame.
        let current_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("skin uniform buffer"),
            size: MAX_JOINTS as u64 * size_of::<Mat4>() as u64,
            usage: buffer_usages,
            mapped_at_creation: false,
        });
        let prev_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("skin uniform buffer"),
            size: MAX_JOINTS as u64 * size_of::<Mat4>() as u64,
            usage: buffer_usages,
            mapped_at_creation: false,
        });

        Self {
            current_staging_buffer: vec![],
            current_buffer,
            prev_buffer,
            allocator: Allocator::new(MAX_TOTAL_JOINTS),
            skin_uniform_info: MainEntityHashMap::default(),
            joint_to_skins: MainEntityHashMap::default(),
            total_joints: 0,
        }
    }
}

impl SkinUniforms {
    /// Returns the current offset in joints of the skin in the buffer.
    pub fn skin_index(&self, skin: MainEntity) -> Option<u32> {
        self.skin_uniform_info
            .get(&skin)
            .map(SkinUniformInfo::offset)
    }

    /// Returns the current offset in bytes of the skin in the buffer.
    pub fn skin_byte_offset(&self, skin: MainEntity) -> Option<SkinByteOffset> {
        self.skin_uniform_info.get(&skin).map(|skin_uniform_info| {
            SkinByteOffset::from_index(skin_uniform_info.offset() as usize)
        })
    }

    /// Returns an iterator over all skins in the scene.
    pub fn all_skins(&self) -> impl Iterator<Item = &MainEntity> {
        self.skin_uniform_info.keys()
    }
}

/// Allocation information about each skin.
pub struct SkinUniformInfo {
    /// The allocation of the joints within the [`SkinUniforms::current_buffer`].
    pub allocation: Allocation,
    /// The entities that comprise the joints.
    pub joints: Vec<MainEntity>,
}

impl SkinUniformInfo {
    /// The offset in joints within the [`SkinUniforms::current_staging_buffer`].
    pub(super) fn offset(&self) -> u32 {
        self.allocation.offset * JOINTS_PER_ALLOCATION_UNIT
    }
}

/// The location of the first joint matrix in the skin uniform buffer.
#[derive(Clone, Copy)]
pub struct SkinByteOffset {
    /// The byte offset of the first joint matrix.
    pub byte_offset: u32,
}

impl SkinByteOffset {
    /// Index to be in address space based on the size of a skin uniform.
    const fn from_index(index: usize) -> Self {
        SkinByteOffset {
            byte_offset: (index * size_of::<Mat4>()) as u32,
        }
    }

    /// Returns this skin index in elements (not bytes).
    ///
    /// Each element is a 4x4 matrix.
    pub fn index(&self) -> u32 {
        self.byte_offset / size_of::<Mat4>() as u32
    }
}

/// Returns true if skinning must use uniforms (and dynamic offsets) because
/// storage buffers aren't supported on the current platform.
pub fn skins_use_uniform_buffers(render_device: &RenderDevice) -> bool {
    static SKINS_USE_UNIFORM_BUFFERS: OnceLock<bool> = OnceLock::new();
    *SKINS_USE_UNIFORM_BUFFERS
        .get_or_init(|| render_device.limits().max_storage_buffers_per_shader_stage == 0)
}
