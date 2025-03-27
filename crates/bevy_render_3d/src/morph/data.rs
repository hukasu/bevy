use bevy_ecs::{component::Component, resource::Resource};
use bevy_render::{
    render_resource::{BufferUsages, RawBufferVec},
    sync_world::MainEntityHashMap,
};

#[derive(Component)]
pub struct MorphIndex {
    pub(super) index: u32,
}

/// Maps each mesh affected by morph targets to the applicable offset within the
/// [`MorphUniforms`] buffer.
///
/// We store both the current frame's mapping and the previous frame's mapping
/// for the purposes of motion vector calculation.
#[derive(Default, Resource)]
pub struct MorphIndices {
    /// Maps each entity with a morphed mesh to the appropriate offset within
    /// [`MorphUniforms::current_buffer`].
    pub current: MainEntityHashMap<MorphIndex>,

    /// Maps each entity with a morphed mesh to the appropriate offset within
    /// [`MorphUniforms::prev_buffer`].
    pub prev: MainEntityHashMap<MorphIndex>,
}

/// The GPU buffers containing morph weights for all meshes with morph targets.
///
/// This is double-buffered: we store the weights of the previous frame in
/// addition to those of the current frame. This is for motion vector
/// calculation. Every frame, we swap buffers and reuse the morph target weight
/// buffer from two frames ago for the current frame.
#[derive(Resource)]
pub struct MorphUniforms {
    /// The morph weights for the current frame.
    pub current_buffer: RawBufferVec<f32>,
    /// The morph weights for the previous frame.
    pub prev_buffer: RawBufferVec<f32>,
}

impl Default for MorphUniforms {
    fn default() -> Self {
        Self {
            current_buffer: RawBufferVec::new(BufferUsages::UNIFORM),
            prev_buffer: RawBufferVec::new(BufferUsages::UNIFORM),
        }
    }
}
