use core::{iter, mem};

use bevy_ecs::{
    entity::Entity,
    query::{With, Without},
    system::{Commands, Query, Res, ResMut},
};
use bevy_render::{
    batching::NoAutomaticBatching,
    mesh::morph::{MeshMorphWeights, MAX_MORPH_WEIGHTS},
    render_resource::RawBufferVec,
    renderer::{RenderDevice, RenderQueue},
    view::ViewVisibility,
    Extract,
};

use bytemuck::NoUninit;

use super::data::{MorphIndex, MorphIndices, MorphUniforms};

pub fn prepare_morphs(
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut uniform: ResMut<MorphUniforms>,
) {
    if uniform.current_buffer.is_empty() {
        return;
    }
    let len = uniform.current_buffer.len();
    uniform.current_buffer.reserve(len, &render_device);
    uniform
        .current_buffer
        .write_buffer(&render_device, &render_queue);

    // We don't need to write `uniform.prev_buffer` because we already wrote it
    // last frame, and the data should still be on the GPU.
}

const fn can_align(step: usize, target: usize) -> bool {
    step % target == 0 || target % step == 0
}

const WGPU_MIN_ALIGN: usize = 256;

/// Align a [`RawBufferVec`] to `N` bytes by padding the end with `T::default()` values.
fn add_to_alignment<T: NoUninit + Default>(buffer: &mut RawBufferVec<T>) {
    let n = WGPU_MIN_ALIGN;
    let t_size = size_of::<T>();
    if !can_align(n, t_size) {
        // This panic is stripped at compile time, due to n, t_size and can_align being const
        panic!(
            "RawBufferVec should contain only types with a size multiple or divisible by {n}, \
            {} has a size of {t_size}, which is neither multiple or divisible by {n}",
            core::any::type_name::<T>()
        );
    }

    let buffer_size = buffer.len();
    let byte_size = t_size * buffer_size;
    let bytes_over_n = byte_size % n;
    if bytes_over_n == 0 {
        return;
    }
    let bytes_to_add = n - bytes_over_n;
    let ts_to_add = bytes_to_add / t_size;
    buffer.extend(iter::repeat_with(T::default).take(ts_to_add));
}

// Notes on implementation: see comment on top of the extract_skins system in skin module.
// This works similarly, but for `f32` instead of `Mat4`
pub fn extract_morphs(
    morph_indices: ResMut<MorphIndices>,
    uniform: ResMut<MorphUniforms>,
    query: Extract<Query<(Entity, &ViewVisibility, &MeshMorphWeights)>>,
) {
    // Borrow check workaround.
    let (morph_indices, uniform) = (morph_indices.into_inner(), uniform.into_inner());

    // Swap buffers. We need to keep the previous frame's buffer around for the
    // purposes of motion vector computation.
    mem::swap(&mut morph_indices.current, &mut morph_indices.prev);
    mem::swap(&mut uniform.current_buffer, &mut uniform.prev_buffer);
    morph_indices.current.clear();
    uniform.current_buffer.clear();

    for (entity, view_visibility, morph_weights) in &query {
        if !view_visibility.get() {
            continue;
        }
        let start = uniform.current_buffer.len();
        let weights = morph_weights.weights();
        let legal_weights = weights.iter().take(MAX_MORPH_WEIGHTS).copied();
        uniform.current_buffer.extend(legal_weights);
        add_to_alignment::<f32>(&mut uniform.current_buffer);

        let index = (start * size_of::<f32>()) as u32;
        morph_indices
            .current
            .insert(entity.into(), MorphIndex { index });
    }
}

// NOTE: Because morph targets require per-morph target texture bindings, they cannot
// currently be batched.
pub fn no_automatic_morph_batching(
    mut commands: Commands,
    query: Query<Entity, (With<MeshMorphWeights>, Without<NoAutomaticBatching>)>,
) {
    for entity in &query {
        commands.entity(entity).try_insert(NoAutomaticBatching);
    }
}
