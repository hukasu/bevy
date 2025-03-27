use core::num::NonZero;

use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Query, Res, ResMut},
};
use bevy_render::{
    render_resource::{
        binding_types, BindGroupLayoutEntryBuilder, SamplerBindingType, TextureSampleType,
    },
    renderer::{RenderAdapter, RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    view::ViewVisibility,
    Extract,
};
use bevy_transform::components::GlobalTransform;

use crate::{cluster::GlobalClusterableObjectMeta, decal::ClusteredDecal};

use super::{
    clustered_decals_are_usable,
    decals::{DecalsBuffer, RenderClusteredDecal, RenderClusteredDecals},
    MAX_VIEW_DECALS,
};

/// Extracts decals from the main world into the render world.
pub fn extract_decals(
    decals: Extract<
        Query<(
            RenderEntity,
            &ClusteredDecal,
            &GlobalTransform,
            &ViewVisibility,
        )>,
    >,
    mut render_decals: ResMut<RenderClusteredDecals>,
) {
    // Clear out the `RenderDecals` in preparation for a new frame.
    render_decals.clear();

    // Loop over each decal.
    for (decal_entity, clustered_decal, global_transform, view_visibility) in &decals {
        // If the decal is invisible, skip it.
        if !view_visibility.get() {
            continue;
        }

        // Insert or add the image.
        let image_index = render_decals.get_or_insert_image(&clustered_decal.image.id());

        // Record the decal.
        let decal_index = render_decals.decals.len();
        render_decals
            .entity_to_decal_index
            .insert(decal_entity, decal_index);

        render_decals.decals.push(RenderClusteredDecal {
            local_from_world: global_transform.affine().inverse().into(),
            image_index,
            tag: clustered_decal.tag,
            pad_a: 0,
            pad_b: 0,
        });
    }
}

/// Adds all decals in the scene to the [`GlobalClusterableObjectMeta`] table.
pub fn prepare_decals(
    decals: Query<Entity, With<ClusteredDecal>>,
    mut global_clusterable_object_meta: ResMut<GlobalClusterableObjectMeta>,
    render_decals: Res<RenderClusteredDecals>,
) {
    for decal_entity in &decals {
        if let Some(index) = render_decals.entity_to_decal_index.get(&decal_entity) {
            global_clusterable_object_meta
                .entity_to_index
                .insert(decal_entity, *index);
        }
    }
}

/// Uploads the list of decals from [`RenderClusteredDecals::decals`] to the
/// GPU.
pub fn upload_decals(
    render_decals: Res<RenderClusteredDecals>,
    mut decals_buffer: ResMut<DecalsBuffer>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) {
    decals_buffer.clear();

    for &decal in &render_decals.decals {
        decals_buffer.push(decal);
    }

    // Make sure the buffer is non-empty.
    // Otherwise there won't be a buffer to bind.
    if decals_buffer.is_empty() {
        decals_buffer.push(RenderClusteredDecal::default());
    }

    decals_buffer.write_buffer(&render_device, &render_queue);
}

/// Returns the layout for the clustered-decal-related bind group entries for a
/// single view.
pub fn get_bind_group_layout_entries(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> Option<[BindGroupLayoutEntryBuilder; 3]> {
    // If binding arrays aren't supported on the current platform, we have no
    // bind group layout entries.
    if !clustered_decals_are_usable(render_device, render_adapter) {
        return None;
    }

    let Some(max_view_decals) = u32::try_from(MAX_VIEW_DECALS)
        .ok()
        .and_then(|max_view_decals| NonZero::<u32>::new(max_view_decals))
    else {
        unreachable!("`MAX_VIEW_DECALS` should never be zero or exceed u32.");
    };

    Some([
        // `decals`
        binding_types::storage_buffer_read_only::<RenderClusteredDecal>(false),
        // `decal_textures`
        binding_types::texture_2d(TextureSampleType::Float { filterable: true })
            .count(max_view_decals),
        // `decal_sampler`
        binding_types::sampler(SamplerBindingType::Filtering),
    ])
}
