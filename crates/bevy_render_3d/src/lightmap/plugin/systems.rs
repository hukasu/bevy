use bevy_ecs::{
    entity::Entity,
    query::{Changed, Or},
    removal_detection::RemovedComponents,
    system::{Query, Res, ResMut},
};
use bevy_render::{
    render_asset::RenderAssets,
    sync_world::MainEntity,
    texture::{FallbackImage, GpuImage},
    view::ViewVisibility,
    Extract,
};
use tracing::error;

use crate::lightmap::{
    lightmap::{RenderLightmap, RenderLightmaps},
    Lightmap,
};

/// Extracts all lightmaps from the scene and populates the [`RenderLightmaps`]
/// resource.
pub fn extract_lightmaps(
    render_lightmaps: ResMut<RenderLightmaps>,
    changed_lightmaps_query: Extract<
        Query<
            (Entity, &ViewVisibility, &Lightmap),
            Or<(Changed<ViewVisibility>, Changed<Lightmap>)>,
        >,
    >,
    mut removed_lightmaps_query: Extract<RemovedComponents<Lightmap>>,
    images: Res<RenderAssets<GpuImage>>,
    fallback_images: Res<FallbackImage>,
) {
    let render_lightmaps = render_lightmaps.into_inner();

    // Loop over each entity.
    for (entity, view_visibility, lightmap) in changed_lightmaps_query.iter() {
        if render_lightmaps
            .render_lightmaps
            .contains_key(&MainEntity::from(entity))
        {
            continue;
        }

        // Only process visible entities.
        if !view_visibility.get() {
            continue;
        }

        let (slab_index, slot_index) =
            render_lightmaps.allocate(&fallback_images, lightmap.image.id());
        render_lightmaps.render_lightmaps.insert(
            entity.into(),
            RenderLightmap::new(
                lightmap.uv_rect,
                slab_index,
                slot_index,
                lightmap.bicubic_sampling,
            ),
        );

        render_lightmaps
            .pending_lightmaps
            .insert((slab_index, slot_index));
    }

    for entity in removed_lightmaps_query.read() {
        if changed_lightmaps_query.contains(entity) {
            continue;
        }

        let Some(RenderLightmap {
            slab_index,
            slot_index,
            ..
        }) = render_lightmaps
            .render_lightmaps
            .remove(&MainEntity::from(entity))
        else {
            continue;
        };

        render_lightmaps.remove(&fallback_images, slab_index, slot_index);
        render_lightmaps
            .pending_lightmaps
            .remove(&(slab_index, slot_index));
    }

    render_lightmaps
        .pending_lightmaps
        .retain(|&(slab_index, slot_index)| {
            let Some(asset_id) = render_lightmaps.slabs[usize::from(slab_index)].lightmaps
                [usize::from(slot_index)]
            .asset_id
            else {
                error!(
                    "Allocated lightmap should have been removed from `pending_lightmaps` by now"
                );
                return false;
            };

            let Some(gpu_image) = images.get(asset_id) else {
                return true;
            };
            render_lightmaps.slabs[usize::from(slab_index)].insert(slot_index, gpu_image.clone());
            false
        });
}
