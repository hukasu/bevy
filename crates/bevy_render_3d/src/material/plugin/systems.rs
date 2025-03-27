use core::hash::Hash;

use bevy_asset::prelude::AssetChanged;
use bevy_core_pipeline::{
    core_3d::{
        AlphaMask3d, Opaque3d, Opaque3dBatchSetKey, Opaque3dBinKey, Transmissive3d, Transparent3d,
    },
    prepass::{OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey},
};
use bevy_ecs::{
    change_detection::DetectChangesMut,
    entity::Entity,
    query::{Changed, Or},
    removal_detection::RemovedComponents,
    system::{Query, Res, ResMut, SystemChangeTick},
};
use bevy_platform_support::{collections::HashSet, hash::FixedHasher};
use bevy_render::{
    batching::gpu_preprocessing::GpuPreprocessingSupport,
    mesh::{allocator::MeshAllocator, Mesh3d, RenderMesh},
    render_asset::RenderAssets,
    render_phase::{
        BinnedRenderPhaseType, PhaseItemExtraIndex, ViewBinnedRenderPhases, ViewSortedRenderPhases,
    },
    render_resource::{PipelineCache, SpecializedMeshPipelines},
    renderer::{RenderDevice, RenderQueue},
    sync_world::MainEntity,
    texture::FallbackImage,
    view::{
        ExtractedView, Msaa, RenderVisibilityRanges, RenderVisibleEntities, RetainedViewEntity,
        ViewVisibility,
    },
    Extract,
};

use tracing::error;

use crate::{
    lightmap::RenderLightmaps,
    material::{
        alpha_mode_pipeline_key,
        bindless::FallbackBindlessResources,
        material::{
            MaterialBindGroupAllocator, PreparedMaterial, RenderMaterialInstances, RenderPhaseType,
            SpecializedMaterialPipelineCache,
        },
        MaterialPipeline, MaterialPipelineKey,
    },
    mesh_pipeline::{
        render::{
            instance::{RenderMeshInstanceFlags, RenderMeshInstances},
            pipeline::MeshPipelineKey,
            RenderMeshMaterialIds,
        },
        render_method::OpaqueRendererMethod,
        specialization::{
            EntitiesNeedingSpecialization, EntitySpecializationTicks, ViewSpecializationTicks,
        },
        ViewKeyCache,
    },
};

use super::{Material, MeshMaterial3d};

/// A system that ensures that
/// [`crate::render::mesh::extract_meshes_for_gpu_building`] re-extracts meshes
/// whose materials changed.
///
/// As [`crate::render::mesh::collect_meshes_for_gpu_building`] only considers
/// meshes that were newly extracted, and it writes information from the
/// [`RenderMeshMaterialIds`] into the
/// [`crate::render::mesh::MeshInputUniform`], we must tell
/// [`crate::render::mesh::extract_meshes_for_gpu_building`] to re-extract a
/// mesh if its material changed. Otherwise, the material binding information in
/// the [`crate::render::mesh::MeshInputUniform`] might not be updated properly.
/// The easiest way to ensure that
/// [`crate::render::mesh::extract_meshes_for_gpu_building`] re-extracts a mesh
/// is to mark its [`Mesh3d`] as changed, so that's what this system does.
pub fn mark_meshes_as_changed_if_their_materials_changed<M>(
    mut changed_meshes_query: Query<
        &mut Mesh3d,
        Or<(Changed<MeshMaterial3d<M>>, AssetChanged<MeshMaterial3d<M>>)>,
    >,
) where
    M: Material,
{
    for mut mesh in &mut changed_meshes_query {
        mesh.set_changed();
    }
}

/// For each view, iterates over all the meshes visible from that view and adds
/// them to [`BinnedRenderPhase`]s or [`SortedRenderPhase`]s as appropriate.
pub fn queue_material_meshes<M: Material>(
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mesh_allocator: Res<MeshAllocator>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    mut opaque_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3d>>,
    mut alpha_mask_render_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3d>>,
    mut transmissive_render_phases: ResMut<ViewSortedRenderPhases<Transmissive3d>>,
    mut transparent_render_phases: ResMut<ViewSortedRenderPhases<Transparent3d>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    specialized_material_pipeline_cache: ResMut<SpecializedMaterialPipelineCache<M>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    for (view, visible_entities) in &views {
        let (
            Some(opaque_phase),
            Some(alpha_mask_phase),
            Some(transmissive_phase),
            Some(transparent_phase),
        ) = (
            opaque_render_phases.get_mut(&view.retained_view_entity),
            alpha_mask_render_phases.get_mut(&view.retained_view_entity),
            transmissive_render_phases.get_mut(&view.retained_view_entity),
            transparent_render_phases.get_mut(&view.retained_view_entity),
        )
        else {
            continue;
        };

        let Some(view_specialized_material_pipeline_cache) =
            specialized_material_pipeline_cache.get(&view.retained_view_entity)
        else {
            continue;
        };

        let rangefinder = view.rangefinder3d();
        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            let Some((current_change_tick, pipeline_id)) = view_specialized_material_pipeline_cache
                .get(visible_entity)
                .map(|(current_change_tick, pipeline_id)| (*current_change_tick, *pipeline_id))
            else {
                continue;
            };

            // Skip the entity if it's cached in a bin and up to date.
            if opaque_phase.validate_cached_entity(*visible_entity, current_change_tick)
                || alpha_mask_phase.validate_cached_entity(*visible_entity, current_change_tick)
            {
                continue;
            }

            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };

            // Fetch the slabs that this mesh resides in.
            let (vertex_slab, index_slab) = mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);

            match material.properties.render_phase_type {
                RenderPhaseType::Transmissive => {
                    let distance = rangefinder.distance_translation(&mesh_instance.translation)
                        + material.properties.depth_bias;
                    transmissive_phase.add(Transmissive3d {
                        entity: (*render_entity, *visible_entity),
                        draw_function: material.properties.draw_function_id,
                        pipeline: pipeline_id,
                        distance,
                        batch_range: 0..1,
                        extra_index: PhaseItemExtraIndex::None,
                        indexed: index_slab.is_some(),
                    });
                }
                RenderPhaseType::Opaque => {
                    if material.properties.render_method == OpaqueRendererMethod::Deferred {
                        // Even though we aren't going to insert the entity into
                        // a bin, we still want to update its cache entry. That
                        // way, we know we don't need to re-examine it in future
                        // frames.
                        opaque_phase.update_cache(*visible_entity, None, current_change_tick);
                        continue;
                    }
                    let batch_set_key = Opaque3dBatchSetKey {
                        pipeline: pipeline_id,
                        draw_function: material.properties.draw_function_id,
                        material_bind_group_index: Some(material.binding.group.0),
                        vertex_slab: vertex_slab.unwrap_or_default(),
                        index_slab,
                        lightmap_slab: mesh_instance.shared.lightmap_slab_index.map(|index| *index),
                    };
                    let bin_key = Opaque3dBinKey {
                        asset_id: mesh_instance.mesh_asset_id.into(),
                    };
                    opaque_phase.add(
                        batch_set_key,
                        bin_key,
                        (*render_entity, *visible_entity),
                        mesh_instance.current_uniform_index,
                        BinnedRenderPhaseType::mesh(
                            mesh_instance.should_batch(),
                            &gpu_preprocessing_support,
                        ),
                        current_change_tick,
                    );
                }
                // Alpha mask
                RenderPhaseType::AlphaMask => {
                    let batch_set_key = OpaqueNoLightmap3dBatchSetKey {
                        draw_function: material.properties.draw_function_id,
                        pipeline: pipeline_id,
                        material_bind_group_index: Some(material.binding.group.0),
                        vertex_slab: vertex_slab.unwrap_or_default(),
                        index_slab,
                    };
                    let bin_key = OpaqueNoLightmap3dBinKey {
                        asset_id: mesh_instance.mesh_asset_id.into(),
                    };
                    alpha_mask_phase.add(
                        batch_set_key,
                        bin_key,
                        (*render_entity, *visible_entity),
                        mesh_instance.current_uniform_index,
                        BinnedRenderPhaseType::mesh(
                            mesh_instance.should_batch(),
                            &gpu_preprocessing_support,
                        ),
                        current_change_tick,
                    );
                }
                RenderPhaseType::Transparent => {
                    let distance = rangefinder.distance_translation(&mesh_instance.translation)
                        + material.properties.depth_bias;
                    transparent_phase.add(Transparent3d {
                        entity: (*render_entity, *visible_entity),
                        draw_function: material.properties.draw_function_id,
                        pipeline: pipeline_id,
                        distance,
                        batch_range: 0..1,
                        extra_index: PhaseItemExtraIndex::None,
                        indexed: index_slab.is_some(),
                    });
                }
            }
        }
    }
}

/// Fills the [`RenderMaterialInstances`] and [`RenderMeshMaterialIds`]
/// resources from the meshes in the scene.
pub fn extract_mesh_materials<M: Material>(
    mut material_instances: ResMut<RenderMaterialInstances<M>>,
    mut material_ids: ResMut<RenderMeshMaterialIds>,
    changed_meshes_query: Extract<
        Query<
            (Entity, &ViewVisibility, &MeshMaterial3d<M>),
            Or<(Changed<ViewVisibility>, Changed<MeshMaterial3d<M>>)>,
        >,
    >,
    mut removed_visibilities_query: Extract<RemovedComponents<ViewVisibility>>,
    mut removed_materials_query: Extract<RemovedComponents<MeshMaterial3d<M>>>,
) {
    for (entity, view_visibility, material) in &changed_meshes_query {
        if view_visibility.get() {
            material_instances.insert(entity.into(), material.id());
            material_ids.insert(entity.into(), material.id().into());
        } else {
            material_instances.remove(&MainEntity::from(entity));
            material_ids.remove(entity.into());
        }
    }

    for entity in removed_visibilities_query
        .read()
        .chain(removed_materials_query.read())
    {
        // Only queue a mesh for removal if we didn't pick it up above.
        // It's possible that a necessary component was removed and re-added in
        // the same frame.
        if !changed_meshes_query.contains(entity) {
            material_instances.remove(&MainEntity::from(entity));
            material_ids.remove(entity.into());
        }
    }
}

pub fn extract_entities_needs_specialization<M>(
    entities_needing_specialization: Extract<Res<EntitiesNeedingSpecialization<M>>>,
    mut entity_specialization_ticks: ResMut<EntitySpecializationTicks<M>>,
    ticks: SystemChangeTick,
) where
    M: Material,
{
    for entity in entities_needing_specialization.iter() {
        // Update the entity's specialization tick with this run's tick
        entity_specialization_ticks.insert((*entity).into(), ticks.this_run());
    }
}

pub fn check_entities_needing_specialization<M>(
    needs_specialization: Query<
        Entity,
        Or<(
            Changed<Mesh3d>,
            AssetChanged<Mesh3d>,
            Changed<MeshMaterial3d<M>>,
            AssetChanged<MeshMaterial3d<M>>,
        )>,
    >,
    mut entities_needing_specialization: ResMut<EntitiesNeedingSpecialization<M>>,
) where
    M: Material,
{
    entities_needing_specialization.clear();
    for entity in &needs_specialization {
        entities_needing_specialization.push(entity);
    }
}

pub fn specialize_material_meshes<M: Material>(
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    render_lightmaps: Res<RenderLightmaps>,
    render_visibility_ranges: Res<RenderVisibilityRanges>,
    (
        material_bind_group_allocator,
        opaque_render_phases,
        alpha_mask_render_phases,
        transmissive_render_phases,
        transparent_render_phases,
    ): (
        Res<MaterialBindGroupAllocator<M>>,
        Res<ViewBinnedRenderPhases<Opaque3d>>,
        Res<ViewBinnedRenderPhases<AlphaMask3d>>,
        Res<ViewSortedRenderPhases<Transmissive3d>>,
        Res<ViewSortedRenderPhases<Transparent3d>>,
    ),
    views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    view_key_cache: Res<ViewKeyCache>,
    entity_specialization_ticks: Res<EntitySpecializationTicks<M>>,
    view_specialization_ticks: Res<ViewSpecializationTicks>,
    mut specialized_material_pipeline_cache: ResMut<SpecializedMaterialPipelineCache<M>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<MaterialPipeline<M>>>,
    pipeline: Res<MaterialPipeline<M>>,
    pipeline_cache: Res<PipelineCache>,
    ticks: SystemChangeTick,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    // Record the retained IDs of all shadow views so that we can expire old
    // pipeline IDs.
    let mut all_views: HashSet<RetainedViewEntity, FixedHasher> = HashSet::default();

    for (view, visible_entities) in &views {
        all_views.insert(view.retained_view_entity);

        if !transparent_render_phases.contains_key(&view.retained_view_entity)
            && !opaque_render_phases.contains_key(&view.retained_view_entity)
            && !alpha_mask_render_phases.contains_key(&view.retained_view_entity)
            && !transmissive_render_phases.contains_key(&view.retained_view_entity)
        {
            continue;
        }

        let Some(view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let view_tick = view_specialization_ticks
            .get(&view.retained_view_entity)
            .unwrap();
        let view_specialized_material_pipeline_cache = specialized_material_pipeline_cache
            .entry(view.retained_view_entity)
            .or_default();

        for (_, visible_entity) in visible_entities.iter::<Mesh3d>() {
            let Some(material_asset_id) = render_material_instances.get(visible_entity) else {
                continue;
            };
            let entity_tick = entity_specialization_ticks.get(visible_entity).unwrap();
            let last_specialized_tick = view_specialized_material_pipeline_cache
                .get(visible_entity)
                .map(|(tick, _)| *tick);
            let needs_specialization = last_specialized_tick.is_none_or(|tick| {
                view_tick.is_newer_than(tick, ticks.this_run())
                    || entity_tick.is_newer_than(tick, ticks.this_run())
            });
            if !needs_specialization {
                continue;
            }
            let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(*visible_entity)
            else {
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };
            let Some(material_bind_group) =
                material_bind_group_allocator.get(material.binding.group)
            else {
                continue;
            };

            let mut mesh_pipeline_key_bits = material.properties.mesh_pipeline_key_bits;
            mesh_pipeline_key_bits.insert(alpha_mode_pipeline_key(
                material.properties.alpha_mode,
                &Msaa::from_samples(view_key.msaa_samples()),
            ));
            let mut mesh_key = *view_key
                | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits())
                | mesh_pipeline_key_bits;

            if let Some(lightmap) = render_lightmaps.render_lightmaps.get(visible_entity) {
                mesh_key |= MeshPipelineKey::LIGHTMAPPED;

                if lightmap.bicubic_sampling {
                    mesh_key |= MeshPipelineKey::LIGHTMAP_BICUBIC_SAMPLING;
                }
            }

            if render_visibility_ranges.entity_has_crossfading_visibility_ranges(*visible_entity) {
                mesh_key |= MeshPipelineKey::VISIBILITY_RANGE_DITHER;
            }

            if view_key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
                // If the previous frame have skins or morph targets, note that.
                if mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_SKIN)
                {
                    mesh_key |= MeshPipelineKey::HAS_PREVIOUS_SKIN;
                }
                if mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_MORPH)
                {
                    mesh_key |= MeshPipelineKey::HAS_PREVIOUS_MORPH;
                }
            }

            let key = MaterialPipelineKey {
                mesh_key,
                bind_group_data: material_bind_group
                    .get_extra_data(material.binding.slot)
                    .clone(),
            };
            let pipeline_id = pipelines.specialize(&pipeline_cache, &pipeline, key, &mesh.layout);
            let pipeline_id = match pipeline_id {
                Ok(id) => id,
                Err(err) => {
                    error!("{}", err);
                    continue;
                }
            };

            view_specialized_material_pipeline_cache
                .insert(*visible_entity, (ticks.this_run(), pipeline_id));
        }
    }

    // Delete specialized pipelines belonging to views that have expired.
    specialized_material_pipeline_cache
        .retain(|retained_view_entity, _| all_views.contains(retained_view_entity));
}

/// Creates and/or recreates any bind groups that contain materials that were
/// modified this frame.
pub fn prepare_material_bind_groups<M>(
    mut allocator: ResMut<MaterialBindGroupAllocator<M>>,
    render_device: Res<RenderDevice>,
    fallback_image: Res<FallbackImage>,
    fallback_resources: Res<FallbackBindlessResources>,
) where
    M: Material,
{
    allocator.prepare_bind_groups(&render_device, &fallback_resources, &fallback_image);
}

/// Uploads the contents of all buffers that the [`MaterialBindGroupAllocator`]
/// manages to the GPU.
///
/// Non-bindless allocators don't currently manage any buffers, so this method
/// only has an effect for bindless allocators.
pub fn write_material_bind_group_buffers<M>(
    mut allocator: ResMut<MaterialBindGroupAllocator<M>>,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
) where
    M: Material,
{
    allocator.write_buffers(&render_device, &render_queue);
}
