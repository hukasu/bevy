use std::hash::Hash;

use bevy_ecs::{
    entity::Entity,
    query::With,
    system::{Query, Res, ResMut, SystemChangeTick},
};
use bevy_platform_support::{collections::HashSet, hash::FixedHasher};
use bevy_render::{
    alpha::AlphaMode,
    batching::gpu_preprocessing::GpuPreprocessingSupport,
    mesh::{allocator::MeshAllocator, RenderMesh},
    render_asset::RenderAssets,
    render_phase::{BinnedRenderPhaseType, DrawFunctions, ViewBinnedRenderPhases},
    render_resource::{PipelineCache, SpecializedMeshPipelines},
    view::{ExtractedView, RetainedViewEntity},
};
use tracing::error;

use crate::{
    light::{
        render::{
            ExtractedDirectionalLight, ExtractedPointLight, LightEntity, LightKeyCache,
            LightSpecializationTicks, RenderCascadesVisibleEntities, RenderCubemapVisibleEntities,
            RenderVisibleMeshEntities,
        },
        ViewLightEntities,
    },
    lightmap::RenderLightmaps,
    material::{
        Material, MaterialBindGroupAllocator, MaterialPipelineKey, PreparedMaterial,
        RenderMaterialInstances,
    },
    mesh_pipeline::{
        render::{
            instance::{RenderMeshInstanceFlags, RenderMeshInstances},
            pipeline::MeshPipelineKey,
        },
        specialization::EntitySpecializationTicks,
    },
    prepass::{commands::DrawPrepass, render::PrepassPipeline},
    shadow::{
        phase_item::Shadow,
        render::{ShadowBatchSetKey, ShadowBinKey, SpecializedShadowMaterialPipelineCache},
    },
};

pub fn specialize_shadows<M: Material>(
    prepass_pipeline: Res<PrepassPipeline<M>>,
    (
        render_meshes,
        render_mesh_instances,
        render_materials,
        render_material_instances,
        material_bind_group_allocator,
    ): (
        Res<RenderAssets<RenderMesh>>,
        Res<RenderMeshInstances>,
        Res<RenderAssets<PreparedMaterial<M>>>,
        Res<RenderMaterialInstances<M>>,
        Res<MaterialBindGroupAllocator<M>>,
    ),
    shadow_render_phases: Res<ViewBinnedRenderPhases<Shadow>>,
    mut pipelines: ResMut<SpecializedMeshPipelines<PrepassPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    render_lightmaps: Res<RenderLightmaps>,
    view_lights: Query<(Entity, &ViewLightEntities), With<ExtractedView>>,
    view_light_entities: Query<(&LightEntity, &ExtractedView)>,
    point_light_entities: Query<&RenderCubemapVisibleEntities, With<ExtractedPointLight>>,
    directional_light_entities: Query<
        &RenderCascadesVisibleEntities,
        With<ExtractedDirectionalLight>,
    >,
    spot_light_entities: Query<&RenderVisibleMeshEntities, With<ExtractedPointLight>>,
    light_key_cache: Res<LightKeyCache>,
    mut specialized_material_pipeline_cache: ResMut<SpecializedShadowMaterialPipelineCache<M>>,
    light_specialization_ticks: Res<LightSpecializationTicks>,
    entity_specialization_ticks: Res<EntitySpecializationTicks<M>>,
    ticks: SystemChangeTick,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    // Record the retained IDs of all shadow views so that we can expire old
    // pipeline IDs.
    let mut all_shadow_views: HashSet<RetainedViewEntity, FixedHasher> = HashSet::default();

    for (entity, view_lights) in &view_lights {
        for view_light_entity in view_lights.lights.iter().copied() {
            let Ok((light_entity, extracted_view_light)) =
                view_light_entities.get(view_light_entity)
            else {
                continue;
            };

            all_shadow_views.insert(extracted_view_light.retained_view_entity);

            if !shadow_render_phases.contains_key(&extracted_view_light.retained_view_entity) {
                continue;
            }
            let Some(light_key) = light_key_cache.get(&extracted_view_light.retained_view_entity)
            else {
                continue;
            };

            let visible_entities = match light_entity {
                LightEntity::Directional {
                    light_entity,
                    cascade_index,
                } => directional_light_entities
                    .get(*light_entity)
                    .expect("Failed to get directional light visible entities")
                    .entities
                    .get(&entity)
                    .expect("Failed to get directional light visible entities for view")
                    .get(*cascade_index)
                    .expect("Failed to get directional light visible entities for cascade"),
                LightEntity::Point {
                    light_entity,
                    face_index,
                } => point_light_entities
                    .get(*light_entity)
                    .expect("Failed to get point light visible entities")
                    .get(*face_index),
                LightEntity::Spot { light_entity } => spot_light_entities
                    .get(*light_entity)
                    .expect("Failed to get spot light visible entities"),
            };

            // NOTE: Lights with shadow mapping disabled will have no visible entities
            // so no meshes will be queued

            let view_tick = light_specialization_ticks
                .get(&extracted_view_light.retained_view_entity)
                .unwrap();
            let view_specialized_material_pipeline_cache = specialized_material_pipeline_cache
                .entry(extracted_view_light.retained_view_entity)
                .or_default();

            for (_, visible_entity) in visible_entities.iter().copied() {
                let Some(material_asset_id) = render_material_instances.get(&visible_entity) else {
                    continue;
                };
                let entity_tick = entity_specialization_ticks.get(&visible_entity).unwrap();
                let last_specialized_tick = view_specialized_material_pipeline_cache
                    .get(&visible_entity)
                    .map(|(tick, _)| *tick);
                let needs_specialization = last_specialized_tick.is_none_or(|tick| {
                    view_tick.is_newer_than(tick, ticks.this_run())
                        || entity_tick.is_newer_than(tick, ticks.this_run())
                });
                if !needs_specialization {
                    continue;
                }
                let Some(material) = render_materials.get(*material_asset_id) else {
                    continue;
                };
                let Some(mesh_instance) =
                    render_mesh_instances.render_mesh_queue_data(visible_entity)
                else {
                    continue;
                };
                if !mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::SHADOW_CASTER)
                {
                    continue;
                }
                let Some(material_bind_group) =
                    material_bind_group_allocator.get(material.binding.group)
                else {
                    continue;
                };
                let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                    continue;
                };

                let mut mesh_key =
                    *light_key | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());

                // Even though we don't use the lightmap in the shadow map, the
                // `SetMeshBindGroup` render command will bind the data for it. So
                // we need to include the appropriate flag in the mesh pipeline key
                // to ensure that the necessary bind group layout entries are
                // present.
                if render_lightmaps
                    .render_lightmaps
                    .contains_key(&visible_entity)
                {
                    mesh_key |= MeshPipelineKey::LIGHTMAPPED;
                }

                mesh_key |= match material.properties.alpha_mode {
                    AlphaMode::Mask(_)
                    | AlphaMode::Blend
                    | AlphaMode::Premultiplied
                    | AlphaMode::Add
                    | AlphaMode::AlphaToCoverage => MeshPipelineKey::MAY_DISCARD,
                    _ => MeshPipelineKey::NONE,
                };
                let pipeline_id = pipelines.specialize(
                    &pipeline_cache,
                    &prepass_pipeline,
                    MaterialPipelineKey {
                        mesh_key,
                        bind_group_data: material_bind_group
                            .get_extra_data(material.binding.slot)
                            .clone(),
                    },
                    &mesh.layout,
                );

                let pipeline_id = match pipeline_id {
                    Ok(id) => id,
                    Err(err) => {
                        error!("{}", err);
                        continue;
                    }
                };

                view_specialized_material_pipeline_cache
                    .insert(visible_entity, (ticks.this_run(), pipeline_id));
            }
        }
    }

    // Delete specialized pipelines belonging to views that have expired.
    specialized_material_pipeline_cache.retain(|view, _| all_shadow_views.contains(view));
}

/// For each shadow cascade, iterates over all the meshes "visible" from it and
/// adds them to [`BinnedRenderPhase`]s or [`SortedRenderPhase`]s as
/// appropriate.
pub fn queue_shadows<M: Material>(
    shadow_draw_functions: Res<DrawFunctions<Shadow>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mut shadow_render_phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    mesh_allocator: Res<MeshAllocator>,
    view_lights: Query<(Entity, &ViewLightEntities), With<ExtractedView>>,
    view_light_entities: Query<(&LightEntity, &ExtractedView)>,
    point_light_entities: Query<&RenderCubemapVisibleEntities, With<ExtractedPointLight>>,
    directional_light_entities: Query<
        &RenderCascadesVisibleEntities,
        With<ExtractedDirectionalLight>,
    >,
    spot_light_entities: Query<&RenderVisibleMeshEntities, With<ExtractedPointLight>>,
    specialized_material_pipeline_cache: Res<SpecializedShadowMaterialPipelineCache<M>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let draw_shadow_mesh = shadow_draw_functions.read().id::<DrawPrepass<M>>();
    for (entity, view_lights) in &view_lights {
        for view_light_entity in view_lights.lights.iter().copied() {
            let Ok((light_entity, extracted_view_light)) =
                view_light_entities.get(view_light_entity)
            else {
                continue;
            };
            let Some(shadow_phase) =
                shadow_render_phases.get_mut(&extracted_view_light.retained_view_entity)
            else {
                continue;
            };

            let Some(view_specialized_material_pipeline_cache) =
                specialized_material_pipeline_cache.get(&extracted_view_light.retained_view_entity)
            else {
                continue;
            };

            let visible_entities = match light_entity {
                LightEntity::Directional {
                    light_entity,
                    cascade_index,
                } => directional_light_entities
                    .get(*light_entity)
                    .expect("Failed to get directional light visible entities")
                    .entities
                    .get(&entity)
                    .expect("Failed to get directional light visible entities for view")
                    .get(*cascade_index)
                    .expect("Failed to get directional light visible entities for cascade"),
                LightEntity::Point {
                    light_entity,
                    face_index,
                } => point_light_entities
                    .get(*light_entity)
                    .expect("Failed to get point light visible entities")
                    .get(*face_index),
                LightEntity::Spot { light_entity } => spot_light_entities
                    .get(*light_entity)
                    .expect("Failed to get spot light visible entities"),
            };

            for (entity, main_entity) in visible_entities.iter().copied() {
                let Some((current_change_tick, pipeline_id)) =
                    view_specialized_material_pipeline_cache.get(&main_entity)
                else {
                    continue;
                };

                // Skip the entity if it's cached in a bin and up to date.
                if shadow_phase.validate_cached_entity(main_entity, *current_change_tick) {
                    continue;
                }

                let Some(mesh_instance) = render_mesh_instances.render_mesh_queue_data(main_entity)
                else {
                    continue;
                };
                if !mesh_instance
                    .flags
                    .contains(RenderMeshInstanceFlags::SHADOW_CASTER)
                {
                    continue;
                }

                let Some(material_asset_id) = render_material_instances.get(&main_entity) else {
                    continue;
                };
                let Some(material) = render_materials.get(*material_asset_id) else {
                    continue;
                };

                let (vertex_slab, index_slab) =
                    mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);

                let batch_set_key = ShadowBatchSetKey {
                    pipeline: *pipeline_id,
                    draw_function: draw_shadow_mesh,
                    material_bind_group_index: Some(material.binding.group.0),
                    vertex_slab: vertex_slab.unwrap_or_default(),
                    index_slab,
                };

                shadow_phase.add(
                    batch_set_key,
                    ShadowBinKey {
                        asset_id: mesh_instance.mesh_asset_id.into(),
                    },
                    (entity, main_entity),
                    mesh_instance.current_uniform_index,
                    BinnedRenderPhaseType::mesh(
                        mesh_instance.should_batch(),
                        &gpu_preprocessing_support,
                    ),
                    *current_change_tick,
                );
            }
        }
    }
}
