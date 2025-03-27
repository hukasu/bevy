use core::hash::Hash;

use bevy_core_pipeline::{
    core_3d::Camera3d,
    deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
    prepass::{
        AlphaMask3dPrepass, DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass,
        Opaque3dPrepass, OpaqueNoLightmap3dBatchSetKey, OpaqueNoLightmap3dBinKey, PreviousViewData,
        PreviousViewUniformOffset, PreviousViewUniforms,
    },
};
use bevy_ecs::{
    entity::Entity,
    query::{Or, With},
    system::{Commands, Query, Res, ResMut, SystemChangeTick},
};
use bevy_render::{
    alpha::AlphaMode,
    batching::gpu_preprocessing::GpuPreprocessingSupport,
    camera::Camera,
    globals::GlobalsBuffer,
    mesh::{allocator::MeshAllocator, Mesh3d, RenderMesh},
    render_asset::RenderAssets,
    render_phase::{BinnedRenderPhaseType, ViewBinnedRenderPhases},
    render_resource::{BindGroupEntries, PipelineCache, SpecializedMeshPipelines},
    renderer::{RenderDevice, RenderQueue},
    sync_world::RenderEntity,
    view::{ExtractedView, Msaa, RenderVisibilityRanges, RenderVisibleEntities, ViewUniforms},
    Extract,
};
use bevy_transform::components::GlobalTransform;

use tracing::{error, warn};

use crate::{
    lightmap::RenderLightmaps,
    material::{
        alpha_mode_pipeline_key, material::RenderPhaseType, Material, MaterialBindGroupAllocator,
        MaterialPipelineKey, PreparedMaterial, RenderMaterialInstances,
    },
    mesh_pipeline::{
        render::{
            instance::{RenderMeshInstanceFlags, RenderMeshInstances},
            pipeline::MeshPipelineKey,
        },
        specialization::EntitySpecializationTicks,
    },
    prepass::{
        render::{
            PrepassPipeline, PrepassViewBindGroup, SpecializedPrepassMaterialPipelineCache,
            ViewKeyPrepassCache, ViewPrepassSpecializationTicks,
        },
        PreviousGlobalTransform,
    },
    render_method::OpaqueRendererMethod,
    shadow::render::ShadowView,
};

#[cfg(feature = "meshlet")]
use crate::meshlet::MeshletMesh3d;

#[cfg(not(feature = "meshlet"))]
type PreviousMeshFilter = With<Mesh3d>;
#[cfg(feature = "meshlet")]
type PreviousMeshFilter = Or<(With<Mesh3d>, With<MeshletMesh3d>)>;

pub fn update_previous_view_data(
    mut commands: Commands,
    query: Query<(Entity, &Camera, &GlobalTransform), Or<(With<Camera3d>, With<ShadowView>)>>,
) {
    for (entity, camera, camera_transform) in &query {
        let view_from_world = camera_transform.compute_matrix().inverse();
        commands.entity(entity).try_insert(PreviousViewData {
            view_from_world,
            clip_from_world: camera.clip_from_view() * view_from_world,
            clip_from_view: camera.clip_from_view(),
        });
    }
}

pub fn update_mesh_previous_global_transforms(
    mut commands: Commands,
    views: Query<&Camera, Or<(With<Camera3d>, With<ShadowView>)>>,
    meshes: Query<(Entity, &GlobalTransform, Option<&PreviousGlobalTransform>), PreviousMeshFilter>,
) {
    let should_run = views.iter().any(|camera| camera.is_active);

    if should_run {
        for (entity, transform, old_previous_transform) in &meshes {
            let new_previous_transform = PreviousGlobalTransform(transform.affine());
            // Make sure not to trigger change detection on
            // `PreviousGlobalTransform` if the previous transform hasn't
            // changed.
            if old_previous_transform != Some(&new_previous_transform) {
                commands.entity(entity).try_insert(new_previous_transform);
            }
        }
    }
}

// Extract the render phases for the prepass
pub fn extract_camera_previous_view_data(
    mut commands: Commands,
    cameras_3d: Extract<Query<(RenderEntity, &Camera, Option<&PreviousViewData>), With<Camera3d>>>,
) {
    for (entity, camera, maybe_previous_view_data) in cameras_3d.iter() {
        let mut entity = commands
            .get_entity(entity)
            .expect("Camera entity wasn't synced.");
        if camera.is_active {
            if let Some(previous_view_data) = maybe_previous_view_data {
                entity.insert(previous_view_data.clone());
            }
        } else {
            entity.remove::<PreviousViewData>();
        }
    }
}

pub fn prepare_previous_view_uniforms(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    render_queue: Res<RenderQueue>,
    mut previous_view_uniforms: ResMut<PreviousViewUniforms>,
    views: Query<
        (Entity, &ExtractedView, Option<&PreviousViewData>),
        Or<(With<Camera3d>, With<ShadowView>)>,
    >,
) {
    let views_iter = views.iter();
    let view_count = views_iter.len();
    let Some(mut writer) =
        previous_view_uniforms
            .uniforms
            .get_writer(view_count, &render_device, &render_queue)
    else {
        return;
    };

    for (entity, camera, maybe_previous_view_uniforms) in views_iter {
        let prev_view_data = match maybe_previous_view_uniforms {
            Some(previous_view) => previous_view.clone(),
            None => {
                let view_from_world = camera.world_from_view.compute_matrix().inverse();
                PreviousViewData {
                    view_from_world,
                    clip_from_world: camera.clip_from_view * view_from_world,
                    clip_from_view: camera.clip_from_view,
                }
            }
        };

        commands.entity(entity).insert(PreviousViewUniformOffset {
            offset: writer.write(&prev_view_data),
        });
    }
}

pub fn prepare_prepass_view_bind_group<M: Material>(
    render_device: Res<RenderDevice>,
    prepass_pipeline: Res<PrepassPipeline<M>>,
    view_uniforms: Res<ViewUniforms>,
    globals_buffer: Res<GlobalsBuffer>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    visibility_ranges: Res<RenderVisibilityRanges>,
    mut prepass_view_bind_group: ResMut<PrepassViewBindGroup>,
) {
    if let (Some(view_binding), Some(globals_binding), Some(visibility_ranges_buffer)) = (
        view_uniforms.uniforms.binding(),
        globals_buffer.buffer.binding(),
        visibility_ranges.buffer().buffer(),
    ) {
        prepass_view_bind_group.no_motion_vectors = Some(render_device.create_bind_group(
            "prepass_view_no_motion_vectors_bind_group",
            &prepass_pipeline.view_layout_no_motion_vectors,
            &BindGroupEntries::with_indices((
                (0, view_binding.clone()),
                (1, globals_binding.clone()),
                (14, visibility_ranges_buffer.as_entire_binding()),
            )),
        ));

        if let Some(previous_view_uniforms_binding) = previous_view_uniforms.uniforms.binding() {
            prepass_view_bind_group.motion_vectors = Some(render_device.create_bind_group(
                "prepass_view_motion_vectors_bind_group",
                &prepass_pipeline.view_layout_motion_vectors,
                &BindGroupEntries::with_indices((
                    (0, view_binding),
                    (1, globals_binding),
                    (2, previous_view_uniforms_binding),
                    (14, visibility_ranges_buffer.as_entire_binding()),
                )),
            ));
        }
    }
}

pub fn check_prepass_views_need_specialization(
    mut view_key_cache: ResMut<ViewKeyPrepassCache>,
    mut view_specialization_ticks: ResMut<ViewPrepassSpecializationTicks>,
    mut views: Query<(
        &ExtractedView,
        &Msaa,
        Option<&DepthPrepass>,
        Option<&NormalPrepass>,
        Option<&MotionVectorPrepass>,
    )>,
    ticks: SystemChangeTick,
) {
    for (view, msaa, depth_prepass, normal_prepass, motion_vector_prepass) in views.iter_mut() {
        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples());
        if depth_prepass.is_some() {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }
        if normal_prepass.is_some() {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }
        if motion_vector_prepass.is_some() {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if let Some(current_key) = view_key_cache.get_mut(&view.retained_view_entity) {
            if *current_key != view_key {
                view_key_cache.insert(view.retained_view_entity, view_key);
                view_specialization_ticks.insert(view.retained_view_entity, ticks.this_run());
            }
        } else {
            view_key_cache.insert(view.retained_view_entity, view_key);
            view_specialization_ticks.insert(view.retained_view_entity, ticks.this_run());
        }
    }
}

pub fn specialize_prepass_material_meshes<M>(
    render_meshes: Res<RenderAssets<RenderMesh>>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_mesh_instances: Res<RenderMeshInstances>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    render_lightmaps: Res<RenderLightmaps>,
    render_visibility_ranges: Res<RenderVisibilityRanges>,
    material_bind_group_allocator: Res<MaterialBindGroupAllocator<M>>,
    view_key_cache: Res<ViewKeyPrepassCache>,
    views: Query<(
        &ExtractedView,
        &RenderVisibleEntities,
        &Msaa,
        Option<&MotionVectorPrepass>,
        Option<&DeferredPrepass>,
    )>,
    (
        opaque_prepass_render_phases,
        alpha_mask_prepass_render_phases,
        opaque_deferred_render_phases,
        alpha_mask_deferred_render_phases,
    ): (
        Res<ViewBinnedRenderPhases<Opaque3dPrepass>>,
        Res<ViewBinnedRenderPhases<AlphaMask3dPrepass>>,
        Res<ViewBinnedRenderPhases<Opaque3dDeferred>>,
        Res<ViewBinnedRenderPhases<AlphaMask3dDeferred>>,
    ),
    (
        mut specialized_material_pipeline_cache,
        ticks,
        prepass_pipeline,
        mut pipelines,
        pipeline_cache,
        view_specialization_ticks,
        entity_specialization_ticks,
    ): (
        ResMut<SpecializedPrepassMaterialPipelineCache<M>>,
        SystemChangeTick,
        Res<PrepassPipeline<M>>,
        ResMut<SpecializedMeshPipelines<PrepassPipeline<M>>>,
        Res<PipelineCache>,
        Res<ViewPrepassSpecializationTicks>,
        Res<EntitySpecializationTicks<M>>,
    ),
) where
    M: Material,
    M::Data: PartialEq + Eq + Hash + Clone,
{
    for (extracted_view, visible_entities, msaa, motion_vector_prepass, deferred_prepass) in &views
    {
        if !opaque_deferred_render_phases.contains_key(&extracted_view.retained_view_entity)
            && !alpha_mask_deferred_render_phases.contains_key(&extracted_view.retained_view_entity)
            && !opaque_prepass_render_phases.contains_key(&extracted_view.retained_view_entity)
            && !alpha_mask_prepass_render_phases.contains_key(&extracted_view.retained_view_entity)
        {
            continue;
        }

        let Some(view_key) = view_key_cache.get(&extracted_view.retained_view_entity) else {
            continue;
        };

        let view_tick = view_specialization_ticks
            .get(&extracted_view.retained_view_entity)
            .unwrap();
        let view_specialized_material_pipeline_cache = specialized_material_pipeline_cache
            .entry(extracted_view.retained_view_entity)
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
            let Some(material) = render_materials.get(*material_asset_id) else {
                continue;
            };
            let Some(material_bind_group) =
                material_bind_group_allocator.get(material.binding.group)
            else {
                warn!("Couldn't get bind group for material");
                continue;
            };
            let Some(mesh) = render_meshes.get(mesh_instance.mesh_asset_id) else {
                continue;
            };

            let mut mesh_key = *view_key | MeshPipelineKey::from_bits_retain(mesh.key_bits.bits());

            let alpha_mode = material.properties.alpha_mode;
            match alpha_mode {
                AlphaMode::Opaque | AlphaMode::AlphaToCoverage | AlphaMode::Mask(_) => {
                    mesh_key |= alpha_mode_pipeline_key(alpha_mode, msaa);
                }
                AlphaMode::Blend
                | AlphaMode::Premultiplied
                | AlphaMode::Add
                | AlphaMode::Multiply => continue,
            }

            if material.properties.reads_view_transmission_texture {
                // No-op: Materials reading from `ViewTransmissionTexture` are not rendered in the `Opaque3d`
                // phase, and are therefore also excluded from the prepass much like alpha-blended materials.
                continue;
            }

            let forward = match material.properties.render_method {
                OpaqueRendererMethod::Forward => true,
                OpaqueRendererMethod::Deferred => false,
                OpaqueRendererMethod::Auto => unreachable!(),
            };

            let deferred = deferred_prepass.is_some() && !forward;

            if deferred {
                mesh_key |= MeshPipelineKey::DEFERRED_PREPASS;
            }

            if let Some(lightmap) = render_lightmaps.render_lightmaps.get(visible_entity) {
                // Even though we don't use the lightmap in the forward prepass, the
                // `SetMeshBindGroup` render command will bind the data for it. So
                // we need to include the appropriate flag in the mesh pipeline key
                // to ensure that the necessary bind group layout entries are
                // present.
                mesh_key |= MeshPipelineKey::LIGHTMAPPED;

                if lightmap.bicubic_sampling && deferred {
                    mesh_key |= MeshPipelineKey::LIGHTMAP_BICUBIC_SAMPLING;
                }
            }

            if render_visibility_ranges.entity_has_crossfading_visibility_ranges(*visible_entity) {
                mesh_key |= MeshPipelineKey::VISIBILITY_RANGE_DITHER;
            }

            // If the previous frame has skins or morph targets, note that.
            if motion_vector_prepass.is_some() {
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
                .insert(*visible_entity, (ticks.this_run(), pipeline_id));
        }
    }
}

pub fn queue_prepass_material_meshes<M: Material>(
    render_mesh_instances: Res<RenderMeshInstances>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mesh_allocator: Res<MeshAllocator>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
    mut opaque_prepass_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3dPrepass>>,
    mut alpha_mask_prepass_render_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3dPrepass>>,
    mut opaque_deferred_render_phases: ResMut<ViewBinnedRenderPhases<Opaque3dDeferred>>,
    mut alpha_mask_deferred_render_phases: ResMut<ViewBinnedRenderPhases<AlphaMask3dDeferred>>,
    views: Query<(&ExtractedView, &RenderVisibleEntities)>,
    specialized_material_pipeline_cache: Res<SpecializedPrepassMaterialPipelineCache<M>>,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    for (extracted_view, visible_entities) in &views {
        let (
            mut opaque_phase,
            mut alpha_mask_phase,
            mut opaque_deferred_phase,
            mut alpha_mask_deferred_phase,
        ) = (
            opaque_prepass_render_phases.get_mut(&extracted_view.retained_view_entity),
            alpha_mask_prepass_render_phases.get_mut(&extracted_view.retained_view_entity),
            opaque_deferred_render_phases.get_mut(&extracted_view.retained_view_entity),
            alpha_mask_deferred_render_phases.get_mut(&extracted_view.retained_view_entity),
        );

        let Some(view_specialized_material_pipeline_cache) =
            specialized_material_pipeline_cache.get(&extracted_view.retained_view_entity)
        else {
            continue;
        };

        // Skip if there's no place to put the mesh.
        if opaque_phase.is_none()
            && alpha_mask_phase.is_none()
            && opaque_deferred_phase.is_none()
            && alpha_mask_deferred_phase.is_none()
        {
            continue;
        }

        for (render_entity, visible_entity) in visible_entities.iter::<Mesh3d>() {
            let Some((current_change_tick, pipeline_id)) =
                view_specialized_material_pipeline_cache.get(visible_entity)
            else {
                continue;
            };

            // Skip the entity if it's cached in a bin and up to date.
            if opaque_phase.as_mut().is_some_and(|phase| {
                phase.validate_cached_entity(*visible_entity, *current_change_tick)
            }) || alpha_mask_phase.as_mut().is_some_and(|phase| {
                phase.validate_cached_entity(*visible_entity, *current_change_tick)
            }) || opaque_deferred_phase.as_mut().is_some_and(|phase| {
                phase.validate_cached_entity(*visible_entity, *current_change_tick)
            }) || alpha_mask_deferred_phase.as_mut().is_some_and(|phase| {
                phase.validate_cached_entity(*visible_entity, *current_change_tick)
            }) {
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
            let (vertex_slab, index_slab) = mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);

            let deferred = match material.properties.render_method {
                OpaqueRendererMethod::Forward => false,
                OpaqueRendererMethod::Deferred => true,
                OpaqueRendererMethod::Auto => unreachable!(),
            };

            match material.properties.render_phase_type {
                RenderPhaseType::Opaque => {
                    if deferred {
                        opaque_deferred_phase.as_mut().unwrap().add(
                            OpaqueNoLightmap3dBatchSetKey {
                                draw_function: material
                                    .properties
                                    .deferred_draw_function_id
                                    .unwrap(),
                                pipeline: *pipeline_id,
                                material_bind_group_index: Some(material.binding.group.0),
                                vertex_slab: vertex_slab.unwrap_or_default(),
                                index_slab,
                            },
                            OpaqueNoLightmap3dBinKey {
                                asset_id: mesh_instance.mesh_asset_id.into(),
                            },
                            (*render_entity, *visible_entity),
                            mesh_instance.current_uniform_index,
                            BinnedRenderPhaseType::mesh(
                                mesh_instance.should_batch(),
                                &gpu_preprocessing_support,
                            ),
                            *current_change_tick,
                        );
                    } else if let Some(opaque_phase) = opaque_phase.as_mut() {
                        let (vertex_slab, index_slab) =
                            mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);
                        opaque_phase.add(
                            OpaqueNoLightmap3dBatchSetKey {
                                draw_function: material
                                    .properties
                                    .prepass_draw_function_id
                                    .unwrap(),
                                pipeline: *pipeline_id,
                                material_bind_group_index: Some(material.binding.group.0),
                                vertex_slab: vertex_slab.unwrap_or_default(),
                                index_slab,
                            },
                            OpaqueNoLightmap3dBinKey {
                                asset_id: mesh_instance.mesh_asset_id.into(),
                            },
                            (*render_entity, *visible_entity),
                            mesh_instance.current_uniform_index,
                            BinnedRenderPhaseType::mesh(
                                mesh_instance.should_batch(),
                                &gpu_preprocessing_support,
                            ),
                            *current_change_tick,
                        );
                    }
                }
                RenderPhaseType::AlphaMask => {
                    if deferred {
                        let (vertex_slab, index_slab) =
                            mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);
                        let batch_set_key = OpaqueNoLightmap3dBatchSetKey {
                            draw_function: material.properties.deferred_draw_function_id.unwrap(),
                            pipeline: *pipeline_id,
                            material_bind_group_index: Some(material.binding.group.0),
                            vertex_slab: vertex_slab.unwrap_or_default(),
                            index_slab,
                        };
                        let bin_key = OpaqueNoLightmap3dBinKey {
                            asset_id: mesh_instance.mesh_asset_id.into(),
                        };
                        alpha_mask_deferred_phase.as_mut().unwrap().add(
                            batch_set_key,
                            bin_key,
                            (*render_entity, *visible_entity),
                            mesh_instance.current_uniform_index,
                            BinnedRenderPhaseType::mesh(
                                mesh_instance.should_batch(),
                                &gpu_preprocessing_support,
                            ),
                            *current_change_tick,
                        );
                    } else if let Some(alpha_mask_phase) = alpha_mask_phase.as_mut() {
                        let (vertex_slab, index_slab) =
                            mesh_allocator.mesh_slabs(&mesh_instance.mesh_asset_id);
                        let batch_set_key = OpaqueNoLightmap3dBatchSetKey {
                            draw_function: material.properties.prepass_draw_function_id.unwrap(),
                            pipeline: *pipeline_id,
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
                            *current_change_tick,
                        );
                    }
                }
                _ => {}
            }
        }
    }
}
