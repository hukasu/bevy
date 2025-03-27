use bevy_core_pipeline::{
    core_3d::{Camera3d, ViewTransmissionTexture},
    oit::{OitBuffers, OrderIndependentTransparencySettings},
    prepass::{
        DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass, ViewPrepassTextures,
    },
    tonemapping::{get_lut_bindings, DebandDither, Tonemapping, TonemappingLuts},
};
use bevy_diagnostic::FrameCount;
use bevy_ecs::{
    entity::Entity,
    query::{Changed, Has, Or, QueryData, With, Without},
    removal_detection::RemovedComponents,
    system::{lifetimeless::Read, Commands, Local, Query, Res, ResMut, SystemChangeTick},
};
use bevy_image::BevyDefault;
use bevy_mesh::skinning::SkinnedMesh;
use bevy_render::{
    batching::{
        gpu_preprocessing::{self, InstanceInputUniformBuffer},
        no_gpu_preprocessing, NoAutomaticBatching,
    },
    camera::{Camera, Projection, TemporalJitter},
    globals::GlobalsBuffer,
    mesh::{allocator::MeshAllocator, Mesh3d, MeshTag, RenderMesh},
    primitives::Aabb,
    render_asset::RenderAssets,
    render_resource::{
        BindingResource, DynamicBindGroupEntries, TextureAspect, TextureFormat, TextureView,
        TextureViewDescriptor,
    },
    renderer::{RenderAdapter, RenderDevice},
    sync_world::{MainEntity, MainEntityHashMap},
    texture::{FallbackImage, FallbackImageMsaa, FallbackImageZero, GpuImage},
    view::{
        ExtractedView, Msaa, NoFrustumCulling, NoIndirectDrawing, RenderVisibilityRanges,
        ViewUniforms, ViewVisibility, VisibilityRange,
    },
    Extract,
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::{default, Parallel, TypeIdMap};

use nonmax::NonMaxU32;

use crate::{
    cluster::{cluster::ViewClusterBindings, GlobalClusterableObjectMeta},
    decal::clustered::decals::{
        DecalsBuffer, RenderClusteredDecals, RenderViewClusteredDecalBindGroupEntries,
    },
    distance_fog::{fog::FogMeta, DistanceFog},
    is_skinned,
    light::{render::LightMeta, NotShadowCaster, NotShadowReceiver, TransmittedShadowReceiver},
    light_probe::{
        environment_map::{EnvironmentMapUniformBuffer, RenderViewEnvironmentMapBindGroupEntries},
        irradiance_volume::RenderViewIrradianceVolumeBindGroupEntries,
        light_probes::{LightProbesBuffer, RenderViewLightProbes},
        EnvironmentMapLight, IrradianceVolume, IRRADIANCE_VOLUMES_ARE_USABLE,
    },
    lightmap::{Lightmap, LightmapSlabIndex, RenderLightmaps},
    material::RenderMaterialBindings,
    morph::data::{MorphIndices, MorphUniforms},
    prepass::PreviousGlobalTransform,
    shadow::{
        render::{ShadowSamplers, ViewShadowBindings},
        ShadowFilteringMethod,
    },
    skin::uniforms::SkinUniforms,
    ssao::{ssao::ScreenSpaceAmbientOcclusionResources, ScreenSpaceAmbientOcclusion},
    ssr::render::ScreenSpaceReflectionsBuffer,
};

use super::{
    render::{
        instance::{
            RenderMeshInstanceCpu, RenderMeshInstanceFlags, RenderMeshInstanceGpu,
            RenderMeshInstanceGpuBuilder, RenderMeshInstanceGpuQueue, RenderMeshInstanceGpuQueues,
            RenderMeshInstanceShared, RenderMeshInstances, RenderMeshInstancesGpu,
        },
        pack_lightmap_uv_rect,
        pipeline::{
            screen_space_specular_transmission_pipeline_key, tonemapping_pipeline_key,
            MeshPipeline, MeshPipelineKey, MeshPipelineViewLayoutKey,
        },
        MeshBindGroupPair, MeshBindGroups, MeshCullingData, MeshCullingDataBuffer, MeshFlags,
        MeshInputUniform, MeshPhaseBindGroups, MeshTransforms, MeshUniform, MeshViewBindGroup,
        MeshesToReextractNextFrame, RenderMeshMaterialIds,
    },
    specialization::ViewSpecializationTicks,
    ViewKeyCache,
};

/// All the data that we need from a mesh in the main world.
type GpuMeshExtractionQuery = (
    Entity,
    Read<ViewVisibility>,
    Read<GlobalTransform>,
    Option<Read<PreviousGlobalTransform>>,
    Option<Read<Lightmap>>,
    Option<Read<Aabb>>,
    Read<Mesh3d>,
    Option<Read<MeshTag>>,
    Has<NoFrustumCulling>,
    Has<NotShadowReceiver>,
    Has<TransmittedShadowReceiver>,
    Has<NotShadowCaster>,
    Has<NoAutomaticBatching>,
    Has<VisibilityRange>,
);

/// A system that sets the [`RenderMeshInstanceFlags`] for each mesh based on
/// whether the previous frame had skins and/or morph targets.
///
/// Ordinarily, [`RenderMeshInstanceFlags`] are set during the extraction phase.
/// However, we can't do that for the flags related to skins and morph targets
/// because the previous frame's skin and morph targets are the responsibility
/// of [`extract_skins`] and [`extract_morphs`] respectively. We want to run
/// those systems in parallel with mesh extraction for performance, so we need
/// to defer setting of these mesh instance flags to after extraction, which
/// this system does. An alternative to having skin- and morph-target-related
/// data in [`RenderMeshInstanceFlags`] would be to have
/// [`crate::material::queue_material_meshes`] check the skin and morph target
/// tables for each mesh, but that would be too slow in the hot mesh queuing
/// loop.
pub fn set_mesh_motion_vector_flags(
    mut render_mesh_instances: ResMut<RenderMeshInstances>,
    skin_uniforms: Res<SkinUniforms>,
    morph_indices: Res<MorphIndices>,
) {
    for &entity in skin_uniforms.all_skins() {
        render_mesh_instances
            .insert_mesh_instance_flags(entity, RenderMeshInstanceFlags::HAS_PREVIOUS_SKIN);
    }
    for &entity in morph_indices.prev.keys() {
        render_mesh_instances
            .insert_mesh_instance_flags(entity, RenderMeshInstanceFlags::HAS_PREVIOUS_MORPH);
    }
}

/// Creates the per-mesh bind groups for each type of mesh and each phase.
pub fn prepare_mesh_bind_groups(
    mut commands: Commands,
    meshes: Res<RenderAssets<RenderMesh>>,
    mesh_pipeline: Res<MeshPipeline>,
    render_device: Res<RenderDevice>,
    cpu_batched_instance_buffer: Option<
        Res<no_gpu_preprocessing::BatchedInstanceBuffer<MeshUniform>>,
    >,
    gpu_batched_instance_buffers: Option<
        Res<gpu_preprocessing::BatchedInstanceBuffers<MeshUniform, MeshInputUniform>>,
    >,
    skins_uniform: Res<SkinUniforms>,
    weights_uniform: Res<MorphUniforms>,
    mut render_lightmaps: ResMut<RenderLightmaps>,
) {
    // CPU mesh preprocessing path.
    if let Some(cpu_batched_instance_buffer) = cpu_batched_instance_buffer {
        if let Some(instance_data_binding) = cpu_batched_instance_buffer
            .into_inner()
            .instance_data_binding()
        {
            // In this path, we only have a single set of bind groups for all phases.
            let cpu_preprocessing_mesh_bind_groups = prepare_mesh_bind_groups_for_phase(
                instance_data_binding,
                &meshes,
                &mesh_pipeline,
                &render_device,
                &skins_uniform,
                &weights_uniform,
                &mut render_lightmaps,
            );

            commands.insert_resource(MeshBindGroups::CpuPreprocessing(
                cpu_preprocessing_mesh_bind_groups,
            ));
            return;
        }
    }

    // GPU mesh preprocessing path.
    if let Some(gpu_batched_instance_buffers) = gpu_batched_instance_buffers {
        let mut gpu_preprocessing_mesh_bind_groups = TypeIdMap::default();

        // Loop over each phase.
        for (phase_type_id, batched_phase_instance_buffers) in
            &gpu_batched_instance_buffers.phase_instance_buffers
        {
            let Some(instance_data_binding) =
                batched_phase_instance_buffers.instance_data_binding()
            else {
                continue;
            };

            let mesh_phase_bind_groups = prepare_mesh_bind_groups_for_phase(
                instance_data_binding,
                &meshes,
                &mesh_pipeline,
                &render_device,
                &skins_uniform,
                &weights_uniform,
                &mut render_lightmaps,
            );

            gpu_preprocessing_mesh_bind_groups.insert(*phase_type_id, mesh_phase_bind_groups);
        }

        commands.insert_resource(MeshBindGroups::GpuPreprocessing(
            gpu_preprocessing_mesh_bind_groups,
        ));
    }
}

pub fn prepare_mesh_view_bind_groups(
    mut commands: Commands,
    (render_device, render_adapter): (Res<RenderDevice>, Res<RenderAdapter>),
    mesh_pipeline: Res<MeshPipeline>,
    shadow_samplers: Res<ShadowSamplers>,
    (light_meta, global_light_meta): (Res<LightMeta>, Res<GlobalClusterableObjectMeta>),
    fog_meta: Res<FogMeta>,
    (view_uniforms, environment_map_uniform): (Res<ViewUniforms>, Res<EnvironmentMapUniformBuffer>),
    views: Query<(
        Entity,
        &ViewShadowBindings,
        &ViewClusterBindings,
        &Msaa,
        Option<&ScreenSpaceAmbientOcclusionResources>,
        Option<&ViewPrepassTextures>,
        Option<&ViewTransmissionTexture>,
        &Tonemapping,
        Option<&RenderViewLightProbes<EnvironmentMapLight>>,
        Option<&RenderViewLightProbes<IrradianceVolume>>,
        Has<OrderIndependentTransparencySettings>,
    )>,
    (images, mut fallback_images, fallback_image, fallback_image_zero): (
        Res<RenderAssets<GpuImage>>,
        FallbackImageMsaa,
        Res<FallbackImage>,
        Res<FallbackImageZero>,
    ),
    globals_buffer: Res<GlobalsBuffer>,
    tonemapping_luts: Res<TonemappingLuts>,
    light_probes_buffer: Res<LightProbesBuffer>,
    visibility_ranges: Res<RenderVisibilityRanges>,
    ssr_buffer: Res<ScreenSpaceReflectionsBuffer>,
    oit_buffers: Res<OitBuffers>,
    (decals_buffer, render_decals): (Res<DecalsBuffer>, Res<RenderClusteredDecals>),
) {
    if let (
        Some(view_binding),
        Some(light_binding),
        Some(clusterable_objects_binding),
        Some(globals),
        Some(fog_binding),
        Some(light_probes_binding),
        Some(visibility_ranges_buffer),
        Some(ssr_binding),
        Some(environment_map_binding),
    ) = (
        view_uniforms.uniforms.binding(),
        light_meta.view_gpu_lights.binding(),
        global_light_meta.gpu_clusterable_objects.binding(),
        globals_buffer.buffer.binding(),
        fog_meta.gpu_fogs.binding(),
        light_probes_buffer.binding(),
        visibility_ranges.buffer().buffer(),
        ssr_buffer.binding(),
        environment_map_uniform.binding(),
    ) {
        for (
            entity,
            shadow_bindings,
            cluster_bindings,
            msaa,
            ssao_resources,
            prepass_textures,
            transmission_texture,
            tonemapping,
            render_view_environment_maps,
            render_view_irradiance_volumes,
            has_oit,
        ) in &views
        {
            let fallback_ssao = fallback_images
                .image_for_samplecount(1, TextureFormat::bevy_default())
                .texture_view
                .clone();
            let ssao_view = ssao_resources
                .map(|t| &t.screen_space_ambient_occlusion_texture.default_view)
                .unwrap_or(&fallback_ssao);

            let mut layout_key = MeshPipelineViewLayoutKey::from(*msaa)
                | MeshPipelineViewLayoutKey::from(prepass_textures);
            if has_oit {
                layout_key |= MeshPipelineViewLayoutKey::OIT_ENABLED;
            }

            let layout = &mesh_pipeline.get_view_layout(layout_key);

            let mut entries = DynamicBindGroupEntries::new_with_indices((
                (0, view_binding.clone()),
                (1, light_binding.clone()),
                (2, &shadow_bindings.point_light_depth_texture_view),
                (3, &shadow_samplers.point_light_comparison_sampler),
                #[cfg(feature = "experimental_pbr_pcss")]
                (4, &shadow_samplers.point_light_linear_sampler),
                (5, &shadow_bindings.directional_light_depth_texture_view),
                (6, &shadow_samplers.directional_light_comparison_sampler),
                #[cfg(feature = "experimental_pbr_pcss")]
                (7, &shadow_samplers.directional_light_linear_sampler),
                (8, clusterable_objects_binding.clone()),
                (
                    9,
                    cluster_bindings
                        .clusterable_object_index_lists_binding()
                        .unwrap(),
                ),
                (10, cluster_bindings.offsets_and_counts_binding().unwrap()),
                (11, globals.clone()),
                (12, fog_binding.clone()),
                (13, light_probes_binding.clone()),
                (14, visibility_ranges_buffer.as_entire_binding()),
                (15, ssr_binding.clone()),
                (16, ssao_view),
            ));

            let environment_map_bind_group_entries = RenderViewEnvironmentMapBindGroupEntries::get(
                render_view_environment_maps,
                &images,
                &fallback_image,
                &render_device,
                &render_adapter,
            );

            match environment_map_bind_group_entries {
                RenderViewEnvironmentMapBindGroupEntries::Single {
                    diffuse_texture_view,
                    specular_texture_view,
                    sampler,
                } => {
                    entries = entries.extend_with_indices((
                        (17, diffuse_texture_view),
                        (18, specular_texture_view),
                        (19, sampler),
                        (20, environment_map_binding.clone()),
                    ));
                }
                RenderViewEnvironmentMapBindGroupEntries::Multiple {
                    ref diffuse_texture_views,
                    ref specular_texture_views,
                    sampler,
                } => {
                    entries = entries.extend_with_indices((
                        (17, diffuse_texture_views.as_slice()),
                        (18, specular_texture_views.as_slice()),
                        (19, sampler),
                        (20, environment_map_binding.clone()),
                    ));
                }
            }

            let irradiance_volume_bind_group_entries = if IRRADIANCE_VOLUMES_ARE_USABLE {
                Some(RenderViewIrradianceVolumeBindGroupEntries::get(
                    render_view_irradiance_volumes,
                    &images,
                    &fallback_image,
                    &render_device,
                    &render_adapter,
                ))
            } else {
                None
            };

            match irradiance_volume_bind_group_entries {
                Some(RenderViewIrradianceVolumeBindGroupEntries::Single {
                    texture_view,
                    sampler,
                }) => {
                    entries = entries.extend_with_indices(((21, texture_view), (22, sampler)));
                }
                Some(RenderViewIrradianceVolumeBindGroupEntries::Multiple {
                    ref texture_views,
                    sampler,
                }) => {
                    entries = entries
                        .extend_with_indices(((21, texture_views.as_slice()), (22, sampler)));
                }
                None => {}
            }

            let decal_bind_group_entries = RenderViewClusteredDecalBindGroupEntries::get(
                &render_decals,
                &decals_buffer,
                &images,
                &fallback_image,
                &render_device,
                &render_adapter,
            );

            // Add the decal bind group entries.
            if let Some(ref render_view_decal_bind_group_entries) = decal_bind_group_entries {
                entries = entries.extend_with_indices((
                    // `clustered_decals`
                    (
                        23,
                        render_view_decal_bind_group_entries
                            .decals
                            .as_entire_binding(),
                    ),
                    // `clustered_decal_textures`
                    (
                        24,
                        render_view_decal_bind_group_entries
                            .texture_views
                            .as_slice(),
                    ),
                    // `clustered_decal_sampler`
                    (25, render_view_decal_bind_group_entries.sampler),
                ));
            }

            let lut_bindings =
                get_lut_bindings(&images, &tonemapping_luts, tonemapping, &fallback_image);
            entries = entries.extend_with_indices(((26, lut_bindings.0), (27, lut_bindings.1)));

            // When using WebGL, we can't have a depth texture with multisampling
            let prepass_bindings;
            if cfg!(any(not(feature = "webgl"), not(target_arch = "wasm32"))) || msaa.samples() == 1
            {
                prepass_bindings = get_bindings(prepass_textures);
                for (binding, index) in prepass_bindings
                    .iter()
                    .map(Option::as_ref)
                    .zip([28, 29, 30, 31])
                    .flat_map(|(b, i)| b.map(|b| (b, i)))
                {
                    entries = entries.extend_with_indices(((index, binding),));
                }
            };

            let transmission_view = transmission_texture
                .map(|transmission| &transmission.view)
                .unwrap_or(&fallback_image_zero.texture_view);

            let transmission_sampler = transmission_texture
                .map(|transmission| &transmission.sampler)
                .unwrap_or(&fallback_image_zero.sampler);

            entries =
                entries.extend_with_indices(((32, transmission_view), (33, transmission_sampler)));

            if has_oit {
                if let (
                    Some(oit_layers_binding),
                    Some(oit_layer_ids_binding),
                    Some(oit_settings_binding),
                ) = (
                    oit_buffers.layers.binding(),
                    oit_buffers.layer_ids.binding(),
                    oit_buffers.settings.binding(),
                ) {
                    entries = entries.extend_with_indices((
                        (34, oit_layers_binding.clone()),
                        (35, oit_layer_ids_binding.clone()),
                        (36, oit_settings_binding.clone()),
                    ));
                }
            }

            commands.entity(entity).insert(MeshViewBindGroup {
                value: render_device.create_bind_group("mesh_view_bind_group", layout, &entries),
            });
        }
    }
}

pub fn check_views_need_specialization(
    mut view_key_cache: ResMut<ViewKeyCache>,
    mut view_specialization_ticks: ResMut<ViewSpecializationTicks>,
    mut views: Query<(
        &ExtractedView,
        &Msaa,
        Option<&Tonemapping>,
        Option<&DebandDither>,
        Option<&ShadowFilteringMethod>,
        Has<ScreenSpaceAmbientOcclusion>,
        (
            Has<NormalPrepass>,
            Has<DepthPrepass>,
            Has<MotionVectorPrepass>,
            Has<DeferredPrepass>,
        ),
        Option<&Camera3d>,
        Has<TemporalJitter>,
        Option<&Projection>,
        Has<DistanceFog>,
        (
            Has<RenderViewLightProbes<EnvironmentMapLight>>,
            Has<RenderViewLightProbes<IrradianceVolume>>,
        ),
        Has<OrderIndependentTransparencySettings>,
    )>,
    ticks: SystemChangeTick,
) {
    for (
        view,
        msaa,
        tonemapping,
        dither,
        shadow_filter_method,
        ssao,
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
        camera_3d,
        temporal_jitter,
        projection,
        distance_fog,
        (has_environment_maps, has_irradiance_volumes),
        has_oit,
    ) in views.iter_mut()
    {
        let mut view_key = MeshPipelineKey::from_msaa_samples(msaa.samples())
            | MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }

        if depth_prepass {
            view_key |= MeshPipelineKey::DEPTH_PREPASS;
        }

        if motion_vector_prepass {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        if deferred_prepass {
            view_key |= MeshPipelineKey::DEFERRED_PREPASS;
        }

        if temporal_jitter {
            view_key |= MeshPipelineKey::TEMPORAL_JITTER;
        }

        if has_environment_maps {
            view_key |= MeshPipelineKey::ENVIRONMENT_MAP;
        }

        if has_irradiance_volumes {
            view_key |= MeshPipelineKey::IRRADIANCE_VOLUME;
        }

        if has_oit {
            view_key |= MeshPipelineKey::OIT_ENABLED;
        }

        if let Some(projection) = projection {
            view_key |= match projection {
                Projection::Perspective(_) => MeshPipelineKey::VIEW_PROJECTION_PERSPECTIVE,
                Projection::Orthographic(_) => MeshPipelineKey::VIEW_PROJECTION_ORTHOGRAPHIC,
                Projection::Custom(_) => MeshPipelineKey::VIEW_PROJECTION_NONSTANDARD,
            };
        }

        match shadow_filter_method.unwrap_or(&ShadowFilteringMethod::default()) {
            ShadowFilteringMethod::Hardware2x2 => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_HARDWARE_2X2;
            }
            ShadowFilteringMethod::Gaussian => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_GAUSSIAN;
            }
            ShadowFilteringMethod::Temporal => {
                view_key |= MeshPipelineKey::SHADOW_FILTER_METHOD_TEMPORAL;
            }
        }

        if !view.hdr {
            if let Some(tonemapping) = tonemapping {
                view_key |= MeshPipelineKey::TONEMAP_IN_SHADER;
                view_key |= tonemapping_pipeline_key(*tonemapping);
            }
            if let Some(DebandDither::Enabled) = dither {
                view_key |= MeshPipelineKey::DEBAND_DITHER;
            }
        }
        if ssao {
            view_key |= MeshPipelineKey::SCREEN_SPACE_AMBIENT_OCCLUSION;
        }
        if distance_fog {
            view_key |= MeshPipelineKey::DISTANCE_FOG;
        }
        if let Some(camera_3d) = camera_3d {
            view_key |= screen_space_specular_transmission_pipeline_key(
                camera_3d.screen_space_specular_transmission_quality,
            );
        }
        if !view_key_cache
            .get_mut(&view.retained_view_entity)
            .is_some_and(|current_key| *current_key == view_key)
        {
            view_key_cache.insert(view.retained_view_entity, view_key);
            view_specialization_ticks.insert(view.retained_view_entity, ticks.this_run());
        }
    }
}

/// Extracts meshes from the main world into the render world and queues
/// [`MeshInputUniform`]s to be uploaded to the GPU.
///
/// This is optimized to only look at entities that have changed since the last
/// frame.
///
/// This is the variant of the system that runs when we're using GPU
/// [`MeshUniform`] building.
pub fn extract_meshes_for_gpu_building(
    mut render_mesh_instances: ResMut<RenderMeshInstances>,
    render_visibility_ranges: Res<RenderVisibilityRanges>,
    mut render_mesh_instance_queues: ResMut<RenderMeshInstanceGpuQueues>,
    changed_meshes_query: Extract<
        Query<
            GpuMeshExtractionQuery,
            Or<(
                Changed<ViewVisibility>,
                Changed<GlobalTransform>,
                Changed<PreviousGlobalTransform>,
                Changed<Lightmap>,
                Changed<Aabb>,
                Changed<Mesh3d>,
                Changed<NoFrustumCulling>,
                Changed<NotShadowReceiver>,
                Changed<TransmittedShadowReceiver>,
                Changed<NotShadowCaster>,
                Changed<NoAutomaticBatching>,
                Changed<VisibilityRange>,
                Changed<SkinnedMesh>,
            )>,
        >,
    >,
    all_meshes_query: Extract<Query<GpuMeshExtractionQuery>>,
    mut removed_visibilities_query: Extract<RemovedComponents<ViewVisibility>>,
    mut removed_global_transforms_query: Extract<RemovedComponents<GlobalTransform>>,
    mut removed_meshes_query: Extract<RemovedComponents<Mesh3d>>,
    gpu_culling_query: Extract<Query<(), (With<Camera>, Without<NoIndirectDrawing>)>>,
    meshes_to_reextract_next_frame: ResMut<MeshesToReextractNextFrame>,
) {
    let any_gpu_culling = !gpu_culling_query.is_empty();

    for render_mesh_instance_queue in render_mesh_instance_queues.iter_mut() {
        render_mesh_instance_queue.init(any_gpu_culling);
    }

    // Collect render mesh instances. Build up the uniform buffer.

    let RenderMeshInstances::GpuBuilding(ref mut render_mesh_instances) = *render_mesh_instances
    else {
        panic!(
            "`extract_meshes_for_gpu_building` should only be called if we're \
            using GPU `MeshUniform` building"
        );
    };

    // Find all meshes that have changed, and record information needed to
    // construct the `MeshInputUniform` for them.
    changed_meshes_query.par_iter().for_each_init(
        || render_mesh_instance_queues.borrow_local_mut(),
        |queue, query_row| {
            extract_mesh_for_gpu_building(
                query_row,
                &render_visibility_ranges,
                render_mesh_instances,
                queue,
                any_gpu_culling,
            );
        },
    );

    // Process materials that `collect_meshes_for_gpu_building` marked as
    // needing to be reextracted. This will happen when we extracted a mesh on
    // some previous frame, but its material hadn't been prepared yet, perhaps
    // because the material hadn't yet been loaded. We reextract such materials
    // on subsequent frames so that `collect_meshes_for_gpu_building` will check
    // to see if their materials have been prepared.
    let mut queue = render_mesh_instance_queues.borrow_local_mut();
    for &mesh_entity in &**meshes_to_reextract_next_frame {
        if let Ok(query_row) = all_meshes_query.get(*mesh_entity) {
            extract_mesh_for_gpu_building(
                query_row,
                &render_visibility_ranges,
                render_mesh_instances,
                &mut queue,
                any_gpu_culling,
            );
        }
    }

    // Also record info about each mesh that became invisible.
    for entity in removed_visibilities_query
        .read()
        .chain(removed_global_transforms_query.read())
        .chain(removed_meshes_query.read())
    {
        // Only queue a mesh for removal if we didn't pick it up above.
        // It's possible that a necessary component was removed and re-added in
        // the same frame.
        let entity = MainEntity::from(entity);
        if !changed_meshes_query.contains(*entity)
            && !meshes_to_reextract_next_frame.contains(&entity)
        {
            queue.remove(entity, any_gpu_culling);
        }
    }
}

fn get_bindings(prepass_textures: Option<&ViewPrepassTextures>) -> [Option<TextureView>; 4] {
    let depth_desc = TextureViewDescriptor {
        label: Some("prepass_depth"),
        aspect: TextureAspect::DepthOnly,
        ..default()
    };
    let depth_view = prepass_textures
        .and_then(|x| x.depth.as_ref())
        .map(|texture| texture.texture.texture.create_view(&depth_desc));

    [
        depth_view,
        prepass_textures.and_then(|pt| pt.normal_view().cloned()),
        prepass_textures.and_then(|pt| pt.motion_vectors_view().cloned()),
        prepass_textures.and_then(|pt| pt.deferred_view().cloned()),
    ]
}

/// Creates the per-mesh bind groups for each type of mesh, for a single phase.
fn prepare_mesh_bind_groups_for_phase(
    model: BindingResource,
    meshes: &RenderAssets<RenderMesh>,
    mesh_pipeline: &MeshPipeline,
    render_device: &RenderDevice,
    skins_uniform: &SkinUniforms,
    weights_uniform: &MorphUniforms,
    render_lightmaps: &mut RenderLightmaps,
) -> MeshPhaseBindGroups {
    let layouts = &mesh_pipeline.mesh_layouts;

    // TODO: Reuse allocations.
    let mut groups = MeshPhaseBindGroups {
        model_only: Some(layouts.model_only(render_device, &model)),
        ..default()
    };

    // Create the skinned mesh bind group with the current and previous buffers
    // (the latter being for motion vector computation).
    let (skin, prev_skin) = (&skins_uniform.current_buffer, &skins_uniform.prev_buffer);
    groups.skinned = Some(MeshBindGroupPair {
        motion_vectors: layouts.skinned_motion(render_device, &model, skin, prev_skin),
        no_motion_vectors: layouts.skinned(render_device, &model, skin),
    });

    // Create the morphed bind groups just like we did for the skinned bind
    // group.
    if let Some(weights) = weights_uniform.current_buffer.buffer() {
        let prev_weights = weights_uniform.prev_buffer.buffer().unwrap_or(weights);
        for (id, gpu_mesh) in meshes.iter() {
            if let Some(targets) = gpu_mesh.morph_targets.as_ref() {
                let bind_group_pair = if is_skinned(&gpu_mesh.layout) {
                    let prev_skin = &skins_uniform.prev_buffer;
                    MeshBindGroupPair {
                        motion_vectors: layouts.morphed_skinned_motion(
                            render_device,
                            &model,
                            skin,
                            weights,
                            targets,
                            prev_skin,
                            prev_weights,
                        ),
                        no_motion_vectors: layouts.morphed_skinned(
                            render_device,
                            &model,
                            skin,
                            weights,
                            targets,
                        ),
                    }
                } else {
                    MeshBindGroupPair {
                        motion_vectors: layouts.morphed_motion(
                            render_device,
                            &model,
                            weights,
                            targets,
                            prev_weights,
                        ),
                        no_motion_vectors: layouts.morphed(render_device, &model, weights, targets),
                    }
                };
                groups.morph_targets.insert(id, bind_group_pair);
            }
        }
    }

    // Create lightmap bindgroups. There will be one bindgroup for each slab.
    let bindless_supported = render_lightmaps.bindless_supported;
    for (lightmap_slab_id, lightmap_slab) in render_lightmaps.slabs.iter_mut().enumerate() {
        groups.lightmaps.insert(
            LightmapSlabIndex(NonMaxU32::new(lightmap_slab_id as u32).unwrap()),
            layouts.lightmapped(render_device, &model, lightmap_slab, bindless_supported),
        );
    }

    groups
}

/// Extracts meshes from the main world into the render world, populating the
/// [`RenderMeshInstances`].
///
/// This is the variant of the system that runs when we're *not* using GPU
/// [`MeshUniform`] building.
pub fn extract_meshes_for_cpu_building(
    mut render_mesh_instances: ResMut<RenderMeshInstances>,
    render_visibility_ranges: Res<RenderVisibilityRanges>,
    mut render_mesh_instance_queues: Local<Parallel<Vec<(Entity, RenderMeshInstanceCpu)>>>,
    meshes_query: Extract<
        Query<(
            Entity,
            &ViewVisibility,
            &GlobalTransform,
            Option<&PreviousGlobalTransform>,
            &Mesh3d,
            Option<&MeshTag>,
            Has<NoFrustumCulling>,
            Has<NotShadowReceiver>,
            Has<TransmittedShadowReceiver>,
            Has<NotShadowCaster>,
            Has<NoAutomaticBatching>,
            Has<VisibilityRange>,
        )>,
    >,
) {
    meshes_query.par_iter().for_each_init(
        || render_mesh_instance_queues.borrow_local_mut(),
        |queue,
         (
            entity,
            view_visibility,
            transform,
            previous_transform,
            mesh,
            tag,
            no_frustum_culling,
            not_shadow_receiver,
            transmitted_receiver,
            not_shadow_caster,
            no_automatic_batching,
            visibility_range,
        )| {
            if !view_visibility.get() {
                return;
            }

            let mut lod_index = None;
            if visibility_range {
                lod_index = render_visibility_ranges.lod_index_for_entity(entity.into());
            }

            let mesh_flags = MeshFlags::from_components(
                transform,
                lod_index,
                no_frustum_culling,
                not_shadow_receiver,
                transmitted_receiver,
            );

            let shared = RenderMeshInstanceShared::from_components(
                previous_transform,
                mesh,
                tag,
                not_shadow_caster,
                no_automatic_batching,
            );

            let world_from_local = transform.affine();
            queue.push((
                entity,
                RenderMeshInstanceCpu {
                    transforms: MeshTransforms {
                        world_from_local: (&world_from_local).into(),
                        previous_world_from_local: (&previous_transform
                            .map(|t| t.0)
                            .unwrap_or(world_from_local))
                            .into(),
                        flags: mesh_flags.bits(),
                    },
                    shared,
                },
            ));
        },
    );

    // Collect the render mesh instances.
    let RenderMeshInstances::CpuBuilding(ref mut render_mesh_instances) = *render_mesh_instances
    else {
        panic!(
            "`extract_meshes_for_cpu_building` should only be called if we're using CPU \
            `MeshUniform` building"
        );
    };

    render_mesh_instances.clear();
    for queue in render_mesh_instance_queues.iter_mut() {
        for (entity, render_mesh_instance) in queue.drain(..) {
            render_mesh_instances.insert(entity.into(), render_mesh_instance);
        }
    }
}

fn extract_mesh_for_gpu_building(
    (
        entity,
        view_visibility,
        transform,
        previous_transform,
        lightmap,
        aabb,
        mesh,
        tag,
        no_frustum_culling,
        not_shadow_receiver,
        transmitted_receiver,
        not_shadow_caster,
        no_automatic_batching,
        visibility_range,
    ): <GpuMeshExtractionQuery as QueryData>::Item<'_>,
    render_visibility_ranges: &RenderVisibilityRanges,
    render_mesh_instances: &RenderMeshInstancesGpu,
    queue: &mut RenderMeshInstanceGpuQueue,
    any_gpu_culling: bool,
) {
    if !view_visibility.get() {
        queue.remove(entity.into(), any_gpu_culling);
        return;
    }

    let mut lod_index = None;
    if visibility_range {
        lod_index = render_visibility_ranges.lod_index_for_entity(entity.into());
    }

    let mesh_flags = MeshFlags::from_components(
        transform,
        lod_index,
        no_frustum_culling,
        not_shadow_receiver,
        transmitted_receiver,
    );

    let shared = RenderMeshInstanceShared::from_components(
        previous_transform,
        mesh,
        tag,
        not_shadow_caster,
        no_automatic_batching,
    );

    let lightmap_uv_rect = pack_lightmap_uv_rect(lightmap.map(|lightmap| lightmap.uv_rect));

    let gpu_mesh_culling_data = any_gpu_culling.then(|| MeshCullingData::new(aabb));

    let previous_input_index = if shared
        .flags
        .contains(RenderMeshInstanceFlags::HAS_PREVIOUS_TRANSFORM)
    {
        render_mesh_instances
            .get(&MainEntity::from(entity))
            .map(|render_mesh_instance| render_mesh_instance.current_uniform_index)
    } else {
        None
    };

    let gpu_mesh_instance_builder = RenderMeshInstanceGpuBuilder {
        shared,
        world_from_local: (&transform.affine()).into(),
        lightmap_uv_rect,
        mesh_flags,
        previous_input_index,
    };

    queue.push(
        entity.into(),
        gpu_mesh_instance_builder,
        gpu_mesh_culling_data,
    );
}

/// Creates the [`RenderMeshInstanceGpu`]s and [`MeshInputUniform`]s when GPU
/// mesh uniforms are built.
pub fn collect_meshes_for_gpu_building(
    render_mesh_instances: ResMut<RenderMeshInstances>,
    batched_instance_buffers: ResMut<
        gpu_preprocessing::BatchedInstanceBuffers<MeshUniform, MeshInputUniform>,
    >,
    mut mesh_culling_data_buffer: ResMut<MeshCullingDataBuffer>,
    mut render_mesh_instance_queues: ResMut<RenderMeshInstanceGpuQueues>,
    mesh_allocator: Res<MeshAllocator>,
    mesh_material_ids: Res<RenderMeshMaterialIds>,
    render_material_bindings: Res<RenderMaterialBindings>,
    render_lightmaps: Res<RenderLightmaps>,
    skin_uniforms: Res<SkinUniforms>,
    frame_count: Res<FrameCount>,
    mut meshes_to_reextract_next_frame: ResMut<MeshesToReextractNextFrame>,
) {
    let RenderMeshInstances::GpuBuilding(render_mesh_instances) =
        render_mesh_instances.into_inner()
    else {
        return;
    };

    // We're going to rebuild `meshes_to_reextract_next_frame`.
    meshes_to_reextract_next_frame.clear();

    // Collect render mesh instances. Build up the uniform buffer.
    let gpu_preprocessing::BatchedInstanceBuffers {
        current_input_buffer,
        previous_input_buffer,
        ..
    } = batched_instance_buffers.into_inner();

    previous_input_buffer.clear();

    // Build the [`RenderMeshInstance`]s and [`MeshInputUniform`]s.

    for queue in render_mesh_instance_queues.iter_mut() {
        match *queue {
            RenderMeshInstanceGpuQueue::None => {
                // This can only happen if the queue is empty.
            }

            RenderMeshInstanceGpuQueue::CpuCulling {
                ref mut changed,
                ref mut removed,
            } => {
                for (entity, mesh_instance_builder) in changed.drain(..) {
                    mesh_instance_builder.update(
                        entity,
                        &mut *render_mesh_instances,
                        current_input_buffer,
                        previous_input_buffer,
                        &mesh_allocator,
                        &mesh_material_ids,
                        &render_material_bindings,
                        &render_lightmaps,
                        &skin_uniforms,
                        *frame_count,
                        &mut meshes_to_reextract_next_frame,
                    );
                }

                for entity in removed.drain(..) {
                    remove_mesh_input_uniform(
                        entity,
                        &mut *render_mesh_instances,
                        current_input_buffer,
                    );
                }
            }

            RenderMeshInstanceGpuQueue::GpuCulling {
                ref mut changed,
                ref mut removed,
            } => {
                for (entity, mesh_instance_builder, mesh_culling_builder) in changed.drain(..) {
                    let Some(instance_data_index) = mesh_instance_builder.update(
                        entity,
                        &mut *render_mesh_instances,
                        current_input_buffer,
                        previous_input_buffer,
                        &mesh_allocator,
                        &mesh_material_ids,
                        &render_material_bindings,
                        &render_lightmaps,
                        &skin_uniforms,
                        *frame_count,
                        &mut meshes_to_reextract_next_frame,
                    ) else {
                        continue;
                    };
                    mesh_culling_builder
                        .update(&mut mesh_culling_data_buffer, instance_data_index as usize);
                }

                for entity in removed.drain(..) {
                    remove_mesh_input_uniform(
                        entity,
                        &mut *render_mesh_instances,
                        current_input_buffer,
                    );
                }
            }
        }
    }

    // Buffers can't be empty. Make sure there's something in the previous input buffer.
    previous_input_buffer.ensure_nonempty();
}

/// Removes a [`MeshInputUniform`] corresponding to an entity that became
/// invisible from the buffer.
fn remove_mesh_input_uniform(
    entity: MainEntity,
    render_mesh_instances: &mut MainEntityHashMap<RenderMeshInstanceGpu>,
    current_input_buffer: &mut InstanceInputUniformBuffer<MeshInputUniform>,
) -> Option<u32> {
    // Remove the uniform data.
    let removed_render_mesh_instance = render_mesh_instances.remove(&entity)?;

    let removed_uniform_index = removed_render_mesh_instance.current_uniform_index.get();
    current_input_buffer.remove(removed_uniform_index);
    Some(removed_uniform_index)
}
