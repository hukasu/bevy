use core::ops::DerefMut;

use bevy_color::{ColorToComponents, LinearRgba};
use bevy_core_pipeline::core_3d::{Camera3d, CORE_3D_DEPTH_FORMAT};
use bevy_ecs::{
    change_detection::DetectChanges,
    entity::{hash_map::EntityHashMap, hash_set::EntityHashSet, Entity},
    observer::Trigger,
    query::{AnyOf, Changed, Has, Or, With, Without},
    removal_detection::RemovedComponents,
    system::{Commands, Local, Query, Res, ResMut, SystemChangeTick},
    world::{OnAdd, OnRemove, World},
};
use bevy_math::{ops, Mat4, UVec4, Vec3A, Vec3Swizzles, Vec4, Vec4Swizzles};
use bevy_platform_support::collections::{HashMap, HashSet};
use bevy_render::{
    batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
    camera::{Camera, SortedCameras},
    experimental::occlusion_culling::{
        OcclusionCulling, OcclusionCullingSubview, OcclusionCullingSubviewEntities,
    },
    mesh::Mesh3d,
    primitives::{Aabb, CascadesFrusta, CubemapFrusta, Frustum, HalfSpace, Sphere},
    render_phase::ViewBinnedRenderPhases,
    render_resource::{
        Extent3d, TextureAspect, TextureDescriptor, TextureDimension, TextureUsages,
        TextureViewDescriptor, TextureViewDimension,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_world::{MainEntity, RenderEntity},
    texture::{DepthAttachment, TextureCache},
    view::{
        ExtractedView, InheritedVisibility, NoFrustumCulling, NoIndirectDrawing,
        PreviousVisibleEntities, RenderLayers, RetainedViewEntity, ViewVisibility, VisibilityRange,
        VisibleEntityRanges,
    },
    Extract,
};
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_utils::Parallel;
use tracing::warn;

use crate::{
    cluster::{
        cluster::ExtractedClusterConfig,
        clusterable_objects::{
            GlobalVisibleClusterableObjects, GpuClusterableObject, VisibleClusterableObjects,
        },
        plugin::calculate_cluster_factors,
        GlobalClusterableObjectMeta,
    },
    light::{
        plugin::{MAX_CASCADES_PER_LIGHT, MAX_DIRECTIONAL_LIGHTS},
        point_light::{CubeMapFace, CUBE_MAP_FACES},
        render::{
            DirectionalLightFlags, ExtractedDirectionalLight, ExtractedPointLight,
            GpuDirectionalCascade, GpuDirectionalLight, GpuLights, LightEntity, LightKeyCache,
            LightMeta, LightSpecializationTicks, LightViewEntities, PointLightFlags,
            RenderCascadesVisibleEntities, RenderCubemapVisibleEntities, RenderVisibleMeshEntities,
        },
        AmbientLight, CascadeShadowConfig, Cascades, CascadesVisibleEntities,
        CubemapVisibleEntities, DirectionalLight, DirectionalLightShadowMap, NotShadowCaster,
        PointLight, PointLightShadowMap, SpotLight, ViewLightEntities, ViewLightsUniformOffset,
        VisibleMeshEntities,
    },
    material::{Material, MeshMaterial3d},
    mesh_pipeline::{
        render::pipeline::MeshPipelineKey, specialization::EntitiesNeedingSpecialization,
    },
    shadow::{
        phase_item::Shadow,
        render::{ShadowView, ViewShadowBindings},
    },
    volumetric_fog::VolumetricLight,
};

pub fn update_directional_light_frusta(
    mut views: Query<
        (
            &Cascades,
            &DirectionalLight,
            &ViewVisibility,
            &mut CascadesFrusta,
        ),
        (
            // Prevents this query from conflicting with camera queries.
            Without<Camera>,
        ),
    >,
) {
    for (cascades, directional_light, visibility, mut frusta) in &mut views {
        // The frustum is used for culling meshes to the light for shadow mapping
        // so if shadow mapping is disabled for this light, then the frustum is
        // not needed.
        if !directional_light.shadows_enabled || !visibility.get() {
            continue;
        }

        frusta.frusta = cascades
            .cascades
            .iter()
            .map(|(view, cascades)| {
                (
                    *view,
                    cascades
                        .iter()
                        .map(|c| Frustum::from_clip_from_world(&c.clip_from_world))
                        .collect::<Vec<_>>(),
                )
            })
            .collect();
    }
}

pub fn update_spot_light_frusta(
    global_lights: Res<GlobalVisibleClusterableObjects>,
    mut views: Query<
        (Entity, &GlobalTransform, &SpotLight, &mut Frustum),
        Or<(Changed<GlobalTransform>, Changed<SpotLight>)>,
    >,
) {
    for (entity, transform, spot_light, mut frustum) in &mut views {
        // The frusta are used for culling meshes to the light for shadow mapping
        // so if shadow mapping is disabled for this light, then the frusta are
        // not needed.
        // Also, if the light is not relevant for any cluster, it will not be in the
        // global lights set and so there is no need to update its frusta.
        if !spot_light.shadows_enabled || !global_lights.entities.contains(&entity) {
            continue;
        }

        // ignore scale because we don't want to effectively scale light radius and range
        // by applying those as a view transform to shadow map rendering of objects
        let view_backward = transform.back();

        let spot_world_from_view = spot_light_world_from_view(transform);
        let spot_clip_from_view =
            spot_light_clip_from_view(spot_light.outer_angle, spot_light.shadow_map_near_z);
        let clip_from_world = spot_clip_from_view * spot_world_from_view.inverse();

        *frustum = Frustum::from_clip_from_world_custom_far(
            &clip_from_world,
            &transform.translation(),
            &view_backward,
            spot_light.range,
        );
    }
}
// NOTE: Run this after assign_lights_to_clusters!
pub fn update_point_light_frusta(
    global_lights: Res<GlobalVisibleClusterableObjects>,
    mut views: Query<
        (Entity, &GlobalTransform, &PointLight, &mut CubemapFrusta),
        Or<(Changed<GlobalTransform>, Changed<PointLight>)>,
    >,
) {
    let view_rotations = CUBE_MAP_FACES
        .iter()
        .map(|CubeMapFace { target, up }| Transform::IDENTITY.looking_at(*target, *up))
        .collect::<Vec<_>>();

    for (entity, transform, point_light, mut cubemap_frusta) in &mut views {
        // The frusta are used for culling meshes to the light for shadow mapping
        // so if shadow mapping is disabled for this light, then the frusta are
        // not needed.
        // Also, if the light is not relevant for any cluster, it will not be in the
        // global lights set and so there is no need to update its frusta.
        if !point_light.shadows_enabled || !global_lights.entities.contains(&entity) {
            continue;
        }

        let clip_from_view = Mat4::perspective_infinite_reverse_rh(
            core::f32::consts::FRAC_PI_2,
            1.0,
            point_light.shadow_map_near_z,
        );

        // ignore scale because we don't want to effectively scale light radius and range
        // by applying those as a view transform to shadow map rendering of objects
        // and ignore rotation because we want the shadow map projections to align with the axes
        let view_translation = Transform::from_translation(transform.translation());
        let view_backward = transform.back();

        for (view_rotation, frustum) in view_rotations.iter().zip(cubemap_frusta.iter_mut()) {
            let world_from_view = view_translation * *view_rotation;
            let clip_from_world = clip_from_view * world_from_view.compute_matrix().inverse();

            *frustum = Frustum::from_clip_from_world_custom_far(
                &clip_from_world,
                &transform.translation(),
                &view_backward,
                point_light.range,
            );
        }
    }
}

pub fn check_dir_light_mesh_visibility(
    mut commands: Commands,
    mut directional_lights: Query<
        (
            &DirectionalLight,
            &CascadesFrusta,
            &mut CascadesVisibleEntities,
            Option<&RenderLayers>,
            &ViewVisibility,
        ),
        Without<SpotLight>,
    >,
    visible_entity_query: Query<
        (
            Entity,
            &InheritedVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
            Has<VisibilityRange>,
            Has<NoFrustumCulling>,
        ),
        (
            Without<NotShadowCaster>,
            Without<DirectionalLight>,
            With<Mesh3d>,
        ),
    >,
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    mut defer_visible_entities_queue: Local<Parallel<Vec<Entity>>>,
    mut view_visible_entities_queue: Local<Parallel<Vec<Vec<Entity>>>>,
) {
    let visible_entity_ranges = visible_entity_ranges.as_deref();

    for (directional_light, frusta, mut visible_entities, maybe_view_mask, light_view_visibility) in
        &mut directional_lights
    {
        let mut views_to_remove = Vec::new();
        for (view, cascade_view_entities) in &mut visible_entities.entities {
            match frusta.frusta.get(view) {
                Some(view_frusta) => {
                    cascade_view_entities.resize(view_frusta.len(), Default::default());
                    cascade_view_entities.iter_mut().for_each(|x| x.clear());
                }
                None => views_to_remove.push(*view),
            };
        }
        for (view, frusta) in &frusta.frusta {
            visible_entities
                .entities
                .entry(*view)
                .or_insert_with(|| vec![VisibleMeshEntities::default(); frusta.len()]);
        }

        for v in views_to_remove {
            visible_entities.entities.remove(&v);
        }

        // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
        if !directional_light.shadows_enabled || !light_view_visibility.get() {
            continue;
        }

        let view_mask = maybe_view_mask.unwrap_or_default();

        for (view, view_frusta) in &frusta.frusta {
            visible_entity_query.par_iter().for_each_init(
                || {
                    let mut entities = view_visible_entities_queue.borrow_local_mut();
                    entities.resize(view_frusta.len(), Vec::default());
                    (defer_visible_entities_queue.borrow_local_mut(), entities)
                },
                |(defer_visible_entities_local_queue, view_visible_entities_local_queue),
                 (
                    entity,
                    inherited_visibility,
                    maybe_entity_mask,
                    maybe_aabb,
                    maybe_transform,
                    has_visibility_range,
                    has_no_frustum_culling,
                )| {
                    if !inherited_visibility.get() {
                        return;
                    }

                    let entity_mask = maybe_entity_mask.unwrap_or_default();
                    if !view_mask.intersects(entity_mask) {
                        return;
                    }

                    // Check visibility ranges.
                    if has_visibility_range
                        && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                            !visible_entity_ranges.entity_is_in_range_of_view(entity, *view)
                        })
                    {
                        return;
                    }

                    if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                        let mut visible = false;
                        for (frustum, frustum_visible_entities) in view_frusta
                            .iter()
                            .zip(view_visible_entities_local_queue.iter_mut())
                        {
                            // Disable near-plane culling, as a shadow caster could lie before the near plane.
                            if !has_no_frustum_culling
                                && !frustum.intersects_obb(aabb, &transform.affine(), false, true)
                            {
                                continue;
                            }
                            visible = true;

                            frustum_visible_entities.push(entity);
                        }
                        if visible {
                            defer_visible_entities_local_queue.push(entity);
                        }
                    } else {
                        defer_visible_entities_local_queue.push(entity);
                        for frustum_visible_entities in view_visible_entities_local_queue.iter_mut()
                        {
                            frustum_visible_entities.push(entity);
                        }
                    }
                },
            );
            // collect entities from parallel queue
            for entities in view_visible_entities_queue.iter_mut() {
                visible_entities
                    .entities
                    .get_mut(view)
                    .unwrap()
                    .iter_mut()
                    .zip(entities.iter_mut())
                    .for_each(|(dst, source)| {
                        dst.append(source);
                    });
            }
        }

        for (_, cascade_view_entities) in &mut visible_entities.entities {
            cascade_view_entities
                .iter_mut()
                .map(DerefMut::deref_mut)
                .for_each(shrink_entities);
        }
    }

    // Defer marking view visibility so this system can run in parallel with check_point_light_mesh_visibility
    // TODO: use resource to avoid unnecessary memory alloc
    let mut defer_queue = core::mem::take(defer_visible_entities_queue.deref_mut());
    commands.queue(move |world: &mut World| {
        world.resource_scope::<PreviousVisibleEntities, _>(
            |world, mut previous_visible_entities| {
                let mut query = world.query::<(Entity, &mut ViewVisibility)>();
                for entities in defer_queue.iter_mut() {
                    let mut iter = query.iter_many_mut(world, entities.iter());
                    while let Some((entity, mut view_visibility)) = iter.fetch_next() {
                        if !**view_visibility {
                            view_visibility.set();
                        }

                        // Remove any entities that were discovered to be
                        // visible from the `PreviousVisibleEntities` resource.
                        previous_visible_entities.remove(&entity);
                    }
                }
            },
        );
    });
}

pub fn check_point_light_mesh_visibility(
    visible_point_lights: Query<&VisibleClusterableObjects>,
    mut point_lights: Query<(
        &PointLight,
        &GlobalTransform,
        &CubemapFrusta,
        &mut CubemapVisibleEntities,
        Option<&RenderLayers>,
    )>,
    mut spot_lights: Query<(
        &SpotLight,
        &GlobalTransform,
        &Frustum,
        &mut VisibleMeshEntities,
        Option<&RenderLayers>,
    )>,
    mut visible_entity_query: Query<
        (
            Entity,
            &InheritedVisibility,
            &mut ViewVisibility,
            Option<&RenderLayers>,
            Option<&Aabb>,
            Option<&GlobalTransform>,
            Has<VisibilityRange>,
            Has<NoFrustumCulling>,
        ),
        (
            Without<NotShadowCaster>,
            Without<DirectionalLight>,
            With<Mesh3d>,
        ),
    >,
    visible_entity_ranges: Option<Res<VisibleEntityRanges>>,
    mut previous_visible_entities: ResMut<PreviousVisibleEntities>,
    mut cubemap_visible_entities_queue: Local<Parallel<[Vec<Entity>; 6]>>,
    mut spot_visible_entities_queue: Local<Parallel<Vec<Entity>>>,
    mut checked_lights: Local<EntityHashSet>,
) {
    checked_lights.clear();

    let visible_entity_ranges = visible_entity_ranges.as_deref();
    for visible_lights in &visible_point_lights {
        for light_entity in visible_lights.entities.iter().copied() {
            if !checked_lights.insert(light_entity) {
                continue;
            }

            // Point lights
            if let Ok((
                point_light,
                transform,
                cubemap_frusta,
                mut cubemap_visible_entities,
                maybe_view_mask,
            )) = point_lights.get_mut(light_entity)
            {
                for visible_entities in cubemap_visible_entities.iter_mut() {
                    visible_entities.entities.clear();
                }

                // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
                if !point_light.shadows_enabled {
                    continue;
                }

                let view_mask = maybe_view_mask.unwrap_or_default();
                let light_sphere = Sphere {
                    center: Vec3A::from(transform.translation()),
                    radius: point_light.range,
                };

                visible_entity_query.par_iter_mut().for_each_init(
                    || cubemap_visible_entities_queue.borrow_local_mut(),
                    |cubemap_visible_entities_local_queue,
                     (
                        entity,
                        inherited_visibility,
                        mut view_visibility,
                        maybe_entity_mask,
                        maybe_aabb,
                        maybe_transform,
                        has_visibility_range,
                        has_no_frustum_culling,
                    )| {
                        if !inherited_visibility.get() {
                            return;
                        }
                        let entity_mask = maybe_entity_mask.unwrap_or_default();
                        if !view_mask.intersects(entity_mask) {
                            return;
                        }
                        if has_visibility_range
                            && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                                !visible_entity_ranges.entity_is_in_range_of_any_view(entity)
                            })
                        {
                            return;
                        }

                        // If we have an aabb and transform, do frustum culling
                        if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                            let model_to_world = transform.affine();
                            // Do a cheap sphere vs obb test to prune out most meshes outside the sphere of the light
                            if !has_no_frustum_culling
                                && !light_sphere.intersects_obb(aabb, &model_to_world)
                            {
                                return;
                            }

                            for (frustum, visible_entities) in cubemap_frusta
                                .iter()
                                .zip(cubemap_visible_entities_local_queue.iter_mut())
                            {
                                if has_no_frustum_culling
                                    || frustum.intersects_obb(aabb, &model_to_world, true, true)
                                {
                                    if !**view_visibility {
                                        view_visibility.set();
                                    }
                                    visible_entities.push(entity);
                                }
                            }
                        } else {
                            if !**view_visibility {
                                view_visibility.set();
                            }
                            for visible_entities in cubemap_visible_entities_local_queue.iter_mut()
                            {
                                visible_entities.push(entity);
                            }
                        }
                    },
                );

                for entities in cubemap_visible_entities_queue.iter_mut() {
                    for (dst, source) in
                        cubemap_visible_entities.iter_mut().zip(entities.iter_mut())
                    {
                        // Remove any entities that were discovered to be
                        // visible from the `PreviousVisibleEntities` resource.
                        for entity in source.iter() {
                            previous_visible_entities.remove(entity);
                        }

                        dst.entities.append(source);
                    }
                }

                for visible_entities in cubemap_visible_entities.iter_mut() {
                    shrink_entities(visible_entities);
                }
            }

            // Spot lights
            if let Ok((point_light, transform, frustum, mut visible_entities, maybe_view_mask)) =
                spot_lights.get_mut(light_entity)
            {
                visible_entities.clear();

                // NOTE: If shadow mapping is disabled for the light then it must have no visible entities
                if !point_light.shadows_enabled {
                    continue;
                }

                let view_mask = maybe_view_mask.unwrap_or_default();
                let light_sphere = Sphere {
                    center: Vec3A::from(transform.translation()),
                    radius: point_light.range,
                };

                visible_entity_query.par_iter_mut().for_each_init(
                    || spot_visible_entities_queue.borrow_local_mut(),
                    |spot_visible_entities_local_queue,
                     (
                        entity,
                        inherited_visibility,
                        mut view_visibility,
                        maybe_entity_mask,
                        maybe_aabb,
                        maybe_transform,
                        has_visibility_range,
                        has_no_frustum_culling,
                    )| {
                        if !inherited_visibility.get() {
                            return;
                        }

                        let entity_mask = maybe_entity_mask.unwrap_or_default();
                        if !view_mask.intersects(entity_mask) {
                            return;
                        }
                        // Check visibility ranges.
                        if has_visibility_range
                            && visible_entity_ranges.is_some_and(|visible_entity_ranges| {
                                !visible_entity_ranges.entity_is_in_range_of_any_view(entity)
                            })
                        {
                            return;
                        }

                        if let (Some(aabb), Some(transform)) = (maybe_aabb, maybe_transform) {
                            let model_to_world = transform.affine();
                            // Do a cheap sphere vs obb test to prune out most meshes outside the sphere of the light
                            if !has_no_frustum_culling
                                && !light_sphere.intersects_obb(aabb, &model_to_world)
                            {
                                return;
                            }

                            if has_no_frustum_culling
                                || frustum.intersects_obb(aabb, &model_to_world, true, true)
                            {
                                if !**view_visibility {
                                    view_visibility.set();
                                }
                                spot_visible_entities_local_queue.push(entity);
                            }
                        } else {
                            if !**view_visibility {
                                view_visibility.set();
                            }
                            spot_visible_entities_local_queue.push(entity);
                        }
                    },
                );

                for entities in spot_visible_entities_queue.iter_mut() {
                    visible_entities.append(entities);

                    // Remove any entities that were discovered to be visible
                    // from the `PreviousVisibleEntities` resource.
                    for entity in entities {
                        previous_visible_entities.remove(entity);
                    }
                }

                shrink_entities(visible_entities.deref_mut());
            }
        }
    }
}

pub fn prepare_lights(
    mut commands: Commands,
    mut texture_cache: ResMut<TextureCache>,
    (render_device, render_queue): (Res<RenderDevice>, Res<RenderQueue>),
    mut global_light_meta: ResMut<GlobalClusterableObjectMeta>,
    mut light_meta: ResMut<LightMeta>,
    views: Query<
        (
            Entity,
            MainEntity,
            &ExtractedView,
            &ExtractedClusterConfig,
            Option<&RenderLayers>,
            Has<NoIndirectDrawing>,
            Option<&AmbientLight>,
        ),
        With<Camera3d>,
    >,
    ambient_light: Res<AmbientLight>,
    point_light_shadow_map: Res<PointLightShadowMap>,
    directional_light_shadow_map: Res<DirectionalLightShadowMap>,
    mut shadow_render_phases: ResMut<ViewBinnedRenderPhases<Shadow>>,
    (
        mut max_directional_lights_warning_emitted,
        mut max_cascades_per_light_warning_emitted,
        mut live_shadow_mapping_lights,
    ): (Local<bool>, Local<bool>, Local<HashSet<RetainedViewEntity>>),
    point_lights: Query<(
        Entity,
        &MainEntity,
        &ExtractedPointLight,
        AnyOf<(&CubemapFrusta, &Frustum)>,
    )>,
    directional_lights: Query<(Entity, &MainEntity, &ExtractedDirectionalLight)>,
    mut light_view_entities: Query<&mut LightViewEntities>,
    sorted_cameras: Res<SortedCameras>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    let views_iter = views.iter();
    let views_count = views_iter.len();
    let Some(mut view_gpu_lights_writer) =
        light_meta
            .view_gpu_lights
            .get_writer(views_count, &render_device, &render_queue)
    else {
        return;
    };

    // Pre-calculate for PointLights
    let cube_face_rotations = CUBE_MAP_FACES
        .iter()
        .map(|CubeMapFace { target, up }| Transform::IDENTITY.looking_at(*target, *up))
        .collect::<Vec<_>>();

    global_light_meta.entity_to_index.clear();

    let mut point_lights: Vec<_> = point_lights.iter().collect::<Vec<_>>();
    let mut directional_lights: Vec<_> = directional_lights.iter().collect::<Vec<_>>();

    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    let max_texture_array_layers = render_device.limits().max_texture_array_layers as usize;
    #[cfg(any(
        not(feature = "webgl"),
        not(target_arch = "wasm32"),
        feature = "webgpu"
    ))]
    let max_texture_cubes = max_texture_array_layers / 6;
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    let max_texture_array_layers = 1;
    #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
    let max_texture_cubes = 1;

    if !*max_directional_lights_warning_emitted && directional_lights.len() > MAX_DIRECTIONAL_LIGHTS
    {
        warn!(
            "The amount of directional lights of {} is exceeding the supported limit of {}.",
            directional_lights.len(),
            MAX_DIRECTIONAL_LIGHTS
        );
        *max_directional_lights_warning_emitted = true;
    }

    if !*max_cascades_per_light_warning_emitted
        && directional_lights
            .iter()
            .any(|(_, _, light)| light.cascade_shadow_config.bounds.len() > MAX_CASCADES_PER_LIGHT)
    {
        warn!(
            "The number of cascades configured for a directional light exceeds the supported limit of {}.",
            MAX_CASCADES_PER_LIGHT
        );
        *max_cascades_per_light_warning_emitted = true;
    }

    let point_light_count = point_lights
        .iter()
        .filter(|light| light.2.spot_light_angles.is_none())
        .count();

    let point_light_volumetric_enabled_count = point_lights
        .iter()
        .filter(|(_, _, light, _)| light.volumetric && light.spot_light_angles.is_none())
        .count()
        .min(max_texture_cubes);

    let point_light_shadow_maps_count = point_lights
        .iter()
        .filter(|light| light.2.shadows_enabled && light.2.spot_light_angles.is_none())
        .count()
        .min(max_texture_cubes);

    let directional_volumetric_enabled_count = directional_lights
        .iter()
        .take(MAX_DIRECTIONAL_LIGHTS)
        .filter(|(_, _, light)| light.volumetric)
        .count()
        .min(max_texture_array_layers / MAX_CASCADES_PER_LIGHT);

    let directional_shadow_enabled_count = directional_lights
        .iter()
        .take(MAX_DIRECTIONAL_LIGHTS)
        .filter(|(_, _, light)| light.shadows_enabled)
        .count()
        .min(max_texture_array_layers / MAX_CASCADES_PER_LIGHT);

    let spot_light_count = point_lights
        .iter()
        .filter(|(_, _, light, _)| light.spot_light_angles.is_some())
        .count()
        .min(max_texture_array_layers - directional_shadow_enabled_count * MAX_CASCADES_PER_LIGHT);

    let spot_light_volumetric_enabled_count = point_lights
        .iter()
        .filter(|(_, _, light, _)| light.volumetric && light.spot_light_angles.is_some())
        .count()
        .min(max_texture_array_layers - directional_shadow_enabled_count * MAX_CASCADES_PER_LIGHT);

    let spot_light_shadow_maps_count = point_lights
        .iter()
        .filter(|(_, _, light, _)| light.shadows_enabled && light.spot_light_angles.is_some())
        .count()
        .min(max_texture_array_layers - directional_shadow_enabled_count * MAX_CASCADES_PER_LIGHT);

    // Sort lights by
    // - point-light vs spot-light, so that we can iterate point lights and spot lights in contiguous blocks in the fragment shader,
    // - then those with shadows enabled first, so that the index can be used to render at most `point_light_shadow_maps_count`
    //   point light shadows and `spot_light_shadow_maps_count` spot light shadow maps,
    // - then by entity as a stable key to ensure that a consistent set of lights are chosen if the light count limit is exceeded.
    point_lights.sort_by_cached_key(|(entity, _, light, _)| {
        (
            match light.spot_light_angles {
                Some(_) => (1, !light.shadows_enabled, !light.volumetric),
                None => (0, !light.shadows_enabled, !light.volumetric),
            },
            *entity,
        )
    });

    // Sort lights by
    // - those with volumetric (and shadows) enabled first, so that the
    //   volumetric lighting pass can quickly find the volumetric lights;
    // - then those with shadows enabled second, so that the index can be used
    //   to render at most `directional_light_shadow_maps_count` directional light
    //   shadows
    // - then by entity as a stable key to ensure that a consistent set of
    //   lights are chosen if the light count limit is exceeded.
    // - because entities are unique, we can use `sort_unstable_by_key`
    //   and still end up with a stable order.
    directional_lights.sort_unstable_by_key(|(entity, _, light)| {
        (light.volumetric, light.shadows_enabled, *entity)
    });

    if global_light_meta.entity_to_index.capacity() < point_lights.len() {
        global_light_meta
            .entity_to_index
            .reserve(point_lights.len());
    }

    let mut gpu_point_lights = Vec::new();
    for (index, &(entity, _, light, _)) in point_lights.iter().enumerate() {
        let mut flags = PointLightFlags::NONE;

        // Lights are sorted, shadow enabled lights are first
        if light.shadows_enabled
            && (index < point_light_shadow_maps_count
                || (light.spot_light_angles.is_some()
                    && index - point_light_count < spot_light_shadow_maps_count))
        {
            flags |= PointLightFlags::SHADOWS_ENABLED;
        }

        let cube_face_projection = Mat4::perspective_infinite_reverse_rh(
            core::f32::consts::FRAC_PI_2,
            1.0,
            light.shadow_map_near_z,
        );
        if light.shadows_enabled
            && light.volumetric
            && (index < point_light_volumetric_enabled_count
                || (light.spot_light_angles.is_some()
                    && index - point_light_count < spot_light_volumetric_enabled_count))
        {
            flags |= PointLightFlags::VOLUMETRIC;
        }

        if light.affects_lightmapped_mesh_diffuse {
            flags |= PointLightFlags::AFFECTS_LIGHTMAPPED_MESH_DIFFUSE;
        }

        let (light_custom_data, spot_light_tan_angle) = match light.spot_light_angles {
            Some((inner, outer)) => {
                let light_direction = light.transform.forward();
                if light_direction.y.is_sign_negative() {
                    flags |= PointLightFlags::SPOT_LIGHT_Y_NEGATIVE;
                }

                let cos_outer = ops::cos(outer);
                let spot_scale = 1.0 / f32::max(ops::cos(inner) - cos_outer, 1e-4);
                let spot_offset = -cos_outer * spot_scale;

                (
                    // For spot lights: the direction (x,z), spot_scale and spot_offset
                    light_direction.xz().extend(spot_scale).extend(spot_offset),
                    ops::tan(outer),
                )
            }
            None => {
                (
                    // For point lights: the lower-right 2x2 values of the projection matrix [2][2] [2][3] [3][2] [3][3]
                    Vec4::new(
                        cube_face_projection.z_axis.z,
                        cube_face_projection.z_axis.w,
                        cube_face_projection.w_axis.z,
                        cube_face_projection.w_axis.w,
                    ),
                    // unused
                    0.0,
                )
            }
        };

        gpu_point_lights.push(GpuClusterableObject {
            light_custom_data,
            // premultiply color by intensity
            // we don't use the alpha at all, so no reason to multiply only [0..3]
            color_inverse_square_range: (Vec4::from_slice(&light.color.to_f32_array())
                * light.intensity)
                .xyz()
                .extend(1.0 / (light.range * light.range)),
            position_radius: light.transform.translation().extend(light.radius),
            flags: flags.bits(),
            shadow_depth_bias: light.shadow_depth_bias,
            shadow_normal_bias: light.shadow_normal_bias,
            shadow_map_near_z: light.shadow_map_near_z,
            spot_light_tan_angle,
            pad_a: 0.0,
            pad_b: 0.0,
            soft_shadow_size: if light.soft_shadows_enabled {
                light.radius
            } else {
                0.0
            },
        });
        global_light_meta.entity_to_index.insert(entity, index);
    }

    let mut gpu_directional_lights = [GpuDirectionalLight::default(); MAX_DIRECTIONAL_LIGHTS];
    let mut num_directional_cascades_enabled = 0usize;
    for (index, (_light_entity, _, light)) in directional_lights
        .iter()
        .enumerate()
        .take(MAX_DIRECTIONAL_LIGHTS)
    {
        let mut flags = DirectionalLightFlags::NONE;

        // Lights are sorted, volumetric and shadow enabled lights are first
        if light.volumetric
            && light.shadows_enabled
            && (index < directional_volumetric_enabled_count)
        {
            flags |= DirectionalLightFlags::VOLUMETRIC;
        }
        // Shadow enabled lights are second
        if light.shadows_enabled && (index < directional_shadow_enabled_count) {
            flags |= DirectionalLightFlags::SHADOWS_ENABLED;
        }

        if light.affects_lightmapped_mesh_diffuse {
            flags |= DirectionalLightFlags::AFFECTS_LIGHTMAPPED_MESH_DIFFUSE;
        }

        let num_cascades = light
            .cascade_shadow_config
            .bounds
            .len()
            .min(MAX_CASCADES_PER_LIGHT);
        gpu_directional_lights[index] = GpuDirectionalLight {
            // Set to true later when necessary.
            skip: 0u32,
            // Filled in later.
            cascades: [GpuDirectionalCascade::default(); MAX_CASCADES_PER_LIGHT],
            // premultiply color by illuminance
            // we don't use the alpha at all, so no reason to multiply only [0..3]
            color: Vec4::from_slice(&light.color.to_f32_array()) * light.illuminance,
            // direction is negated to be ready for N.L
            dir_to_light: light.transform.back().into(),
            flags: flags.bits(),
            soft_shadow_size: light.soft_shadow_size.unwrap_or_default(),
            shadow_depth_bias: light.shadow_depth_bias,
            shadow_normal_bias: light.shadow_normal_bias,
            num_cascades: num_cascades as u32,
            cascades_overlap_proportion: light.cascade_shadow_config.overlap_proportion,
            depth_texture_base_index: num_directional_cascades_enabled as u32,
        };
        if index < directional_shadow_enabled_count {
            num_directional_cascades_enabled += num_cascades;
        }
    }

    global_light_meta
        .gpu_clusterable_objects
        .set(gpu_point_lights);
    global_light_meta
        .gpu_clusterable_objects
        .write_buffer(&render_device, &render_queue);

    live_shadow_mapping_lights.clear();

    let mut point_light_depth_attachments = HashMap::<u32, DepthAttachment>::default();
    let mut directional_light_depth_attachments = HashMap::<u32, DepthAttachment>::default();

    let point_light_depth_texture = texture_cache.get(
        &render_device,
        TextureDescriptor {
            size: Extent3d {
                width: point_light_shadow_map.size as u32,
                height: point_light_shadow_map.size as u32,
                depth_or_array_layers: point_light_shadow_maps_count.max(1) as u32 * 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            label: Some("point_light_shadow_map_texture"),
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    let point_light_depth_texture_view =
        point_light_depth_texture
            .texture
            .create_view(&TextureViewDescriptor {
                label: Some("point_light_shadow_map_array_texture_view"),
                format: None,
                // NOTE: iOS Simulator is missing CubeArray support so we use Cube instead.
                // See https://github.com/bevyengine/bevy/pull/12052 - remove if support is added.
                #[cfg(all(
                    not(target_abi = "sim"),
                    any(
                        not(feature = "webgl"),
                        not(target_arch = "wasm32"),
                        feature = "webgpu"
                    )
                ))]
                dimension: Some(TextureViewDimension::CubeArray),
                #[cfg(any(
                    target_abi = "sim",
                    all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu"))
                ))]
                dimension: Some(TextureViewDimension::Cube),
                usage: None,
                aspect: TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            });

    let directional_light_depth_texture = texture_cache.get(
        &render_device,
        TextureDescriptor {
            size: Extent3d {
                width: (directional_light_shadow_map.size as u32)
                    .min(render_device.limits().max_texture_dimension_2d),
                height: (directional_light_shadow_map.size as u32)
                    .min(render_device.limits().max_texture_dimension_2d),
                depth_or_array_layers: (num_directional_cascades_enabled
                    + spot_light_shadow_maps_count)
                    .max(1) as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: CORE_3D_DEPTH_FORMAT,
            label: Some("directional_light_shadow_map_texture"),
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
    );

    let directional_light_depth_texture_view =
        directional_light_depth_texture
            .texture
            .create_view(&TextureViewDescriptor {
                label: Some("directional_light_shadow_map_array_texture_view"),
                format: None,
                #[cfg(any(
                    not(feature = "webgl"),
                    not(target_arch = "wasm32"),
                    feature = "webgpu"
                ))]
                dimension: Some(TextureViewDimension::D2Array),
                #[cfg(all(feature = "webgl", target_arch = "wasm32", not(feature = "webgpu")))]
                dimension: Some(TextureViewDimension::D2),
                usage: None,
                aspect: TextureAspect::DepthOnly,
                base_mip_level: 0,
                mip_level_count: None,
                base_array_layer: 0,
                array_layer_count: None,
            });

    let mut live_views = EntityHashSet::with_capacity(views_count);

    // set up light data for each view
    for (
        entity,
        camera_main_entity,
        extracted_view,
        clusters,
        maybe_layers,
        no_indirect_drawing,
        maybe_ambient_override,
    ) in sorted_cameras
        .0
        .iter()
        .filter_map(|sorted_camera| views.get(sorted_camera.entity).ok())
    {
        live_views.insert(entity);

        let mut view_lights = Vec::new();
        let mut view_occlusion_culling_lights = Vec::new();

        let gpu_preprocessing_mode = gpu_preprocessing_support.min(if !no_indirect_drawing {
            GpuPreprocessingMode::Culling
        } else {
            GpuPreprocessingMode::PreprocessingOnly
        });

        let is_orthographic = extracted_view.clip_from_view.w_axis.w == 1.0;
        let cluster_factors_zw = calculate_cluster_factors(
            clusters.near,
            clusters.far,
            clusters.dimensions.z as f32,
            is_orthographic,
        );

        let n_clusters = clusters.dimensions.x * clusters.dimensions.y * clusters.dimensions.z;
        let ambient_light = maybe_ambient_override.unwrap_or(&ambient_light);
        let mut gpu_lights = GpuLights {
            directional_lights: gpu_directional_lights,
            ambient_color: Vec4::from_slice(&LinearRgba::from(ambient_light.color).to_f32_array())
                * ambient_light.brightness,
            cluster_factors: Vec4::new(
                clusters.dimensions.x as f32 / extracted_view.viewport.z as f32,
                clusters.dimensions.y as f32 / extracted_view.viewport.w as f32,
                cluster_factors_zw.x,
                cluster_factors_zw.y,
            ),
            cluster_dimensions: clusters.dimensions.extend(n_clusters),
            n_directional_lights: directional_lights.iter().len().min(MAX_DIRECTIONAL_LIGHTS)
                as u32,
            // spotlight shadow maps are stored in the directional light array, starting at num_directional_cascades_enabled.
            // the spot lights themselves start in the light array at point_light_count. so to go from light
            // index to shadow map index, we need to subtract point light count and add directional shadowmap count.
            spot_light_shadowmap_offset: num_directional_cascades_enabled as i32
                - point_light_count as i32,
            ambient_light_affects_lightmapped_meshes: ambient_light.affects_lightmapped_meshes
                as u32,
        };

        // TODO: this should select lights based on relevance to the view instead of the first ones that show up in a query
        for &(light_entity, light_main_entity, light, (point_light_frusta, _)) in point_lights
            .iter()
            // Lights are sorted, shadow enabled lights are first
            .take(point_light_count.min(max_texture_cubes))
        {
            let Ok(mut light_view_entities) = light_view_entities.get_mut(light_entity) else {
                continue;
            };

            if !light.shadows_enabled {
                if let Some(entities) = light_view_entities.remove(&entity) {
                    despawn_entities(&mut commands, entities);
                }
                continue;
            }

            let light_index = *global_light_meta
                .entity_to_index
                .get(&light_entity)
                .unwrap();
            // ignore scale because we don't want to effectively scale light radius and range
            // by applying those as a view transform to shadow map rendering of objects
            // and ignore rotation because we want the shadow map projections to align with the axes
            let view_translation = GlobalTransform::from_translation(light.transform.translation());

            // for each face of a cube and each view we spawn a light entity
            let light_view_entities = light_view_entities
                .entry(entity)
                .or_insert_with(|| (0..6).map(|_| commands.spawn_empty().id()).collect());

            let cube_face_projection = Mat4::perspective_infinite_reverse_rh(
                core::f32::consts::FRAC_PI_2,
                1.0,
                light.shadow_map_near_z,
            );

            for (face_index, ((view_rotation, frustum), view_light_entity)) in cube_face_rotations
                .iter()
                .zip(&point_light_frusta.unwrap().frusta)
                .zip(light_view_entities.iter().copied())
                .enumerate()
            {
                let mut first = false;
                let base_array_layer = (light_index * 6 + face_index) as u32;

                let depth_attachment = point_light_depth_attachments
                    .entry(base_array_layer)
                    .or_insert_with(|| {
                        first = true;

                        let depth_texture_view =
                            point_light_depth_texture
                                .texture
                                .create_view(&TextureViewDescriptor {
                                    label: Some("point_light_shadow_map_texture_view"),
                                    format: None,
                                    dimension: Some(TextureViewDimension::D2),
                                    usage: None,
                                    aspect: TextureAspect::All,
                                    base_mip_level: 0,
                                    mip_level_count: None,
                                    base_array_layer,
                                    array_layer_count: Some(1u32),
                                });

                        DepthAttachment::new(depth_texture_view, Some(0.0))
                    })
                    .clone();

                let retained_view_entity = RetainedViewEntity::new(
                    *light_main_entity,
                    Some(camera_main_entity.into()),
                    face_index as u32,
                );

                commands.entity(view_light_entity).insert((
                    ShadowView {
                        depth_attachment,
                        pass_name: format!(
                            "shadow pass point light {} {}",
                            light_index,
                            face_index_to_name(face_index)
                        ),
                    },
                    ExtractedView {
                        retained_view_entity,
                        viewport: UVec4::new(
                            0,
                            0,
                            point_light_shadow_map.size as u32,
                            point_light_shadow_map.size as u32,
                        ),
                        world_from_view: view_translation * *view_rotation,
                        clip_from_world: None,
                        clip_from_view: cube_face_projection,
                        hdr: false,
                        color_grading: Default::default(),
                    },
                    *frustum,
                    LightEntity::Point {
                        light_entity,
                        face_index,
                    },
                ));

                if !matches!(gpu_preprocessing_mode, GpuPreprocessingMode::Culling) {
                    commands.entity(view_light_entity).insert(NoIndirectDrawing);
                }

                view_lights.push(view_light_entity);

                if first {
                    // Subsequent views with the same light entity will reuse the same shadow map
                    shadow_render_phases
                        .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
                    live_shadow_mapping_lights.insert(retained_view_entity);
                }
            }
        }

        // spot lights
        for (light_index, &(light_entity, light_main_entity, light, (_, spot_light_frustum))) in
            point_lights
                .iter()
                .skip(point_light_count)
                .take(spot_light_count)
                .enumerate()
        {
            let Ok(mut light_view_entities) = light_view_entities.get_mut(light_entity) else {
                continue;
            };

            if !light.shadows_enabled {
                if let Some(entities) = light_view_entities.remove(&entity) {
                    despawn_entities(&mut commands, entities);
                }
                continue;
            }

            let spot_world_from_view = spot_light_world_from_view(&light.transform);
            let spot_world_from_view = spot_world_from_view.into();

            let angle = light.spot_light_angles.expect("lights should be sorted so that \
                [point_light_count..point_light_count + spot_light_shadow_maps_count] are spot lights").1;
            let spot_projection = spot_light_clip_from_view(angle, light.shadow_map_near_z);

            let mut first = false;
            let base_array_layer = (num_directional_cascades_enabled + light_index) as u32;

            let depth_attachment = directional_light_depth_attachments
                .entry(base_array_layer)
                .or_insert_with(|| {
                    first = true;

                    let depth_texture_view = directional_light_depth_texture.texture.create_view(
                        &TextureViewDescriptor {
                            label: Some("spot_light_shadow_map_texture_view"),
                            format: None,
                            dimension: Some(TextureViewDimension::D2),
                            usage: None,
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: None,
                            base_array_layer,
                            array_layer_count: Some(1u32),
                        },
                    );

                    DepthAttachment::new(depth_texture_view, Some(0.0))
                })
                .clone();

            let light_view_entities = light_view_entities
                .entry(entity)
                .or_insert_with(|| vec![commands.spawn_empty().id()]);

            let view_light_entity = light_view_entities[0];

            let retained_view_entity =
                RetainedViewEntity::new(*light_main_entity, Some(camera_main_entity.into()), 0);

            commands.entity(view_light_entity).insert((
                ShadowView {
                    depth_attachment,
                    pass_name: format!("shadow pass spot light {light_index}"),
                },
                ExtractedView {
                    retained_view_entity,
                    viewport: UVec4::new(
                        0,
                        0,
                        directional_light_shadow_map.size as u32,
                        directional_light_shadow_map.size as u32,
                    ),
                    world_from_view: spot_world_from_view,
                    clip_from_view: spot_projection,
                    clip_from_world: None,
                    hdr: false,
                    color_grading: Default::default(),
                },
                *spot_light_frustum.unwrap(),
                LightEntity::Spot { light_entity },
            ));

            if !matches!(gpu_preprocessing_mode, GpuPreprocessingMode::Culling) {
                commands.entity(view_light_entity).insert(NoIndirectDrawing);
            }

            view_lights.push(view_light_entity);

            if first {
                // Subsequent views with the same light entity will reuse the same shadow map
                shadow_render_phases
                    .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
                live_shadow_mapping_lights.insert(retained_view_entity);
            }
        }

        // directional lights
        let mut directional_depth_texture_array_index = 0u32;
        let view_layers = maybe_layers.unwrap_or_default();
        for (light_index, &(light_entity, light_main_entity, light)) in directional_lights
            .iter()
            .enumerate()
            .take(MAX_DIRECTIONAL_LIGHTS)
        {
            let gpu_light = &mut gpu_lights.directional_lights[light_index];

            let Ok(mut light_view_entities) = light_view_entities.get_mut(light_entity) else {
                continue;
            };

            // Check if the light intersects with the view.
            if !view_layers.intersects(&light.render_layers) {
                gpu_light.skip = 1u32;
                if let Some(entities) = light_view_entities.remove(&entity) {
                    despawn_entities(&mut commands, entities);
                }
                continue;
            }

            // Only deal with cascades when shadows are enabled.
            if (gpu_light.flags & DirectionalLightFlags::SHADOWS_ENABLED.bits()) == 0u32 {
                if let Some(entities) = light_view_entities.remove(&entity) {
                    despawn_entities(&mut commands, entities);
                }
                continue;
            }

            let cascades = light
                .cascades
                .get(&entity)
                .unwrap()
                .iter()
                .take(MAX_CASCADES_PER_LIGHT);
            let frusta = light
                .frusta
                .get(&entity)
                .unwrap()
                .iter()
                .take(MAX_CASCADES_PER_LIGHT);

            let iter = cascades
                .zip(frusta)
                .zip(&light.cascade_shadow_config.bounds);

            let light_view_entities = light_view_entities.entry(entity).or_insert_with(|| {
                (0..iter.len())
                    .map(|_| commands.spawn_empty().id())
                    .collect()
            });
            if light_view_entities.len() != iter.len() {
                let entities = core::mem::take(light_view_entities);
                despawn_entities(&mut commands, entities);
                light_view_entities.extend((0..iter.len()).map(|_| commands.spawn_empty().id()));
            }

            for (cascade_index, (((cascade, frustum), bound), view_light_entity)) in
                iter.zip(light_view_entities.iter().copied()).enumerate()
            {
                gpu_lights.directional_lights[light_index].cascades[cascade_index] =
                    GpuDirectionalCascade {
                        clip_from_world: cascade.clip_from_world,
                        texel_size: cascade.texel_size,
                        far_bound: *bound,
                    };

                let depth_texture_view =
                    directional_light_depth_texture
                        .texture
                        .create_view(&TextureViewDescriptor {
                            label: Some("directional_light_shadow_map_array_texture_view"),
                            format: None,
                            dimension: Some(TextureViewDimension::D2),
                            usage: None,
                            aspect: TextureAspect::All,
                            base_mip_level: 0,
                            mip_level_count: None,
                            base_array_layer: directional_depth_texture_array_index,
                            array_layer_count: Some(1u32),
                        });

                // NOTE: For point and spotlights, we reuse the same depth attachment for all views.
                // However, for directional lights, we want a new depth attachment for each view,
                // so that the view is cleared for each view.
                let depth_attachment = DepthAttachment::new(depth_texture_view.clone(), Some(0.0));

                directional_depth_texture_array_index += 1;

                let mut frustum = *frustum;
                // Push the near clip plane out to infinity for directional lights
                frustum.half_spaces[4] =
                    HalfSpace::new(frustum.half_spaces[4].normal().extend(f32::INFINITY));

                let retained_view_entity = RetainedViewEntity::new(
                    *light_main_entity,
                    Some(camera_main_entity.into()),
                    cascade_index as u32,
                );

                commands.entity(view_light_entity).insert((
                    ShadowView {
                        depth_attachment,
                        pass_name: format!(
                            "shadow pass directional light {light_index} cascade {cascade_index}"
                        ),
                    },
                    ExtractedView {
                        retained_view_entity,
                        viewport: UVec4::new(
                            0,
                            0,
                            directional_light_shadow_map.size as u32,
                            directional_light_shadow_map.size as u32,
                        ),
                        world_from_view: GlobalTransform::from(cascade.world_from_cascade),
                        clip_from_view: cascade.clip_from_cascade,
                        clip_from_world: Some(cascade.clip_from_world),
                        hdr: false,
                        color_grading: Default::default(),
                    },
                    frustum,
                    LightEntity::Directional {
                        light_entity,
                        cascade_index,
                    },
                ));

                if !matches!(gpu_preprocessing_mode, GpuPreprocessingMode::Culling) {
                    commands.entity(view_light_entity).insert(NoIndirectDrawing);
                }

                view_lights.push(view_light_entity);

                // If this light is using occlusion culling, add the appropriate components.
                if light.occlusion_culling {
                    commands.entity(view_light_entity).insert((
                        OcclusionCulling,
                        OcclusionCullingSubview {
                            depth_texture_view,
                            depth_texture_size: directional_light_shadow_map.size as u32,
                        },
                    ));
                    view_occlusion_culling_lights.push(view_light_entity);
                }

                // Subsequent views with the same light entity will **NOT** reuse the same shadow map
                // (Because the cascades are unique to each view)
                // TODO: Implement GPU culling for shadow passes.
                shadow_render_phases
                    .prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);
                live_shadow_mapping_lights.insert(retained_view_entity);
            }
        }

        commands.entity(entity).insert((
            ViewShadowBindings {
                point_light_depth_texture: point_light_depth_texture.texture.clone(),
                point_light_depth_texture_view: point_light_depth_texture_view.clone(),
                directional_light_depth_texture: directional_light_depth_texture.texture.clone(),
                directional_light_depth_texture_view: directional_light_depth_texture_view.clone(),
            },
            ViewLightEntities {
                lights: view_lights,
            },
            ViewLightsUniformOffset {
                offset: view_gpu_lights_writer.write(&gpu_lights),
            },
        ));

        // Make a link from the camera to all shadow cascades with occlusion
        // culling enabled.
        if !view_occlusion_culling_lights.is_empty() {
            commands
                .entity(entity)
                .insert(OcclusionCullingSubviewEntities(
                    view_occlusion_culling_lights,
                ));
        }
    }

    // Despawn light-view entities for views that no longer exist
    for mut entities in &mut light_view_entities {
        for (_, light_view_entities) in
            entities.extract_if(|entity, _| !live_views.contains(entity))
        {
            despawn_entities(&mut commands, light_view_entities);
        }
    }

    shadow_render_phases.retain(|entity, _| live_shadow_mapping_lights.contains(entity));
}

// this method of constructing a basis from a vec3 is used by glam::Vec3::any_orthonormal_pair
// we will also construct it in the fragment shader and need our implementations to match,
// so we reproduce it here to avoid a mismatch if glam changes. we also switch the handedness
// could move this onto transform but it's pretty niche
fn spot_light_world_from_view(transform: &GlobalTransform) -> Mat4 {
    // the matrix z_local (opposite of transform.forward())
    let fwd_dir = transform.back().extend(0.0);

    let sign = 1f32.copysign(fwd_dir.z);
    let a = -1.0 / (fwd_dir.z + sign);
    let b = fwd_dir.x * fwd_dir.y * a;
    let up_dir = Vec4::new(
        1.0 + sign * fwd_dir.x * fwd_dir.x * a,
        sign * b,
        -sign * fwd_dir.x,
        0.0,
    );
    let right_dir = Vec4::new(-b, -sign - fwd_dir.y * fwd_dir.y * a, fwd_dir.y, 0.0);

    Mat4::from_cols(
        right_dir,
        up_dir,
        fwd_dir,
        transform.translation().extend(1.0),
    )
}

fn spot_light_clip_from_view(angle: f32, near_z: f32) -> Mat4 {
    // spot light projection FOV is 2x the angle from spot light center to outer edge
    Mat4::perspective_infinite_reverse_rh(angle * 2.0, 1.0, near_z)
}

fn shrink_entities(visible_entities: &mut Vec<Entity>) {
    // Check that visible entities capacity() is no more than two times greater than len()
    let capacity = visible_entities.capacity();
    let reserved = capacity
        .checked_div(visible_entities.len())
        .map_or(0, |reserve| {
            if reserve > 2 {
                capacity / (reserve / 2)
            } else {
                capacity
            }
        });

    visible_entities.shrink_to(reserved);
}

fn face_index_to_name(face_index: usize) -> &'static str {
    match face_index {
        0 => "+x",
        1 => "-x",
        2 => "+y",
        3 => "-y",
        4 => "+z",
        5 => "-z",
        _ => "invalid",
    }
}

fn despawn_entities(commands: &mut Commands, entities: Vec<Entity>) {
    if entities.is_empty() {
        return;
    }
    commands.queue(move |world: &mut World| {
        for entity in entities {
            world.despawn(entity);
        }
    });
}

pub fn extract_lights(
    mut commands: Commands,
    point_light_shadow_map: Extract<Res<PointLightShadowMap>>,
    directional_light_shadow_map: Extract<Res<DirectionalLightShadowMap>>,
    global_point_lights: Extract<Res<GlobalVisibleClusterableObjects>>,
    point_lights: Extract<
        Query<(
            Entity,
            RenderEntity,
            &PointLight,
            &CubemapVisibleEntities,
            &GlobalTransform,
            &ViewVisibility,
            &CubemapFrusta,
            Option<&VolumetricLight>,
        )>,
    >,
    spot_lights: Extract<
        Query<(
            Entity,
            RenderEntity,
            &SpotLight,
            &VisibleMeshEntities,
            &GlobalTransform,
            &ViewVisibility,
            &Frustum,
            Option<&VolumetricLight>,
        )>,
    >,
    directional_lights: Extract<
        Query<
            (
                Entity,
                RenderEntity,
                &DirectionalLight,
                &CascadesVisibleEntities,
                &Cascades,
                &CascadeShadowConfig,
                &CascadesFrusta,
                &GlobalTransform,
                &ViewVisibility,
                Option<&RenderLayers>,
                Option<&VolumetricLight>,
                Has<OcclusionCulling>,
            ),
            Without<SpotLight>,
        >,
    >,
    mapper: Extract<Query<RenderEntity>>,
    mut previous_point_lights_len: Local<usize>,
    mut previous_spot_lights_len: Local<usize>,
) {
    // NOTE: These shadow map resources are extracted here as they are used here too so this avoids
    // races between scheduling of ExtractResourceSystems and this system.
    if point_light_shadow_map.is_changed() {
        commands.insert_resource(point_light_shadow_map.clone());
    }
    if directional_light_shadow_map.is_changed() {
        commands.insert_resource(directional_light_shadow_map.clone());
    }
    // This is the point light shadow map texel size for one face of the cube as a distance of 1.0
    // world unit from the light.
    // point_light_texel_size = 2.0 * 1.0 * tan(PI / 4.0) / cube face width in texels
    // PI / 4.0 is half the cube face fov, tan(PI / 4.0) = 1.0, so this simplifies to:
    // point_light_texel_size = 2.0 / cube face width in texels
    // NOTE: When using various PCF kernel sizes, this will need to be adjusted, according to:
    // https://catlikecoding.com/unity/tutorials/custom-srp/point-and-spot-shadows/
    let point_light_texel_size = 2.0 / point_light_shadow_map.size as f32;

    let mut point_lights_values = Vec::with_capacity(*previous_point_lights_len);
    for entity in global_point_lights.iter().copied() {
        let Ok((
            main_entity,
            render_entity,
            point_light,
            cubemap_visible_entities,
            transform,
            view_visibility,
            frusta,
            volumetric_light,
        )) = point_lights.get(entity)
        else {
            continue;
        };
        if !view_visibility.get() {
            continue;
        }
        let render_cubemap_visible_entities = RenderCubemapVisibleEntities {
            data: cubemap_visible_entities
                .iter()
                .map(|v| create_render_visible_mesh_entities(&mapper, v))
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        };

        let extracted_point_light = ExtractedPointLight {
            color: point_light.color.into(),
            // NOTE: Map from luminous power in lumens to luminous intensity in lumens per steradian
            // for a point light. See https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminousPower
            // for details.
            intensity: point_light.intensity / (4.0 * core::f32::consts::PI),
            range: point_light.range,
            radius: point_light.radius,
            transform: *transform,
            shadows_enabled: point_light.shadows_enabled,
            shadow_depth_bias: point_light.shadow_depth_bias,
            // The factor of SQRT_2 is for the worst-case diagonal offset
            shadow_normal_bias: point_light.shadow_normal_bias
                * point_light_texel_size
                * core::f32::consts::SQRT_2,
            shadow_map_near_z: point_light.shadow_map_near_z,
            spot_light_angles: None,
            volumetric: volumetric_light.is_some(),
            affects_lightmapped_mesh_diffuse: point_light.affects_lightmapped_mesh_diffuse,
            #[cfg(feature = "experimental_pbr_pcss")]
            soft_shadows_enabled: point_light.soft_shadows_enabled,
            #[cfg(not(feature = "experimental_pbr_pcss"))]
            soft_shadows_enabled: false,
        };
        point_lights_values.push((
            render_entity,
            (
                extracted_point_light,
                render_cubemap_visible_entities,
                (*frusta).clone(),
                MainEntity::from(main_entity),
            ),
        ));
    }
    *previous_point_lights_len = point_lights_values.len();
    commands.try_insert_batch(point_lights_values);

    let mut spot_lights_values = Vec::with_capacity(*previous_spot_lights_len);
    for entity in global_point_lights.iter().copied() {
        if let Ok((
            main_entity,
            render_entity,
            spot_light,
            visible_entities,
            transform,
            view_visibility,
            frustum,
            volumetric_light,
        )) = spot_lights.get(entity)
        {
            if !view_visibility.get() {
                continue;
            }
            let render_visible_entities =
                create_render_visible_mesh_entities(&mapper, visible_entities);

            let texel_size =
                2.0 * ops::tan(spot_light.outer_angle) / directional_light_shadow_map.size as f32;

            spot_lights_values.push((
                render_entity,
                (
                    ExtractedPointLight {
                        color: spot_light.color.into(),
                        // NOTE: Map from luminous power in lumens to luminous intensity in lumens per steradian
                        // for a point light. See https://google.github.io/filament/Filament.html#mjx-eqn-pointLightLuminousPower
                        // for details.
                        // Note: Filament uses a divisor of PI for spot lights. We choose to use the same 4*PI divisor
                        // in both cases so that toggling between point light and spot light keeps lit areas lit equally,
                        // which seems least surprising for users
                        intensity: spot_light.intensity / (4.0 * core::f32::consts::PI),
                        range: spot_light.range,
                        radius: spot_light.radius,
                        transform: *transform,
                        shadows_enabled: spot_light.shadows_enabled,
                        shadow_depth_bias: spot_light.shadow_depth_bias,
                        // The factor of SQRT_2 is for the worst-case diagonal offset
                        shadow_normal_bias: spot_light.shadow_normal_bias
                            * texel_size
                            * core::f32::consts::SQRT_2,
                        shadow_map_near_z: spot_light.shadow_map_near_z,
                        spot_light_angles: Some((spot_light.inner_angle, spot_light.outer_angle)),
                        volumetric: volumetric_light.is_some(),
                        affects_lightmapped_mesh_diffuse: spot_light
                            .affects_lightmapped_mesh_diffuse,
                        #[cfg(feature = "experimental_pbr_pcss")]
                        soft_shadows_enabled: spot_light.soft_shadows_enabled,
                        #[cfg(not(feature = "experimental_pbr_pcss"))]
                        soft_shadows_enabled: false,
                    },
                    render_visible_entities,
                    *frustum,
                    MainEntity::from(main_entity),
                ),
            ));
        }
    }
    *previous_spot_lights_len = spot_lights_values.len();
    commands.try_insert_batch(spot_lights_values);

    for (
        main_entity,
        entity,
        directional_light,
        visible_entities,
        cascades,
        cascade_config,
        frusta,
        transform,
        view_visibility,
        maybe_layers,
        volumetric_light,
        occlusion_culling,
    ) in &directional_lights
    {
        if !view_visibility.get() {
            commands
                .get_entity(entity)
                .expect("Light entity wasn't synced.")
                .remove::<(ExtractedDirectionalLight, RenderCascadesVisibleEntities)>();
            continue;
        }

        // TODO: update in place instead of reinserting.
        let mut extracted_cascades = EntityHashMap::default();
        let mut extracted_frusta = EntityHashMap::default();
        let mut cascade_visible_entities = EntityHashMap::default();
        for (e, v) in cascades.cascades.iter() {
            if let Ok(entity) = mapper.get(*e) {
                extracted_cascades.insert(entity, v.clone());
            } else {
                break;
            }
        }
        for (e, v) in frusta.frusta.iter() {
            if let Ok(entity) = mapper.get(*e) {
                extracted_frusta.insert(entity, v.clone());
            } else {
                break;
            }
        }
        for (e, v) in visible_entities.entities.iter() {
            if let Ok(entity) = mapper.get(*e) {
                cascade_visible_entities.insert(
                    entity,
                    v.iter()
                        .map(|v| create_render_visible_mesh_entities(&mapper, v))
                        .collect(),
                );
            } else {
                break;
            }
        }

        commands
            .get_entity(entity)
            .expect("Light entity wasn't synced.")
            .insert((
                ExtractedDirectionalLight {
                    color: directional_light.color.into(),
                    illuminance: directional_light.illuminance,
                    transform: *transform,
                    volumetric: volumetric_light.is_some(),
                    affects_lightmapped_mesh_diffuse: directional_light
                        .affects_lightmapped_mesh_diffuse,
                    #[cfg(feature = "experimental_pbr_pcss")]
                    soft_shadow_size: directional_light.soft_shadow_size,
                    #[cfg(not(feature = "experimental_pbr_pcss"))]
                    soft_shadow_size: None,
                    shadows_enabled: directional_light.shadows_enabled,
                    shadow_depth_bias: directional_light.shadow_depth_bias,
                    // The factor of SQRT_2 is for the worst-case diagonal offset
                    shadow_normal_bias: directional_light.shadow_normal_bias
                        * core::f32::consts::SQRT_2,
                    cascade_shadow_config: cascade_config.clone(),
                    cascades: extracted_cascades,
                    frusta: extracted_frusta,
                    render_layers: maybe_layers.unwrap_or_default().clone(),
                    occlusion_culling,
                },
                RenderCascadesVisibleEntities {
                    entities: cascade_visible_entities,
                },
                MainEntity::from(main_entity),
            ));
    }
}

pub fn create_render_visible_mesh_entities(
    mapper: &Extract<Query<RenderEntity>>,
    visible_entities: &VisibleMeshEntities,
) -> RenderVisibleMeshEntities {
    RenderVisibleMeshEntities {
        entities: visible_entities
            .iter()
            .map(|e| {
                let render_entity = mapper.get(*e).unwrap_or(Entity::PLACEHOLDER);
                (render_entity, MainEntity::from(*e))
            })
            .collect(),
    }
}

// TODO: using required component
pub fn add_light_view_entities(
    trigger: Trigger<OnAdd, (ExtractedDirectionalLight, ExtractedPointLight)>,
    mut commands: Commands,
) {
    if let Ok(mut v) = commands.get_entity(trigger.target()) {
        v.insert(LightViewEntities::default());
    }
}

/// Removes [`LightViewEntities`] when light is removed. See [`add_light_view_entities`].
pub fn extracted_light_removed(
    trigger: Trigger<OnRemove, (ExtractedDirectionalLight, ExtractedPointLight)>,
    mut commands: Commands,
) {
    if let Ok(mut v) = commands.get_entity(trigger.target()) {
        v.try_remove::<LightViewEntities>();
    }
}

pub fn remove_light_view_entities(
    trigger: Trigger<OnRemove, LightViewEntities>,
    query: Query<&LightViewEntities>,
    mut commands: Commands,
) {
    if let Ok(entities) = query.get(trigger.target()) {
        for v in entities.0.values() {
            for e in v.iter().copied() {
                if let Ok(mut v) = commands.get_entity(e) {
                    v.despawn();
                }
            }
        }
    }
}

pub fn clear_directional_light_cascades(mut lights: Query<(&DirectionalLight, &mut Cascades)>) {
    for (directional_light, mut cascades) in lights.iter_mut() {
        if !directional_light.shadows_enabled {
            continue;
        }
        cascades.cascades.clear();
    }
}

// These will be extracted in the material extraction, which will also clear the needs_specialization
// collection.
pub fn check_light_entities_needing_specialization<M: Material>(
    needs_specialization: Query<Entity, (With<MeshMaterial3d<M>>, Changed<NotShadowCaster>)>,
    mut entities_needing_specialization: ResMut<EntitiesNeedingSpecialization<M>>,
    mut removed_components: RemovedComponents<NotShadowCaster>,
) {
    for entity in &needs_specialization {
        entities_needing_specialization.push(entity);
    }

    for removed in removed_components.read() {
        entities_needing_specialization.entities.push(removed);
    }
}

pub fn check_views_lights_need_specialization(
    view_lights: Query<&ViewLightEntities, With<ExtractedView>>,
    view_light_entities: Query<(&LightEntity, &ExtractedView)>,
    shadow_render_phases: Res<ViewBinnedRenderPhases<Shadow>>,
    mut light_key_cache: ResMut<LightKeyCache>,
    mut light_specialization_ticks: ResMut<LightSpecializationTicks>,
    ticks: SystemChangeTick,
) {
    for view_lights in &view_lights {
        for view_light_entity in view_lights.lights.iter().copied() {
            let Ok((light_entity, extracted_view_light)) =
                view_light_entities.get(view_light_entity)
            else {
                continue;
            };
            if !shadow_render_phases.contains_key(&extracted_view_light.retained_view_entity) {
                continue;
            }

            let is_directional_light = matches!(light_entity, LightEntity::Directional { .. });
            let mut light_key = MeshPipelineKey::DEPTH_PREPASS;
            light_key.set(MeshPipelineKey::UNCLIPPED_DEPTH_ORTHO, is_directional_light);
            if let Some(current_key) =
                light_key_cache.get_mut(&extracted_view_light.retained_view_entity)
            {
                if *current_key != light_key {
                    light_key_cache.insert(extracted_view_light.retained_view_entity, light_key);
                    light_specialization_ticks
                        .insert(extracted_view_light.retained_view_entity, ticks.this_run());
                }
            } else {
                light_key_cache.insert(extracted_view_light.retained_view_entity, light_key);
                light_specialization_ticks
                    .insert(extracted_view_light.retained_view_entity, ticks.this_run());
            }
        }
    }
}
