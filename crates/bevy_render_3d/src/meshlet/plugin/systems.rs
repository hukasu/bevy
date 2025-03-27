use core::{hash::Hash, ops::DerefMut};

use bevy_asset::{AssetEvent, AssetServer, Assets};
use bevy_core_pipeline::{
    core_3d::Camera3d,
    experimental::mip_generation::ViewDepthPyramid,
    prepass::{
        DeferredPrepass, DepthPrepass, MotionVectorPrepass, NormalPrepass, PreviousViewUniforms,
    },
    tonemapping::{DebandDither, Tonemapping},
};
use bevy_ecs::{
    entity::{Entities, Entity},
    event::EventReader,
    query::{AnyOf, Has, With},
    system::{Commands, Local, Query, Res, ResMut, SystemState},
};
use bevy_math::Vec4Swizzles;
use bevy_mesh::{
    Mesh, MeshVertexBufferLayout, MeshVertexBufferLayoutRef, MeshVertexBufferLayouts,
    PrimitiveTopology, VertexBufferLayout,
};
use bevy_platform_support::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
};
use bevy_render::{
    alpha::AlphaMode,
    camera::{Projection, TemporalJitter},
    render_asset::RenderAssets,
    render_resource::{
        encase::internal::WriteInto, BindGroupEntries, BufferAddress, BufferDescriptor,
        BufferInitDescriptor, BufferUsages, CachedRenderPipelineId, CompareFunction,
        DepthBiasState, DepthStencilState, DispatchIndirectArgs, DrawIndirectArgs, Extent3d,
        FragmentState, MultisampleState, PipelineCache, PrimitiveState, RenderPipelineDescriptor,
        ShaderRef, ShaderSize, ShaderType, SpecializedMeshPipeline, StencilState, StorageBuffer,
        TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, VertexAttribute,
        VertexState, VertexStepMode,
    },
    renderer::{RenderDevice, RenderQueue},
    texture::TextureCache,
    view::{ExtractedView, Msaa, RenderLayers, ViewUniforms},
    MainWorld,
};
use bevy_transform::components::GlobalTransform;

use tracing::error;

use crate::{
    distance_fog::DistanceFog,
    light::{NotShadowCaster, NotShadowReceiver},
    light_probe::{light_probes::RenderViewLightProbes, EnvironmentMapLight, IrradianceVolume},
    material::{
        self, MaterialBindGroupAllocator, MaterialPipeline, MaterialPipelineKey, PreparedMaterial,
        RenderMaterialBindings, RenderMaterialInstances,
    },
    mesh_pipeline::render::{
        pipeline::{tonemapping_pipeline_key, MeshPipeline, MeshPipelineKey},
        RenderMeshMaterialIds,
    },
    meshlet::{
        render::{
            InstanceManager, MeshletMeshManager, MeshletViewBindGroups,
            MeshletViewMaterialsDeferredGBufferPrepass, MeshletViewMaterialsMainOpaquePass,
            MeshletViewMaterialsPrepass, MeshletViewResources, ResourceManager,
        },
        MeshletMesh, MeshletMesh3d,
    },
    prepass::{render::PrepassPipeline, PreviousGlobalTransform},
    render_method::OpaqueRendererMethod,
    shadow::{render::ShadowView, ShadowFilteringMethod},
    ssao::ScreenSpaceAmbientOcclusion,
};

use super::MESHLET_MESH_MATERIAL_SHADER_HANDLE;

pub fn extract_meshlet_mesh_entities(
    mut meshlet_mesh_manager: ResMut<MeshletMeshManager>,
    mut instance_manager: ResMut<InstanceManager>,
    // TODO: Replace main_world and system_state when Extract<ResMut<Assets<MeshletMesh>>> is possible
    mut main_world: ResMut<MainWorld>,
    mesh_material_ids: Res<RenderMeshMaterialIds>,
    render_material_bindings: Res<RenderMaterialBindings>,
    mut system_state: Local<
        Option<
            SystemState<(
                Query<(
                    Entity,
                    &MeshletMesh3d,
                    &GlobalTransform,
                    Option<&PreviousGlobalTransform>,
                    Option<&RenderLayers>,
                    Has<NotShadowReceiver>,
                    Has<NotShadowCaster>,
                )>,
                Res<AssetServer>,
                ResMut<Assets<MeshletMesh>>,
                EventReader<AssetEvent<MeshletMesh>>,
            )>,
        >,
    >,
    render_entities: &Entities,
) {
    // Get instances query
    if system_state.is_none() {
        *system_state = Some(SystemState::new(&mut main_world));
    }
    let system_state = system_state.as_mut().unwrap();
    let (instances_query, asset_server, mut assets, mut asset_events) =
        system_state.get_mut(&mut main_world);

    // Reset per-frame data
    instance_manager.reset(render_entities);

    // Free GPU buffer space for any modified or dropped MeshletMesh assets
    for asset_event in asset_events.read() {
        if let AssetEvent::Unused { id } | AssetEvent::Modified { id } = asset_event {
            meshlet_mesh_manager.remove(id);
        }
    }

    // Iterate over every instance
    for (
        instance,
        meshlet_mesh,
        transform,
        previous_transform,
        render_layers,
        not_shadow_receiver,
        not_shadow_caster,
    ) in &instances_query
    {
        // Skip instances with an unloaded MeshletMesh asset
        // TODO: This is a semi-expensive check
        if asset_server.is_managed(meshlet_mesh.id())
            && !asset_server.is_loaded_with_dependencies(meshlet_mesh.id())
        {
            continue;
        }

        // Upload the instance's MeshletMesh asset data if not done already done
        let meshlets_slice =
            meshlet_mesh_manager.queue_upload_if_needed(meshlet_mesh.id(), &mut assets);

        // Add the instance's data to the instance manager
        instance_manager.add_instance(
            instance.into(),
            meshlets_slice,
            transform,
            previous_transform,
            render_layers,
            &mesh_material_ids,
            &render_material_bindings,
            not_shadow_receiver,
            not_shadow_caster,
        );
    }
}

/// For each entity in the scene, record what material ID its material was assigned in the `prepare_material_meshlet_meshes` systems,
/// and note that the material is used by at least one entity in the scene.
pub fn queue_material_meshlet_meshes<M: material::Material>(
    mut instance_manager: ResMut<InstanceManager>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
) {
    let instance_manager = instance_manager.deref_mut();

    for (i, (instance, _, _)) in instance_manager.instances.iter().enumerate() {
        if let Some(material_asset_id) = render_material_instances.get(instance) {
            if let Some(material_id) = instance_manager
                .material_id_lookup
                .get(&material_asset_id.untyped())
            {
                instance_manager
                    .material_ids_present_in_scene
                    .insert(*material_id);
                instance_manager.instance_material_ids.get_mut()[i] = *material_id;
            }
        }
    }
}

/// Prepare [`Material`] pipelines for [`super::MeshletMesh`] entities for use in [`super::MeshletMainOpaquePass3dNode`],
/// and register the material with [`InstanceManager`].
pub fn prepare_material_meshlet_meshes_main_opaque_pass<M: material::Material>(
    resource_manager: ResMut<ResourceManager>,
    mut instance_manager: ResMut<InstanceManager>,
    mut cache: Local<HashMap<MeshPipelineKey, CachedRenderPipelineId>>,
    pipeline_cache: Res<PipelineCache>,
    material_pipeline: Res<MaterialPipeline<M>>,
    mesh_pipeline: Res<MeshPipeline>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    material_bind_group_allocator: Res<MaterialBindGroupAllocator<M>>,
    asset_server: Res<AssetServer>,
    mut mesh_vertex_buffer_layouts: ResMut<MeshVertexBufferLayouts>,
    mut views: Query<
        (
            &mut MeshletViewMaterialsMainOpaquePass,
            &ExtractedView,
            Option<&Tonemapping>,
            Option<&DebandDither>,
            Option<&ShadowFilteringMethod>,
            (Has<ScreenSpaceAmbientOcclusion>, Has<DistanceFog>),
            (
                Has<NormalPrepass>,
                Has<DepthPrepass>,
                Has<MotionVectorPrepass>,
                Has<DeferredPrepass>,
            ),
            Has<TemporalJitter>,
            Option<&Projection>,
            Has<RenderViewLightProbes<EnvironmentMapLight>>,
            Has<RenderViewLightProbes<IrradianceVolume>>,
        ),
        With<Camera3d>,
    >,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let fake_vertex_buffer_layout = &fake_vertex_buffer_layout(&mut mesh_vertex_buffer_layouts);

    for (
        mut materials,
        view,
        tonemapping,
        dither,
        shadow_filter_method,
        (ssao, distance_fog),
        (normal_prepass, depth_prepass, motion_vector_prepass, deferred_prepass),
        temporal_jitter,
        projection,
        has_environment_maps,
        has_irradiance_volumes,
    ) in &mut views
    {
        let mut view_key =
            MeshPipelineKey::from_msaa_samples(1) | MeshPipelineKey::from_hdr(view.hdr);

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

        view_key |= MeshPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);

        for material_id in render_material_instances.values().collect::<HashSet<_>>() {
            let Some(material) = render_materials.get(*material_id) else {
                continue;
            };
            let Some(material_bind_group) =
                material_bind_group_allocator.get(material.binding.group)
            else {
                continue;
            };

            if material.properties.render_method != OpaqueRendererMethod::Forward
                || material.properties.alpha_mode != AlphaMode::Opaque
                || material.properties.reads_view_transmission_texture
            {
                continue;
            }

            let Ok(material_pipeline_descriptor) = material_pipeline.specialize(
                MaterialPipelineKey {
                    mesh_key: view_key,
                    bind_group_data: material_bind_group
                        .get_extra_data(material.binding.slot)
                        .clone(),
                },
                fake_vertex_buffer_layout,
            ) else {
                continue;
            };
            let material_fragment = material_pipeline_descriptor.fragment.unwrap();

            let mut shader_defs = material_fragment.shader_defs;
            shader_defs.push("MESHLET_MESH_MATERIAL_PASS".into());

            let pipeline_descriptor = RenderPipelineDescriptor {
                label: material_pipeline_descriptor.label,
                layout: vec![
                    mesh_pipeline.get_view_layout(view_key.into()).clone(),
                    resource_manager.material_shade_bind_group_layout.clone(),
                    material_pipeline.material_layout.clone(),
                ],
                push_constant_ranges: vec![],
                vertex: VertexState {
                    shader: MESHLET_MESH_MATERIAL_SHADER_HANDLE,
                    shader_defs: shader_defs.clone(),
                    entry_point: material_pipeline_descriptor.vertex.entry_point,
                    buffers: Vec::new(),
                },
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth16Unorm,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Equal,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: match M::meshlet_mesh_fragment_shader() {
                        ShaderRef::Default => MESHLET_MESH_MATERIAL_SHADER_HANDLE,
                        ShaderRef::Handle(handle) => handle,
                        ShaderRef::Path(path) => asset_server.load(path),
                    },
                    shader_defs,
                    entry_point: material_fragment.entry_point,
                    targets: material_fragment.targets,
                }),
                zero_initialize_workgroup_memory: false,
            };

            let material_id = instance_manager.get_material_id(material_id.untyped());

            let pipeline_id = *cache.entry(view_key).or_insert_with(|| {
                pipeline_cache.queue_render_pipeline(pipeline_descriptor.clone())
            });

            let Some(material_bind_group) =
                material_bind_group_allocator.get(material.binding.group)
            else {
                continue;
            };
            let Some(bind_group) = material_bind_group.bind_group() else {
                continue;
            };

            materials.push((material_id, pipeline_id, (*bind_group).clone()));
        }
    }
}

/// Prepare [`Material`] pipelines for [`super::MeshletMesh`] entities for use in [`super::MeshletPrepassNode`],
/// and [`super::MeshletDeferredGBufferPrepassNode`] and register the material with [`InstanceManager`].
pub fn prepare_material_meshlet_meshes_prepass<M: material::Material>(
    resource_manager: ResMut<ResourceManager>,
    mut instance_manager: ResMut<InstanceManager>,
    mut cache: Local<HashMap<MeshPipelineKey, CachedRenderPipelineId>>,
    pipeline_cache: Res<PipelineCache>,
    prepass_pipeline: Res<PrepassPipeline<M>>,
    render_materials: Res<RenderAssets<PreparedMaterial<M>>>,
    render_material_instances: Res<RenderMaterialInstances<M>>,
    mut mesh_vertex_buffer_layouts: ResMut<MeshVertexBufferLayouts>,
    material_bind_group_allocator: Res<MaterialBindGroupAllocator<M>>,
    asset_server: Res<AssetServer>,
    mut views: Query<
        (
            &mut MeshletViewMaterialsPrepass,
            &mut MeshletViewMaterialsDeferredGBufferPrepass,
            &ExtractedView,
            AnyOf<(&NormalPrepass, &MotionVectorPrepass, &DeferredPrepass)>,
        ),
        With<Camera3d>,
    >,
) where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let fake_vertex_buffer_layout = &fake_vertex_buffer_layout(&mut mesh_vertex_buffer_layouts);

    for (
        mut materials,
        mut deferred_materials,
        view,
        (normal_prepass, motion_vector_prepass, deferred_prepass),
    ) in &mut views
    {
        let mut view_key =
            MeshPipelineKey::from_msaa_samples(1) | MeshPipelineKey::from_hdr(view.hdr);

        if normal_prepass.is_some() {
            view_key |= MeshPipelineKey::NORMAL_PREPASS;
        }
        if motion_vector_prepass.is_some() {
            view_key |= MeshPipelineKey::MOTION_VECTOR_PREPASS;
        }

        view_key |= MeshPipelineKey::from_primitive_topology(PrimitiveTopology::TriangleList);

        for material_id in render_material_instances.values().collect::<HashSet<_>>() {
            let Some(material) = render_materials.get(*material_id) else {
                continue;
            };
            let Some(material_bind_group) =
                material_bind_group_allocator.get(material.binding.group)
            else {
                continue;
            };

            if material.properties.alpha_mode != AlphaMode::Opaque
                || material.properties.reads_view_transmission_texture
            {
                continue;
            }

            let material_wants_deferred = matches!(
                material.properties.render_method,
                OpaqueRendererMethod::Deferred
            );
            if deferred_prepass.is_some() && material_wants_deferred {
                view_key |= MeshPipelineKey::DEFERRED_PREPASS;
            } else if normal_prepass.is_none() && motion_vector_prepass.is_none() {
                continue;
            }

            let Ok(material_pipeline_descriptor) = prepass_pipeline.specialize(
                MaterialPipelineKey {
                    mesh_key: view_key,
                    bind_group_data: material_bind_group
                        .get_extra_data(material.binding.slot)
                        .clone(),
                },
                fake_vertex_buffer_layout,
            ) else {
                continue;
            };
            let material_fragment = material_pipeline_descriptor.fragment.unwrap();

            let mut shader_defs = material_fragment.shader_defs;
            shader_defs.push("MESHLET_MESH_MATERIAL_PASS".into());

            let view_layout = if view_key.contains(MeshPipelineKey::MOTION_VECTOR_PREPASS) {
                prepass_pipeline.view_layout_motion_vectors.clone()
            } else {
                prepass_pipeline.view_layout_no_motion_vectors.clone()
            };

            let fragment_shader = if view_key.contains(MeshPipelineKey::DEFERRED_PREPASS) {
                M::meshlet_mesh_deferred_fragment_shader()
            } else {
                M::meshlet_mesh_prepass_fragment_shader()
            };

            let entry_point = match fragment_shader {
                ShaderRef::Default => "prepass_fragment".into(),
                _ => material_fragment.entry_point,
            };

            let pipeline_descriptor = RenderPipelineDescriptor {
                label: material_pipeline_descriptor.label,
                layout: vec![
                    view_layout,
                    resource_manager.material_shade_bind_group_layout.clone(),
                    prepass_pipeline.material_layout.clone(),
                ],
                push_constant_ranges: vec![],
                vertex: VertexState {
                    shader: MESHLET_MESH_MATERIAL_SHADER_HANDLE,
                    shader_defs: shader_defs.clone(),
                    entry_point: material_pipeline_descriptor.vertex.entry_point,
                    buffers: Vec::new(),
                },
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: TextureFormat::Depth16Unorm,
                    depth_write_enabled: false,
                    depth_compare: CompareFunction::Equal,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: match fragment_shader {
                        ShaderRef::Default => MESHLET_MESH_MATERIAL_SHADER_HANDLE,
                        ShaderRef::Handle(handle) => handle,
                        ShaderRef::Path(path) => asset_server.load(path),
                    },
                    shader_defs,
                    entry_point,
                    targets: material_fragment.targets,
                }),
                zero_initialize_workgroup_memory: false,
            };

            let material_id = instance_manager.get_material_id(material_id.untyped());

            let pipeline_id = *cache.entry(view_key).or_insert_with(|| {
                pipeline_cache.queue_render_pipeline(pipeline_descriptor.clone())
            });

            let Some(material_bind_group) =
                material_bind_group_allocator.get(material.binding.group)
            else {
                continue;
            };
            let Some(bind_group) = material_bind_group.bind_group() else {
                continue;
            };

            let item = (material_id, pipeline_id, (*bind_group).clone());
            if view_key.contains(MeshPipelineKey::DEFERRED_PREPASS) {
                deferred_materials.push(item);
            } else {
                materials.push(item);
            }
        }
    }
}

// TODO: Cache things per-view and skip running this system / optimize this system
pub fn prepare_meshlet_per_frame_resources(
    mut resource_manager: ResMut<ResourceManager>,
    mut instance_manager: ResMut<InstanceManager>,
    views: Query<(
        Entity,
        &ExtractedView,
        Option<&RenderLayers>,
        AnyOf<(&Camera3d, &ShadowView)>,
    )>,
    mut texture_cache: ResMut<TextureCache>,
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    if instance_manager.scene_cluster_count == 0 {
        return;
    }

    let instance_manager = instance_manager.as_mut();

    // TODO: Move this and the submit to a separate system and remove pub from the fields
    instance_manager
        .instance_uniforms
        .write_buffer(&render_device, &render_queue);
    upload_storage_buffer(
        &mut instance_manager.instance_material_ids,
        &render_device,
        &render_queue,
    );
    upload_storage_buffer(
        &mut instance_manager.instance_meshlet_counts,
        &render_device,
        &render_queue,
    );
    upload_storage_buffer(
        &mut instance_manager.instance_meshlet_slice_starts,
        &render_device,
        &render_queue,
    );

    let needed_buffer_size = 4 * instance_manager.scene_cluster_count as u64;
    match &mut resource_manager.cluster_instance_ids {
        Some(buffer) if buffer.size() >= needed_buffer_size => buffer.clone(),
        slot => {
            let buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("meshlet_cluster_instance_ids"),
                size: needed_buffer_size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            *slot = Some(buffer.clone());
            buffer
        }
    };
    match &mut resource_manager.cluster_meshlet_ids {
        Some(buffer) if buffer.size() >= needed_buffer_size => buffer.clone(),
        slot => {
            let buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("meshlet_cluster_meshlet_ids"),
                size: needed_buffer_size,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            *slot = Some(buffer.clone());
            buffer
        }
    };

    let needed_buffer_size =
        instance_manager.scene_cluster_count.div_ceil(u32::BITS) as u64 * size_of::<u32>() as u64;
    for (view_entity, view, render_layers, (_, shadow_view)) in &views {
        let not_shadow_view = shadow_view.is_none();

        let instance_visibility = instance_manager
            .view_instance_visibility
            .entry(view_entity)
            .or_insert_with(|| {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_view_instance_visibility"));
                buffer
            });
        for (instance_index, (_, layers, not_shadow_caster)) in
            instance_manager.instances.iter().enumerate()
        {
            // If either the layers don't match the view's layers or this is a shadow view
            // and the instance is not a shadow caster, hide the instance for this view
            if !render_layers
                .unwrap_or(&RenderLayers::default())
                .intersects(layers)
                || (shadow_view.is_some() && *not_shadow_caster)
            {
                let vec = instance_visibility.get_mut();
                let index = instance_index / 32;
                let bit = instance_index - index * 32;
                if vec.len() <= index {
                    vec.extend(core::iter::repeat_n(0, index - vec.len() + 1));
                }
                vec[index] |= 1 << bit;
            }
        }
        upload_storage_buffer(instance_visibility, &render_device, &render_queue);
        let instance_visibility = instance_visibility.buffer().unwrap().clone();

        let second_pass_candidates_buffer =
            match &mut resource_manager.second_pass_candidates_buffer {
                Some(buffer) if buffer.size() >= needed_buffer_size => buffer.clone(),
                slot => {
                    let buffer = render_device.create_buffer(&BufferDescriptor {
                        label: Some("meshlet_second_pass_candidates"),
                        size: needed_buffer_size,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    *slot = Some(buffer.clone());
                    buffer
                }
            };

        // TODO: Remove this once wgpu allows render passes with no attachments
        let dummy_render_target = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("meshlet_dummy_render_target"),
                size: Extent3d {
                    width: view.viewport.z,
                    height: view.viewport.w,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: TextureFormat::R8Uint,
                usage: TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            },
        );

        let visibility_buffer = texture_cache.get(
            &render_device,
            TextureDescriptor {
                label: Some("meshlet_visibility_buffer"),
                size: Extent3d {
                    width: view.viewport.z,
                    height: view.viewport.w,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format: if not_shadow_view {
                    TextureFormat::R64Uint
                } else {
                    TextureFormat::R32Uint
                },
                usage: TextureUsages::STORAGE_ATOMIC | TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            },
        );

        let visibility_buffer_software_raster_indirect_args_first = render_device
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("meshlet_visibility_buffer_software_raster_indirect_args_first"),
                contents: DispatchIndirectArgs { x: 0, y: 1, z: 1 }.as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            });
        let visibility_buffer_software_raster_indirect_args_second = render_device
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("visibility_buffer_software_raster_indirect_args_second"),
                contents: DispatchIndirectArgs { x: 0, y: 1, z: 1 }.as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            });

        let visibility_buffer_hardware_raster_indirect_args_first = render_device
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("meshlet_visibility_buffer_hardware_raster_indirect_args_first"),
                contents: DrawIndirectArgs {
                    vertex_count: 128 * 3,
                    instance_count: 0,
                    first_vertex: 0,
                    first_instance: 0,
                }
                .as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            });
        let visibility_buffer_hardware_raster_indirect_args_second = render_device
            .create_buffer_with_data(&BufferInitDescriptor {
                label: Some("visibility_buffer_hardware_raster_indirect_args_second"),
                contents: DrawIndirectArgs {
                    vertex_count: 128 * 3,
                    instance_count: 0,
                    first_vertex: 0,
                    first_instance: 0,
                }
                .as_bytes(),
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
            });

        let depth_pyramid = ViewDepthPyramid::new(
            &render_device,
            &mut texture_cache,
            &resource_manager.depth_pyramid_dummy_texture,
            view.viewport.zw(),
            "meshlet_depth_pyramid",
            "meshlet_depth_pyramid_texture_view",
        );

        let previous_depth_pyramid =
            match resource_manager.previous_depth_pyramids.get(&view_entity) {
                Some(texture_view) => texture_view.clone(),
                None => depth_pyramid.all_mips.clone(),
            };
        resource_manager
            .previous_depth_pyramids
            .insert(view_entity, depth_pyramid.all_mips.clone());

        let material_depth = TextureDescriptor {
            label: Some("meshlet_material_depth"),
            size: Extent3d {
                width: view.viewport.z,
                height: view.viewport.w,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Depth16Unorm,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };

        commands.entity(view_entity).insert(MeshletViewResources {
            scene_instance_count: instance_manager.scene_instance_count,
            scene_cluster_count: instance_manager.scene_cluster_count,
            second_pass_candidates_buffer,
            instance_visibility,
            dummy_render_target,
            visibility_buffer,
            visibility_buffer_software_raster_indirect_args_first,
            visibility_buffer_software_raster_indirect_args_second,
            visibility_buffer_hardware_raster_indirect_args_first,
            visibility_buffer_hardware_raster_indirect_args_second,
            depth_pyramid,
            previous_depth_pyramid,
            material_depth: not_shadow_view
                .then(|| texture_cache.get(&render_device, material_depth)),
            view_size: view.viewport.zw(),
            raster_cluster_rightmost_slot: resource_manager.raster_cluster_rightmost_slot,
            not_shadow_view,
        });
    }
}

pub fn prepare_meshlet_view_bind_groups(
    meshlet_mesh_manager: Res<MeshletMeshManager>,
    resource_manager: Res<ResourceManager>,
    instance_manager: Res<InstanceManager>,
    views: Query<(Entity, &MeshletViewResources)>,
    view_uniforms: Res<ViewUniforms>,
    previous_view_uniforms: Res<PreviousViewUniforms>,
    render_device: Res<RenderDevice>,
    mut commands: Commands,
) {
    let (
        Some(cluster_instance_ids),
        Some(cluster_meshlet_ids),
        Some(view_uniforms),
        Some(previous_view_uniforms),
    ) = (
        resource_manager.cluster_instance_ids.as_ref(),
        resource_manager.cluster_meshlet_ids.as_ref(),
        view_uniforms.uniforms.binding(),
        previous_view_uniforms.uniforms.binding(),
    )
    else {
        return;
    };

    let first_node = Arc::new(AtomicBool::new(true));

    let fill_cluster_buffers_global_cluster_count =
        render_device.create_buffer(&BufferDescriptor {
            label: Some("meshlet_fill_cluster_buffers_global_cluster_count"),
            size: 4,
            usage: BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

    // TODO: Some of these bind groups can be reused across multiple views
    for (view_entity, view_resources) in &views {
        let entries = BindGroupEntries::sequential((
            instance_manager.instance_meshlet_counts.binding().unwrap(),
            instance_manager
                .instance_meshlet_slice_starts
                .binding()
                .unwrap(),
            cluster_instance_ids.as_entire_binding(),
            cluster_meshlet_ids.as_entire_binding(),
            fill_cluster_buffers_global_cluster_count.as_entire_binding(),
        ));
        let fill_cluster_buffers = render_device.create_bind_group(
            "meshlet_fill_cluster_buffers",
            &resource_manager.fill_cluster_buffers_bind_group_layout,
            &entries,
        );

        let clear_visibility_buffer = render_device.create_bind_group(
            "meshlet_clear_visibility_buffer_bind_group",
            if view_resources.not_shadow_view {
                &resource_manager.clear_visibility_buffer_bind_group_layout
            } else {
                &resource_manager.clear_visibility_buffer_shadow_view_bind_group_layout
            },
            &BindGroupEntries::single(&view_resources.visibility_buffer.default_view),
        );

        let entries = BindGroupEntries::sequential((
            cluster_meshlet_ids.as_entire_binding(),
            meshlet_mesh_manager.meshlet_bounding_spheres.binding(),
            meshlet_mesh_manager.meshlet_simplification_errors.binding(),
            cluster_instance_ids.as_entire_binding(),
            instance_manager.instance_uniforms.binding().unwrap(),
            view_resources.instance_visibility.as_entire_binding(),
            view_resources
                .second_pass_candidates_buffer
                .as_entire_binding(),
            view_resources
                .visibility_buffer_software_raster_indirect_args_first
                .as_entire_binding(),
            view_resources
                .visibility_buffer_hardware_raster_indirect_args_first
                .as_entire_binding(),
            resource_manager
                .visibility_buffer_raster_clusters
                .as_entire_binding(),
            &view_resources.previous_depth_pyramid,
            view_uniforms.clone(),
            previous_view_uniforms.clone(),
        ));
        let culling_first = render_device.create_bind_group(
            "meshlet_culling_first_bind_group",
            &resource_manager.culling_bind_group_layout,
            &entries,
        );

        let entries = BindGroupEntries::sequential((
            cluster_meshlet_ids.as_entire_binding(),
            meshlet_mesh_manager.meshlet_bounding_spheres.binding(),
            meshlet_mesh_manager.meshlet_simplification_errors.binding(),
            cluster_instance_ids.as_entire_binding(),
            instance_manager.instance_uniforms.binding().unwrap(),
            view_resources.instance_visibility.as_entire_binding(),
            view_resources
                .second_pass_candidates_buffer
                .as_entire_binding(),
            view_resources
                .visibility_buffer_software_raster_indirect_args_second
                .as_entire_binding(),
            view_resources
                .visibility_buffer_hardware_raster_indirect_args_second
                .as_entire_binding(),
            resource_manager
                .visibility_buffer_raster_clusters
                .as_entire_binding(),
            &view_resources.depth_pyramid.all_mips,
            view_uniforms.clone(),
            previous_view_uniforms.clone(),
        ));
        let culling_second = render_device.create_bind_group(
            "meshlet_culling_second_bind_group",
            &resource_manager.culling_bind_group_layout,
            &entries,
        );

        let downsample_depth = view_resources.depth_pyramid.create_bind_group(
            &render_device,
            "meshlet_downsample_depth_bind_group",
            if view_resources.not_shadow_view {
                &resource_manager.downsample_depth_bind_group_layout
            } else {
                &resource_manager.downsample_depth_shadow_view_bind_group_layout
            },
            &view_resources.visibility_buffer.default_view,
            &resource_manager.depth_pyramid_sampler,
        );

        let entries = BindGroupEntries::sequential((
            cluster_meshlet_ids.as_entire_binding(),
            meshlet_mesh_manager.meshlets.binding(),
            meshlet_mesh_manager.indices.binding(),
            meshlet_mesh_manager.vertex_positions.binding(),
            cluster_instance_ids.as_entire_binding(),
            instance_manager.instance_uniforms.binding().unwrap(),
            resource_manager
                .visibility_buffer_raster_clusters
                .as_entire_binding(),
            resource_manager
                .software_raster_cluster_count
                .as_entire_binding(),
            &view_resources.visibility_buffer.default_view,
            view_uniforms.clone(),
        ));
        let visibility_buffer_raster = render_device.create_bind_group(
            "meshlet_visibility_raster_buffer_bind_group",
            if view_resources.not_shadow_view {
                &resource_manager.visibility_buffer_raster_bind_group_layout
            } else {
                &resource_manager.visibility_buffer_raster_shadow_view_bind_group_layout
            },
            &entries,
        );

        let resolve_depth = render_device.create_bind_group(
            "meshlet_resolve_depth_bind_group",
            if view_resources.not_shadow_view {
                &resource_manager.resolve_depth_bind_group_layout
            } else {
                &resource_manager.resolve_depth_shadow_view_bind_group_layout
            },
            &BindGroupEntries::single(&view_resources.visibility_buffer.default_view),
        );

        let resolve_material_depth = view_resources.material_depth.as_ref().map(|_| {
            let entries = BindGroupEntries::sequential((
                &view_resources.visibility_buffer.default_view,
                cluster_instance_ids.as_entire_binding(),
                instance_manager.instance_material_ids.binding().unwrap(),
            ));
            render_device.create_bind_group(
                "meshlet_resolve_material_depth_bind_group",
                &resource_manager.resolve_material_depth_bind_group_layout,
                &entries,
            )
        });

        let material_shade = view_resources.material_depth.as_ref().map(|_| {
            let entries = BindGroupEntries::sequential((
                &view_resources.visibility_buffer.default_view,
                cluster_meshlet_ids.as_entire_binding(),
                meshlet_mesh_manager.meshlets.binding(),
                meshlet_mesh_manager.indices.binding(),
                meshlet_mesh_manager.vertex_positions.binding(),
                meshlet_mesh_manager.vertex_normals.binding(),
                meshlet_mesh_manager.vertex_uvs.binding(),
                cluster_instance_ids.as_entire_binding(),
                instance_manager.instance_uniforms.binding().unwrap(),
            ));
            render_device.create_bind_group(
                "meshlet_mesh_material_shade_bind_group",
                &resource_manager.material_shade_bind_group_layout,
                &entries,
            )
        });

        let remap_1d_to_2d_dispatch = resource_manager
            .remap_1d_to_2d_dispatch_bind_group_layout
            .as_ref()
            .map(|layout| {
                (
                    render_device.create_bind_group(
                        "meshlet_remap_1d_to_2d_dispatch_first_bind_group",
                        layout,
                        &BindGroupEntries::sequential((
                            view_resources
                                .visibility_buffer_software_raster_indirect_args_first
                                .as_entire_binding(),
                            resource_manager
                                .software_raster_cluster_count
                                .as_entire_binding(),
                        )),
                    ),
                    render_device.create_bind_group(
                        "meshlet_remap_1d_to_2d_dispatch_second_bind_group",
                        layout,
                        &BindGroupEntries::sequential((
                            view_resources
                                .visibility_buffer_software_raster_indirect_args_second
                                .as_entire_binding(),
                            resource_manager
                                .software_raster_cluster_count
                                .as_entire_binding(),
                        )),
                    ),
                )
            });

        commands.entity(view_entity).insert(MeshletViewBindGroups {
            first_node: Arc::clone(&first_node),
            fill_cluster_buffers,
            clear_visibility_buffer,
            culling_first,
            culling_second,
            downsample_depth,
            visibility_buffer_raster,
            resolve_depth,
            resolve_material_depth,
            material_shade,
            remap_1d_to_2d_dispatch,
        });
    }
}

/// Upload all newly queued [`MeshletMesh`] asset data to the GPU.
pub fn perform_pending_meshlet_mesh_writes(
    mut meshlet_mesh_manager: ResMut<MeshletMeshManager>,
    render_queue: Res<RenderQueue>,
    render_device: Res<RenderDevice>,
) {
    meshlet_mesh_manager
        .vertex_positions
        .perform_writes(&render_queue, &render_device);
    meshlet_mesh_manager
        .vertex_normals
        .perform_writes(&render_queue, &render_device);
    meshlet_mesh_manager
        .vertex_uvs
        .perform_writes(&render_queue, &render_device);
    meshlet_mesh_manager
        .indices
        .perform_writes(&render_queue, &render_device);
    meshlet_mesh_manager
        .meshlets
        .perform_writes(&render_queue, &render_device);
    meshlet_mesh_manager
        .meshlet_bounding_spheres
        .perform_writes(&render_queue, &render_device);
    meshlet_mesh_manager
        .meshlet_simplification_errors
        .perform_writes(&render_queue, &render_device);
}

pub fn configure_meshlet_views(
    mut views_3d: Query<(
        Entity,
        &Msaa,
        Has<NormalPrepass>,
        Has<MotionVectorPrepass>,
        Has<DeferredPrepass>,
    )>,
    mut commands: Commands,
) {
    for (entity, msaa, normal_prepass, motion_vector_prepass, deferred_prepass) in &mut views_3d {
        if *msaa != Msaa::Off {
            error!("MeshletPlugin can't be used with MSAA. Add Msaa::Off to your camera to use this plugin.");
            std::process::exit(1);
        }

        if !(normal_prepass || motion_vector_prepass || deferred_prepass) {
            commands
                .entity(entity)
                .insert(MeshletViewMaterialsMainOpaquePass::default());
        } else {
            // TODO: Should we add both Prepass and DeferredGBufferPrepass materials here, and in other systems/nodes?
            commands.entity(entity).insert((
                MeshletViewMaterialsMainOpaquePass::default(),
                MeshletViewMaterialsPrepass::default(),
                MeshletViewMaterialsDeferredGBufferPrepass::default(),
            ));
        }
    }
}

// TODO: Try using Queue::write_buffer_with() in queue_meshlet_mesh_upload() to reduce copies
fn upload_storage_buffer<T: ShaderSize + bytemuck::NoUninit>(
    buffer: &mut StorageBuffer<Vec<T>>,
    render_device: &RenderDevice,
    render_queue: &RenderQueue,
) where
    Vec<T>: WriteInto,
{
    let inner = buffer.buffer();
    let capacity = inner.map_or(0, |b| b.size());
    let size = buffer.get().size().get() as BufferAddress;

    if capacity >= size {
        let inner = inner.unwrap();
        let bytes = bytemuck::must_cast_slice(buffer.get().as_slice());
        render_queue.write_buffer(inner, 0, bytes);
    } else {
        buffer.write_buffer(render_device, render_queue);
    }
}

// Meshlet materials don't use a traditional vertex buffer, but the material specialization requires one.
fn fake_vertex_buffer_layout(layouts: &mut MeshVertexBufferLayouts) -> MeshVertexBufferLayoutRef {
    layouts.insert(MeshVertexBufferLayout::new(
        vec![
            Mesh::ATTRIBUTE_POSITION.id,
            Mesh::ATTRIBUTE_NORMAL.id,
            Mesh::ATTRIBUTE_UV_0.id,
            Mesh::ATTRIBUTE_TANGENT.id,
        ],
        VertexBufferLayout {
            array_stride: 48,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    format: Mesh::ATTRIBUTE_POSITION.format,
                    offset: 0,
                    shader_location: 0,
                },
                VertexAttribute {
                    format: Mesh::ATTRIBUTE_NORMAL.format,
                    offset: 12,
                    shader_location: 1,
                },
                VertexAttribute {
                    format: Mesh::ATTRIBUTE_UV_0.format,
                    offset: 24,
                    shader_location: 2,
                },
                VertexAttribute {
                    format: Mesh::ATTRIBUTE_TANGENT.format,
                    offset: 32,
                    shader_location: 3,
                },
            ],
        },
    ))
}
