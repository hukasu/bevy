use core::{num::NonZero, ops::Range, sync::atomic::Ordering};

use bevy_asset::{AssetId, Assets, UntypedAssetId};
use bevy_color::LinearRgba;
use bevy_core_pipeline::{
    core_3d::CORE_3D_DEPTH_FORMAT,
    experimental::mip_generation::{self, ViewDepthPyramid, DOWNSAMPLE_DEPTH_SHADER_HANDLE},
    fullscreen_vertex_shader::fullscreen_shader_vertex_state,
    prepass::{
        MotionVectorPrepass, PreviousViewData, PreviousViewUniformOffset, ViewPrepassTextures,
    },
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component,
    entity::{hash_map::EntityHashMap, Entities},
    query::{Has, QueryItem, QueryState},
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_math::{ops, UVec2, Vec2};
use bevy_mesh::PrimitiveTopology;
use bevy_platform_support::{
    collections::{HashMap, HashSet},
    sync::{atomic::AtomicBool, Arc},
};
use bevy_render::{
    camera::ExtractedCamera,
    render_graph::{Node, NodeRunError, RenderGraphContext, ViewNode},
    render_resource::{
        binding_types::{
            sampler, storage_buffer_read_only_sized, storage_buffer_sized, texture_2d,
            texture_storage_2d, uniform_buffer,
        },
        BindGroup, BindGroupLayout, BindGroupLayoutEntries, BindingResource, Buffer, BufferAddress,
        BufferDescriptor, BufferUsages, CachedRenderPipelineId, CommandEncoderDescriptor,
        CompareFunction, DepthBiasState, DepthStencilState, FragmentState, LoadOp,
        MultisampleState, Operations, PipelineCache, PrimitiveState,
        RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPipelineDescriptor, Sampler,
        SamplerBindingType, SamplerDescriptor, ShaderStages, StencilState, StorageBuffer,
        StorageTextureAccess, StoreOp, TextureFormat, TextureSampleType, TextureView, VertexState,
        COPY_BUFFER_ALIGNMENT, *,
    },
    renderer::{RenderContext, RenderDevice, RenderQueue},
    sync_world::MainEntity,
    texture::CachedTexture,
    view::{RenderLayers, ViewDepthTexture, ViewTarget, ViewUniform, ViewUniformOffset},
};
use bevy_transform::components::GlobalTransform;

use range_alloc::RangeAllocator;

use crate::{
    distance_fog::fog::ViewFogUniformOffset,
    light::{render::LightEntity, ViewLightEntities, ViewLightsUniformOffset},
    light_probe::{
        environment_map::ViewEnvironmentMapUniformOffset,
        light_probes::ViewLightProbesUniformOffset,
    },
    material::RenderMaterialBindings,
    mesh_pipeline::render::{
        MeshFlags, MeshTransforms, MeshUniform, MeshViewBindGroup, RenderMeshMaterialIds,
    },
    prepass::{render::PrepassViewBindGroup, PreviousGlobalTransform},
    shadow::render::ShadowView,
    ssr::render::ViewScreenSpaceReflectionsUniformOffset,
};

use super::{
    asset::{Meshlet, MeshletBoundingSpheres, MeshletSimplificationError},
    plugin::{
        MESHLET_CLEAR_VISIBILITY_BUFFER_SHADER_HANDLE, MESHLET_CULLING_SHADER_HANDLE,
        MESHLET_FILL_CLUSTER_BUFFERS_SHADER_HANDLE, MESHLET_REMAP_1D_TO_2D_DISPATCH_SHADER_HANDLE,
        MESHLET_RESOLVE_RENDER_TARGETS_SHADER_HANDLE,
        MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
        MESHLET_VISIBILITY_BUFFER_SOFTWARE_RASTER_SHADER_HANDLE,
    },
    MeshletMesh,
};

/// Manages data for each entity with a [`MeshletMesh`].
#[derive(Resource)]
pub struct InstanceManager {
    /// Amount of instances in the scene.
    pub scene_instance_count: u32,
    /// Amount of clusters in the scene.
    pub scene_cluster_count: u32,

    /// Per-instance [`MainEntity`], [`RenderLayers`], and [`NotShadowCaster`].
    pub instances: Vec<(MainEntity, RenderLayers, bool)>,
    /// Per-instance [`MeshUniform`].
    pub instance_uniforms: StorageBuffer<Vec<MeshUniform>>,
    /// Per-instance material ID.
    pub instance_material_ids: StorageBuffer<Vec<u32>>,
    /// Per-instance count of meshlets in the instance's [`MeshletMesh`].
    pub instance_meshlet_counts: StorageBuffer<Vec<u32>>,
    /// Per-instance index to the start of the instance's slice of the meshlets buffer.
    pub instance_meshlet_slice_starts: StorageBuffer<Vec<u32>>,
    /// Per-view per-instance visibility bit. Used for [`RenderLayers`] and [`NotShadowCaster`] support.
    pub view_instance_visibility: EntityHashMap<StorageBuffer<Vec<u32>>>,

    /// Next material ID available for a [`Material`].
    next_material_id: u32,
    /// Map of [`Material`] to material ID.
    pub(crate) material_id_lookup: HashMap<UntypedAssetId, u32>,
    /// Set of material IDs used in the scene.
    pub(crate) material_ids_present_in_scene: HashSet<u32>,
}

impl InstanceManager {
    pub fn new() -> Self {
        Self {
            scene_instance_count: 0,
            scene_cluster_count: 0,

            instances: Vec::new(),
            instance_uniforms: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_uniforms"));
                buffer
            },
            instance_material_ids: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_material_ids"));
                buffer
            },
            instance_meshlet_counts: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_meshlet_counts"));
                buffer
            },
            instance_meshlet_slice_starts: {
                let mut buffer = StorageBuffer::default();
                buffer.set_label(Some("meshlet_instance_meshlet_slice_starts"));
                buffer
            },
            view_instance_visibility: EntityHashMap::default(),

            next_material_id: 0,
            material_id_lookup: HashMap::default(),
            material_ids_present_in_scene: HashSet::default(),
        }
    }

    pub fn add_instance(
        &mut self,
        instance: MainEntity,
        meshlets_slice: Range<u32>,
        transform: &GlobalTransform,
        previous_transform: Option<&PreviousGlobalTransform>,
        render_layers: Option<&RenderLayers>,
        mesh_material_ids: &RenderMeshMaterialIds,
        render_material_bindings: &RenderMaterialBindings,
        not_shadow_receiver: bool,
        not_shadow_caster: bool,
    ) {
        // Build a MeshUniform for the instance
        let transform = transform.affine();
        let previous_transform = previous_transform.map(|t| t.0).unwrap_or(transform);
        let mut flags = if not_shadow_receiver {
            MeshFlags::empty()
        } else {
            MeshFlags::SHADOW_RECEIVER
        };
        if transform.matrix3.determinant().is_sign_positive() {
            flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }
        let transforms = MeshTransforms {
            world_from_local: (&transform).into(),
            previous_world_from_local: (&previous_transform).into(),
            flags: flags.bits(),
        };

        let mesh_material = mesh_material_ids.mesh_material(instance);
        let mesh_material_binding_id = render_material_bindings
            .get(&mesh_material)
            .cloned()
            .unwrap_or_default();

        let mesh_uniform = MeshUniform::new(
            &transforms,
            0,
            mesh_material_binding_id.slot,
            None,
            None,
            None,
        );

        // Append instance data
        self.instances.push((
            instance,
            render_layers.cloned().unwrap_or(RenderLayers::default()),
            not_shadow_caster,
        ));
        self.instance_uniforms.get_mut().push(mesh_uniform);
        self.instance_material_ids.get_mut().push(0);
        self.instance_meshlet_counts
            .get_mut()
            .push(meshlets_slice.len() as u32);
        self.instance_meshlet_slice_starts
            .get_mut()
            .push(meshlets_slice.start);

        self.scene_instance_count += 1;
        self.scene_cluster_count += meshlets_slice.len() as u32;
    }

    /// Get the material ID for a [`crate::Material`].
    pub fn get_material_id(&mut self, material_asset_id: UntypedAssetId) -> u32 {
        *self
            .material_id_lookup
            .entry(material_asset_id)
            .or_insert_with(|| {
                self.next_material_id += 1;
                self.next_material_id
            })
    }

    pub fn material_present_in_scene(&self, material_id: &u32) -> bool {
        self.material_ids_present_in_scene.contains(material_id)
    }

    pub fn reset(&mut self, entities: &Entities) {
        self.scene_instance_count = 0;
        self.scene_cluster_count = 0;

        self.instances.clear();
        self.instance_uniforms.get_mut().clear();
        self.instance_material_ids.get_mut().clear();
        self.instance_meshlet_counts.get_mut().clear();
        self.instance_meshlet_slice_starts.get_mut().clear();
        self.view_instance_visibility
            .retain(|view_entity, _| entities.contains(*view_entity));
        self.view_instance_visibility
            .values_mut()
            .for_each(|b| b.get_mut().clear());

        self.next_material_id = 0;
        self.material_id_lookup.clear();
        self.material_ids_present_in_scene.clear();
    }
}

/// Manages uploading [`MeshletMesh`] asset data to the GPU.
#[derive(Resource)]
pub struct MeshletMeshManager {
    pub vertex_positions: PersistentGpuBuffer<Arc<[u32]>>,
    pub vertex_normals: PersistentGpuBuffer<Arc<[u32]>>,
    pub vertex_uvs: PersistentGpuBuffer<Arc<[Vec2]>>,
    pub indices: PersistentGpuBuffer<Arc<[u8]>>,
    pub meshlets: PersistentGpuBuffer<Arc<[Meshlet]>>,
    pub meshlet_bounding_spheres: PersistentGpuBuffer<Arc<[MeshletBoundingSpheres]>>,
    pub meshlet_simplification_errors: PersistentGpuBuffer<Arc<[MeshletSimplificationError]>>,
    meshlet_mesh_slices: HashMap<AssetId<MeshletMesh>, [Range<BufferAddress>; 7]>,
}

impl FromWorld for MeshletMeshManager {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        Self {
            vertex_positions: PersistentGpuBuffer::new("meshlet_vertex_positions", render_device),
            vertex_normals: PersistentGpuBuffer::new("meshlet_vertex_normals", render_device),
            vertex_uvs: PersistentGpuBuffer::new("meshlet_vertex_uvs", render_device),
            indices: PersistentGpuBuffer::new("meshlet_indices", render_device),
            meshlets: PersistentGpuBuffer::new("meshlets", render_device),
            meshlet_bounding_spheres: PersistentGpuBuffer::new(
                "meshlet_bounding_spheres",
                render_device,
            ),
            meshlet_simplification_errors: PersistentGpuBuffer::new(
                "meshlet_simplification_errors",
                render_device,
            ),
            meshlet_mesh_slices: HashMap::default(),
        }
    }
}

impl MeshletMeshManager {
    pub fn queue_upload_if_needed(
        &mut self,
        asset_id: AssetId<MeshletMesh>,
        assets: &mut Assets<MeshletMesh>,
    ) -> Range<u32> {
        let queue_meshlet_mesh = |asset_id: &AssetId<MeshletMesh>| {
            let meshlet_mesh = assets.remove_untracked(*asset_id).expect(
                "MeshletMesh asset was already unloaded but is not registered with MeshletMeshManager",
            );

            let vertex_positions_slice = self
                .vertex_positions
                .queue_write(Arc::clone(&meshlet_mesh.vertex_positions), ());
            let vertex_normals_slice = self
                .vertex_normals
                .queue_write(Arc::clone(&meshlet_mesh.vertex_normals), ());
            let vertex_uvs_slice = self
                .vertex_uvs
                .queue_write(Arc::clone(&meshlet_mesh.vertex_uvs), ());
            let indices_slice = self
                .indices
                .queue_write(Arc::clone(&meshlet_mesh.indices), ());
            let meshlets_slice = self.meshlets.queue_write(
                Arc::clone(&meshlet_mesh.meshlets),
                (
                    vertex_positions_slice.start,
                    vertex_normals_slice.start,
                    indices_slice.start,
                ),
            );
            let meshlet_bounding_spheres_slice = self
                .meshlet_bounding_spheres
                .queue_write(Arc::clone(&meshlet_mesh.meshlet_bounding_spheres), ());
            let meshlet_simplification_errors_slice = self
                .meshlet_simplification_errors
                .queue_write(Arc::clone(&meshlet_mesh.meshlet_simplification_errors), ());

            [
                vertex_positions_slice,
                vertex_normals_slice,
                vertex_uvs_slice,
                indices_slice,
                meshlets_slice,
                meshlet_bounding_spheres_slice,
                meshlet_simplification_errors_slice,
            ]
        };

        // If the MeshletMesh asset has not been uploaded to the GPU yet, queue it for uploading
        let [_, _, _, _, meshlets_slice, _, _] = self
            .meshlet_mesh_slices
            .entry(asset_id)
            .or_insert_with_key(queue_meshlet_mesh)
            .clone();

        let meshlets_slice_start = meshlets_slice.start as u32 / size_of::<Meshlet>() as u32;
        let meshlets_slice_end = meshlets_slice.end as u32 / size_of::<Meshlet>() as u32;
        meshlets_slice_start..meshlets_slice_end
    }

    pub fn remove(&mut self, asset_id: &AssetId<MeshletMesh>) {
        if let Some(
            [vertex_positions_slice, vertex_normals_slice, vertex_uvs_slice, indices_slice, meshlets_slice, meshlet_bounding_spheres_slice, meshlet_simplification_errors_slice],
        ) = self.meshlet_mesh_slices.remove(asset_id)
        {
            self.vertex_positions
                .mark_slice_unused(vertex_positions_slice);
            self.vertex_normals.mark_slice_unused(vertex_normals_slice);
            self.vertex_uvs.mark_slice_unused(vertex_uvs_slice);
            self.indices.mark_slice_unused(indices_slice);
            self.meshlets.mark_slice_unused(meshlets_slice);
            self.meshlet_bounding_spheres
                .mark_slice_unused(meshlet_bounding_spheres_slice);
            self.meshlet_simplification_errors
                .mark_slice_unused(meshlet_simplification_errors_slice);
        }
    }
}

// ------------ TODO: Everything under here needs to be rewritten and cached ------------
/// Rasterize meshlets into a depth buffer, and optional visibility buffer + material depth buffer for shading passes.
pub struct MeshletVisibilityBufferRasterPassNode {
    main_view_query: QueryState<(
        &'static ExtractedCamera,
        &'static ViewDepthTexture,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
        &'static ViewLightEntities,
    )>,
    view_light_query: QueryState<(
        &'static ShadowView,
        &'static LightEntity,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
    )>,
}

impl FromWorld for MeshletVisibilityBufferRasterPassNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            main_view_query: QueryState::new(world),
            view_light_query: QueryState::new(world),
        }
    }
}

impl Node for MeshletVisibilityBufferRasterPassNode {
    fn update(&mut self, world: &mut World) {
        self.main_view_query.update_archetypes(world);
        self.view_light_query.update_archetypes(world);
    }

    // TODO: Reuse compute/render passes between logical passes where possible, as they're expensive
    fn run(
        &self,
        graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let Ok((
            camera,
            view_depth,
            view_offset,
            previous_view_offset,
            meshlet_view_bind_groups,
            meshlet_view_resources,
            lights,
        )) = self.main_view_query.get_manual(world, graph.view_entity())
        else {
            return Ok(());
        };

        let Some((
            fill_cluster_buffers_pipeline,
            clear_visibility_buffer_pipeline,
            clear_visibility_buffer_shadow_view_pipeline,
            culling_first_pipeline,
            culling_second_pipeline,
            downsample_depth_first_pipeline,
            downsample_depth_second_pipeline,
            downsample_depth_first_shadow_view_pipeline,
            downsample_depth_second_shadow_view_pipeline,
            visibility_buffer_software_raster_pipeline,
            visibility_buffer_software_raster_shadow_view_pipeline,
            visibility_buffer_hardware_raster_pipeline,
            visibility_buffer_hardware_raster_shadow_view_pipeline,
            visibility_buffer_hardware_raster_shadow_view_unclipped_pipeline,
            resolve_depth_pipeline,
            resolve_depth_shadow_view_pipeline,
            resolve_material_depth_pipeline,
            remap_1d_to_2d_dispatch_pipeline,
        )) = MeshletPipelines::get(world)
        else {
            return Ok(());
        };

        let first_node = meshlet_view_bind_groups
            .first_node
            .fetch_and(false, Ordering::SeqCst);

        let div_ceil = meshlet_view_resources.scene_cluster_count.div_ceil(128);
        let thread_per_cluster_workgroups = ops::cbrt(div_ceil as f32).ceil() as u32;

        render_context
            .command_encoder()
            .push_debug_group("meshlet_visibility_buffer_raster");
        if first_node {
            fill_cluster_buffers_pass(
                render_context,
                &meshlet_view_bind_groups.fill_cluster_buffers,
                fill_cluster_buffers_pipeline,
                meshlet_view_resources.scene_instance_count,
            );
        }
        clear_visibility_buffer_pass(
            render_context,
            &meshlet_view_bind_groups.clear_visibility_buffer,
            clear_visibility_buffer_pipeline,
            meshlet_view_resources.view_size,
        );
        render_context.command_encoder().clear_buffer(
            &meshlet_view_resources.second_pass_candidates_buffer,
            0,
            None,
        );
        cull_pass(
            "culling_first",
            render_context,
            &meshlet_view_bind_groups.culling_first,
            view_offset,
            previous_view_offset,
            culling_first_pipeline,
            thread_per_cluster_workgroups,
            meshlet_view_resources.scene_cluster_count,
            meshlet_view_resources.raster_cluster_rightmost_slot,
            meshlet_view_bind_groups
                .remap_1d_to_2d_dispatch
                .as_ref()
                .map(|(bg1, _)| bg1),
            remap_1d_to_2d_dispatch_pipeline,
        );
        raster_pass(
            true,
            render_context,
            &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_first,
            &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_first,
            &meshlet_view_resources.dummy_render_target.default_view,
            meshlet_view_bind_groups,
            view_offset,
            visibility_buffer_software_raster_pipeline,
            visibility_buffer_hardware_raster_pipeline,
            Some(camera),
            meshlet_view_resources.raster_cluster_rightmost_slot,
        );
        meshlet_view_resources.depth_pyramid.downsample_depth(
            "downsample_depth",
            render_context,
            meshlet_view_resources.view_size,
            &meshlet_view_bind_groups.downsample_depth,
            downsample_depth_first_pipeline,
            downsample_depth_second_pipeline,
        );
        cull_pass(
            "culling_second",
            render_context,
            &meshlet_view_bind_groups.culling_second,
            view_offset,
            previous_view_offset,
            culling_second_pipeline,
            thread_per_cluster_workgroups,
            meshlet_view_resources.scene_cluster_count,
            meshlet_view_resources.raster_cluster_rightmost_slot,
            meshlet_view_bind_groups
                .remap_1d_to_2d_dispatch
                .as_ref()
                .map(|(_, bg2)| bg2),
            remap_1d_to_2d_dispatch_pipeline,
        );
        raster_pass(
            false,
            render_context,
            &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_second,
            &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_second,
            &meshlet_view_resources.dummy_render_target.default_view,
            meshlet_view_bind_groups,
            view_offset,
            visibility_buffer_software_raster_pipeline,
            visibility_buffer_hardware_raster_pipeline,
            Some(camera),
            meshlet_view_resources.raster_cluster_rightmost_slot,
        );
        resolve_depth(
            render_context,
            view_depth.get_attachment(StoreOp::Store),
            meshlet_view_bind_groups,
            resolve_depth_pipeline,
            camera,
        );
        resolve_material_depth(
            render_context,
            meshlet_view_resources,
            meshlet_view_bind_groups,
            resolve_material_depth_pipeline,
            camera,
        );
        meshlet_view_resources.depth_pyramid.downsample_depth(
            "downsample_depth",
            render_context,
            meshlet_view_resources.view_size,
            &meshlet_view_bind_groups.downsample_depth,
            downsample_depth_first_pipeline,
            downsample_depth_second_pipeline,
        );
        render_context.command_encoder().pop_debug_group();

        for light_entity in &lights.lights {
            let Ok((
                shadow_view,
                light_type,
                view_offset,
                previous_view_offset,
                meshlet_view_bind_groups,
                meshlet_view_resources,
            )) = self.view_light_query.get_manual(world, *light_entity)
            else {
                continue;
            };

            let shadow_visibility_buffer_hardware_raster_pipeline =
                if let LightEntity::Directional { .. } = light_type {
                    visibility_buffer_hardware_raster_shadow_view_unclipped_pipeline
                } else {
                    visibility_buffer_hardware_raster_shadow_view_pipeline
                };

            render_context.command_encoder().push_debug_group(&format!(
                "meshlet_visibility_buffer_raster: {}",
                shadow_view.pass_name
            ));
            clear_visibility_buffer_pass(
                render_context,
                &meshlet_view_bind_groups.clear_visibility_buffer,
                clear_visibility_buffer_shadow_view_pipeline,
                meshlet_view_resources.view_size,
            );
            render_context.command_encoder().clear_buffer(
                &meshlet_view_resources.second_pass_candidates_buffer,
                0,
                None,
            );
            cull_pass(
                "culling_first",
                render_context,
                &meshlet_view_bind_groups.culling_first,
                view_offset,
                previous_view_offset,
                culling_first_pipeline,
                thread_per_cluster_workgroups,
                meshlet_view_resources.scene_cluster_count,
                meshlet_view_resources.raster_cluster_rightmost_slot,
                meshlet_view_bind_groups
                    .remap_1d_to_2d_dispatch
                    .as_ref()
                    .map(|(bg1, _)| bg1),
                remap_1d_to_2d_dispatch_pipeline,
            );
            raster_pass(
                true,
                render_context,
                &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_first,
                &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_first,
                &meshlet_view_resources.dummy_render_target.default_view,
                meshlet_view_bind_groups,
                view_offset,
                visibility_buffer_software_raster_shadow_view_pipeline,
                shadow_visibility_buffer_hardware_raster_pipeline,
                None,
                meshlet_view_resources.raster_cluster_rightmost_slot,
            );
            meshlet_view_resources.depth_pyramid.downsample_depth(
                "downsample_depth",
                render_context,
                meshlet_view_resources.view_size,
                &meshlet_view_bind_groups.downsample_depth,
                downsample_depth_first_shadow_view_pipeline,
                downsample_depth_second_shadow_view_pipeline,
            );
            cull_pass(
                "culling_second",
                render_context,
                &meshlet_view_bind_groups.culling_second,
                view_offset,
                previous_view_offset,
                culling_second_pipeline,
                thread_per_cluster_workgroups,
                meshlet_view_resources.scene_cluster_count,
                meshlet_view_resources.raster_cluster_rightmost_slot,
                meshlet_view_bind_groups
                    .remap_1d_to_2d_dispatch
                    .as_ref()
                    .map(|(_, bg2)| bg2),
                remap_1d_to_2d_dispatch_pipeline,
            );
            raster_pass(
                false,
                render_context,
                &meshlet_view_resources.visibility_buffer_software_raster_indirect_args_second,
                &meshlet_view_resources.visibility_buffer_hardware_raster_indirect_args_second,
                &meshlet_view_resources.dummy_render_target.default_view,
                meshlet_view_bind_groups,
                view_offset,
                visibility_buffer_software_raster_shadow_view_pipeline,
                shadow_visibility_buffer_hardware_raster_pipeline,
                None,
                meshlet_view_resources.raster_cluster_rightmost_slot,
            );
            resolve_depth(
                render_context,
                shadow_view.depth_attachment.get_attachment(StoreOp::Store),
                meshlet_view_bind_groups,
                resolve_depth_shadow_view_pipeline,
                camera,
            );
            meshlet_view_resources.depth_pyramid.downsample_depth(
                "downsample_depth",
                render_context,
                meshlet_view_resources.view_size,
                &meshlet_view_bind_groups.downsample_depth,
                downsample_depth_first_shadow_view_pipeline,
                downsample_depth_second_shadow_view_pipeline,
            );
            render_context.command_encoder().pop_debug_group();
        }

        Ok(())
    }
}

fn fill_cluster_buffers_pass(
    render_context: &mut RenderContext,
    fill_cluster_buffers_bind_group: &BindGroup,
    fill_cluster_buffers_pass_pipeline: &ComputePipeline,
    scene_instance_count: u32,
) {
    let mut fill_cluster_buffers_pass_workgroups_x = scene_instance_count;
    let mut fill_cluster_buffers_pass_workgroups_y = 1;
    if scene_instance_count
        > render_context
            .render_device()
            .limits()
            .max_compute_workgroups_per_dimension
    {
        fill_cluster_buffers_pass_workgroups_x = (scene_instance_count as f32).sqrt().ceil() as u32;
        fill_cluster_buffers_pass_workgroups_y = fill_cluster_buffers_pass_workgroups_x;
    }

    let command_encoder = render_context.command_encoder();
    let mut fill_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some("fill_cluster_buffers"),
        timestamp_writes: None,
    });
    fill_pass.set_pipeline(fill_cluster_buffers_pass_pipeline);
    fill_pass.set_push_constants(0, &scene_instance_count.to_le_bytes());
    fill_pass.set_bind_group(0, fill_cluster_buffers_bind_group, &[]);
    fill_pass.dispatch_workgroups(
        fill_cluster_buffers_pass_workgroups_x,
        fill_cluster_buffers_pass_workgroups_y,
        1,
    );
}

// TODO: Replace this with vkCmdClearColorImage once wgpu supports it
fn clear_visibility_buffer_pass(
    render_context: &mut RenderContext,
    clear_visibility_buffer_bind_group: &BindGroup,
    clear_visibility_buffer_pipeline: &ComputePipeline,
    view_size: UVec2,
) {
    let command_encoder = render_context.command_encoder();
    let mut clear_visibility_buffer_pass =
        command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("clear_visibility_buffer"),
            timestamp_writes: None,
        });
    clear_visibility_buffer_pass.set_pipeline(clear_visibility_buffer_pipeline);
    clear_visibility_buffer_pass.set_push_constants(0, bytemuck::bytes_of(&view_size));
    clear_visibility_buffer_pass.set_bind_group(0, clear_visibility_buffer_bind_group, &[]);
    clear_visibility_buffer_pass.dispatch_workgroups(
        view_size.x.div_ceil(16),
        view_size.y.div_ceil(16),
        1,
    );
}

fn cull_pass(
    label: &'static str,
    render_context: &mut RenderContext,
    culling_bind_group: &BindGroup,
    view_offset: &ViewUniformOffset,
    previous_view_offset: &PreviousViewUniformOffset,
    culling_pipeline: &ComputePipeline,
    culling_workgroups: u32,
    scene_cluster_count: u32,
    raster_cluster_rightmost_slot: u32,
    remap_1d_to_2d_dispatch_bind_group: Option<&BindGroup>,
    remap_1d_to_2d_dispatch_pipeline: Option<&ComputePipeline>,
) {
    let max_compute_workgroups_per_dimension = render_context
        .render_device()
        .limits()
        .max_compute_workgroups_per_dimension;

    let command_encoder = render_context.command_encoder();
    let mut cull_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    cull_pass.set_pipeline(culling_pipeline);
    cull_pass.set_push_constants(
        0,
        bytemuck::cast_slice(&[scene_cluster_count, raster_cluster_rightmost_slot]),
    );
    cull_pass.set_bind_group(
        0,
        culling_bind_group,
        &[view_offset.offset, previous_view_offset.offset],
    );
    cull_pass.dispatch_workgroups(culling_workgroups, culling_workgroups, culling_workgroups);

    if let (Some(remap_1d_to_2d_dispatch_pipeline), Some(remap_1d_to_2d_dispatch_bind_group)) = (
        remap_1d_to_2d_dispatch_pipeline,
        remap_1d_to_2d_dispatch_bind_group,
    ) {
        cull_pass.set_pipeline(remap_1d_to_2d_dispatch_pipeline);
        cull_pass.set_push_constants(0, &max_compute_workgroups_per_dimension.to_be_bytes());
        cull_pass.set_bind_group(0, remap_1d_to_2d_dispatch_bind_group, &[]);
        cull_pass.dispatch_workgroups(1, 1, 1);
    }
}

fn raster_pass(
    first_pass: bool,
    render_context: &mut RenderContext,
    visibility_buffer_hardware_software_indirect_args: &Buffer,
    visibility_buffer_hardware_raster_indirect_args: &Buffer,
    dummy_render_target: &TextureView,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    view_offset: &ViewUniformOffset,
    visibility_buffer_hardware_software_pipeline: &ComputePipeline,
    visibility_buffer_hardware_raster_pipeline: &RenderPipeline,
    camera: Option<&ExtractedCamera>,
    raster_cluster_rightmost_slot: u32,
) {
    let command_encoder = render_context.command_encoder();
    let mut software_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
        label: Some(if first_pass {
            "raster_software_first"
        } else {
            "raster_software_second"
        }),
        timestamp_writes: None,
    });
    software_pass.set_pipeline(visibility_buffer_hardware_software_pipeline);
    software_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.visibility_buffer_raster,
        &[view_offset.offset],
    );
    software_pass
        .dispatch_workgroups_indirect(visibility_buffer_hardware_software_indirect_args, 0);
    drop(software_pass);

    let mut hardware_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some(if first_pass {
            "raster_hardware_first"
        } else {
            "raster_hardware_second"
        }),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: dummy_render_target,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(LinearRgba::BLACK.into()),
                store: StoreOp::Discard,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(viewport) = camera.and_then(|camera| camera.viewport.as_ref()) {
        hardware_pass.set_camera_viewport(viewport);
    }
    hardware_pass.set_render_pipeline(visibility_buffer_hardware_raster_pipeline);
    hardware_pass.set_push_constants(
        ShaderStages::VERTEX,
        0,
        &raster_cluster_rightmost_slot.to_le_bytes(),
    );
    hardware_pass.set_bind_group(
        0,
        &meshlet_view_bind_groups.visibility_buffer_raster,
        &[view_offset.offset],
    );
    hardware_pass.draw_indirect(visibility_buffer_hardware_raster_indirect_args, 0);
}

fn resolve_depth(
    render_context: &mut RenderContext,
    depth_stencil_attachment: RenderPassDepthStencilAttachment,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    resolve_depth_pipeline: &RenderPipeline,
    camera: &ExtractedCamera,
) {
    let mut resolve_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("resolve_depth"),
        color_attachments: &[],
        depth_stencil_attachment: Some(depth_stencil_attachment),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    if let Some(viewport) = &camera.viewport {
        resolve_pass.set_camera_viewport(viewport);
    }
    resolve_pass.set_render_pipeline(resolve_depth_pipeline);
    resolve_pass.set_bind_group(0, &meshlet_view_bind_groups.resolve_depth, &[]);
    resolve_pass.draw(0..3, 0..1);
}

fn resolve_material_depth(
    render_context: &mut RenderContext,
    meshlet_view_resources: &MeshletViewResources,
    meshlet_view_bind_groups: &MeshletViewBindGroups,
    resolve_material_depth_pipeline: &RenderPipeline,
    camera: &ExtractedCamera,
) {
    if let (Some(material_depth), Some(resolve_material_depth_bind_group)) = (
        meshlet_view_resources.material_depth.as_ref(),
        meshlet_view_bind_groups.resolve_material_depth.as_ref(),
    ) {
        let mut resolve_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("resolve_material_depth"),
            color_attachments: &[],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &material_depth.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Clear(0.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = &camera.viewport {
            resolve_pass.set_camera_viewport(viewport);
        }
        resolve_pass.set_render_pipeline(resolve_material_depth_pipeline);
        resolve_pass.set_bind_group(0, resolve_material_depth_bind_group, &[]);
        resolve_pass.draw(0..3, 0..1);
    }
}

impl PersistentGpuBufferable for Arc<[Meshlet]> {
    type Metadata = (u64, u64, u64);

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<Meshlet>()
    }

    fn write_bytes_le(
        &self,
        (vertex_position_offset, vertex_attribute_offset, index_offset): Self::Metadata,
        buffer_slice: &mut [u8],
    ) {
        let vertex_position_offset = (vertex_position_offset * 8) as u32;
        let vertex_attribute_offset = (vertex_attribute_offset as usize / size_of::<u32>()) as u32;
        let index_offset = index_offset as u32;

        for (i, meshlet) in self.iter().enumerate() {
            let size = size_of::<Meshlet>();
            let i = i * size;
            let bytes = bytemuck::cast::<_, [u8; size_of::<Meshlet>()]>(Meshlet {
                start_vertex_position_bit: meshlet.start_vertex_position_bit
                    + vertex_position_offset,
                start_vertex_attribute_id: meshlet.start_vertex_attribute_id
                    + vertex_attribute_offset,
                start_index_id: meshlet.start_index_id + index_offset,
                ..*meshlet
            });
            buffer_slice[i..(i + size)].clone_from_slice(&bytes);
        }
    }
}

impl PersistentGpuBufferable for Arc<[u8]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len()
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer_slice: &mut [u8]) {
        buffer_slice.clone_from_slice(self);
    }
}

impl PersistentGpuBufferable for Arc<[u32]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<u32>()
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer_slice: &mut [u8]) {
        buffer_slice.clone_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[Vec2]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<Vec2>()
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer_slice: &mut [u8]) {
        buffer_slice.clone_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[MeshletBoundingSpheres]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<MeshletBoundingSpheres>()
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer_slice: &mut [u8]) {
        buffer_slice.clone_from_slice(bytemuck::cast_slice(self));
    }
}

impl PersistentGpuBufferable for Arc<[MeshletSimplificationError]> {
    type Metadata = ();

    fn size_in_bytes(&self) -> usize {
        self.len() * size_of::<MeshletSimplificationError>()
    }

    fn write_bytes_le(&self, _: Self::Metadata, buffer_slice: &mut [u8]) {
        buffer_slice.clone_from_slice(bytemuck::cast_slice(self));
    }
}

/// Wrapper for a GPU buffer holding a large amount of data that persists across frames.
pub struct PersistentGpuBuffer<T: PersistentGpuBufferable> {
    /// Debug label for the buffer.
    label: &'static str,
    /// Handle to the GPU buffer.
    buffer: Buffer,
    /// Tracks free slices of the buffer.
    allocation_planner: RangeAllocator<BufferAddress>,
    /// Queue of pending writes, and associated metadata.
    write_queue: Vec<(T, T::Metadata, Range<BufferAddress>)>,
}

impl<T: PersistentGpuBufferable> PersistentGpuBuffer<T> {
    /// Create a new persistent buffer.
    pub fn new(label: &'static str, render_device: &RenderDevice) -> Self {
        Self {
            label,
            buffer: render_device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size: 0,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            allocation_planner: RangeAllocator::new(0..0),
            write_queue: Vec::new(),
        }
    }

    /// Queue an item of type T to be added to the buffer, returning the byte range within the buffer that it will be located at.
    pub fn queue_write(&mut self, data: T, metadata: T::Metadata) -> Range<BufferAddress> {
        let data_size = data.size_in_bytes() as u64;
        debug_assert!(data_size % COPY_BUFFER_ALIGNMENT == 0);
        if let Ok(buffer_slice) = self.allocation_planner.allocate_range(data_size) {
            self.write_queue
                .push((data, metadata, buffer_slice.clone()));
            return buffer_slice;
        }

        let buffer_size = self.allocation_planner.initial_range();
        let double_buffer_size = (buffer_size.end - buffer_size.start) * 2;
        let new_size = double_buffer_size.max(data_size);
        self.allocation_planner.grow_to(buffer_size.end + new_size);

        let buffer_slice = self.allocation_planner.allocate_range(data_size).unwrap();
        self.write_queue
            .push((data, metadata, buffer_slice.clone()));
        buffer_slice
    }

    /// Upload all pending data to the GPU buffer.
    pub fn perform_writes(&mut self, render_queue: &RenderQueue, render_device: &RenderDevice) {
        if self.allocation_planner.initial_range().end > self.buffer.size() {
            self.expand_buffer(render_device, render_queue);
        }

        let queue_count = self.write_queue.len();

        for (data, metadata, buffer_slice) in self.write_queue.drain(..) {
            let buffer_slice_size =
                NonZero::<u64>::new(buffer_slice.end - buffer_slice.start).unwrap();
            let mut buffer_view = render_queue
                .write_buffer_with(&self.buffer, buffer_slice.start, buffer_slice_size)
                .unwrap();
            data.write_bytes_le(metadata, &mut buffer_view);
        }

        let queue_saturation = queue_count as f32 / self.write_queue.capacity() as f32;
        if queue_saturation < 0.3 {
            self.write_queue = Vec::new();
        }
    }

    /// Mark a section of the GPU buffer as no longer needed.
    pub fn mark_slice_unused(&mut self, buffer_slice: Range<BufferAddress>) {
        self.allocation_planner.free_range(buffer_slice);
    }

    pub fn binding(&self) -> BindingResource<'_> {
        self.buffer.as_entire_binding()
    }

    /// Expand the buffer by creating a new buffer and copying old data over.
    fn expand_buffer(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        let size = self.allocation_planner.initial_range();
        let new_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some(self.label),
            size: size.end - size.start,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut command_encoder = render_device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("persistent_gpu_buffer_expand"),
        });
        command_encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, self.buffer.size());
        render_queue.submit([command_encoder.finish()]);

        self.buffer = new_buffer;
    }
}

/// A trait representing data that can be written to a [`PersistentGpuBuffer`].
pub trait PersistentGpuBufferable {
    /// Additional metadata associated with each item, made available during `write_bytes_le`.
    type Metadata;

    /// The size in bytes of `self`. This will determine the size of the buffer passed into
    /// `write_bytes_le`.
    ///
    /// All data written must be in a multiple of `wgpu::COPY_BUFFER_ALIGNMENT` bytes. Failure to do so will
    /// result in a panic when using [`PersistentGpuBuffer`].
    fn size_in_bytes(&self) -> usize;

    /// Convert `self` + `metadata` into bytes (little-endian), and write to the provided buffer slice.
    /// Any bytes not written to in the slice will be zeroed out when uploaded to the GPU.
    fn write_bytes_le(&self, metadata: Self::Metadata, buffer_slice: &mut [u8]);
}

#[derive(Resource)]
pub struct MeshletPipelines {
    fill_cluster_buffers: CachedComputePipelineId,
    clear_visibility_buffer: CachedComputePipelineId,
    clear_visibility_buffer_shadow_view: CachedComputePipelineId,
    cull_first: CachedComputePipelineId,
    cull_second: CachedComputePipelineId,
    downsample_depth_first: CachedComputePipelineId,
    downsample_depth_second: CachedComputePipelineId,
    downsample_depth_first_shadow_view: CachedComputePipelineId,
    downsample_depth_second_shadow_view: CachedComputePipelineId,
    visibility_buffer_software_raster: CachedComputePipelineId,
    visibility_buffer_software_raster_shadow_view: CachedComputePipelineId,
    visibility_buffer_hardware_raster: CachedRenderPipelineId,
    visibility_buffer_hardware_raster_shadow_view: CachedRenderPipelineId,
    visibility_buffer_hardware_raster_shadow_view_unclipped: CachedRenderPipelineId,
    resolve_depth: CachedRenderPipelineId,
    resolve_depth_shadow_view: CachedRenderPipelineId,
    resolve_material_depth: CachedRenderPipelineId,
    remap_1d_to_2d_dispatch: Option<CachedComputePipelineId>,
}

impl FromWorld for MeshletPipelines {
    fn from_world(world: &mut World) -> Self {
        let resource_manager = world.resource::<ResourceManager>();
        let fill_cluster_buffers_bind_group_layout = resource_manager
            .fill_cluster_buffers_bind_group_layout
            .clone();
        let clear_visibility_buffer_bind_group_layout = resource_manager
            .clear_visibility_buffer_bind_group_layout
            .clone();
        let clear_visibility_buffer_shadow_view_bind_group_layout = resource_manager
            .clear_visibility_buffer_shadow_view_bind_group_layout
            .clone();
        let cull_layout = resource_manager.culling_bind_group_layout.clone();
        let downsample_depth_layout = resource_manager.downsample_depth_bind_group_layout.clone();
        let downsample_depth_shadow_view_layout = resource_manager
            .downsample_depth_shadow_view_bind_group_layout
            .clone();
        let visibility_buffer_raster_layout = resource_manager
            .visibility_buffer_raster_bind_group_layout
            .clone();
        let visibility_buffer_raster_shadow_view_layout = resource_manager
            .visibility_buffer_raster_shadow_view_bind_group_layout
            .clone();
        let resolve_depth_layout = resource_manager.resolve_depth_bind_group_layout.clone();
        let resolve_depth_shadow_view_layout = resource_manager
            .resolve_depth_shadow_view_bind_group_layout
            .clone();
        let resolve_material_depth_layout = resource_manager
            .resolve_material_depth_bind_group_layout
            .clone();
        let remap_1d_to_2d_dispatch_layout = resource_manager
            .remap_1d_to_2d_dispatch_bind_group_layout
            .clone();
        let pipeline_cache = world.resource_mut::<PipelineCache>();

        Self {
            fill_cluster_buffers: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_fill_cluster_buffers_pipeline".into()),
                    layout: vec![fill_cluster_buffers_bind_group_layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: MESHLET_FILL_CLUSTER_BUFFERS_SHADER_HANDLE,
                    shader_defs: vec!["MESHLET_FILL_CLUSTER_BUFFERS_PASS".into()],
                    entry_point: "fill_cluster_buffers".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            clear_visibility_buffer: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_clear_visibility_buffer_pipeline".into()),
                    layout: vec![clear_visibility_buffer_bind_group_layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..8,
                    }],
                    shader: MESHLET_CLEAR_VISIBILITY_BUFFER_SHADER_HANDLE,
                    shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into()],
                    entry_point: "clear_visibility_buffer".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            clear_visibility_buffer_shadow_view: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_clear_visibility_buffer_shadow_view_pipeline".into()),
                    layout: vec![clear_visibility_buffer_shadow_view_bind_group_layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..8,
                    }],
                    shader: MESHLET_CLEAR_VISIBILITY_BUFFER_SHADER_HANDLE,
                    shader_defs: vec![],
                    entry_point: "clear_visibility_buffer".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            cull_first: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("meshlet_culling_first_pipeline".into()),
                layout: vec![cull_layout.clone()],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..8,
                }],
                shader: MESHLET_CULLING_SHADER_HANDLE,
                shader_defs: vec![
                    "MESHLET_CULLING_PASS".into(),
                    "MESHLET_FIRST_CULLING_PASS".into(),
                ],
                entry_point: "cull_clusters".into(),
                zero_initialize_workgroup_memory: false,
            }),

            cull_second: pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some("meshlet_culling_second_pipeline".into()),
                layout: vec![cull_layout],
                push_constant_ranges: vec![PushConstantRange {
                    stages: ShaderStages::COMPUTE,
                    range: 0..8,
                }],
                shader: MESHLET_CULLING_SHADER_HANDLE,
                shader_defs: vec![
                    "MESHLET_CULLING_PASS".into(),
                    "MESHLET_SECOND_CULLING_PASS".into(),
                ],
                entry_point: "cull_clusters".into(),
                zero_initialize_workgroup_memory: false,
            }),

            downsample_depth_first: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_downsample_depth_first_pipeline".into()),
                    layout: vec![downsample_depth_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                    shader_defs: vec![
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into(),
                        "MESHLET".into(),
                    ],
                    entry_point: "downsample_depth_first".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            downsample_depth_second: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_downsample_depth_second_pipeline".into()),
                    layout: vec![downsample_depth_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                    shader_defs: vec![
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into(),
                        "MESHLET".into(),
                    ],
                    entry_point: "downsample_depth_second".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            downsample_depth_first_shadow_view: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_downsample_depth_first_pipeline".into()),
                    layout: vec![downsample_depth_shadow_view_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                    shader_defs: vec!["MESHLET".into()],
                    entry_point: "downsample_depth_first".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            downsample_depth_second_shadow_view: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_downsample_depth_second_pipeline".into()),
                    layout: vec![downsample_depth_shadow_view_layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: DOWNSAMPLE_DEPTH_SHADER_HANDLE,
                    shader_defs: vec!["MESHLET".into()],
                    entry_point: "downsample_depth_second".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            visibility_buffer_software_raster: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some("meshlet_visibility_buffer_software_raster_pipeline".into()),
                    layout: vec![visibility_buffer_raster_layout.clone()],
                    push_constant_ranges: vec![],
                    shader: MESHLET_VISIBILITY_BUFFER_SOFTWARE_RASTER_SHADER_HANDLE,
                    shader_defs: vec![
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into(),
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into(),
                        if remap_1d_to_2d_dispatch_layout.is_some() {
                            "MESHLET_2D_DISPATCH"
                        } else {
                            ""
                        }
                        .into(),
                    ],
                    entry_point: "rasterize_cluster".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            visibility_buffer_software_raster_shadow_view: pipeline_cache.queue_compute_pipeline(
                ComputePipelineDescriptor {
                    label: Some(
                        "meshlet_visibility_buffer_software_raster_shadow_view_pipeline".into(),
                    ),
                    layout: vec![visibility_buffer_raster_shadow_view_layout.clone()],
                    push_constant_ranges: vec![],
                    shader: MESHLET_VISIBILITY_BUFFER_SOFTWARE_RASTER_SHADER_HANDLE,
                    shader_defs: vec![
                        "MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into(),
                        if remap_1d_to_2d_dispatch_layout.is_some() {
                            "MESHLET_2D_DISPATCH"
                        } else {
                            ""
                        }
                        .into(),
                    ],
                    entry_point: "rasterize_cluster".into(),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            visibility_buffer_hardware_raster: pipeline_cache.queue_render_pipeline(
                RenderPipelineDescriptor {
                    label: Some("meshlet_visibility_buffer_hardware_raster_pipeline".into()),
                    layout: vec![visibility_buffer_raster_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        range: 0..4,
                    }],
                    vertex: VertexState {
                        shader: MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
                        shader_defs: vec![
                            "MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into(),
                            "MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into(),
                        ],
                        entry_point: "vertex".into(),
                        buffers: vec![],
                    },
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: FrontFace::Ccw,
                        cull_mode: Some(Face::Back),
                        unclipped_depth: false,
                        polygon_mode: PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
                        shader_defs: vec![
                            "MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into(),
                            "MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into(),
                        ],
                        entry_point: "fragment".into(),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::R8Uint,
                            blend: None,
                            write_mask: ColorWrites::empty(),
                        })],
                    }),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            visibility_buffer_hardware_raster_shadow_view: pipeline_cache.queue_render_pipeline(
                RenderPipelineDescriptor {
                    label: Some(
                        "meshlet_visibility_buffer_hardware_raster_shadow_view_pipeline".into(),
                    ),
                    layout: vec![visibility_buffer_raster_shadow_view_layout.clone()],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        range: 0..4,
                    }],
                    vertex: VertexState {
                        shader: MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
                        shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into()],
                        entry_point: "vertex".into(),
                        buffers: vec![],
                    },
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: FrontFace::Ccw,
                        cull_mode: Some(Face::Back),
                        unclipped_depth: false,
                        polygon_mode: PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
                        shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into()],
                        entry_point: "fragment".into(),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::R8Uint,
                            blend: None,
                            write_mask: ColorWrites::empty(),
                        })],
                    }),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            visibility_buffer_hardware_raster_shadow_view_unclipped: pipeline_cache
                .queue_render_pipeline(RenderPipelineDescriptor {
                    label: Some(
                        "meshlet_visibility_buffer_hardware_raster_shadow_view_unclipped_pipeline"
                            .into(),
                    ),
                    layout: vec![visibility_buffer_raster_shadow_view_layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::VERTEX,
                        range: 0..4,
                    }],
                    vertex: VertexState {
                        shader: MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
                        shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into()],
                        entry_point: "vertex".into(),
                        buffers: vec![],
                    },
                    primitive: PrimitiveState {
                        topology: PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: FrontFace::Ccw,
                        cull_mode: Some(Face::Back),
                        unclipped_depth: true,
                        polygon_mode: PolygonMode::Fill,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: MESHLET_VISIBILITY_BUFFER_HARDWARE_RASTER_SHADER_HANDLE,
                        shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS".into()],
                        entry_point: "fragment".into(),
                        targets: vec![Some(ColorTargetState {
                            format: TextureFormat::R8Uint,
                            blend: None,
                            write_mask: ColorWrites::empty(),
                        })],
                    }),
                    zero_initialize_workgroup_memory: false,
                }),

            resolve_depth: pipeline_cache.queue_render_pipeline(RenderPipelineDescriptor {
                label: Some("meshlet_resolve_depth_pipeline".into()),
                layout: vec![resolve_depth_layout],
                push_constant_ranges: vec![],
                vertex: fullscreen_shader_vertex_state(),
                primitive: PrimitiveState::default(),
                depth_stencil: Some(DepthStencilState {
                    format: CORE_3D_DEPTH_FORMAT,
                    depth_write_enabled: true,
                    depth_compare: CompareFunction::Always,
                    stencil: StencilState::default(),
                    bias: DepthBiasState::default(),
                }),
                multisample: MultisampleState::default(),
                fragment: Some(FragmentState {
                    shader: MESHLET_RESOLVE_RENDER_TARGETS_SHADER_HANDLE,
                    shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into()],
                    entry_point: "resolve_depth".into(),
                    targets: vec![],
                }),
                zero_initialize_workgroup_memory: false,
            }),

            resolve_depth_shadow_view: pipeline_cache.queue_render_pipeline(
                RenderPipelineDescriptor {
                    label: Some("meshlet_resolve_depth_pipeline".into()),
                    layout: vec![resolve_depth_shadow_view_layout],
                    push_constant_ranges: vec![],
                    vertex: fullscreen_shader_vertex_state(),
                    primitive: PrimitiveState::default(),
                    depth_stencil: Some(DepthStencilState {
                        format: CORE_3D_DEPTH_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: CompareFunction::Always,
                        stencil: StencilState::default(),
                        bias: DepthBiasState::default(),
                    }),
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: MESHLET_RESOLVE_RENDER_TARGETS_SHADER_HANDLE,
                        shader_defs: vec![],
                        entry_point: "resolve_depth".into(),
                        targets: vec![],
                    }),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            resolve_material_depth: pipeline_cache.queue_render_pipeline(
                RenderPipelineDescriptor {
                    label: Some("meshlet_resolve_material_depth_pipeline".into()),
                    layout: vec![resolve_material_depth_layout],
                    push_constant_ranges: vec![],
                    vertex: fullscreen_shader_vertex_state(),
                    primitive: PrimitiveState::default(),
                    depth_stencil: Some(DepthStencilState {
                        format: TextureFormat::Depth16Unorm,
                        depth_write_enabled: true,
                        depth_compare: CompareFunction::Always,
                        stencil: StencilState::default(),
                        bias: DepthBiasState::default(),
                    }),
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        shader: MESHLET_RESOLVE_RENDER_TARGETS_SHADER_HANDLE,
                        shader_defs: vec!["MESHLET_VISIBILITY_BUFFER_RASTER_PASS_OUTPUT".into()],
                        entry_point: "resolve_material_depth".into(),
                        targets: vec![],
                    }),
                    zero_initialize_workgroup_memory: false,
                },
            ),

            remap_1d_to_2d_dispatch: remap_1d_to_2d_dispatch_layout.map(|layout| {
                pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                    label: Some("meshlet_remap_1d_to_2d_dispatch_pipeline".into()),
                    layout: vec![layout],
                    push_constant_ranges: vec![PushConstantRange {
                        stages: ShaderStages::COMPUTE,
                        range: 0..4,
                    }],
                    shader: MESHLET_REMAP_1D_TO_2D_DISPATCH_SHADER_HANDLE,
                    shader_defs: vec![],
                    entry_point: "remap_dispatch".into(),
                    zero_initialize_workgroup_memory: false,
                })
            }),
        }
    }
}

impl MeshletPipelines {
    pub fn get(
        world: &World,
    ) -> Option<(
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &ComputePipeline,
        &RenderPipeline,
        &RenderPipeline,
        &RenderPipeline,
        &RenderPipeline,
        &RenderPipeline,
        &RenderPipeline,
        Option<&ComputePipeline>,
    )> {
        let pipeline_cache = world.get_resource::<PipelineCache>()?;
        let pipeline = world.get_resource::<Self>()?;
        Some((
            pipeline_cache.get_compute_pipeline(pipeline.fill_cluster_buffers)?,
            pipeline_cache.get_compute_pipeline(pipeline.clear_visibility_buffer)?,
            pipeline_cache.get_compute_pipeline(pipeline.clear_visibility_buffer_shadow_view)?,
            pipeline_cache.get_compute_pipeline(pipeline.cull_first)?,
            pipeline_cache.get_compute_pipeline(pipeline.cull_second)?,
            pipeline_cache.get_compute_pipeline(pipeline.downsample_depth_first)?,
            pipeline_cache.get_compute_pipeline(pipeline.downsample_depth_second)?,
            pipeline_cache.get_compute_pipeline(pipeline.downsample_depth_first_shadow_view)?,
            pipeline_cache.get_compute_pipeline(pipeline.downsample_depth_second_shadow_view)?,
            pipeline_cache.get_compute_pipeline(pipeline.visibility_buffer_software_raster)?,
            pipeline_cache
                .get_compute_pipeline(pipeline.visibility_buffer_software_raster_shadow_view)?,
            pipeline_cache.get_render_pipeline(pipeline.visibility_buffer_hardware_raster)?,
            pipeline_cache
                .get_render_pipeline(pipeline.visibility_buffer_hardware_raster_shadow_view)?,
            pipeline_cache.get_render_pipeline(
                pipeline.visibility_buffer_hardware_raster_shadow_view_unclipped,
            )?,
            pipeline_cache.get_render_pipeline(pipeline.resolve_depth)?,
            pipeline_cache.get_render_pipeline(pipeline.resolve_depth_shadow_view)?,
            pipeline_cache.get_render_pipeline(pipeline.resolve_material_depth)?,
            match pipeline.remap_1d_to_2d_dispatch {
                Some(id) => Some(pipeline_cache.get_compute_pipeline(id)?),
                None => None,
            },
        ))
    }
}

/// Manages per-view and per-cluster GPU resources for [`super::MeshletPlugin`].
#[derive(Resource)]
pub struct ResourceManager {
    /// Intermediate buffer of cluster IDs for use with rasterizing the visibility buffer
    pub(super) visibility_buffer_raster_clusters: Buffer,
    /// Intermediate buffer of count of clusters to software rasterize
    pub(super) software_raster_cluster_count: Buffer,
    /// Rightmost slot index of [`Self::visibility_buffer_raster_clusters`]
    pub(super) raster_cluster_rightmost_slot: u32,

    /// Per-cluster instance ID
    pub(super) cluster_instance_ids: Option<Buffer>,
    /// Per-cluster meshlet ID
    pub(super) cluster_meshlet_ids: Option<Buffer>,
    /// Per-cluster bitmask of whether or not it's a candidate for the second raster pass
    pub(super) second_pass_candidates_buffer: Option<Buffer>,
    /// Sampler for a depth pyramid
    pub(super) depth_pyramid_sampler: Sampler,
    /// Dummy texture view for binding depth pyramids with less than the maximum amount of mips
    pub(super) depth_pyramid_dummy_texture: TextureView,

    // TODO
    pub(super) previous_depth_pyramids: EntityHashMap<TextureView>,

    // Bind group layouts
    pub fill_cluster_buffers_bind_group_layout: BindGroupLayout,
    pub clear_visibility_buffer_bind_group_layout: BindGroupLayout,
    pub clear_visibility_buffer_shadow_view_bind_group_layout: BindGroupLayout,
    pub culling_bind_group_layout: BindGroupLayout,
    pub visibility_buffer_raster_bind_group_layout: BindGroupLayout,
    pub visibility_buffer_raster_shadow_view_bind_group_layout: BindGroupLayout,
    pub downsample_depth_bind_group_layout: BindGroupLayout,
    pub downsample_depth_shadow_view_bind_group_layout: BindGroupLayout,
    pub resolve_depth_bind_group_layout: BindGroupLayout,
    pub resolve_depth_shadow_view_bind_group_layout: BindGroupLayout,
    pub resolve_material_depth_bind_group_layout: BindGroupLayout,
    pub material_shade_bind_group_layout: BindGroupLayout,
    pub remap_1d_to_2d_dispatch_bind_group_layout: Option<BindGroupLayout>,
}

impl ResourceManager {
    pub fn new(cluster_buffer_slots: u32, render_device: &RenderDevice) -> Self {
        let needs_dispatch_remap =
            cluster_buffer_slots > render_device.limits().max_compute_workgroups_per_dimension;

        Self {
            visibility_buffer_raster_clusters: render_device.create_buffer(&BufferDescriptor {
                label: Some("meshlet_visibility_buffer_raster_clusters"),
                size: cluster_buffer_slots as u64 * size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            software_raster_cluster_count: render_device.create_buffer(&BufferDescriptor {
                label: Some("meshlet_software_raster_cluster_count"),
                size: size_of::<u32>() as u64,
                usage: BufferUsages::STORAGE,
                mapped_at_creation: false,
            }),
            raster_cluster_rightmost_slot: cluster_buffer_slots - 1,

            cluster_instance_ids: None,
            cluster_meshlet_ids: None,
            second_pass_candidates_buffer: None,
            depth_pyramid_sampler: render_device.create_sampler(&SamplerDescriptor {
                label: Some("meshlet_depth_pyramid_sampler"),
                ..SamplerDescriptor::default()
            }),
            depth_pyramid_dummy_texture: mip_generation::create_depth_pyramid_dummy_texture(
                render_device,
                "meshlet_depth_pyramid_dummy_texture",
                "meshlet_depth_pyramid_dummy_texture_view",
            ),

            previous_depth_pyramids: EntityHashMap::default(),

            // TODO: Buffer min sizes
            fill_cluster_buffers_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_fill_cluster_buffers_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                    ),
                ),
            ),
            clear_visibility_buffer_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_clear_visibility_buffer_bind_group_layout",
                &BindGroupLayoutEntries::single(
                    ShaderStages::COMPUTE,
                    texture_storage_2d(TextureFormat::R64Uint, StorageTextureAccess::WriteOnly),
                ),
            ),
            clear_visibility_buffer_shadow_view_bind_group_layout: render_device
                .create_bind_group_layout(
                    "meshlet_clear_visibility_buffer_shadow_view_bind_group_layout",
                    &BindGroupLayoutEntries::single(
                        ShaderStages::COMPUTE,
                        texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::WriteOnly),
                    ),
                ),
            culling_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_culling_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::COMPUTE,
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                        storage_buffer_sized(false, None),
                        texture_2d(TextureSampleType::Float { filterable: false }),
                        uniform_buffer::<ViewUniform>(true),
                        uniform_buffer::<PreviousViewData>(true),
                    ),
                ),
            ),
            downsample_depth_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_downsample_depth_bind_group_layout",
                &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, {
                    let write_only_r32float = || {
                        texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly)
                    };
                    (
                        texture_storage_2d(TextureFormat::R64Uint, StorageTextureAccess::ReadOnly),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        sampler(SamplerBindingType::NonFiltering),
                    )
                }),
            ),
            downsample_depth_shadow_view_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_downsample_depth_shadow_view_bind_group_layout",
                &BindGroupLayoutEntries::sequential(ShaderStages::COMPUTE, {
                    let write_only_r32float = || {
                        texture_storage_2d(TextureFormat::R32Float, StorageTextureAccess::WriteOnly)
                    };
                    (
                        texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::ReadOnly),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        texture_storage_2d(
                            TextureFormat::R32Float,
                            StorageTextureAccess::ReadWrite,
                        ),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        write_only_r32float(),
                        sampler(SamplerBindingType::NonFiltering),
                    )
                }),
            ),
            visibility_buffer_raster_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_visibility_buffer_raster_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::all(),
                    (
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        texture_storage_2d(TextureFormat::R64Uint, StorageTextureAccess::Atomic),
                        uniform_buffer::<ViewUniform>(true),
                    ),
                ),
            ),
            visibility_buffer_raster_shadow_view_bind_group_layout: render_device
                .create_bind_group_layout(
                    "meshlet_visibility_buffer_raster_shadow_view_bind_group_layout",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::all(),
                        (
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            storage_buffer_read_only_sized(false, None),
                            texture_storage_2d(
                                TextureFormat::R32Uint,
                                StorageTextureAccess::Atomic,
                            ),
                            uniform_buffer::<ViewUniform>(true),
                        ),
                    ),
                ),
            resolve_depth_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_resolve_depth_bind_group_layout",
                &BindGroupLayoutEntries::single(
                    ShaderStages::FRAGMENT,
                    texture_storage_2d(TextureFormat::R64Uint, StorageTextureAccess::ReadOnly),
                ),
            ),
            resolve_depth_shadow_view_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_resolve_depth_shadow_view_bind_group_layout",
                &BindGroupLayoutEntries::single(
                    ShaderStages::FRAGMENT,
                    texture_storage_2d(TextureFormat::R32Uint, StorageTextureAccess::ReadOnly),
                ),
            ),
            resolve_material_depth_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_resolve_material_depth_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_storage_2d(TextureFormat::R64Uint, StorageTextureAccess::ReadOnly),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
            material_shade_bind_group_layout: render_device.create_bind_group_layout(
                "meshlet_mesh_material_shade_bind_group_layout",
                &BindGroupLayoutEntries::sequential(
                    ShaderStages::FRAGMENT,
                    (
                        texture_storage_2d(TextureFormat::R64Uint, StorageTextureAccess::ReadOnly),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                        storage_buffer_read_only_sized(false, None),
                    ),
                ),
            ),
            remap_1d_to_2d_dispatch_bind_group_layout: needs_dispatch_remap.then(|| {
                render_device.create_bind_group_layout(
                    "meshlet_remap_1d_to_2d_dispatch_bind_group_layout",
                    &BindGroupLayoutEntries::sequential(
                        ShaderStages::COMPUTE,
                        (
                            storage_buffer_sized(false, None),
                            storage_buffer_sized(false, None),
                        ),
                    ),
                )
            }),
        }
    }
}

/// Fullscreen shading pass based on the visibility buffer generated from rasterizing meshlets.
#[derive(Default)]
pub struct MeshletMainOpaquePass3dNode;

/// Fullscreen pass to generate a gbuffer based on the visibility buffer generated from rasterizing meshlets.
#[derive(Default)]
pub struct MeshletDeferredGBufferPrepassNode;

/// Fullscreen pass to generate prepass textures based on the visibility buffer generated from rasterizing meshlets.
#[derive(Default)]
pub struct MeshletPrepassNode;

impl ViewNode for MeshletMainOpaquePass3dNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewTarget,
        &'static MeshViewBindGroup,
        &'static ViewUniformOffset,
        &'static ViewLightsUniformOffset,
        &'static ViewFogUniformOffset,
        &'static ViewLightProbesUniformOffset,
        &'static ViewScreenSpaceReflectionsUniformOffset,
        &'static ViewEnvironmentMapUniformOffset,
        &'static MeshletViewMaterialsMainOpaquePass,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            camera,
            target,
            mesh_view_bind_group,
            view_uniform_offset,
            view_lights_offset,
            view_fog_offset,
            view_light_probes_offset,
            view_ssr_offset,
            view_environment_map_offset,
            meshlet_view_materials,
            meshlet_view_bind_groups,
            meshlet_view_resources,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if meshlet_view_materials.is_empty() {
            return Ok(());
        }

        let (
            Some(instance_manager),
            Some(pipeline_cache),
            Some(meshlet_material_depth),
            Some(meshlet_material_shade_bind_group),
        ) = (
            world.get_resource::<InstanceManager>(),
            world.get_resource::<PipelineCache>(),
            meshlet_view_resources.material_depth.as_ref(),
            meshlet_view_bind_groups.material_shade.as_ref(),
        )
        else {
            return Ok(());
        };

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("meshlet_main_opaque_pass_3d"),
            color_attachments: &[Some(target.get_color_attachment())],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &meshlet_material_depth.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        render_pass.set_bind_group(
            0,
            &mesh_view_bind_group.value,
            &[
                view_uniform_offset.offset,
                view_lights_offset.offset,
                view_fog_offset.offset,
                **view_light_probes_offset,
                **view_ssr_offset,
                **view_environment_map_offset,
            ],
        );
        render_pass.set_bind_group(1, meshlet_material_shade_bind_group, &[]);

        // 1 fullscreen triangle draw per material
        for (material_id, material_pipeline_id, material_bind_group) in
            meshlet_view_materials.iter()
        {
            if instance_manager.material_present_in_scene(material_id) {
                if let Some(material_pipeline) =
                    pipeline_cache.get_render_pipeline(*material_pipeline_id)
                {
                    let x = *material_id * 3;
                    render_pass.set_render_pipeline(material_pipeline);
                    render_pass.set_bind_group(2, material_bind_group, &[]);
                    render_pass.draw(x..(x + 3), 0..1);
                }
            }
        }

        Ok(())
    }
}

impl ViewNode for MeshletPrepassNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewPrepassTextures,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        Has<MotionVectorPrepass>,
        &'static MeshletViewMaterialsPrepass,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            camera,
            view_prepass_textures,
            view_uniform_offset,
            previous_view_uniform_offset,
            view_has_motion_vector_prepass,
            meshlet_view_materials,
            meshlet_view_bind_groups,
            meshlet_view_resources,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if meshlet_view_materials.is_empty() {
            return Ok(());
        }

        let (
            Some(prepass_view_bind_group),
            Some(instance_manager),
            Some(pipeline_cache),
            Some(meshlet_material_depth),
            Some(meshlet_material_shade_bind_group),
        ) = (
            world.get_resource::<PrepassViewBindGroup>(),
            world.get_resource::<InstanceManager>(),
            world.get_resource::<PipelineCache>(),
            meshlet_view_resources.material_depth.as_ref(),
            meshlet_view_bind_groups.material_shade.as_ref(),
        )
        else {
            return Ok(());
        };

        let color_attachments = vec![
            view_prepass_textures
                .normal
                .as_ref()
                .map(|normals_texture| normals_texture.get_attachment()),
            view_prepass_textures
                .motion_vectors
                .as_ref()
                .map(|motion_vectors_texture| motion_vectors_texture.get_attachment()),
            // Use None in place of Deferred attachments
            None,
            None,
        ];

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("meshlet_prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &meshlet_material_depth.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        if view_has_motion_vector_prepass {
            render_pass.set_bind_group(
                0,
                prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
                &[
                    view_uniform_offset.offset,
                    previous_view_uniform_offset.offset,
                ],
            );
        } else {
            render_pass.set_bind_group(
                0,
                prepass_view_bind_group.no_motion_vectors.as_ref().unwrap(),
                &[view_uniform_offset.offset],
            );
        }

        render_pass.set_bind_group(1, meshlet_material_shade_bind_group, &[]);

        // 1 fullscreen triangle draw per material
        for (material_id, material_pipeline_id, material_bind_group) in
            meshlet_view_materials.iter()
        {
            if instance_manager.material_present_in_scene(material_id) {
                if let Some(material_pipeline) =
                    pipeline_cache.get_render_pipeline(*material_pipeline_id)
                {
                    let x = *material_id * 3;
                    render_pass.set_render_pipeline(material_pipeline);
                    render_pass.set_bind_group(2, material_bind_group, &[]);
                    render_pass.draw(x..(x + 3), 0..1);
                }
            }
        }

        Ok(())
    }
}

impl ViewNode for MeshletDeferredGBufferPrepassNode {
    type ViewQuery = (
        &'static ExtractedCamera,
        &'static ViewPrepassTextures,
        &'static ViewUniformOffset,
        &'static PreviousViewUniformOffset,
        Has<MotionVectorPrepass>,
        &'static MeshletViewMaterialsDeferredGBufferPrepass,
        &'static MeshletViewBindGroups,
        &'static MeshletViewResources,
    );

    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext,
        (
            camera,
            view_prepass_textures,
            view_uniform_offset,
            previous_view_uniform_offset,
            view_has_motion_vector_prepass,
            meshlet_view_materials,
            meshlet_view_bind_groups,
            meshlet_view_resources,
        ): QueryItem<Self::ViewQuery>,
        world: &World,
    ) -> Result<(), NodeRunError> {
        if meshlet_view_materials.is_empty() {
            return Ok(());
        }

        let (
            Some(prepass_view_bind_group),
            Some(instance_manager),
            Some(pipeline_cache),
            Some(meshlet_material_depth),
            Some(meshlet_material_shade_bind_group),
        ) = (
            world.get_resource::<PrepassViewBindGroup>(),
            world.get_resource::<InstanceManager>(),
            world.get_resource::<PipelineCache>(),
            meshlet_view_resources.material_depth.as_ref(),
            meshlet_view_bind_groups.material_shade.as_ref(),
        )
        else {
            return Ok(());
        };

        let color_attachments = vec![
            view_prepass_textures
                .normal
                .as_ref()
                .map(|normals_texture| normals_texture.get_attachment()),
            view_prepass_textures
                .motion_vectors
                .as_ref()
                .map(|motion_vectors_texture| motion_vectors_texture.get_attachment()),
            view_prepass_textures
                .deferred
                .as_ref()
                .map(|deferred_texture| deferred_texture.get_attachment()),
            view_prepass_textures
                .deferred_lighting_pass_id
                .as_ref()
                .map(|deferred_lighting_pass_id| deferred_lighting_pass_id.get_attachment()),
        ];

        let mut render_pass = render_context.begin_tracked_render_pass(RenderPassDescriptor {
            label: Some("meshlet_deferred_prepass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &meshlet_material_depth.default_view,
                depth_ops: Some(Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some(viewport) = camera.viewport.as_ref() {
            render_pass.set_camera_viewport(viewport);
        }

        if view_has_motion_vector_prepass {
            render_pass.set_bind_group(
                0,
                prepass_view_bind_group.motion_vectors.as_ref().unwrap(),
                &[
                    view_uniform_offset.offset,
                    previous_view_uniform_offset.offset,
                ],
            );
        } else {
            render_pass.set_bind_group(
                0,
                prepass_view_bind_group.no_motion_vectors.as_ref().unwrap(),
                &[view_uniform_offset.offset],
            );
        }

        render_pass.set_bind_group(1, meshlet_material_shade_bind_group, &[]);

        // 1 fullscreen triangle draw per material
        for (material_id, material_pipeline_id, material_bind_group) in
            meshlet_view_materials.iter()
        {
            if instance_manager.material_present_in_scene(material_id) {
                if let Some(material_pipeline) =
                    pipeline_cache.get_render_pipeline(*material_pipeline_id)
                {
                    let x = *material_id * 3;
                    render_pass.set_render_pipeline(material_pipeline);
                    render_pass.set_bind_group(2, material_bind_group, &[]);
                    render_pass.draw(x..(x + 3), 0..1);
                }
            }
        }

        Ok(())
    }
}

/// A list of `(Material ID, Pipeline, BindGroup)` for a view for use in [`super::MeshletMainOpaquePass3dNode`].
#[derive(Component, Deref, DerefMut, Default)]
pub struct MeshletViewMaterialsMainOpaquePass(pub Vec<(u32, CachedRenderPipelineId, BindGroup)>);

/// A list of `(Material ID, Pipeline, BindGroup)` for a view for use in [`super::MeshletPrepassNode`].
#[derive(Component, Deref, DerefMut, Default)]
pub struct MeshletViewMaterialsPrepass(pub Vec<(u32, CachedRenderPipelineId, BindGroup)>);

/// A list of `(Material ID, Pipeline, BindGroup)` for a view for use in [`super::MeshletDeferredGBufferPrepassNode`].
#[derive(Component, Deref, DerefMut, Default)]
pub struct MeshletViewMaterialsDeferredGBufferPrepass(
    pub Vec<(u32, CachedRenderPipelineId, BindGroup)>,
);

// ------------ TODO: Everything under here needs to be rewritten and cached ------------
#[derive(Component)]
pub struct MeshletViewResources {
    pub scene_instance_count: u32,
    pub scene_cluster_count: u32,
    pub second_pass_candidates_buffer: Buffer,
    pub(super) instance_visibility: Buffer,
    pub dummy_render_target: CachedTexture,
    pub visibility_buffer: CachedTexture,
    pub visibility_buffer_software_raster_indirect_args_first: Buffer,
    pub visibility_buffer_software_raster_indirect_args_second: Buffer,
    pub visibility_buffer_hardware_raster_indirect_args_first: Buffer,
    pub visibility_buffer_hardware_raster_indirect_args_second: Buffer,
    pub depth_pyramid: ViewDepthPyramid,
    pub(super) previous_depth_pyramid: TextureView,
    pub material_depth: Option<CachedTexture>,
    pub view_size: UVec2,
    pub raster_cluster_rightmost_slot: u32,
    pub(super) not_shadow_view: bool,
}

#[derive(Component)]
pub struct MeshletViewBindGroups {
    pub first_node: Arc<AtomicBool>,
    pub fill_cluster_buffers: BindGroup,
    pub clear_visibility_buffer: BindGroup,
    pub culling_first: BindGroup,
    pub culling_second: BindGroup,
    pub downsample_depth: BindGroup,
    pub visibility_buffer_raster: BindGroup,
    pub resolve_depth: BindGroup,
    pub resolve_material_depth: Option<BindGroup>,
    pub material_shade: Option<BindGroup>,
    pub remap_1d_to_2d_dispatch: Option<(BindGroup, BindGroup)>,
}
