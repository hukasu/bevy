use bevy_asset::AssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_diagnostic::FrameCount;
use bevy_ecs::resource::Resource;
use bevy_math::{Affine3, UVec2, Vec3};
use bevy_mesh::Mesh;
use bevy_platform_support::collections::hash_map::Entry;
use bevy_render::{
    batching::gpu_preprocessing::InstanceInputUniformBuffer,
    mesh::{allocator::MeshAllocator, Mesh3d, MeshTag},
    render_phase::InputUniformIndex,
    sync_world::{MainEntity, MainEntityHashMap},
};
use bevy_utils::{default, Parallel};
use nonmax::NonMaxU32;

use crate::{
    lightmap::{LightmapSlabIndex, RenderLightmaps},
    material::{MaterialBindingId, RenderMaterialBindings},
    prepass::PreviousGlobalTransform,
    skin::uniforms::SkinUniforms,
};

use super::{
    MeshCullingData, MeshFlags, MeshInputUniform, MeshTransforms, MeshesToReextractNextFrame,
    RenderMeshMaterialIds, RenderMeshQueueData,
};

bitflags::bitflags! {
    /// Various useful flags for [`RenderMeshInstance`]s.
    #[derive(Clone, Copy)]
    pub struct RenderMeshInstanceFlags: u8 {
        /// The mesh casts shadows.
        const SHADOW_CASTER           = 1 << 0;
        /// The mesh can participate in automatic batching.
        const AUTOMATIC_BATCHING      = 1 << 1;
        /// The mesh had a transform last frame and so is eligible for motion
        /// vector computation.
        const HAS_PREVIOUS_TRANSFORM  = 1 << 2;
        /// The mesh had a skin last frame and so that skin should be taken into
        /// account for motion vector computation.
        const HAS_PREVIOUS_SKIN       = 1 << 3;
        /// The mesh had morph targets last frame and so they should be taken
        /// into account for motion vector computation.
        const HAS_PREVIOUS_MORPH      = 1 << 4;
    }
}

/// CPU data that the render world keeps for each entity, when *not* using GPU
/// mesh uniform building.
#[derive(Deref, DerefMut)]
pub struct RenderMeshInstanceCpu {
    /// Data shared between both the CPU mesh uniform building and the GPU mesh
    /// uniform building paths.
    #[deref]
    pub shared: RenderMeshInstanceShared,
    /// The transform of the mesh.
    ///
    /// This will be written into the [`MeshUniform`] at the appropriate time.
    pub transforms: MeshTransforms,
}

/// CPU data that the render world needs to keep for each entity that contains a
/// mesh when using GPU mesh uniform building.
#[derive(Deref, DerefMut)]
pub struct RenderMeshInstanceGpu {
    /// Data shared between both the CPU mesh uniform building and the GPU mesh
    /// uniform building paths.
    #[deref]
    pub shared: RenderMeshInstanceShared,
    /// The translation of the mesh.
    ///
    /// This is the only part of the transform that we have to keep on CPU (for
    /// distance sorting).
    pub translation: Vec3,
    /// The index of the [`MeshInputUniform`] in the buffer.
    pub current_uniform_index: NonMaxU32,
}

/// CPU data that the render world needs to keep about each entity that contains
/// a mesh.
pub struct RenderMeshInstanceShared {
    /// The [`AssetId`] of the mesh.
    pub mesh_asset_id: AssetId<Mesh>,
    /// A slot for the material bind group index.
    pub material_bindings_index: MaterialBindingId,
    /// Various flags.
    pub flags: RenderMeshInstanceFlags,
    /// Index of the slab that the lightmap resides in, if a lightmap is
    /// present.
    pub lightmap_slab_index: Option<LightmapSlabIndex>,
    /// User supplied tag to identify this mesh instance.
    pub tag: u32,
}

impl RenderMeshInstanceShared {
    pub(crate) fn from_components(
        previous_transform: Option<&PreviousGlobalTransform>,
        mesh: &Mesh3d,
        tag: Option<&MeshTag>,
        not_shadow_caster: bool,
        no_automatic_batching: bool,
    ) -> Self {
        let mut mesh_instance_flags = RenderMeshInstanceFlags::empty();
        mesh_instance_flags.set(RenderMeshInstanceFlags::SHADOW_CASTER, !not_shadow_caster);
        mesh_instance_flags.set(
            RenderMeshInstanceFlags::AUTOMATIC_BATCHING,
            !no_automatic_batching,
        );
        mesh_instance_flags.set(
            RenderMeshInstanceFlags::HAS_PREVIOUS_TRANSFORM,
            previous_transform.is_some(),
        );

        RenderMeshInstanceShared {
            mesh_asset_id: mesh.id(),
            flags: mesh_instance_flags,
            // This gets filled in later, during `RenderMeshGpuBuilder::update`.
            material_bindings_index: default(),
            lightmap_slab_index: None,
            tag: tag.map_or(0, |i| **i),
        }
    }

    /// Returns true if this entity is eligible to participate in automatic
    /// batching.
    #[inline]
    pub fn should_batch(&self) -> bool {
        self.flags
            .contains(RenderMeshInstanceFlags::AUTOMATIC_BATCHING)
    }
}

/// Information that is gathered during the parallel portion of mesh extraction
/// when GPU mesh uniform building is enabled.
///
/// From this, the [`MeshInputUniform`] and [`RenderMeshInstanceGpu`] are
/// prepared.
pub struct RenderMeshInstanceGpuBuilder {
    /// Data that will be placed on the [`RenderMeshInstanceGpu`].
    pub shared: RenderMeshInstanceShared,
    /// The current transform.
    pub world_from_local: Affine3,
    /// Four 16-bit unsigned normalized UV values packed into a [`UVec2`]:
    ///
    /// ```text
    ///                         <--- MSB                   LSB --->
    ///                         +---- min v ----+ +---- min u ----+
    ///     lightmap_uv_rect.x: vvvvvvvv vvvvvvvv uuuuuuuu uuuuuuuu,
    ///                         +---- max v ----+ +---- max u ----+
    ///     lightmap_uv_rect.y: VVVVVVVV VVVVVVVV UUUUUUUU UUUUUUUU,
    ///
    /// (MSB: most significant bit; LSB: least significant bit.)
    /// ```
    pub lightmap_uv_rect: UVec2,
    /// The index of the previous mesh input.
    pub previous_input_index: Option<NonMaxU32>,
    /// Various flags.
    pub mesh_flags: MeshFlags,
}

/// The per-thread queues used during [`extract_meshes_for_gpu_building`].
///
/// There are two varieties of these: one for when culling happens on CPU and
/// one for when culling happens on GPU. Having the two varieties avoids wasting
/// space if GPU culling is disabled.
#[derive(Default)]
pub enum RenderMeshInstanceGpuQueue {
    /// The default value.
    ///
    /// This becomes [`RenderMeshInstanceGpuQueue::CpuCulling`] or
    /// [`RenderMeshInstanceGpuQueue::GpuCulling`] once extraction starts.
    #[default]
    None,
    /// The version of [`RenderMeshInstanceGpuQueue`] that omits the
    /// [`MeshCullingData`], so that we don't waste space when GPU
    /// culling is disabled.
    CpuCulling {
        /// Stores GPU data for each entity that became visible or changed in
        /// such a way that necessitates updating the [`MeshInputUniform`] (e.g.
        /// changed transform).
        changed: Vec<(MainEntity, RenderMeshInstanceGpuBuilder)>,
        /// Stores the IDs of entities that became invisible this frame.
        removed: Vec<MainEntity>,
    },
    /// The version of [`RenderMeshInstanceGpuQueue`] that contains the
    /// [`MeshCullingData`], used when any view has GPU culling
    /// enabled.
    GpuCulling {
        /// Stores GPU data for each entity that became visible or changed in
        /// such a way that necessitates updating the [`MeshInputUniform`] (e.g.
        /// changed transform).
        changed: Vec<(MainEntity, RenderMeshInstanceGpuBuilder, MeshCullingData)>,
        /// Stores the IDs of entities that became invisible this frame.
        removed: Vec<MainEntity>,
    },
}

impl RenderMeshInstanceGpuQueue {
    /// Clears out a [`RenderMeshInstanceGpuQueue`], creating or recreating it
    /// as necessary.
    ///
    /// `any_gpu_culling` should be set to true if any view has GPU culling
    /// enabled.
    pub(crate) fn init(&mut self, any_gpu_culling: bool) {
        match (any_gpu_culling, &mut *self) {
            (true, RenderMeshInstanceGpuQueue::GpuCulling { changed, removed }) => {
                changed.clear();
                removed.clear();
            }
            (true, _) => {
                *self = RenderMeshInstanceGpuQueue::GpuCulling {
                    changed: vec![],
                    removed: vec![],
                }
            }
            (false, RenderMeshInstanceGpuQueue::CpuCulling { changed, removed }) => {
                changed.clear();
                removed.clear();
            }
            (false, _) => {
                *self = RenderMeshInstanceGpuQueue::CpuCulling {
                    changed: vec![],
                    removed: vec![],
                }
            }
        }
    }

    /// Adds a new mesh to this queue.
    pub(crate) fn push(
        &mut self,
        entity: MainEntity,
        instance_builder: RenderMeshInstanceGpuBuilder,
        culling_data_builder: Option<MeshCullingData>,
    ) {
        match (&mut *self, culling_data_builder) {
            (
                &mut RenderMeshInstanceGpuQueue::CpuCulling {
                    changed: ref mut queue,
                    ..
                },
                None,
            ) => {
                queue.push((entity, instance_builder));
            }
            (
                &mut RenderMeshInstanceGpuQueue::GpuCulling {
                    changed: ref mut queue,
                    ..
                },
                Some(culling_data_builder),
            ) => {
                queue.push((entity, instance_builder, culling_data_builder));
            }
            (_, None) => {
                *self = RenderMeshInstanceGpuQueue::CpuCulling {
                    changed: vec![(entity, instance_builder)],
                    removed: vec![],
                };
            }
            (_, Some(culling_data_builder)) => {
                *self = RenderMeshInstanceGpuQueue::GpuCulling {
                    changed: vec![(entity, instance_builder, culling_data_builder)],
                    removed: vec![],
                };
            }
        }
    }

    /// Adds the given entity to the `removed` list, queuing it for removal.
    ///
    /// The `gpu_culling` parameter specifies whether GPU culling is enabled.
    pub(crate) fn remove(&mut self, entity: MainEntity, gpu_culling: bool) {
        match (&mut *self, gpu_culling) {
            (RenderMeshInstanceGpuQueue::None, false) => {
                *self = RenderMeshInstanceGpuQueue::CpuCulling {
                    changed: vec![],
                    removed: vec![entity],
                }
            }
            (RenderMeshInstanceGpuQueue::None, true) => {
                *self = RenderMeshInstanceGpuQueue::GpuCulling {
                    changed: vec![],
                    removed: vec![entity],
                }
            }
            (RenderMeshInstanceGpuQueue::CpuCulling { removed, .. }, _)
            | (RenderMeshInstanceGpuQueue::GpuCulling { removed, .. }, _) => {
                removed.push(entity);
            }
        }
    }
}

/// The per-thread queues containing mesh instances, populated during the
/// extract phase.
///
/// These are filled in [`extract_meshes_for_gpu_building`] and consumed in
/// [`collect_meshes_for_gpu_building`].
#[derive(Resource, Default, Deref, DerefMut)]
pub struct RenderMeshInstanceGpuQueues(Parallel<RenderMeshInstanceGpuQueue>);

/// Information that the render world keeps about each entity that contains a
/// mesh.
///
/// The set of information needed is different depending on whether CPU or GPU
/// [`MeshUniform`] building is in use.
#[derive(Resource)]
pub enum RenderMeshInstances {
    /// Information needed when using CPU mesh instance data building.
    CpuBuilding(RenderMeshInstancesCpu),
    /// Information needed when using GPU mesh instance data building.
    GpuBuilding(RenderMeshInstancesGpu),
}

impl RenderMeshInstances {
    /// Creates a new [`RenderMeshInstances`] instance.
    pub fn new(use_gpu_instance_buffer_builder: bool) -> RenderMeshInstances {
        if use_gpu_instance_buffer_builder {
            RenderMeshInstances::GpuBuilding(RenderMeshInstancesGpu::default())
        } else {
            RenderMeshInstances::CpuBuilding(RenderMeshInstancesCpu::default())
        }
    }

    /// Returns the ID of the mesh asset attached to the given entity, if any.
    pub(crate) fn mesh_asset_id(&self, entity: MainEntity) -> Option<AssetId<Mesh>> {
        match *self {
            RenderMeshInstances::CpuBuilding(ref instances) => instances.mesh_asset_id(entity),
            RenderMeshInstances::GpuBuilding(ref instances) => instances.mesh_asset_id(entity),
        }
    }

    /// Constructs [`RenderMeshQueueData`] for the given entity, if it has a
    /// mesh attached.
    pub fn render_mesh_queue_data(&self, entity: MainEntity) -> Option<RenderMeshQueueData> {
        match *self {
            RenderMeshInstances::CpuBuilding(ref instances) => {
                instances.render_mesh_queue_data(entity)
            }
            RenderMeshInstances::GpuBuilding(ref instances) => {
                instances.render_mesh_queue_data(entity)
            }
        }
    }

    /// Inserts the given flags into the CPU or GPU render mesh instance data
    /// for the given mesh as appropriate.
    pub fn insert_mesh_instance_flags(
        &mut self,
        entity: MainEntity,
        flags: RenderMeshInstanceFlags,
    ) {
        match *self {
            RenderMeshInstances::CpuBuilding(ref mut instances) => {
                instances.insert_mesh_instance_flags(entity, flags);
            }
            RenderMeshInstances::GpuBuilding(ref mut instances) => {
                instances.insert_mesh_instance_flags(entity, flags);
            }
        }
    }
}

/// Information that the render world keeps about each entity that contains a
/// mesh, when using CPU mesh instance data building.
#[derive(Default, Deref, DerefMut)]
pub struct RenderMeshInstancesCpu(MainEntityHashMap<RenderMeshInstanceCpu>);

impl RenderMeshInstancesCpu {
    fn mesh_asset_id(&self, entity: MainEntity) -> Option<AssetId<Mesh>> {
        self.get(&entity)
            .map(|render_mesh_instance| render_mesh_instance.mesh_asset_id)
    }

    fn render_mesh_queue_data(&self, entity: MainEntity) -> Option<RenderMeshQueueData> {
        self.get(&entity)
            .map(|render_mesh_instance| RenderMeshQueueData {
                shared: &render_mesh_instance.shared,
                translation: render_mesh_instance.transforms.world_from_local.translation,
                current_uniform_index: InputUniformIndex::default(),
            })
    }

    /// Inserts the given flags into the render mesh instance data for the given
    /// mesh.
    fn insert_mesh_instance_flags(&mut self, entity: MainEntity, flags: RenderMeshInstanceFlags) {
        if let Some(instance) = self.get_mut(&entity) {
            instance.flags.insert(flags);
        }
    }
}

/// Information that the render world keeps about each entity that contains a
/// mesh, when using GPU mesh instance data building.
#[derive(Default, Deref, DerefMut)]
pub struct RenderMeshInstancesGpu(MainEntityHashMap<RenderMeshInstanceGpu>);

impl RenderMeshInstancesGpu {
    fn mesh_asset_id(&self, entity: MainEntity) -> Option<AssetId<Mesh>> {
        self.get(&entity)
            .map(|render_mesh_instance| render_mesh_instance.mesh_asset_id)
    }

    fn render_mesh_queue_data(&self, entity: MainEntity) -> Option<RenderMeshQueueData> {
        self.get(&entity)
            .map(|render_mesh_instance| RenderMeshQueueData {
                shared: &render_mesh_instance.shared,
                translation: render_mesh_instance.translation,
                current_uniform_index: InputUniformIndex(
                    render_mesh_instance.current_uniform_index.into(),
                ),
            })
    }

    /// Inserts the given flags into the render mesh instance data for the given
    /// mesh.
    fn insert_mesh_instance_flags(&mut self, entity: MainEntity, flags: RenderMeshInstanceFlags) {
        if let Some(instance) = self.get_mut(&entity) {
            instance.flags.insert(flags);
        }
    }
}

impl RenderMeshInstanceGpuBuilder {
    /// Flushes this mesh instance to the [`RenderMeshInstanceGpu`] and
    /// [`MeshInputUniform`] tables, replacing the existing entry if applicable.
    pub fn update(
        mut self,
        entity: MainEntity,
        render_mesh_instances: &mut MainEntityHashMap<RenderMeshInstanceGpu>,
        current_input_buffer: &mut InstanceInputUniformBuffer<MeshInputUniform>,
        previous_input_buffer: &mut InstanceInputUniformBuffer<MeshInputUniform>,
        mesh_allocator: &MeshAllocator,
        mesh_material_ids: &RenderMeshMaterialIds,
        render_material_bindings: &RenderMaterialBindings,
        render_lightmaps: &RenderLightmaps,
        skin_uniforms: &SkinUniforms,
        timestamp: FrameCount,
        meshes_to_reextract_next_frame: &mut MeshesToReextractNextFrame,
    ) -> Option<u32> {
        let (first_vertex_index, vertex_count) =
            match mesh_allocator.mesh_vertex_slice(&self.shared.mesh_asset_id) {
                Some(mesh_vertex_slice) => (
                    mesh_vertex_slice.range.start,
                    mesh_vertex_slice.range.end - mesh_vertex_slice.range.start,
                ),
                None => (0, 0),
            };
        let (mesh_is_indexed, first_index_index, index_count) =
            match mesh_allocator.mesh_index_slice(&self.shared.mesh_asset_id) {
                Some(mesh_index_slice) => (
                    true,
                    mesh_index_slice.range.start,
                    mesh_index_slice.range.end - mesh_index_slice.range.start,
                ),
                None => (false, 0, 0),
            };
        let current_skin_index = match skin_uniforms.skin_byte_offset(entity) {
            Some(skin_index) => skin_index.index(),
            None => u32::MAX,
        };

        // Look up the material index. If we couldn't fetch the material index,
        // then the material hasn't been prepared yet, perhaps because it hasn't
        // yet loaded. In that case, add the mesh to
        // `meshes_to_reextract_next_frame` and bail.
        let mesh_material = mesh_material_ids.mesh_material(entity);
        let mesh_material_binding_id = match render_material_bindings.get(&mesh_material) {
            Some(binding_id) => *binding_id,
            None => {
                meshes_to_reextract_next_frame.insert(entity);
                return None;
            }
        };
        self.shared.material_bindings_index = mesh_material_binding_id;

        let lightmap_slot = match render_lightmaps.render_lightmaps.get(&entity) {
            Some(render_lightmap) => u16::from(*render_lightmap.slot_index),
            None => u16::MAX,
        };
        let lightmap_slab_index = render_lightmaps
            .render_lightmaps
            .get(&entity)
            .map(|lightmap| lightmap.slab_index);
        self.shared.lightmap_slab_index = lightmap_slab_index;

        // Create the mesh input uniform.
        let mut mesh_input_uniform = MeshInputUniform {
            world_from_local: self.world_from_local.to_transpose(),
            lightmap_uv_rect: self.lightmap_uv_rect,
            flags: self.mesh_flags.bits(),
            previous_input_index: u32::MAX,
            timestamp: timestamp.0,
            first_vertex_index,
            first_index_index,
            index_count: if mesh_is_indexed {
                index_count
            } else {
                vertex_count
            },
            current_skin_index,
            material_and_lightmap_bind_group_slot: u32::from(
                self.shared.material_bindings_index.slot,
            ) | ((lightmap_slot as u32) << 16),
            tag: self.shared.tag,
            pad: 0,
        };

        // Did the last frame contain this entity as well?
        let current_uniform_index;
        match render_mesh_instances.entry(entity) {
            Entry::Occupied(mut occupied_entry) => {
                // Yes, it did. Replace its entry with the new one.

                // Reserve a slot.
                current_uniform_index = u32::from(occupied_entry.get_mut().current_uniform_index);

                // Save the old mesh input uniform. The mesh preprocessing
                // shader will need it to compute motion vectors.
                let previous_mesh_input_uniform =
                    current_input_buffer.get_unchecked(current_uniform_index);
                let previous_input_index = previous_input_buffer.add(previous_mesh_input_uniform);
                mesh_input_uniform.previous_input_index = previous_input_index;

                // Write in the new mesh input uniform.
                current_input_buffer.set(current_uniform_index, mesh_input_uniform);

                occupied_entry.replace_entry_with(|_, _| {
                    Some(RenderMeshInstanceGpu {
                        translation: self.world_from_local.translation,
                        shared: self.shared,
                        current_uniform_index: NonMaxU32::new(current_uniform_index)
                            .unwrap_or_default(),
                    })
                });
            }

            Entry::Vacant(vacant_entry) => {
                // No, this is a new entity. Push its data on to the buffer.
                current_uniform_index = current_input_buffer.add(mesh_input_uniform);

                vacant_entry.insert(RenderMeshInstanceGpu {
                    translation: self.world_from_local.translation,
                    shared: self.shared,
                    current_uniform_index: NonMaxU32::new(current_uniform_index)
                        .unwrap_or_default(),
                });
            }
        }

        Some(current_uniform_index)
    }
}
