pub mod instance;
pub mod pipeline;

use bevy_asset::{AssetId, UntypedAssetId};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{component::Component, resource::Resource};
use bevy_math::{uvec2, vec4, Affine3, Mat4, Rect, UVec2, Vec3, Vec4};
use bevy_mesh::{morph::MAX_MORPH_WEIGHTS, BaseMeshPipelineKey, Mesh};
use bevy_platform_support::collections::HashMap;
use bevy_render::{
    primitives::Aabb,
    render_phase::InputUniformIndex,
    render_resource::{
        BindGroup, BindGroupLayout, BindGroupLayoutEntries, BindingResource, Buffer, BufferUsages,
        RawBufferVec, ShaderStages, ShaderType, TextureView,
    },
    renderer::{RenderAdapter, RenderDevice},
    sync_world::{MainEntity, MainEntityHashMap, MainEntityHashSet},
};
use bevy_transform::components::GlobalTransform;
use bevy_utils::TypeIdMap;

use bytemuck::{Pod, Zeroable};
use nonmax::NonMaxU16;
use static_assertions::const_assert_eq;

use crate::{
    binding_arrays_are_usable,
    lightmap::{
        lightmap::{LightmapSlab, LightmapSlotIndex},
        LightmapSlabIndex,
    },
    material::material::MaterialBindGroupSlot,
    skin::MAX_JOINTS,
};

use instance::RenderMeshInstanceShared;
use pipeline::MeshPipelineKey;

pub const TONEMAPPING_LUT_TEXTURE_BINDING_INDEX: u32 = 26;
pub const TONEMAPPING_LUT_SAMPLER_BINDING_INDEX: u32 = 27;

/// This is used to allocate buffers.
/// The correctness of the value depends on the GPU/platform.
/// The current value is chosen because it is guaranteed to work everywhere.
/// To allow for bigger values, a check must be made for the limits
/// of the GPU at runtime, which would mean not using consts anymore.
pub const MORPH_BUFFER_SIZE: usize = MAX_MORPH_WEIGHTS * MORPH_WEIGHT_SIZE;
const MORPH_WEIGHT_SIZE: usize = size_of::<f32>();

pub(crate) const JOINT_BUFFER_SIZE: usize = MAX_JOINTS * JOINT_SIZE;
const JOINT_SIZE: usize = size_of::<Mat4>();

/// Data that [`crate::material::queue_material_meshes`] and similar systems
/// need in order to place entities that contain meshes in the right batch.
#[derive(Deref)]
pub struct RenderMeshQueueData<'a> {
    /// General information about the mesh instance.
    #[deref]
    pub shared: &'a RenderMeshInstanceShared,
    /// The translation of the mesh instance.
    pub translation: Vec3,
    /// The index of the [`MeshInputUniform`] in the GPU buffer for this mesh
    /// instance.
    pub current_uniform_index: InputUniformIndex,
}

/// The bind groups for meshes currently loaded.
///
/// If GPU mesh preprocessing isn't in use, these are global to the scene. If
/// GPU mesh preprocessing is in use, these are specific to a single phase.
#[derive(Default)]
pub struct MeshPhaseBindGroups {
    pub model_only: Option<BindGroup>,
    pub skinned: Option<MeshBindGroupPair>,
    pub morph_targets: HashMap<AssetId<Mesh>, MeshBindGroupPair>,
    pub lightmaps: HashMap<LightmapSlabIndex, BindGroup>,
}

pub struct MeshBindGroupPair {
    pub motion_vectors: BindGroup,
    pub no_motion_vectors: BindGroup,
}

/// All bind groups for meshes currently loaded.
#[derive(Resource)]
pub enum MeshBindGroups {
    /// The bind groups for the meshes for the entire scene, if GPU mesh
    /// preprocessing isn't in use.
    CpuPreprocessing(MeshPhaseBindGroups),
    /// A mapping from the type ID of a phase (e.g. [`Opaque3d`]) to the mesh
    /// bind groups for that phase.
    GpuPreprocessing(TypeIdMap<MeshPhaseBindGroups>),
}

impl MeshPhaseBindGroups {
    pub fn reset(&mut self) {
        self.model_only = None;
        self.skinned = None;
        self.morph_targets.clear();
        self.lightmaps.clear();
    }
    /// Get the `BindGroup` for `RenderMesh` with given `handle_id` and lightmap
    /// key `lightmap`.
    pub fn get(
        &self,
        asset_id: AssetId<Mesh>,
        lightmap: Option<LightmapSlabIndex>,
        is_skinned: bool,
        morph: bool,
        motion_vectors: bool,
    ) -> Option<&BindGroup> {
        match (is_skinned, morph, lightmap) {
            (_, true, _) => self
                .morph_targets
                .get(&asset_id)
                .map(|bind_group_pair| bind_group_pair.get(motion_vectors)),
            (true, false, _) => self
                .skinned
                .as_ref()
                .map(|bind_group_pair| bind_group_pair.get(motion_vectors)),
            (false, false, Some(lightmap_slab)) => self.lightmaps.get(&lightmap_slab),
            (false, false, None) => self.model_only.as_ref(),
        }
    }
}

impl MeshBindGroupPair {
    fn get(&self, motion_vectors: bool) -> &BindGroup {
        if motion_vectors {
            &self.motion_vectors
        } else {
            &self.no_motion_vectors
        }
    }
}

/// Holds a list of meshes that couldn't be extracted this frame because their
/// materials weren't prepared yet.
///
/// On subsequent frames, we try to reextract those meshes.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct MeshesToReextractNextFrame(MainEntityHashSet);

#[derive(Component)]
pub struct MeshTransforms {
    pub world_from_local: Affine3,
    pub previous_world_from_local: Affine3,
    pub flags: u32,
}

// NOTE: These must match the bit flags in bevy_pbr/src/render/mesh_types.wgsl!
bitflags::bitflags! {
    /// Various flags and tightly-packed values on a mesh.
    ///
    /// Flags grow from the top bit down; other values grow from the bottom bit
    /// up.
    #[repr(transparent)]
    pub struct MeshFlags: u32 {
        /// Bitmask for the 16-bit index into the LOD array.
        ///
        /// This will be `u16::MAX` if this mesh has no LOD.
        const LOD_INDEX_MASK              = (1 << 16) - 1;
        /// Disables frustum culling for this mesh.
        ///
        /// This corresponds to the
        /// [`bevy_render::view::visibility::NoFrustumCulling`] component.
        const NO_FRUSTUM_CULLING          = 1 << 28;
        const SHADOW_RECEIVER             = 1 << 29;
        const TRANSMITTED_SHADOW_RECEIVER = 1 << 30;
        // Indicates the sign of the determinant of the 3x3 model matrix. If the sign is positive,
        // then the flag should be set, else it should not be set.
        const SIGN_DETERMINANT_MODEL_3X3  = 1 << 31;
        const NONE                        = 0;
        const UNINITIALIZED               = 0xFFFFFFFF;
    }
}

impl MeshFlags {
    pub(super) fn from_components(
        transform: &GlobalTransform,
        lod_index: Option<NonMaxU16>,
        no_frustum_culling: bool,
        not_shadow_receiver: bool,
        transmitted_receiver: bool,
    ) -> MeshFlags {
        let mut mesh_flags = if not_shadow_receiver {
            MeshFlags::empty()
        } else {
            MeshFlags::SHADOW_RECEIVER
        };
        if no_frustum_culling {
            mesh_flags |= MeshFlags::NO_FRUSTUM_CULLING;
        }
        if transmitted_receiver {
            mesh_flags |= MeshFlags::TRANSMITTED_SHADOW_RECEIVER;
        }
        if transform.affine().matrix3.determinant().is_sign_positive() {
            mesh_flags |= MeshFlags::SIGN_DETERMINANT_MODEL_3X3;
        }

        let lod_index_bits = match lod_index {
            None => u16::MAX,
            Some(lod_index) => u16::from(lod_index),
        };
        mesh_flags |=
            MeshFlags::from_bits_retain((lod_index_bits as u32) << MeshFlags::LOD_INDEX_SHIFT);

        mesh_flags
    }

    /// The first bit of the LOD index.
    pub const LOD_INDEX_SHIFT: u32 = 0;
}
#[derive(ShaderType, Clone)]
pub struct MeshUniform {
    // Affine 4x3 matrices transposed to 3x4
    pub world_from_local: [Vec4; 3],
    pub previous_world_from_local: [Vec4; 3],
    // 3x3 matrix packed in mat2x4 and f32 as:
    //   [0].xyz, [1].x,
    //   [1].yz, [2].xy
    //   [2].z
    pub local_from_world_transpose_a: [Vec4; 2],
    pub local_from_world_transpose_b: f32,
    pub flags: u32,
    // Four 16-bit unsigned normalized UV values packed into a `UVec2`:
    //
    //                         <--- MSB                   LSB --->
    //                         +---- min v ----+ +---- min u ----+
    //     lightmap_uv_rect.x: vvvvvvvv vvvvvvvv uuuuuuuu uuuuuuuu,
    //                         +---- max v ----+ +---- max u ----+
    //     lightmap_uv_rect.y: VVVVVVVV VVVVVVVV UUUUUUUU UUUUUUUU,
    //
    // (MSB: most significant bit; LSB: least significant bit.)
    pub lightmap_uv_rect: UVec2,
    /// The index of this mesh's first vertex in the vertex buffer.
    ///
    /// Multiple meshes can be packed into a single vertex buffer (see
    /// [`MeshAllocator`]). This value stores the offset of the first vertex in
    /// this mesh in that buffer.
    pub first_vertex_index: u32,
    /// The current skin index, or `u32::MAX` if there's no skin.
    pub current_skin_index: u32,
    /// The material and lightmap indices, packed into 32 bits.
    ///
    /// Low 16 bits: index of the material inside the bind group data.
    /// High 16 bits: index of the lightmap in the binding array.
    pub material_and_lightmap_bind_group_slot: u32,
    /// User supplied tag to identify this mesh instance.
    pub tag: u32,
    /// Padding.
    pub pad: u32,
}

impl MeshUniform {
    pub fn new(
        mesh_transforms: &MeshTransforms,
        first_vertex_index: u32,
        material_bind_group_slot: MaterialBindGroupSlot,
        maybe_lightmap: Option<(LightmapSlotIndex, Rect)>,
        current_skin_index: Option<u32>,
        tag: Option<u32>,
    ) -> Self {
        let (local_from_world_transpose_a, local_from_world_transpose_b) =
            mesh_transforms.world_from_local.inverse_transpose_3x3();
        let lightmap_bind_group_slot = match maybe_lightmap {
            None => u16::MAX,
            Some((slot_index, _)) => slot_index.into(),
        };

        Self {
            world_from_local: mesh_transforms.world_from_local.to_transpose(),
            previous_world_from_local: mesh_transforms.previous_world_from_local.to_transpose(),
            lightmap_uv_rect: pack_lightmap_uv_rect(maybe_lightmap.map(|(_, uv_rect)| uv_rect)),
            local_from_world_transpose_a,
            local_from_world_transpose_b,
            flags: mesh_transforms.flags,
            first_vertex_index,
            current_skin_index: current_skin_index.unwrap_or(u32::MAX),
            material_and_lightmap_bind_group_slot: u32::from(material_bind_group_slot)
                | ((lightmap_bind_group_slot as u32) << 16),
            tag: tag.unwrap_or(0),
            pad: 0,
        }
    }
}

/// Information that has to be transferred from CPU to GPU in order to produce
/// the full [`MeshUniform`].
///
/// This is essentially a subset of the fields in [`MeshUniform`] above.
#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default, Debug)]
#[repr(C)]
pub struct MeshInputUniform {
    /// Affine 4x3 matrix transposed to 3x4.
    pub world_from_local: [Vec4; 3],
    /// Four 16-bit unsigned normalized UV values packed into a `UVec2`:
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
    /// Various [`MeshFlags`].
    pub flags: u32,
    /// The index of this mesh's [`MeshInputUniform`] in the previous frame's
    /// buffer, if applicable.
    ///
    /// This is used for TAA. If not present, this will be `u32::MAX`.
    pub previous_input_index: u32,
    /// The index of this mesh's first vertex in the vertex buffer.
    ///
    /// Multiple meshes can be packed into a single vertex buffer (see
    /// [`MeshAllocator`]). This value stores the offset of the first vertex in
    /// this mesh in that buffer.
    pub first_vertex_index: u32,
    /// The index of this mesh's first index in the index buffer, if any.
    ///
    /// Multiple meshes can be packed into a single index buffer (see
    /// [`MeshAllocator`]). This value stores the offset of the first index in
    /// this mesh in that buffer.
    ///
    /// If this mesh isn't indexed, this value is ignored.
    pub first_index_index: u32,
    /// For an indexed mesh, the number of indices that make it up; for a
    /// non-indexed mesh, the number of vertices in it.
    pub index_count: u32,
    /// The current skin index, or `u32::MAX` if there's no skin.
    pub current_skin_index: u32,
    /// The material and lightmap indices, packed into 32 bits.
    ///
    /// Low 16 bits: index of the material inside the bind group data.
    /// High 16 bits: index of the lightmap in the binding array.
    pub material_and_lightmap_bind_group_slot: u32,
    /// The number of the frame on which this [`MeshInputUniform`] was built.
    ///
    /// This is used to validate the previous transform and skin. If this
    /// [`MeshInputUniform`] wasn't updated on this frame, then we know that
    /// neither this mesh's transform nor that of its joints have been updated
    /// on this frame, and therefore the transforms of both this mesh and its
    /// joints must be identical to those for the previous frame.
    pub timestamp: u32,
    /// User supplied tag to identify this mesh instance.
    pub tag: u32,
    /// Padding.
    pub pad: u32,
}

/// All possible [`BindGroupLayout`]s in bevy's default mesh shader (`mesh.wgsl`).
#[derive(Clone)]
pub struct MeshLayouts {
    /// The mesh model uniform (transform) and nothing else.
    pub model_only: BindGroupLayout,

    /// Includes the lightmap texture and uniform.
    pub lightmapped: BindGroupLayout,

    /// Also includes the uniform for skinning
    pub skinned: BindGroupLayout,

    /// Like [`MeshLayouts::skinned`], but includes slots for the previous
    /// frame's joint matrices, so that we can compute motion vectors.
    pub skinned_motion: BindGroupLayout,

    /// Also includes the uniform and [`MorphAttributes`] for morph targets.
    ///
    /// [`MorphAttributes`]: bevy_render::mesh::morph::MorphAttributes
    pub morphed: BindGroupLayout,

    /// Like [`MeshLayouts::morphed`], but includes a slot for the previous
    /// frame's morph weights, so that we can compute motion vectors.
    pub morphed_motion: BindGroupLayout,

    /// Also includes both uniforms for skinning and morph targets, also the
    /// morph target [`MorphAttributes`] binding.
    ///
    /// [`MorphAttributes`]: bevy_render::mesh::morph::MorphAttributes
    pub morphed_skinned: BindGroupLayout,

    /// Like [`MeshLayouts::morphed_skinned`], but includes slots for the
    /// previous frame's joint matrices and morph weights, so that we can
    /// compute motion vectors.
    pub morphed_skinned_motion: BindGroupLayout,
}

impl MeshLayouts {
    /// Prepare the layouts used by the default bevy [`Mesh`].
    ///
    /// [`Mesh`]: bevy_render::prelude::Mesh
    pub fn new(render_device: &RenderDevice, render_adapter: &RenderAdapter) -> Self {
        MeshLayouts {
            model_only: Self::model_only_layout(render_device),
            lightmapped: Self::lightmapped_layout(render_device, render_adapter),
            skinned: Self::skinned_layout(render_device),
            skinned_motion: Self::skinned_motion_layout(render_device),
            morphed: Self::morphed_layout(render_device),
            morphed_motion: Self::morphed_motion_layout(render_device),
            morphed_skinned: Self::morphed_skinned_layout(render_device),
            morphed_skinned_motion: Self::morphed_skinned_motion_layout(render_device),
        }
    }

    // ---------- create individual BindGroupLayouts ----------

    fn model_only_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "mesh_layout",
            &BindGroupLayoutEntries::single(
                ShaderStages::empty(),
                layout_entry::model(render_device),
            ),
        )
    }

    /// Creates the layout for skinned meshes.
    fn skinned_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "skinned_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                ),
            ),
        )
    }

    /// Creates the layout for skinned meshes with the infrastructure to compute
    /// motion vectors.
    fn skinned_motion_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "skinned_motion_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                    // The previous frame's joint matrix buffer.
                    (6, layout_entry::skinning(render_device)),
                ),
            ),
        )
    }

    /// Creates the layout for meshes with morph targets.
    fn morphed_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "morphed_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                ),
            ),
        )
    }

    /// Creates the layout for meshes with morph targets and the infrastructure
    /// to compute motion vectors.
    fn morphed_motion_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "morphed_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                    // The previous frame's morph weight buffer.
                    (7, layout_entry::weights()),
                ),
            ),
        )
    }

    /// Creates the bind group layout for meshes with both skins and morph
    /// targets.
    fn morphed_skinned_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "morphed_skinned_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                ),
            ),
        )
    }

    /// Creates the bind group layout for meshes with both skins and morph
    /// targets, in addition to the infrastructure to compute motion vectors.
    fn morphed_skinned_motion_layout(render_device: &RenderDevice) -> BindGroupLayout {
        render_device.create_bind_group_layout(
            "morphed_skinned_motion_mesh_layout",
            &BindGroupLayoutEntries::with_indices(
                ShaderStages::VERTEX,
                (
                    (0, layout_entry::model(render_device)),
                    // The current frame's joint matrix buffer.
                    (1, layout_entry::skinning(render_device)),
                    // The current frame's morph weight buffer.
                    (2, layout_entry::weights()),
                    (3, layout_entry::targets()),
                    // The previous frame's joint matrix buffer.
                    (6, layout_entry::skinning(render_device)),
                    // The previous frame's morph weight buffer.
                    (7, layout_entry::weights()),
                ),
            ),
        )
    }

    fn lightmapped_layout(
        render_device: &RenderDevice,
        render_adapter: &RenderAdapter,
    ) -> BindGroupLayout {
        if binding_arrays_are_usable(render_device, render_adapter) {
            render_device.create_bind_group_layout(
                "lightmapped_mesh_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::VERTEX,
                    (
                        (0, layout_entry::model(render_device)),
                        (4, layout_entry::lightmaps_texture_view_array()),
                        (5, layout_entry::lightmaps_sampler_array()),
                    ),
                ),
            )
        } else {
            render_device.create_bind_group_layout(
                "lightmapped_mesh_layout",
                &BindGroupLayoutEntries::with_indices(
                    ShaderStages::VERTEX,
                    (
                        (0, layout_entry::model(render_device)),
                        (4, layout_entry::lightmaps_texture_view()),
                        (5, layout_entry::lightmaps_sampler()),
                    ),
                ),
            )
        }
    }

    // ---------- BindGroup methods ----------

    pub fn model_only(&self, render_device: &RenderDevice, model: &BindingResource) -> BindGroup {
        render_device.create_bind_group(
            "model_only_mesh_bind_group",
            &self.model_only,
            &[entry::model(0, model.clone())],
        )
    }

    pub fn lightmapped(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        lightmap_slab: &LightmapSlab,
        bindless_lightmaps: bool,
    ) -> BindGroup {
        if bindless_lightmaps {
            let (texture_views, samplers) = lightmap_slab.build_binding_arrays();
            render_device.create_bind_group(
                "lightmapped_mesh_bind_group",
                &self.lightmapped,
                &[
                    entry::model(0, model.clone()),
                    entry::lightmaps_texture_view_array(4, &texture_views),
                    entry::lightmaps_sampler_array(5, &samplers),
                ],
            )
        } else {
            let (texture_view, sampler) = lightmap_slab.bindings_for_first_lightmap();
            render_device.create_bind_group(
                "lightmapped_mesh_bind_group",
                &self.lightmapped,
                &[
                    entry::model(0, model.clone()),
                    entry::lightmaps_texture_view(4, texture_view),
                    entry::lightmaps_sampler(5, sampler),
                ],
            )
        }
    }

    /// Creates the bind group for skinned meshes with no morph targets.
    pub fn skinned(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        current_skin: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "skinned_mesh_bind_group",
            &self.skinned,
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
            ],
        )
    }

    /// Creates the bind group for skinned meshes with no morph targets, with
    /// the infrastructure to compute motion vectors.
    ///
    /// `current_skin` is the buffer of joint matrices for this frame;
    /// `prev_skin` is the buffer for the previous frame. The latter is used for
    /// motion vector computation. If there is no such applicable buffer,
    /// `current_skin` and `prev_skin` will reference the same buffer.
    pub fn skinned_motion(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        current_skin: &Buffer,
        prev_skin: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "skinned_motion_mesh_bind_group",
            &self.skinned_motion,
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
                entry::skinning(render_device, 6, prev_skin),
            ],
        )
    }

    /// Creates the bind group for meshes with no skins but morph targets.
    pub fn morphed(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        current_weights: &Buffer,
        targets: &TextureView,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_mesh_bind_group",
            &self.morphed,
            &[
                entry::model(0, model.clone()),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
            ],
        )
    }

    /// Creates the bind group for meshes with no skins but morph targets, in
    /// addition to the infrastructure to compute motion vectors.
    ///
    /// `current_weights` is the buffer of morph weights for this frame;
    /// `prev_weights` is the buffer for the previous frame. The latter is used
    /// for motion vector computation. If there is no such applicable buffer,
    /// `current_weights` and `prev_weights` will reference the same buffer.
    pub fn morphed_motion(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        current_weights: &Buffer,
        targets: &TextureView,
        prev_weights: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_motion_mesh_bind_group",
            &self.morphed_motion,
            &[
                entry::model(0, model.clone()),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
                entry::weights(7, prev_weights),
            ],
        )
    }

    /// Creates the bind group for meshes with skins and morph targets.
    pub fn morphed_skinned(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        current_skin: &Buffer,
        current_weights: &Buffer,
        targets: &TextureView,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_skinned_mesh_bind_group",
            &self.morphed_skinned,
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
            ],
        )
    }

    /// Creates the bind group for meshes with skins and morph targets, in
    /// addition to the infrastructure to compute motion vectors.
    ///
    /// See the documentation for [`MeshLayouts::skinned_motion`] and
    /// [`MeshLayouts::morphed_motion`] above for more information about the
    /// `current_skin`, `prev_skin`, `current_weights`, and `prev_weights`
    /// buffers.
    pub fn morphed_skinned_motion(
        &self,
        render_device: &RenderDevice,
        model: &BindingResource,
        current_skin: &Buffer,
        current_weights: &Buffer,
        targets: &TextureView,
        prev_skin: &Buffer,
        prev_weights: &Buffer,
    ) -> BindGroup {
        render_device.create_bind_group(
            "morphed_skinned_motion_mesh_bind_group",
            &self.morphed_skinned_motion,
            &[
                entry::model(0, model.clone()),
                entry::skinning(render_device, 1, current_skin),
                entry::weights(2, current_weights),
                entry::targets(3, targets),
                entry::skinning(render_device, 6, prev_skin),
                entry::weights(7, prev_weights),
            ],
        )
    }
}

/// Information about each mesh instance needed to cull it on GPU.
///
/// This consists of its axis-aligned bounding box (AABB).
#[derive(ShaderType, Pod, Zeroable, Clone, Copy, Default)]
#[repr(C)]
pub struct MeshCullingData {
    /// The 3D center of the AABB in model space, padded with an extra unused
    /// float value.
    pub aabb_center: Vec4,
    /// The 3D extents of the AABB in model space, divided by two, padded with
    /// an extra unused float value.
    pub aabb_half_extents: Vec4,
}

impl MeshCullingData {
    /// Returns a new [`MeshCullingData`] initialized with the given AABB.
    ///
    /// If no AABB is provided, an infinitely-large one is conservatively
    /// chosen.
    pub(super) fn new(aabb: Option<&Aabb>) -> Self {
        match aabb {
            Some(aabb) => MeshCullingData {
                aabb_center: aabb.center.extend(0.0),
                aabb_half_extents: aabb.half_extents.extend(0.0),
            },
            None => MeshCullingData {
                aabb_center: Vec3::ZERO.extend(0.0),
                aabb_half_extents: Vec3::INFINITY.extend(0.0),
            },
        }
    }

    /// Flushes this mesh instance culling data to the
    /// [`MeshCullingDataBuffer`], replacing the existing entry if applicable.
    pub fn update(
        &self,
        mesh_culling_data_buffer: &mut MeshCullingDataBuffer,
        instance_data_index: usize,
    ) {
        while mesh_culling_data_buffer.len() < instance_data_index + 1 {
            mesh_culling_data_buffer.push(MeshCullingData::default());
        }
        mesh_culling_data_buffer.values_mut()[instance_data_index] = *self;
    }
}

/// A GPU buffer that holds the information needed to cull meshes on GPU.
///
/// At the moment, this simply holds each mesh's AABB.
///
/// To avoid wasting CPU time in the CPU culling case, this buffer will be empty
/// if GPU culling isn't in use.
#[derive(Resource, Deref, DerefMut)]
pub struct MeshCullingDataBuffer(RawBufferVec<MeshCullingData>);

impl Default for MeshCullingDataBuffer {
    #[inline]
    fn default() -> Self {
        Self(RawBufferVec::new(BufferUsages::STORAGE))
    }
}
/// Maps each mesh instance to the material ID, and allocated binding ID,
/// associated with that mesh instance.
#[derive(Resource, Default)]
pub struct RenderMeshMaterialIds {
    /// Maps the mesh instance to the material ID.
    mesh_to_material: MainEntityHashMap<UntypedAssetId>,
}

impl RenderMeshMaterialIds {
    /// Returns the mesh material ID for the entity with the given mesh, or a
    /// dummy mesh material ID if the mesh has no material ID.
    ///
    /// Meshes almost always have materials, but in very specific circumstances
    /// involving custom pipelines they won't. (See the
    /// `specialized_mesh_pipelines` example.)
    pub(crate) fn mesh_material(&self, entity: MainEntity) -> UntypedAssetId {
        self.mesh_to_material
            .get(&entity)
            .cloned()
            // All `Asset`s have the same invalid `AssetId`, so
            // any `Asset` can be used here
            .unwrap_or(AssetId::<Mesh>::invalid().into())
    }

    pub(crate) fn insert(&mut self, mesh_entity: MainEntity, material_id: UntypedAssetId) {
        self.mesh_to_material.insert(mesh_entity, material_id);
    }

    pub(crate) fn remove(&mut self, main_entity: MainEntity) {
        self.mesh_to_material.remove(&main_entity);
    }
}

#[derive(Component)]
pub struct MeshViewBindGroup {
    pub value: BindGroup,
}

/// Packs the lightmap UV rect into 64 bits (4 16-bit unsigned integers).
pub(crate) fn pack_lightmap_uv_rect(maybe_rect: Option<Rect>) -> UVec2 {
    match maybe_rect {
        Some(rect) => {
            let rect_uvec4 = (vec4(rect.min.x, rect.min.y, rect.max.x, rect.max.y) * 65535.0)
                .round()
                .as_uvec4();
            uvec2(
                rect_uvec4.x | (rect_uvec4.y << 16),
                rect_uvec4.z | (rect_uvec4.w << 16),
            )
        }
        None => UVec2::ZERO,
    }
}

// Ensure that we didn't overflow the number of bits available in `MeshPipelineKey`.
const_assert_eq!(
    (((MeshPipelineKey::LAST_FLAG.bits() << 1) - 1) | MeshPipelineKey::ALL_RESERVED_BITS.bits())
        & BaseMeshPipelineKey::all().bits(),
    0
);

// Ensure that the reserved bits don't overlap with the topology bits
const_assert_eq!(
    (BaseMeshPipelineKey::PRIMITIVE_TOPOLOGY_MASK_BITS
        << BaseMeshPipelineKey::PRIMITIVE_TOPOLOGY_SHIFT_BITS)
        & MeshPipelineKey::ALL_RESERVED_BITS.bits(),
    0
);

#[cfg(test)]
mod tests {
    use super::MeshPipelineKey;

    #[test]
    fn mesh_key_msaa_samples() {
        for i in [1, 2, 4, 8, 16, 32, 64, 128] {
            assert_eq!(MeshPipelineKey::from_msaa_samples(i).msaa_samples(), i);
        }
    }
}

/// Individual layout entries.
mod layout_entry {
    use core::num::NonZeroU32;

    use bevy_render::{
        render_resource::{
            binding_types::{
                sampler, storage_buffer_read_only_sized, texture_2d, texture_3d,
                uniform_buffer_sized,
            },
            BindGroupLayoutEntryBuilder, BufferSize, GpuArrayBuffer, SamplerBindingType,
            ShaderStages, TextureSampleType,
        },
        renderer::RenderDevice,
    };

    use crate::{
        lightmap::lightmap::LIGHTMAPS_PER_SLAB, skin::uniforms::skins_use_uniform_buffers,
    };

    use super::{MeshUniform, JOINT_BUFFER_SIZE, MORPH_BUFFER_SIZE};

    pub(super) fn model(render_device: &RenderDevice) -> BindGroupLayoutEntryBuilder {
        GpuArrayBuffer::<MeshUniform>::binding_layout(render_device)
            .visibility(ShaderStages::VERTEX_FRAGMENT)
    }

    pub(super) fn skinning(render_device: &RenderDevice) -> BindGroupLayoutEntryBuilder {
        // If we can use storage buffers, do so. Otherwise, fall back to uniform
        // buffers.
        let size = BufferSize::new(JOINT_BUFFER_SIZE as u64);
        if skins_use_uniform_buffers(render_device) {
            uniform_buffer_sized(true, size)
        } else {
            storage_buffer_read_only_sized(false, size)
        }
    }

    pub(super) fn weights() -> BindGroupLayoutEntryBuilder {
        uniform_buffer_sized(true, BufferSize::new(MORPH_BUFFER_SIZE as u64))
    }

    pub(super) fn targets() -> BindGroupLayoutEntryBuilder {
        texture_3d(TextureSampleType::Float { filterable: false })
    }

    pub(super) fn lightmaps_texture_view() -> BindGroupLayoutEntryBuilder {
        texture_2d(TextureSampleType::Float { filterable: true }).visibility(ShaderStages::FRAGMENT)
    }

    pub(super) fn lightmaps_sampler() -> BindGroupLayoutEntryBuilder {
        sampler(SamplerBindingType::Filtering).visibility(ShaderStages::FRAGMENT)
    }

    pub(super) fn lightmaps_texture_view_array() -> BindGroupLayoutEntryBuilder {
        texture_2d(TextureSampleType::Float { filterable: true })
            .visibility(ShaderStages::FRAGMENT)
            .count(NonZeroU32::new(LIGHTMAPS_PER_SLAB as u32).unwrap())
    }

    pub(super) fn lightmaps_sampler_array() -> BindGroupLayoutEntryBuilder {
        sampler(SamplerBindingType::Filtering)
            .visibility(ShaderStages::FRAGMENT)
            .count(NonZeroU32::new(LIGHTMAPS_PER_SLAB as u32).unwrap())
    }
}

/// Individual [`BindGroupEntry`]
/// for bind groups.
mod entry {
    use bevy_render::{
        render_resource::{
            BindGroupEntry, BindingResource, Buffer, BufferBinding, BufferSize, Sampler,
            TextureView, WgpuSampler, WgpuTextureView,
        },
        renderer::RenderDevice,
    };

    use crate::skin::uniforms::skins_use_uniform_buffers;

    use super::{JOINT_BUFFER_SIZE, MORPH_BUFFER_SIZE};

    fn entry(binding: u32, size: Option<u64>, buffer: &Buffer) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Buffer(BufferBinding {
                buffer,
                offset: 0,
                size: size.map(|size| BufferSize::new(size).unwrap()),
            }),
        }
    }

    pub(super) fn model(binding: u32, resource: BindingResource) -> BindGroupEntry {
        BindGroupEntry { binding, resource }
    }

    pub(super) fn skinning<'a>(
        render_device: &RenderDevice,
        binding: u32,
        buffer: &'a Buffer,
    ) -> BindGroupEntry<'a> {
        let size = if skins_use_uniform_buffers(render_device) {
            Some(JOINT_BUFFER_SIZE as u64)
        } else {
            None
        };
        entry(binding, size, buffer)
    }

    pub(super) fn weights(binding: u32, buffer: &Buffer) -> BindGroupEntry {
        entry(binding, Some(MORPH_BUFFER_SIZE as u64), buffer)
    }

    pub(super) fn targets(binding: u32, texture: &TextureView) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::TextureView(texture),
        }
    }

    pub(super) fn lightmaps_texture_view(binding: u32, texture: &TextureView) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::TextureView(texture),
        }
    }

    pub(super) fn lightmaps_sampler(binding: u32, sampler: &Sampler) -> BindGroupEntry {
        BindGroupEntry {
            binding,
            resource: BindingResource::Sampler(sampler),
        }
    }

    pub(super) fn lightmaps_texture_view_array<'a>(
        binding: u32,
        textures: &'a [&'a WgpuTextureView],
    ) -> BindGroupEntry<'a> {
        BindGroupEntry {
            binding,
            resource: BindingResource::TextureViewArray(textures),
        }
    }

    pub(super) fn lightmaps_sampler_array<'a>(
        binding: u32,
        samplers: &'a [&'a WgpuSampler],
    ) -> BindGroupEntry<'a> {
        BindGroupEntry {
            binding,
            resource: BindingResource::SamplerArray(samplers),
        }
    }
}
