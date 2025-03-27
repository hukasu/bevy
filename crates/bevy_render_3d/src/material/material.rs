use core::{hash::Hash, marker::PhantomData};

use bevy_asset::{AssetId, AssetServer, Handle, UntypedAssetId};
use bevy_core_pipeline::{
    core_3d::{AlphaMask3d, Opaque3d, Transmissive3d, Transparent3d},
    deferred::{AlphaMask3dDeferred, Opaque3dDeferred},
    prepass::{AlphaMask3dPrepass, Opaque3dPrepass},
};
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::{Component, Tick},
    resource::Resource,
    system::{
        lifetimeless::{SRes, SResMut},
        SystemParamItem,
    },
    world::{FromWorld, World},
};
use bevy_mesh::MeshVertexBufferLayoutRef;
use bevy_platform_support::collections::{hash_map::Entry, HashMap};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::{
    alpha::AlphaMode,
    render_asset::{PrepareAssetError, RenderAsset},
    render_phase::{DrawFunctionId, DrawFunctions},
    render_resource::{
        AsBindGroupError, BindGroup, BindGroupId, BindGroupLayout, BindingNumber, BufferUsages,
        CachedRenderPipelineId, PreparedBindGroup, RawBufferVec, RenderPipelineDescriptor, Shader,
        ShaderRef, SpecializedMeshPipeline, SpecializedMeshPipelineError, UnpreparedBindGroup,
    },
    renderer::{RenderDevice, RenderQueue},
    sync_world::MainEntityHashMap,
    texture::FallbackImage,
    view::{Msaa, RetainedViewEntity},
};
use bytemuck::Pod;

use crate::{
    mesh_pipeline::render::pipeline::{MeshPipeline, MeshPipelineKey},
    prepass::commands::DrawPrepass,
    render_method::{DefaultOpaqueRendererMethod, OpaqueRendererMethod},
};

use super::{
    bindless::{
        FallbackBindlessResources, MaterialBindGroupBindlessAllocator, MaterialBindlessSlab,
    },
    commands::DrawMaterial,
    material_uses_bindless_resources,
    non_bindless::{MaterialBindGroupNonBindlessAllocator, MaterialNonBindlessSlab},
    Material,
};

/// Render pipeline data for a given [`Material`].
#[derive(Resource)]
pub struct MaterialPipeline<M: Material> {
    pub mesh_pipeline: MeshPipeline,
    pub material_layout: BindGroupLayout,
    pub vertex_shader: Option<Handle<Shader>>,
    pub fragment_shader: Option<Handle<Shader>>,
    /// Whether this material *actually* uses bindless resources, taking the
    /// platform support (or lack thereof) of bindless resources into account.
    pub bindless: bool,
    pub marker: PhantomData<M>,
}

impl<M: Material> Clone for MaterialPipeline<M> {
    fn clone(&self) -> Self {
        Self {
            mesh_pipeline: self.mesh_pipeline.clone(),
            material_layout: self.material_layout.clone(),
            vertex_shader: self.vertex_shader.clone(),
            fragment_shader: self.fragment_shader.clone(),
            bindless: self.bindless,
            marker: PhantomData,
        }
    }
}

impl<M: Material> SpecializedMeshPipeline for MaterialPipeline<M>
where
    M::Data: PartialEq + Eq + Hash + Clone,
{
    type Key = MaterialPipelineKey<M>;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key.mesh_key, layout)?;
        if let Some(vertex_shader) = &self.vertex_shader {
            descriptor.vertex.shader = vertex_shader.clone();
        }

        if let Some(fragment_shader) = &self.fragment_shader {
            descriptor.fragment.as_mut().unwrap().shader = fragment_shader.clone();
        }

        descriptor.layout.insert(2, self.material_layout.clone());

        M::specialize(self, &mut descriptor, layout, key)?;

        // If bindless mode is on, add a `BINDLESS` define.
        if self.bindless {
            descriptor.vertex.shader_defs.push("BINDLESS".into());
            if let Some(ref mut fragment) = descriptor.fragment {
                fragment.shader_defs.push("BINDLESS".into());
            }
        }

        Ok(descriptor)
    }
}

impl<M: Material> FromWorld for MaterialPipeline<M> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let render_device = world.resource::<RenderDevice>();

        MaterialPipeline {
            mesh_pipeline: world.resource::<MeshPipeline>().clone(),
            material_layout: M::bind_group_layout(render_device),
            vertex_shader: match M::vertex_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            fragment_shader: match M::fragment_shader() {
                ShaderRef::Default => None,
                ShaderRef::Handle(handle) => Some(handle),
                ShaderRef::Path(path) => Some(asset_server.load(path)),
            },
            bindless: material_uses_bindless_resources::<M>(render_device),
            marker: PhantomData,
        }
    }
}

impl<M: Material> Eq for MaterialPipelineKey<M> where M::Data: PartialEq {}

impl<M: Material> PartialEq for MaterialPipelineKey<M>
where
    M::Data: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.mesh_key == other.mesh_key && self.bind_group_data == other.bind_group_data
    }
}

impl<M: Material> Clone for MaterialPipelineKey<M>
where
    M::Data: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mesh_key: self.mesh_key,
            bind_group_data: self.bind_group_data.clone(),
        }
    }
}

impl<M: Material> Hash for MaterialPipelineKey<M>
where
    M::Data: Hash,
{
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.mesh_key.hash(state);
        self.bind_group_data.hash(state);
    }
}

/// Data prepared for a [`Material`] instance.
pub struct PreparedMaterial<M: Material> {
    pub binding: MaterialBindingId,
    pub properties: MaterialProperties,
    pub phantom: PhantomData<M>,
}

impl<M: Material> RenderAsset for PreparedMaterial<M> {
    type SourceAsset = M;

    type Param = (
        SRes<RenderDevice>,
        SRes<MaterialPipeline<M>>,
        SRes<DefaultOpaqueRendererMethod>,
        SResMut<MaterialBindGroupAllocator<M>>,
        SResMut<RenderMaterialBindings>,
        SRes<DrawFunctions<Opaque3d>>,
        SRes<DrawFunctions<AlphaMask3d>>,
        SRes<DrawFunctions<Transmissive3d>>,
        SRes<DrawFunctions<Transparent3d>>,
        SRes<DrawFunctions<Opaque3dPrepass>>,
        SRes<DrawFunctions<AlphaMask3dPrepass>>,
        SRes<DrawFunctions<Opaque3dDeferred>>,
        SRes<DrawFunctions<AlphaMask3dDeferred>>,
        M::Param,
    );

    fn prepare_asset(
        material: Self::SourceAsset,
        material_id: AssetId<Self::SourceAsset>,
        (
            render_device,
            pipeline,
            default_opaque_render_method,
            bind_group_allocator,
            render_material_bindings,
            opaque_draw_functions,
            alpha_mask_draw_functions,
            transmissive_draw_functions,
            transparent_draw_functions,
            opaque_prepass_draw_functions,
            alpha_mask_prepass_draw_functions,
            opaque_deferred_draw_functions,
            alpha_mask_deferred_draw_functions,
            material_param,
        ): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let draw_opaque_pbr = opaque_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_alpha_mask_pbr = alpha_mask_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_transmissive_pbr = transmissive_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_transparent_pbr = transparent_draw_functions.read().id::<DrawMaterial<M>>();
        let draw_opaque_prepass = opaque_prepass_draw_functions
            .read()
            .get_id::<DrawPrepass<M>>();
        let draw_alpha_mask_prepass = alpha_mask_prepass_draw_functions
            .read()
            .get_id::<DrawPrepass<M>>();
        let draw_opaque_deferred = opaque_deferred_draw_functions
            .read()
            .get_id::<DrawPrepass<M>>();
        let draw_alpha_mask_deferred = alpha_mask_deferred_draw_functions
            .read()
            .get_id::<DrawPrepass<M>>();

        let render_method = match material.opaque_render_method() {
            OpaqueRendererMethod::Forward => OpaqueRendererMethod::Forward,
            OpaqueRendererMethod::Deferred => OpaqueRendererMethod::Deferred,
            OpaqueRendererMethod::Auto => ***default_opaque_render_method,
        };

        let mut mesh_pipeline_key_bits = MeshPipelineKey::empty();
        mesh_pipeline_key_bits.set(
            MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE,
            material.reads_view_transmission_texture(),
        );

        let reads_view_transmission_texture =
            mesh_pipeline_key_bits.contains(MeshPipelineKey::READS_VIEW_TRANSMISSION_TEXTURE);

        let render_phase_type = match material.alpha_mode() {
            AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Add | AlphaMode::Multiply => {
                RenderPhaseType::Transparent
            }
            _ if reads_view_transmission_texture => RenderPhaseType::Transmissive,
            AlphaMode::Opaque | AlphaMode::AlphaToCoverage => RenderPhaseType::Opaque,
            AlphaMode::Mask(_) => RenderPhaseType::AlphaMask,
        };

        let draw_function_id = match render_phase_type {
            RenderPhaseType::Opaque => draw_opaque_pbr,
            RenderPhaseType::AlphaMask => draw_alpha_mask_pbr,
            RenderPhaseType::Transmissive => draw_transmissive_pbr,
            RenderPhaseType::Transparent => draw_transparent_pbr,
        };
        let prepass_draw_function_id = match render_phase_type {
            RenderPhaseType::Opaque => draw_opaque_prepass,
            RenderPhaseType::AlphaMask => draw_alpha_mask_prepass,
            _ => None,
        };
        let deferred_draw_function_id = match render_phase_type {
            RenderPhaseType::Opaque => draw_opaque_deferred,
            RenderPhaseType::AlphaMask => draw_alpha_mask_deferred,
            _ => None,
        };

        match material.unprepared_bind_group(
            &pipeline.material_layout,
            render_device,
            material_param,
            false,
        ) {
            Ok(unprepared) => {
                // Allocate or update the material.
                let binding = match render_material_bindings.entry(material_id.into()) {
                    Entry::Occupied(mut occupied_entry) => {
                        // TODO: Have a fast path that doesn't require
                        // recreating the bind group if only buffer contents
                        // change. For now, we just delete and recreate the bind
                        // group.
                        bind_group_allocator.free(*occupied_entry.get());
                        let new_binding = bind_group_allocator
                            .allocate_unprepared(unprepared, &pipeline.material_layout);
                        *occupied_entry.get_mut() = new_binding;
                        new_binding
                    }
                    Entry::Vacant(vacant_entry) => *vacant_entry.insert(
                        bind_group_allocator
                            .allocate_unprepared(unprepared, &pipeline.material_layout),
                    ),
                };

                Ok(PreparedMaterial {
                    binding,
                    properties: MaterialProperties {
                        alpha_mode: material.alpha_mode(),
                        depth_bias: material.depth_bias(),
                        reads_view_transmission_texture,
                        render_phase_type,
                        draw_function_id,
                        prepass_draw_function_id,
                        render_method,
                        mesh_pipeline_key_bits,
                        deferred_draw_function_id,
                    },
                    phantom: PhantomData,
                })
            }

            Err(AsBindGroupError::RetryNextUpdate) => {
                Err(PrepareAssetError::RetryNextUpdate(material))
            }

            Err(AsBindGroupError::CreateBindGroupDirectly) => {
                // This material has opted out of automatic bind group creation
                // and is requesting a fully-custom bind group. Invoke
                // `as_bind_group` as requested, and store the resulting bind
                // group in the slot.
                match material.as_bind_group(
                    &pipeline.material_layout,
                    render_device,
                    material_param,
                ) {
                    Ok(prepared_bind_group) => {
                        // Store the resulting bind group directly in the slot.
                        let material_binding_id =
                            bind_group_allocator.allocate_prepared(prepared_bind_group);
                        render_material_bindings.insert(material_id.into(), material_binding_id);

                        Ok(PreparedMaterial {
                            binding: material_binding_id,
                            properties: MaterialProperties {
                                alpha_mode: material.alpha_mode(),
                                depth_bias: material.depth_bias(),
                                reads_view_transmission_texture,
                                render_phase_type,
                                draw_function_id,
                                prepass_draw_function_id,
                                render_method,
                                mesh_pipeline_key_bits,
                                deferred_draw_function_id,
                            },
                            phantom: PhantomData,
                        })
                    }

                    Err(AsBindGroupError::RetryNextUpdate) => {
                        Err(PrepareAssetError::RetryNextUpdate(material))
                    }

                    Err(other) => Err(PrepareAssetError::AsBindGroupError(other)),
                }
            }

            Err(other) => Err(PrepareAssetError::AsBindGroupError(other)),
        }
    }

    fn unload_asset(
        source_asset: AssetId<Self::SourceAsset>,
        (_, _, _, bind_group_allocator, render_material_bindings, ..): &mut SystemParamItem<
            Self::Param,
        >,
    ) {
        let Some(material_binding_id) = render_material_bindings.remove(&source_asset.untyped())
        else {
            return;
        };
        bind_group_allocator.free(material_binding_id);
    }
}

/// Common [`Material`] properties, calculated for a specific material instance.
pub struct MaterialProperties {
    /// Is this material should be rendered by the deferred renderer when.
    /// [`AlphaMode::Opaque`] or [`AlphaMode::Mask`]
    pub render_method: OpaqueRendererMethod,
    /// The [`AlphaMode`] of this material.
    pub alpha_mode: AlphaMode,
    /// The bits in the [`MeshPipelineKey`] for this material.
    ///
    /// These are precalculated so that we can just "or" them together in
    /// [`queue_material_meshes`].
    pub mesh_pipeline_key_bits: MeshPipelineKey,
    /// Add a bias to the view depth of the mesh which can be used to force a specific render order
    /// for meshes with equal depth, to avoid z-fighting.
    /// The bias is in depth-texture units so large values may be needed to overcome small depth differences.
    pub depth_bias: f32,
    /// Whether the material would like to read from [`ViewTransmissionTexture`](bevy_core_pipeline::core_3d::ViewTransmissionTexture).
    ///
    /// This allows taking color output from the [`Opaque3d`] pass as an input, (for screen-space transmission) but requires
    /// rendering to take place in a separate [`Transmissive3d`] pass.
    pub reads_view_transmission_texture: bool,
    pub render_phase_type: RenderPhaseType,
    pub draw_function_id: DrawFunctionId,
    pub prepass_draw_function_id: Option<DrawFunctionId>,
    pub deferred_draw_function_id: Option<DrawFunctionId>,
}

#[derive(Clone, Copy)]
pub enum RenderPhaseType {
    Opaque,
    AlphaMask,
    Transmissive,
    Transparent,
}

/// Stores all extracted instances of a [`Material`] in the render world.
#[derive(Resource, Deref, DerefMut)]
pub struct RenderMaterialInstances<M: Material>(pub MainEntityHashMap<AssetId<M>>);

impl<M: Material> Default for RenderMaterialInstances<M> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Component, Clone, Copy, Default, PartialEq, Eq, Deref, DerefMut)]
pub struct MaterialBindGroupId(pub Option<BindGroupId>);

impl MaterialBindGroupId {
    pub fn new(id: BindGroupId) -> Self {
        Self(Some(id))
    }
}

impl From<BindGroup> for MaterialBindGroupId {
    fn from(value: BindGroup) -> Self {
        Self::new(value.id())
    }
}

/// A resource that places materials into bind groups and tracks their
/// resources.
///
/// Internally, Bevy has separate allocators for bindless and non-bindless
/// materials. This resource provides a common interface to the specific
/// allocator in use.
#[derive(Resource)]
pub enum MaterialBindGroupAllocator<M>
where
    M: Material,
{
    /// The allocator used when the material is bindless.
    Bindless(Box<MaterialBindGroupBindlessAllocator<M>>),
    /// The allocator used when the material is non-bindless.
    NonBindless(Box<MaterialBindGroupNonBindlessAllocator<M>>),
}

impl<M> MaterialBindGroupAllocator<M>
where
    M: Material,
{
    /// Creates a new [`MaterialBindGroupAllocator`] managing the data for a
    /// single material.
    fn new(render_device: &RenderDevice) -> MaterialBindGroupAllocator<M> {
        if material_uses_bindless_resources::<M>(render_device) {
            MaterialBindGroupAllocator::Bindless(Box::new(MaterialBindGroupBindlessAllocator::new(
                render_device,
            )))
        } else {
            MaterialBindGroupAllocator::NonBindless(Box::new(
                MaterialBindGroupNonBindlessAllocator::new(),
            ))
        }
    }

    /// Returns the slab with the given index, if one exists.
    pub fn get(&self, group: MaterialBindGroupIndex) -> Option<MaterialSlab<M>> {
        match *self {
            MaterialBindGroupAllocator::Bindless(ref bindless_allocator) => bindless_allocator
                .get(group)
                .map(|bindless_slab| MaterialSlab(MaterialSlabImpl::Bindless(bindless_slab))),
            MaterialBindGroupAllocator::NonBindless(ref non_bindless_allocator) => {
                non_bindless_allocator.get(group).map(|non_bindless_slab| {
                    MaterialSlab(MaterialSlabImpl::NonBindless(non_bindless_slab))
                })
            }
        }
    }

    /// Allocates an [`UnpreparedBindGroup`] and returns the resulting binding ID.
    ///
    /// This method should generally be preferred over
    /// [`Self::allocate_prepared`], because this method supports both bindless
    /// and non-bindless bind groups. Only use [`Self::allocate_prepared`] if
    /// you need to prepare the bind group yourself.
    pub fn allocate_unprepared(
        &mut self,
        unprepared_bind_group: UnpreparedBindGroup<M::Data>,
        bind_group_layout: &BindGroupLayout,
    ) -> MaterialBindingId {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.allocate_unprepared(unprepared_bind_group),
            MaterialBindGroupAllocator::NonBindless(
                ref mut material_bind_group_non_bindless_allocator,
            ) => material_bind_group_non_bindless_allocator
                .allocate_unprepared(unprepared_bind_group, (*bind_group_layout).clone()),
        }
    }

    /// Places a pre-prepared bind group into a slab.
    ///
    /// For bindless materials, the allocator internally manages the bind
    /// groups, so calling this method will panic if this is a bindless
    /// allocator. Only non-bindless allocators support this method.
    ///
    /// It's generally preferred to use [`Self::allocate_unprepared`], because
    /// that method supports both bindless and non-bindless allocators. Only use
    /// this method if you need to prepare the bind group yourself.
    pub fn allocate_prepared(
        &mut self,
        prepared_bind_group: PreparedBindGroup<M::Data>,
    ) -> MaterialBindingId {
        match *self {
            MaterialBindGroupAllocator::Bindless(_) => {
                panic!(
                    "Bindless resources are incompatible with implementing `as_bind_group` \
                     directly; implement `unprepared_bind_group` instead or disable bindless"
                )
            }
            MaterialBindGroupAllocator::NonBindless(ref mut non_bindless_allocator) => {
                non_bindless_allocator.allocate_prepared(prepared_bind_group)
            }
        }
    }

    /// Deallocates the material with the given binding ID.
    ///
    /// Any resources that are no longer referenced are removed from the slab.
    pub fn free(&mut self, material_binding_id: MaterialBindingId) {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.free(material_binding_id),
            MaterialBindGroupAllocator::NonBindless(
                ref mut material_bind_group_non_bindless_allocator,
            ) => material_bind_group_non_bindless_allocator.free(material_binding_id),
        }
    }

    /// Recreates any bind groups corresponding to slabs that have been modified
    /// since last calling [`MaterialBindGroupAllocator::prepare_bind_groups`].
    pub fn prepare_bind_groups(
        &mut self,
        render_device: &RenderDevice,
        fallback_bindless_resources: &FallbackBindlessResources,
        fallback_image: &FallbackImage,
    ) {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.prepare_bind_groups(
                render_device,
                fallback_bindless_resources,
                fallback_image,
            ),
            MaterialBindGroupAllocator::NonBindless(
                ref mut material_bind_group_non_bindless_allocator,
            ) => material_bind_group_non_bindless_allocator.prepare_bind_groups(render_device),
        }
    }

    /// Uploads the contents of all buffers that this
    /// [`MaterialBindGroupAllocator`] manages to the GPU.
    ///
    /// Non-bindless allocators don't currently manage any buffers, so this
    /// method only has an effect for bindless allocators.
    pub fn write_buffers(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        match *self {
            MaterialBindGroupAllocator::Bindless(
                ref mut material_bind_group_bindless_allocator,
            ) => material_bind_group_bindless_allocator.write_buffers(render_device, render_queue),
            MaterialBindGroupAllocator::NonBindless(_) => {
                // Not applicable.
            }
        }
    }
}

impl<M> FromWorld for MaterialBindGroupAllocator<M>
where
    M: Material,
{
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        MaterialBindGroupAllocator::new(render_device)
    }
}

/// The location of a material (either bindless or non-bindless) within the
/// slabs.
#[derive(Clone, Copy, Debug, Default, Reflect)]
#[reflect(Clone, Default)]
pub struct MaterialBindingId {
    /// The index of the bind group (slab) where the GPU data is located.
    pub group: MaterialBindGroupIndex,
    /// The slot within that bind group.
    ///
    /// Non-bindless materials will always have a slot of 0.
    pub slot: MaterialBindGroupSlot,
}

/// The index of each material bind group.
///
/// In bindless mode, each bind group contains multiple materials. In
/// non-bindless mode, each bind group contains only one material.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Reflect, Deref, DerefMut)]
#[reflect(Default, Clone, PartialEq, Hash)]
pub struct MaterialBindGroupIndex(pub u32);

impl From<u32> for MaterialBindGroupIndex {
    fn from(value: u32) -> Self {
        MaterialBindGroupIndex(value)
    }
}

/// The index of the slot containing material data within each material bind
/// group.
///
/// In bindless mode, this slot is needed to locate the material data in each
/// bind group, since multiple materials are packed into a single slab. In
/// non-bindless mode, this slot is always 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Reflect, Deref, DerefMut)]
#[reflect(Default, Clone, PartialEq)]
pub struct MaterialBindGroupSlot(pub u32);

impl From<u32> for MaterialBindGroupSlot {
    fn from(value: u32) -> Self {
        MaterialBindGroupSlot(value)
    }
}

impl From<MaterialBindGroupSlot> for u32 {
    fn from(value: MaterialBindGroupSlot) -> Self {
        value.0
    }
}

/// A resource that maps each untyped material ID to its binding.
///
/// This duplicates information in `RenderAssets<M>`, but it doesn't have the
/// `M` type parameter, so it can be used in untyped contexts like
/// [`crate::render::mesh::collect_meshes_for_gpu_building`].
#[derive(Resource, Default, Deref, DerefMut)]
pub struct RenderMaterialBindings(HashMap<UntypedAssetId, MaterialBindingId>);

/// The public interface to a slab, which represents a single bind group.
pub struct MaterialSlab<'a, M>(pub MaterialSlabImpl<'a, M>)
where
    M: Material;

impl<'a, M> MaterialSlab<'a, M>
where
    M: Material,
{
    /// Returns the extra data associated with this material.
    ///
    /// When deriving `AsBindGroup`, this data is given by the
    /// `#[bind_group_data(DataType)]` attribute on the material structure.
    pub fn get_extra_data(&self, slot: MaterialBindGroupSlot) -> &M::Data {
        match self.0 {
            MaterialSlabImpl::Bindless(material_bindless_slab) => {
                material_bindless_slab.get_extra_data(slot)
            }
            MaterialSlabImpl::NonBindless(MaterialNonBindlessSlab::Prepared(
                prepared_bind_group,
            )) => &prepared_bind_group.data,
            MaterialSlabImpl::NonBindless(MaterialNonBindlessSlab::Unprepared(
                unprepared_bind_group,
            )) => &unprepared_bind_group.data,
        }
    }

    /// Returns the [`BindGroup`] corresponding to this slab, if it's been
    /// prepared.
    ///
    /// You can prepare bind groups by calling
    /// [`MaterialBindGroupAllocator::prepare_bind_groups`]. If the bind group
    /// isn't ready, this method returns `None`.
    pub fn bind_group(&self) -> Option<&'a BindGroup> {
        match self.0 {
            MaterialSlabImpl::Bindless(material_bindless_slab) => {
                material_bindless_slab.bind_group()
            }
            MaterialSlabImpl::NonBindless(MaterialNonBindlessSlab::Prepared(
                prepared_bind_group,
            )) => Some(&prepared_bind_group.bind_group),
            MaterialSlabImpl::NonBindless(MaterialNonBindlessSlab::Unprepared(_)) => None,
        }
    }
}

/// The actual implementation of a material slab.
///
/// This has bindless and non-bindless variants.
pub enum MaterialSlabImpl<'a, M>
where
    M: Material,
{
    /// The implementation of the slab interface we use when the slab
    /// is bindless.
    Bindless(&'a MaterialBindlessSlab<M>),
    /// The implementation of the slab interface we use when the slab
    /// is non-bindless.
    NonBindless(MaterialNonBindlessSlab<'a, M>),
}

/// Manages an array of untyped plain old data on GPU and allocates individual
/// slots within that array.
///
/// This supports the `#[data]` attribute of `AsBindGroup`.
pub struct MaterialDataBuffer {
    /// The number of the binding that we attach this storage buffer to.
    binding_number: BindingNumber,
    /// The actual data.
    ///
    /// Note that this is untyped (`u8`); the actual aligned size of each
    /// element is given by [`Self::aligned_element_size`];
    buffer: RetainedRawBufferVec<u8>,
    /// The size of each element in the buffer, including padding and alignment
    /// if any.
    aligned_element_size: u32,
    /// A list of free slots within the buffer.
    free_slots: Vec<u32>,
    /// The actual number of slots that have been allocated.
    len: u32,
}

impl MaterialDataBuffer {
    /// Creates a new [`MaterialDataBuffer`] managing a buffer of elements of
    /// size `aligned_element_size` that will be bound to the given binding
    /// number.
    pub fn new(binding_number: BindingNumber, aligned_element_size: u32) -> MaterialDataBuffer {
        MaterialDataBuffer {
            binding_number,
            buffer: RetainedRawBufferVec::new(BufferUsages::STORAGE),
            aligned_element_size,
            free_slots: vec![],
            len: 0,
        }
    }

    /// Allocates a slot for a new piece of data, copies the data into that
    /// slot, and returns the slot ID.
    ///
    /// The size of the piece of data supplied to this method must equal the
    /// [`Self::aligned_element_size`] provided to [`MaterialDataBuffer::new`].
    pub fn insert(&mut self, data: &[u8]) -> u32 {
        // Make the the data is of the right length.
        debug_assert_eq!(data.len(), self.aligned_element_size as usize);

        // Grab a slot.
        let slot = self.free_slots.pop().unwrap_or(self.len);

        // Calculate the range we're going to copy to.
        let start = slot as usize * self.aligned_element_size as usize;
        let end = (slot as usize + 1) * self.aligned_element_size as usize;

        // Resize the buffer if necessary.
        if self.buffer.len() < end {
            self.buffer.reserve_internal(end);
        }
        while self.buffer.values().len() < end {
            self.buffer.push(0);
        }

        // Copy in the data.
        self.buffer.values_mut()[start..end].copy_from_slice(data);

        // Mark the buffer dirty, and finish up.
        self.len += 1;
        self.buffer.dirty = BufferDirtyState::NeedsReserve;
        slot
    }

    /// Marks the given slot as free.
    pub fn remove(&mut self, slot: u32) {
        self.free_slots.push(slot);
        self.len -= 1;
    }

    pub fn binding_number(&self) -> BindingNumber {
        self.binding_number
    }

    pub fn buffer(&self) -> &RetainedRawBufferVec<u8> {
        &self.buffer
    }

    pub fn buffer_mut(&mut self) -> &mut RetainedRawBufferVec<u8> {
        &mut self.buffer
    }
}

/// A buffer containing plain old data, already packed into the appropriate GPU
/// format, and that can be updated incrementally.
///
/// This structure exists in order to encapsulate the lazy update
/// ([`BufferDirtyState`]) logic in a single place.
#[derive(Deref, DerefMut)]
pub struct RetainedRawBufferVec<T>
where
    T: Pod,
{
    /// The contents of the buffer.
    #[deref]
    buffer: RawBufferVec<T>,
    /// Whether the contents of the buffer have been uploaded to the GPU.
    pub dirty: BufferDirtyState,
}

impl<T> RetainedRawBufferVec<T>
where
    T: Pod,
{
    /// Creates a new empty [`RetainedRawBufferVec`] supporting the given
    /// [`BufferUsages`].
    pub fn new(buffer_usages: BufferUsages) -> RetainedRawBufferVec<T> {
        RetainedRawBufferVec {
            buffer: RawBufferVec::new(buffer_usages),
            dirty: BufferDirtyState::NeedsUpload,
        }
    }

    /// Recreates the GPU backing buffer if needed.
    pub fn prepare(&mut self, render_device: &RenderDevice) {
        match self.dirty {
            BufferDirtyState::Clean | BufferDirtyState::NeedsUpload => {}
            BufferDirtyState::NeedsReserve => {
                let capacity = self.buffer.len();
                self.buffer.reserve(capacity, render_device);
                self.dirty = BufferDirtyState::NeedsUpload;
            }
        }
    }

    /// Writes the current contents of the buffer to the GPU if necessary.
    pub fn write(&mut self, render_device: &RenderDevice, render_queue: &RenderQueue) {
        match self.dirty {
            BufferDirtyState::Clean => {}
            BufferDirtyState::NeedsReserve | BufferDirtyState::NeedsUpload => {
                self.buffer.write_buffer(render_device, render_queue);
                self.dirty = BufferDirtyState::Clean;
            }
        }
    }
}

/// Stores the [`SpecializedMaterialViewPipelineCache`] for each view.
#[derive(Resource, Deref, DerefMut)]
pub struct SpecializedMaterialPipelineCache<M> {
    // view entity -> view pipeline cache
    #[deref]
    map: HashMap<RetainedViewEntity, SpecializedMaterialViewPipelineCache<M>>,
    marker: PhantomData<M>,
}

impl<M> Default for SpecializedMaterialPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            marker: PhantomData,
        }
    }
}

/// Stores the cached render pipeline ID for each entity in a single view, as
/// well as the last time it was changed.
#[derive(Deref, DerefMut)]
pub struct SpecializedMaterialViewPipelineCache<M> {
    // material entity -> (tick, pipeline_id)
    #[deref]
    map: MainEntityHashMap<(Tick, CachedRenderPipelineId)>,
    marker: PhantomData<M>,
}

impl<M> Default for SpecializedMaterialViewPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: MainEntityHashMap::default(),
            marker: PhantomData,
        }
    }
}

/// A key uniquely identifying a specialized [`MaterialPipeline`].
pub struct MaterialPipelineKey<M: Material> {
    pub mesh_key: MeshPipelineKey,
    pub bind_group_data: M::Data,
}

/// The CPU/GPU synchronization state of a buffer that we maintain.
///
/// Currently, the only buffer that we maintain is the
/// [`MaterialBindlessIndexTable`].
pub enum BufferDirtyState {
    /// The buffer is currently synchronized between the CPU and GPU.
    Clean,
    /// The buffer hasn't been created yet.
    NeedsReserve,
    /// The buffer exists on both CPU and GPU, but the GPU data is out of date.
    NeedsUpload,
}

pub const fn alpha_mode_pipeline_key(alpha_mode: AlphaMode, msaa: &Msaa) -> MeshPipelineKey {
    match alpha_mode {
        // Premultiplied and Add share the same pipeline key
        // They're made distinct in the PBR shader, via `premultiply_alpha()`
        AlphaMode::Premultiplied | AlphaMode::Add => MeshPipelineKey::BLEND_PREMULTIPLIED_ALPHA,
        AlphaMode::Blend => MeshPipelineKey::BLEND_ALPHA,
        AlphaMode::Multiply => MeshPipelineKey::BLEND_MULTIPLY,
        AlphaMode::Mask(_) => MeshPipelineKey::MAY_DISCARD,
        AlphaMode::AlphaToCoverage => match *msaa {
            Msaa::Off => MeshPipelineKey::MAY_DISCARD,
            _ => MeshPipelineKey::BLEND_ALPHA_TO_COVERAGE,
        },
        _ => MeshPipelineKey::NONE,
    }
}
