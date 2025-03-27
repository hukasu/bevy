use core::marker::PhantomData;

use alloc::string::String;

use bevy_asset::UntypedAssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::{Component, Tick},
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_platform_support::collections::HashMap;
use bevy_render::{
    mesh::allocator::SlabId,
    render_phase::{DrawFunctionId, PhaseItemBatchSetKey},
    render_resource::{
        AddressMode, CachedRenderPipelineId, CompareFunction, FilterMode, Sampler,
        SamplerDescriptor, Texture, TextureView,
    },
    renderer::RenderDevice,
    sync_world::MainEntityHashMap,
    texture::DepthAttachment,
    view::RetainedViewEntity,
};
use bevy_utils::prelude::default;

#[derive(Component)]
pub struct ShadowView {
    pub depth_attachment: DepthAttachment,
    pub pass_name: String,
}

#[derive(Resource, Clone)]
pub struct ShadowSamplers {
    pub point_light_comparison_sampler: Sampler,
    #[cfg(feature = "experimental_pbr_pcss")]
    pub point_light_linear_sampler: Sampler,
    pub directional_light_comparison_sampler: Sampler,
    #[cfg(feature = "experimental_pbr_pcss")]
    pub directional_light_linear_sampler: Sampler,
}

// TODO: this pattern for initializing the shaders / pipeline isn't ideal. this should be handled by the asset system
impl FromWorld for ShadowSamplers {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();

        let base_sampler_descriptor = SamplerDescriptor {
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: FilterMode::Nearest,
            ..default()
        };

        ShadowSamplers {
            point_light_comparison_sampler: render_device.create_sampler(&SamplerDescriptor {
                compare: Some(CompareFunction::GreaterEqual),
                ..base_sampler_descriptor
            }),
            #[cfg(feature = "experimental_pbr_pcss")]
            point_light_linear_sampler: render_device.create_sampler(&base_sampler_descriptor),
            directional_light_comparison_sampler: render_device.create_sampler(
                &SamplerDescriptor {
                    compare: Some(CompareFunction::GreaterEqual),
                    ..base_sampler_descriptor
                },
            ),
            #[cfg(feature = "experimental_pbr_pcss")]
            directional_light_linear_sampler: render_device
                .create_sampler(&base_sampler_descriptor),
        }
    }
}

#[derive(Component)]
pub struct ViewShadowBindings {
    pub point_light_depth_texture: Texture,
    pub point_light_depth_texture_view: TextureView,
    pub directional_light_depth_texture: Texture,
    pub directional_light_depth_texture_view: TextureView,
}

/// Information that must be identical in order to place opaque meshes in the
/// same *batch set*.
///
/// A batch set is a set of batches that can be multi-drawn together, if
/// multi-draw is in use.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShadowBatchSetKey {
    /// The identifier of the render pipeline.
    pub pipeline: CachedRenderPipelineId,

    /// The function used to draw.
    pub draw_function: DrawFunctionId,

    /// The ID of a bind group specific to the material.
    ///
    /// In the case of PBR, this is the `MaterialBindGroupIndex`.
    pub material_bind_group_index: Option<u32>,

    /// The ID of the slab of GPU memory that contains vertex data.
    ///
    /// For non-mesh items, you can fill this with 0 if your items can be
    /// multi-drawn, or with a unique value if they can't.
    pub vertex_slab: SlabId,

    /// The ID of the slab of GPU memory that contains index data, if present.
    ///
    /// For non-mesh items, you can safely fill this with `None`.
    pub index_slab: Option<SlabId>,
}

impl PhaseItemBatchSetKey for ShadowBatchSetKey {
    fn indexed(&self) -> bool {
        self.index_slab.is_some()
    }
}

/// Data used to bin each object in the shadow map phase.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShadowBinKey {
    /// The object.
    pub asset_id: UntypedAssetId,
}

#[derive(Resource, Deref, DerefMut)]
pub struct SpecializedShadowMaterialPipelineCache<M> {
    // view light entity -> view pipeline cache
    #[deref]
    map: HashMap<RetainedViewEntity, SpecializedShadowMaterialViewPipelineCache<M>>,
    marker: PhantomData<M>,
}

#[derive(Deref, DerefMut)]
pub struct SpecializedShadowMaterialViewPipelineCache<M> {
    #[deref]
    map: MainEntityHashMap<(Tick, CachedRenderPipelineId)>,
    marker: PhantomData<M>,
}

impl<M> Default for SpecializedShadowMaterialPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: HashMap::default(),
            marker: PhantomData,
        }
    }
}

impl<M> Default for SpecializedShadowMaterialViewPipelineCache<M> {
    fn default() -> Self {
        Self {
            map: MainEntityHashMap::default(),
            marker: PhantomData,
        }
    }
}
