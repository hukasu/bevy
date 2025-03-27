use bevy_asset::AssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    resource::Resource,
    world::{FromWorld, World},
};
use bevy_image::Image;
use bevy_math::Rect;
use bevy_platform_support::collections::HashSet;
use bevy_render::{
    render_resource::{Sampler, TextureView, WgpuSampler, WgpuTextureView},
    renderer::{RenderAdapter, RenderDevice},
    sync_world::MainEntityHashMap,
    texture::{FallbackImage, GpuImage},
};

use bevy_utils::default;
use fixedbitset::FixedBitSet;
use nonmax::{NonMaxU16, NonMaxU32};

use crate::binding_arrays_are_usable;

/// The number of lightmaps that we store in a single slab, if bindless textures
/// are in use.
///
/// If bindless textures aren't in use, then only a single lightmap can be bound
/// at a time.
pub const LIGHTMAPS_PER_SLAB: usize = 4;

/// Stores data for all lightmaps in the render world.
///
/// This is cleared and repopulated each frame during the `extract_lightmaps`
/// system.
#[derive(Resource)]
pub struct RenderLightmaps {
    /// The mapping from every lightmapped entity to its lightmap info.
    ///
    /// Entities without lightmaps, or for which the mesh or lightmap isn't
    /// loaded, won't have entries in this table.
    pub(crate) render_lightmaps: MainEntityHashMap<RenderLightmap>,

    /// The slabs (binding arrays) containing the lightmaps.
    pub(crate) slabs: Vec<LightmapSlab>,

    free_slabs: FixedBitSet,

    pub(crate) pending_lightmaps: HashSet<(LightmapSlabIndex, LightmapSlotIndex)>,

    /// Whether bindless textures are supported on this platform.
    pub(crate) bindless_supported: bool,
}

impl FromWorld for RenderLightmaps {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let render_adapter = world.resource::<RenderAdapter>();

        let bindless_supported = binding_arrays_are_usable(render_device, render_adapter);

        RenderLightmaps {
            render_lightmaps: default(),
            slabs: vec![],
            free_slabs: FixedBitSet::new(),
            pending_lightmaps: default(),
            bindless_supported,
        }
    }
}

impl RenderLightmaps {
    /// Creates a new slab, appends it to the end of the list, and returns its
    /// slab index.
    fn create_slab(&mut self, fallback_images: &FallbackImage) -> LightmapSlabIndex {
        let slab_index = LightmapSlabIndex::from(self.slabs.len());
        self.free_slabs.grow_and_insert(slab_index.into());
        self.slabs
            .push(LightmapSlab::new(fallback_images, self.bindless_supported));
        slab_index
    }

    pub fn allocate(
        &mut self,
        fallback_images: &FallbackImage,
        image_id: AssetId<Image>,
    ) -> (LightmapSlabIndex, LightmapSlotIndex) {
        let slab_index = match self.free_slabs.minimum() {
            None => self.create_slab(fallback_images),
            Some(slab_index) => slab_index.into(),
        };

        let slab = &mut self.slabs[usize::from(slab_index)];
        let slot_index = slab.allocate(image_id);
        if slab.is_full() {
            self.free_slabs.remove(slab_index.into());
        }

        (slab_index, slot_index)
    }

    pub fn remove(
        &mut self,
        fallback_images: &FallbackImage,
        slab_index: LightmapSlabIndex,
        slot_index: LightmapSlotIndex,
    ) {
        let slab = &mut self.slabs[usize::from(slab_index)];
        slab.remove(fallback_images, slot_index);

        if !slab.is_full() {
            self.free_slabs.grow_and_insert(slot_index.into());
        }
    }
}

/// Lightmap data stored in the render world.
///
/// There is one of these per visible lightmapped mesh instance.
#[derive(Debug)]
pub(crate) struct RenderLightmap {
    /// The rectangle within the lightmap texture that the UVs are relative to.
    ///
    /// The top left coordinate is the `min` part of the rect, and the bottom
    /// right coordinate is the `max` part of the rect. The rect ranges from (0,
    /// 0) to (1, 1).
    pub(crate) uv_rect: Rect,

    /// The index of the slab (i.e. binding array) in which the lightmap is
    /// located.
    pub(crate) slab_index: LightmapSlabIndex,

    /// The index of the slot (i.e. element within the binding array) in which
    /// the lightmap is located.
    ///
    /// If bindless lightmaps aren't in use, this will be 0.
    pub(crate) slot_index: LightmapSlotIndex,

    // Whether or not bicubic sampling should be used for this lightmap.
    pub(crate) bicubic_sampling: bool,
}

impl RenderLightmap {
    /// Creates a new lightmap from a texture, a UV rect, and a slab and slot
    /// index pair.
    pub fn new(
        uv_rect: Rect,
        slab_index: LightmapSlabIndex,
        slot_index: LightmapSlotIndex,
        bicubic_sampling: bool,
    ) -> Self {
        Self {
            uv_rect,
            slab_index,
            slot_index,
            bicubic_sampling,
        }
    }
}

/// A binding array that contains lightmaps.
///
/// This will have a single binding if bindless lightmaps aren't in use.
pub struct LightmapSlab {
    /// The GPU images in this slab.
    pub lightmaps: Vec<AllocatedLightmap>,
    pub free_slots_bitmask: u32,
}

struct AllocatedLightmap {
    pub gpu_image: GpuImage,
    // This will only be present if the lightmap is allocated but not loaded.
    pub asset_id: Option<AssetId<Image>>,
}

/// The index of the slab (binding array) in which a lightmap is located.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deref, DerefMut)]
#[repr(transparent)]
pub struct LightmapSlabIndex(pub(crate) NonMaxU32);

/// The index of the slot (element within the binding array) in the slab in
/// which a lightmap is located.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Deref, DerefMut)]
#[repr(transparent)]
pub struct LightmapSlotIndex(pub(crate) NonMaxU16);

impl LightmapSlab {
    fn new(fallback_images: &FallbackImage, bindless_supported: bool) -> LightmapSlab {
        let count = if bindless_supported {
            LIGHTMAPS_PER_SLAB
        } else {
            1
        };

        LightmapSlab {
            lightmaps: (0..count)
                .map(|_| AllocatedLightmap {
                    gpu_image: fallback_images.d2.clone(),
                    asset_id: None,
                })
                .collect(),
            free_slots_bitmask: (1 << count) - 1,
        }
    }

    fn is_full(&self) -> bool {
        self.free_slots_bitmask == 0
    }

    fn allocate(&mut self, image_id: AssetId<Image>) -> LightmapSlotIndex {
        let index = LightmapSlotIndex::from(self.free_slots_bitmask.trailing_zeros());
        self.free_slots_bitmask &= !(1 << u32::from(index));
        self.lightmaps[usize::from(index)].asset_id = Some(image_id);
        index
    }

    pub fn insert(&mut self, index: LightmapSlotIndex, gpu_image: GpuImage) {
        self.lightmaps[usize::from(index)] = AllocatedLightmap {
            gpu_image,
            asset_id: None,
        }
    }

    fn remove(&mut self, fallback_images: &FallbackImage, index: LightmapSlotIndex) {
        self.lightmaps[usize::from(index)] = AllocatedLightmap {
            gpu_image: fallback_images.d2.clone(),
            asset_id: None,
        };
        self.free_slots_bitmask |= 1 << u32::from(index);
    }

    /// Returns the texture views and samplers for the lightmaps in this slab,
    /// ready to be placed into a bind group.
    ///
    /// This is used when constructing bind groups in bindless mode. Before
    /// returning, this function pads out the arrays with fallback images in
    /// order to fulfill requirements of platforms that require full binding
    /// arrays (e.g. DX12).
    pub(crate) fn build_binding_arrays(&self) -> (Vec<&WgpuTextureView>, Vec<&WgpuSampler>) {
        (
            self.lightmaps
                .iter()
                .map(|allocated_lightmap| &*allocated_lightmap.gpu_image.texture_view)
                .collect(),
            self.lightmaps
                .iter()
                .map(|allocated_lightmap| &*allocated_lightmap.gpu_image.sampler)
                .collect(),
        )
    }

    /// Returns the texture view and sampler corresponding to the first
    /// lightmap, which must exist.
    ///
    /// This is used when constructing bind groups in non-bindless mode.
    pub(crate) fn bindings_for_first_lightmap(&self) -> (&TextureView, &Sampler) {
        (
            &self.lightmaps[0].gpu_image.texture_view,
            &self.lightmaps[0].gpu_image.sampler,
        )
    }
}

impl From<u32> for LightmapSlabIndex {
    fn from(value: u32) -> Self {
        Self(NonMaxU32::new(value).unwrap())
    }
}

impl From<usize> for LightmapSlabIndex {
    fn from(value: usize) -> Self {
        Self::from(value as u32)
    }
}

impl From<u32> for LightmapSlotIndex {
    fn from(value: u32) -> Self {
        Self(NonMaxU16::new(value as u16).unwrap())
    }
}

impl From<usize> for LightmapSlotIndex {
    fn from(value: usize) -> Self {
        Self::from(value as u32)
    }
}

impl From<LightmapSlabIndex> for usize {
    fn from(value: LightmapSlabIndex) -> Self {
        value.0.get() as usize
    }
}

impl From<LightmapSlotIndex> for usize {
    fn from(value: LightmapSlotIndex) -> Self {
        value.0.get() as usize
    }
}

impl From<LightmapSlotIndex> for u16 {
    fn from(value: LightmapSlotIndex) -> Self {
        value.0.get()
    }
}

impl From<LightmapSlotIndex> for u32 {
    fn from(value: LightmapSlotIndex) -> Self {
        value.0.get() as u32
    }
}
