//! Clustered decals, bounding regions that project textures onto surfaces.
//!
//! A *clustered decal* is a bounding box that projects a texture onto any
//! surface within its bounds along the positive Z axis. In Bevy, clustered
//! decals use the *clustered forward* rendering technique.
//!
//! Clustered decals are the highest-quality types of decals that Bevy supports,
//! but they require bindless textures. This means that they presently can't be
//! used on WebGL 2, WebGPU, macOS, or iOS. Bevy's clustered decals can be used
//! with forward or deferred rendering and don't require a prepass.
//!
//! On their own, clustered decals only project the base color of a texture. You
//! can, however, use the built-in *tag* field to customize the appearance of a
//! clustered decal arbitrarily. See the documentation in `clustered.wgsl` for
//! more information and the `clustered_decals` example for an example of use.

use core::ops::Deref;

use bevy_asset::AssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{entity::hash_map::EntityHashMap, resource::Resource};
use bevy_image::Image;
use bevy_math::Mat4;
use bevy_platform_support::collections::HashMap;
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{Buffer, BufferUsages, RawBufferVec, Sampler, ShaderType, TextureView},
    renderer::{RenderAdapter, RenderDevice},
    texture::{FallbackImage, GpuImage},
};

use bytemuck::{Pod, Zeroable};

use super::{clustered_decals_are_usable, MAX_VIEW_DECALS};

/// Stores information about all the clustered decals in the scene.
#[derive(Resource, Default)]
pub struct RenderClusteredDecals {
    /// Maps an index in the shader binding array to the associated decal image.
    ///
    /// [`Self::texture_to_binding_index`] holds the inverse mapping.
    binding_index_to_textures: Vec<AssetId<Image>>,
    /// Maps a decal image to the shader binding array.
    ///
    /// [`Self::binding_index_to_textures`] holds the inverse mapping.
    texture_to_binding_index: HashMap<AssetId<Image>, u32>,
    /// The information concerning each decal that we provide to the shader.
    pub decals: Vec<RenderClusteredDecal>,
    /// Maps the [`bevy_render::sync_world::RenderEntity`] of each decal to the
    /// index of that decal in the [`Self::decals`] list.
    pub entity_to_decal_index: EntityHashMap<usize>,
}

impl RenderClusteredDecals {
    /// Clears out this [`RenderClusteredDecals`] in preparation for a new
    /// frame.
    pub fn clear(&mut self) {
        self.binding_index_to_textures.clear();
        self.texture_to_binding_index.clear();
        self.decals.clear();
        self.entity_to_decal_index.clear();
    }
}

/// The per-view bind group entries pertaining to decals.
pub(crate) struct RenderViewClusteredDecalBindGroupEntries<'a> {
    /// The list of decals, corresponding to `mesh_view_bindings::decals` in the
    /// shader.
    pub(crate) decals: &'a Buffer,
    /// The list of textures, corresponding to
    /// `mesh_view_bindings::decal_textures` in the shader.
    pub(crate) texture_views: Vec<&'a <TextureView as Deref>::Target>,
    /// The sampler that the shader uses to sample decals, corresponding to
    /// `mesh_view_bindings::decal_sampler` in the shader.
    pub(crate) sampler: &'a Sampler,
}

/// A render-world resource that holds the buffer of [`ClusteredDecal`]s ready
/// to upload to the GPU.
#[derive(Resource, Deref, DerefMut)]
pub struct DecalsBuffer(RawBufferVec<RenderClusteredDecal>);

impl Default for DecalsBuffer {
    fn default() -> Self {
        DecalsBuffer(RawBufferVec::new(BufferUsages::STORAGE))
    }
}

/// The GPU data structure that stores information about each decal.
#[derive(Clone, Copy, Default, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct RenderClusteredDecal {
    /// The inverse of the model matrix.
    ///
    /// The shader uses this in order to back-transform world positions into
    /// model space.
    pub local_from_world: Mat4,
    /// The index of the decal texture in the binding array.
    pub image_index: u32,
    /// A custom tag available for application-defined purposes.
    pub tag: u32,
    /// Padding.
    pub pad_a: u32,
    /// Padding.
    pub pad_b: u32,
}

impl<'a> RenderViewClusteredDecalBindGroupEntries<'a> {
    /// Creates and returns the bind group entries for clustered decals for a
    /// single view.
    pub(crate) fn get(
        render_decals: &RenderClusteredDecals,
        decals_buffer: &'a DecalsBuffer,
        images: &'a RenderAssets<GpuImage>,
        fallback_image: &'a FallbackImage,
        render_device: &RenderDevice,
        render_adapter: &RenderAdapter,
    ) -> Option<RenderViewClusteredDecalBindGroupEntries<'a>> {
        // Skip the entries if decals are unsupported on the current platform.
        if !clustered_decals_are_usable(render_device, render_adapter) {
            return None;
        }

        // We use the first sampler among all the images. This assumes that all
        // images use the same sampler, which is a documented restriction. If
        // there's no sampler, we just use the one from the fallback image.
        let sampler = match render_decals
            .binding_index_to_textures
            .iter()
            .filter_map(|image_id| images.get(*image_id))
            .next()
        {
            Some(gpu_image) => &gpu_image.sampler,
            None => &fallback_image.d2.sampler,
        };

        // Gather up the decal textures.
        let mut texture_views = vec![];
        for image_id in &render_decals.binding_index_to_textures {
            match images.get(*image_id) {
                None => texture_views.push(&*fallback_image.d2.texture_view),
                Some(gpu_image) => texture_views.push(&*gpu_image.texture_view),
            }
        }

        // Pad out the binding array to its maximum length, which is
        // required on some platforms.
        while texture_views.len() < MAX_VIEW_DECALS {
            texture_views.push(&*fallback_image.d2.texture_view);
        }

        Some(RenderViewClusteredDecalBindGroupEntries {
            decals: decals_buffer.buffer()?,
            texture_views,
            sampler,
        })
    }
}

impl RenderClusteredDecals {
    /// Returns the index of the given image in the decal texture binding array,
    /// adding it to the list if necessary.
    pub fn get_or_insert_image(&mut self, image_id: &AssetId<Image>) -> u32 {
        *self
            .texture_to_binding_index
            .entry(*image_id)
            .or_insert_with(|| {
                let index = self.binding_index_to_textures.len() as u32;
                self.binding_index_to_textures.push(*image_id);
                index
            })
    }
}
