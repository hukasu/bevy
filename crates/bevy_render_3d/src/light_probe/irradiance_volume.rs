use core::{num::NonZero, ops::Deref};

use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{
        binding_types, BindGroupLayoutEntryBuilder, Sampler, SamplerBindingType, TextureSampleType,
        TextureView,
    },
    renderer::{RenderAdapter, RenderDevice},
    texture::{FallbackImage, GpuImage},
};

use crate::{
    binding_arrays_are_usable,
    light_probe::{IrradianceVolume, MAX_VIEW_LIGHT_PROBES},
};

use super::{add_cubemap_texture_view, light_probes::RenderViewLightProbes};

/// All the bind group entries necessary for PBR shaders to access the
/// irradiance volumes exposed to a view.
pub enum RenderViewIrradianceVolumeBindGroupEntries<'a> {
    /// The version used when binding arrays aren't available on the current platform.
    Single {
        /// The texture view of the closest light probe.
        texture_view: &'a TextureView,
        /// A sampler used to sample voxels of the irradiance volume.
        sampler: &'a Sampler,
    },

    /// The version used when binding arrays are available on the current
    /// platform.
    Multiple {
        /// A texture view of the voxels of each irradiance volume, in the same
        /// order that they are supplied to the view (i.e. in the same order as
        /// `binding_index_to_cubemap` in [`RenderViewLightProbes`]).
        ///
        /// This is a vector of `wgpu::TextureView`s. But we don't want to import
        /// `wgpu` in this crate, so we refer to it indirectly like this.
        texture_views: Vec<&'a <TextureView as Deref>::Target>,

        /// A sampler used to sample voxels of the irradiance volumes.
        sampler: &'a Sampler,
    },
}

impl<'a> RenderViewIrradianceVolumeBindGroupEntries<'a> {
    /// Looks up and returns the bindings for any irradiance volumes visible in
    /// the view, as well as the sampler.
    pub(crate) fn get(
        render_view_irradiance_volumes: Option<&RenderViewLightProbes<IrradianceVolume>>,
        images: &'a RenderAssets<GpuImage>,
        fallback_image: &'a FallbackImage,
        render_device: &RenderDevice,
        render_adapter: &RenderAdapter,
    ) -> RenderViewIrradianceVolumeBindGroupEntries<'a> {
        if binding_arrays_are_usable(render_device, render_adapter) {
            RenderViewIrradianceVolumeBindGroupEntries::get_multiple(
                render_view_irradiance_volumes,
                images,
                fallback_image,
            )
        } else {
            RenderViewIrradianceVolumeBindGroupEntries::single(
                render_view_irradiance_volumes,
                images,
                fallback_image,
            )
        }
    }

    /// Looks up and returns the bindings for any irradiance volumes visible in
    /// the view, as well as the sampler. This is the version used when binding
    /// arrays are available on the current platform.
    fn get_multiple(
        render_view_irradiance_volumes: Option<&RenderViewLightProbes<IrradianceVolume>>,
        images: &'a RenderAssets<GpuImage>,
        fallback_image: &'a FallbackImage,
    ) -> RenderViewIrradianceVolumeBindGroupEntries<'a> {
        let mut texture_views = vec![];
        let mut sampler = None;

        if let Some(irradiance_volumes) = render_view_irradiance_volumes {
            for &cubemap_id in &irradiance_volumes.binding_index_to_textures {
                add_cubemap_texture_view(
                    &mut texture_views,
                    &mut sampler,
                    cubemap_id,
                    images,
                    fallback_image,
                );
            }
        }

        // Pad out the bindings to the size of the binding array using fallback
        // textures. This is necessary on D3D12 and Metal.
        texture_views.resize(MAX_VIEW_LIGHT_PROBES, &*fallback_image.d3.texture_view);

        RenderViewIrradianceVolumeBindGroupEntries::Multiple {
            texture_views,
            sampler: sampler.unwrap_or(&fallback_image.d3.sampler),
        }
    }

    /// Looks up and returns the bindings for any irradiance volumes visible in
    /// the view, as well as the sampler. This is the version used when binding
    /// arrays aren't available on the current platform.
    fn single(
        render_view_irradiance_volumes: Option<&RenderViewLightProbes<IrradianceVolume>>,
        images: &'a RenderAssets<GpuImage>,
        fallback_image: &'a FallbackImage,
    ) -> RenderViewIrradianceVolumeBindGroupEntries<'a> {
        if let Some(irradiance_volumes) = render_view_irradiance_volumes {
            if let Some(irradiance_volume) = irradiance_volumes.render_light_probes.first() {
                if irradiance_volume.texture_index >= 0 {
                    if let Some(image_id) = irradiance_volumes
                        .binding_index_to_textures
                        .get(irradiance_volume.texture_index as usize)
                    {
                        if let Some(image) = images.get(*image_id) {
                            return RenderViewIrradianceVolumeBindGroupEntries::Single {
                                texture_view: &image.texture_view,
                                sampler: &image.sampler,
                            };
                        }
                    }
                }
            }
        }

        RenderViewIrradianceVolumeBindGroupEntries::Single {
            texture_view: &fallback_image.d3.texture_view,
            sampler: &fallback_image.d3.sampler,
        }
    }
}

/// Returns the bind group layout entries for the voxel texture and sampler
/// respectively.
pub fn get_bind_group_layout_entries(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> [BindGroupLayoutEntryBuilder; 2] {
    let mut texture_3d_binding =
        binding_types::texture_3d(TextureSampleType::Float { filterable: true });
    let Ok(max_view_light_probes) = u32::try_from(MAX_VIEW_LIGHT_PROBES) else {
        unreachable!("Maximum number of light probes on view should never exceed a `u32`.");
    };
    if binding_arrays_are_usable(render_device, render_adapter) {
        texture_3d_binding =
            texture_3d_binding.count(NonZero::<u32>::new(max_view_light_probes).unwrap());
    }

    [
        texture_3d_binding,
        binding_types::sampler(SamplerBindingType::Filtering),
    ]
}
