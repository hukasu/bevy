use core::{num::NonZero, ops::Deref};

use bevy_asset::AssetId;
use bevy_derive::{Deref, DerefMut};
use bevy_ecs::{
    component::Component, query::QueryItem, resource::Resource, system::lifetimeless::Read,
};
use bevy_image::Image;
use bevy_math::Mat4;
use bevy_render::{
    extract_instances::ExtractInstance,
    render_asset::RenderAssets,
    render_resource::{
        binding_types::{self, uniform_buffer},
        BindGroupLayoutEntryBuilder, DynamicUniformBuffer, Sampler, SamplerBindingType,
        ShaderStages, ShaderType, TextureSampleType, TextureView,
    },
    renderer::{RenderAdapter, RenderDevice},
    texture::{FallbackImage, GpuImage},
};

use crate::binding_arrays_are_usable;

use super::{
    add_cubemap_texture_view, light_probes::RenderViewLightProbes, EnvironmentMapLight,
    MAX_VIEW_LIGHT_PROBES,
};

/// All the bind group entries necessary for PBR shaders to access the
/// environment maps exposed to a view.
pub enum RenderViewEnvironmentMapBindGroupEntries<'a> {
    /// The version used when binding arrays aren't available on the current
    /// platform.
    Single {
        /// The texture view of the view's diffuse cubemap.
        diffuse_texture_view: &'a TextureView,

        /// The texture view of the view's specular cubemap.
        specular_texture_view: &'a TextureView,

        /// The sampler used to sample elements of both `diffuse_texture_views` and
        /// `specular_texture_views`.
        sampler: &'a Sampler,
    },

    /// The version used when binding arrays are available on the current
    /// platform.
    Multiple {
        /// A texture view of each diffuse cubemap, in the same order that they are
        /// supplied to the view (i.e. in the same order as
        /// `binding_index_to_cubemap` in [`RenderViewLightProbes`]).
        ///
        /// This is a vector of `wgpu::TextureView`s. But we don't want to import
        /// `wgpu` in this crate, so we refer to it indirectly like this.
        diffuse_texture_views: Vec<&'a <TextureView as Deref>::Target>,

        /// As above, but for specular cubemaps.
        specular_texture_views: Vec<&'a <TextureView as Deref>::Target>,

        /// The sampler used to sample elements of both `diffuse_texture_views` and
        /// `specular_texture_views`.
        sampler: &'a Sampler,
    },
}

/// Like [`EnvironmentMapLight`], but contains asset IDs instead of handles.
///
/// This is for use in the render app.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct EnvironmentMapIds {
    /// The blurry image that represents diffuse radiance surrounding a region.
    pub(crate) diffuse: AssetId<Image>,
    /// The typically-sharper, mipmapped image that represents specular radiance
    /// surrounding a region.
    pub(crate) specular: AssetId<Image>,
}

impl ExtractInstance for EnvironmentMapIds {
    type QueryData = Read<EnvironmentMapLight>;
    type QueryFilter = ();

    fn extract(item: QueryItem<'_, Self::QueryData>) -> Option<Self> {
        Some(EnvironmentMapIds {
            diffuse: item.diffuse_map.id(),
            specular: item.specular_map.id(),
        })
    }
}

impl<'a> RenderViewEnvironmentMapBindGroupEntries<'a> {
    /// Looks up and returns the bindings for the environment map diffuse and
    /// specular binding arrays respectively, as well as the sampler.
    pub(crate) fn get(
        render_view_environment_maps: Option<&RenderViewLightProbes<EnvironmentMapLight>>,
        images: &'a RenderAssets<GpuImage>,
        fallback_image: &'a FallbackImage,
        render_device: &RenderDevice,
        render_adapter: &RenderAdapter,
    ) -> RenderViewEnvironmentMapBindGroupEntries<'a> {
        if binding_arrays_are_usable(render_device, render_adapter) {
            let mut diffuse_texture_views = vec![];
            let mut specular_texture_views = vec![];
            let mut sampler = None;

            if let Some(environment_maps) = render_view_environment_maps {
                for &cubemap_id in &environment_maps.binding_index_to_textures {
                    add_cubemap_texture_view(
                        &mut diffuse_texture_views,
                        &mut sampler,
                        cubemap_id.diffuse,
                        images,
                        fallback_image,
                    );
                    add_cubemap_texture_view(
                        &mut specular_texture_views,
                        &mut sampler,
                        cubemap_id.specular,
                        images,
                        fallback_image,
                    );
                }
            }

            // Pad out the bindings to the size of the binding array using fallback
            // textures. This is necessary on D3D12 and Metal.
            diffuse_texture_views.resize(MAX_VIEW_LIGHT_PROBES, &*fallback_image.cube.texture_view);
            specular_texture_views
                .resize(MAX_VIEW_LIGHT_PROBES, &*fallback_image.cube.texture_view);

            return RenderViewEnvironmentMapBindGroupEntries::Multiple {
                diffuse_texture_views,
                specular_texture_views,
                sampler: sampler.unwrap_or(&fallback_image.cube.sampler),
            };
        }

        if let Some(environment_maps) = render_view_environment_maps {
            if let Some(cubemap) = environment_maps.binding_index_to_textures.first() {
                if let (Some(diffuse_image), Some(specular_image)) =
                    (images.get(cubemap.diffuse), images.get(cubemap.specular))
                {
                    return RenderViewEnvironmentMapBindGroupEntries::Single {
                        diffuse_texture_view: &diffuse_image.texture_view,
                        specular_texture_view: &specular_image.texture_view,
                        sampler: &diffuse_image.sampler,
                    };
                }
            }
        }

        RenderViewEnvironmentMapBindGroupEntries::Single {
            diffuse_texture_view: &fallback_image.cube.texture_view,
            specular_texture_view: &fallback_image.cube.texture_view,
            sampler: &fallback_image.cube.sampler,
        }
    }
}

/// Information about the environment map attached to the view, if any. This is
/// a global environment map that lights everything visible in the view, as
/// opposed to a light probe which affects only a specific area.
pub struct EnvironmentMapViewLightProbeInfo {
    /// The index of the diffuse and specular cubemaps in the binding arrays.
    pub(crate) cubemap_index: i32,
    /// The smallest mip level of the specular cubemap.
    pub(crate) smallest_specular_mip_level: u32,
    /// The scale factor applied to the diffuse and specular light in the
    /// cubemap. This is in units of cd/m² (candela per square meter).
    pub(crate) intensity: f32,
    /// Whether this lightmap affects the diffuse lighting of lightmapped
    /// meshes.
    pub(crate) affects_lightmapped_mesh_diffuse: bool,
}

impl Default for EnvironmentMapViewLightProbeInfo {
    fn default() -> Self {
        Self {
            cubemap_index: -1,
            smallest_specular_mip_level: 0,
            intensity: 1.0,
            affects_lightmapped_mesh_diffuse: true,
        }
    }
}

/// A GPU buffer that stores the environment map settings for each view.
#[derive(Resource, Default, Deref, DerefMut)]
pub struct EnvironmentMapUniformBuffer(pub DynamicUniformBuffer<EnvironmentMapUniform>);

/// The uniform struct extracted from [`EnvironmentMapLight`].
/// Will be available for use in the Environment Map shader.
#[derive(Component, ShaderType, Clone)]
pub struct EnvironmentMapUniform {
    /// The world space transformation matrix of the sample ray for environment cubemaps.
    pub transform: Mat4,
}

impl Default for EnvironmentMapUniform {
    fn default() -> Self {
        EnvironmentMapUniform {
            transform: Mat4::IDENTITY,
        }
    }
}

/// A component that stores the offset within the
/// [`EnvironmentMapUniformBuffer`] for each view.
#[derive(Component, Default, Deref, DerefMut)]
pub struct ViewEnvironmentMapUniformOffset(pub u32);

/// Returns the bind group layout entries for the environment map diffuse and
/// specular binding arrays respectively, in addition to the sampler.
pub fn get_bind_group_layout_entries(
    render_device: &RenderDevice,
    render_adapter: &RenderAdapter,
) -> [BindGroupLayoutEntryBuilder; 4] {
    let mut texture_cube_binding =
        binding_types::texture_cube(TextureSampleType::Float { filterable: true });
    let Ok(max_view_light_probes) = u32::try_from(MAX_VIEW_LIGHT_PROBES) else {
        unreachable!("Maximum number of light probes on view should never exceed a `u32`.");
    };
    if binding_arrays_are_usable(render_device, render_adapter) {
        texture_cube_binding =
            texture_cube_binding.count(NonZero::<u32>::new(max_view_light_probes).unwrap());
    }

    [
        texture_cube_binding,
        texture_cube_binding,
        binding_types::sampler(SamplerBindingType::Filtering),
        uniform_buffer::<EnvironmentMapUniform>(true).visibility(ShaderStages::FRAGMENT),
    ]
}
