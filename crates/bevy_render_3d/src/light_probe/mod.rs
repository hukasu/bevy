//! Light probes for baked global illumination.

pub(crate) mod environment_map;
pub(crate) mod irradiance_volume;
pub(crate) mod light_probes;
pub mod plugin;

use core::{hash::Hash, ops::Deref};

use bevy_asset::{AssetId, Handle};
use bevy_ecs::{component::Component, entity::Entity, query::With, reflect::ReflectComponent};
use bevy_image::Image;
use bevy_math::{Quat, Vec3A};
use bevy_platform_support::prelude::Box;
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::{
    render_asset::RenderAssets,
    render_resource::{Sampler, TextureView},
    texture::{FallbackImage, GpuImage},
    view::Visibility,
};
use bevy_transform::components::{GlobalTransform, Transform};
use bevy_utils::default;

use crate::cluster::plugin::ClusterAssignable;

use environment_map::{EnvironmentMapIds, EnvironmentMapViewLightProbeInfo};
use light_probes::RenderViewLightProbes;

/// The maximum number of each type of light probe that each view will consider.
///
/// Because the fragment shader does a linear search through the list for each
/// fragment, this number needs to be relatively small.
pub const MAX_VIEW_LIGHT_PROBES: usize = 8;

/// On WebGL and WebGPU, we must disable irradiance volumes, as otherwise we can
/// overflow the number of texture bindings when deferred rendering is in use
/// (see issue #11885).
pub const IRRADIANCE_VOLUMES_ARE_USABLE: bool = cfg!(not(target_arch = "wasm32"));

/// A marker component for a light probe, which is a cuboid region that provides
/// global illumination to all fragments inside it.
///
/// Note that a light probe will have no effect unless the entity contains some
/// kind of illumination, which can either be an [`EnvironmentMapLight`] or an
/// [`IrradianceVolume`].
///
/// The light probe range is conceptually a unit cube (1×1×1) centered on the
/// origin. The [`Transform`] applied to this entity can scale, rotate, or translate
/// that cube so that it contains all fragments that should take this light probe into account.
///
/// When multiple sources of indirect illumination can be applied to a fragment,
/// the highest-quality one is chosen. Diffuse and specular illumination are
/// considered separately, so, for example, Bevy may decide to sample the
/// diffuse illumination from an irradiance volume and the specular illumination
/// from a reflection probe. From highest priority to lowest priority, the
/// ranking is as follows:
///
/// | Rank | Diffuse              | Specular             |
/// | ---- | -------------------- | -------------------- |
/// | 1    | Lightmap             | Lightmap             |
/// | 2    | Irradiance volume    | Reflection probe     |
/// | 3    | Reflection probe     | View environment map |
/// | 4    | View environment map |                      |
///
/// Note that ambient light is always added to the diffuse component and does
/// not participate in the ranking. That is, ambient light is applied in
/// addition to, not instead of, the light sources above.
///
/// A terminology note: Unfortunately, there is little agreement across game and
/// graphics engines as to what to call the various techniques that Bevy groups
/// under the term *light probe*. In Bevy, a *light probe* is the generic term
/// that encompasses both *reflection probes* and *irradiance volumes*. In
/// object-oriented terms, *light probe* is the superclass, and *reflection
/// probe* and *irradiance volume* are subclasses. In other engines, you may see
/// the term *light probe* refer to an irradiance volume with a single voxel, or
/// perhaps some other technique, while in Bevy *light probe* refers not to a
/// specific technique but rather to a class of techniques. Developers familiar
/// with other engines should be aware of this terminology difference.
#[derive(Component, Debug, Clone, Copy, Default, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
#[require(Transform, Visibility)]
pub struct LightProbe;

impl LightProbe {
    /// Creates a new light probe component.
    #[inline]
    pub fn new() -> Self {
        Self
    }
}

/// A trait implemented by all components that represent light probes.
///
/// Currently, the two light probe types are [`EnvironmentMapLight`] and
/// [`IrradianceVolume`], for reflection probes and irradiance volumes
/// respectively.
///
/// Most light probe systems are written to be generic over the type of light
/// probe. This allows much of the code to be shared and enables easy addition
/// of more light probe types (e.g. real-time reflection planes) in the future.
trait LightProbeComponent: Send + Sync + Component + Sized {
    /// Holds [`AssetId`]s of the texture or textures that this light probe
    /// references.
    ///
    /// This can just be [`AssetId`] if the light probe only references one
    /// texture. If it references multiple textures, it will be a structure
    /// containing those asset IDs.
    type AssetId: Send + Sync + Clone + Eq + Hash;

    /// If the light probe can be attached to the view itself (as opposed to a
    /// cuboid region within the scene), this contains the information that will
    /// be passed to the GPU in order to render it. Otherwise, this will be
    /// `()`.
    ///
    /// Currently, only reflection probes (i.e. [`EnvironmentMapLight`]) can be
    /// attached directly to views.
    type ViewLightProbeInfo: Send + Sync + Default;

    /// Returns the asset ID or asset IDs of the texture or textures referenced
    /// by this light probe.
    fn id(&self, image_assets: &RenderAssets<GpuImage>) -> Option<Self::AssetId>;

    /// Returns the intensity of this light probe.
    ///
    /// This is a scaling factor that will be multiplied by the value or values
    /// sampled from the texture.
    fn intensity(&self) -> f32;

    /// Returns true if this light probe contributes diffuse lighting to meshes
    /// with lightmaps or false otherwise.
    fn affects_lightmapped_mesh_diffuse(&self) -> bool;

    /// Creates an instance of [`RenderViewLightProbes`] containing all the
    /// information needed to render this light probe.
    ///
    /// This is called for every light probe in view every frame.
    fn create_render_view_light_probes(
        view_component: Option<&Self>,
        image_assets: &RenderAssets<GpuImage>,
    ) -> RenderViewLightProbes<Self>;
}

/// A pair of cubemap textures that represent the surroundings of a specific
/// area in space.
///
/// See [`crate::environment_map`] for detailed information.
#[derive(Clone, Component, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct EnvironmentMapLight {
    /// The blurry image that represents diffuse radiance surrounding a region.
    pub diffuse_map: Handle<Image>,

    /// The typically-sharper, mipmapped image that represents specular radiance
    /// surrounding a region.
    pub specular_map: Handle<Image>,

    /// Scale factor applied to the diffuse and specular light generated by this component.
    ///
    /// After applying this multiplier, the resulting values should
    /// be in units of [cd/m^2](https://en.wikipedia.org/wiki/Candela_per_square_metre).
    ///
    /// See also <https://google.github.io/filament/Filament.html#lighting/imagebasedlights/iblunit>.
    pub intensity: f32,

    /// World space rotation applied to the environment light cubemaps.
    /// This is useful for users who require a different axis, such as the Z-axis, to serve
    /// as the vertical axis.
    pub rotation: Quat,

    /// Whether the light from this environment map contributes diffuse lighting
    /// to meshes with lightmaps.
    ///
    /// Set this to false if your lightmap baking tool bakes the diffuse light
    /// from this environment light into the lightmaps in order to avoid
    /// counting the radiance from this environment map twice.
    ///
    /// By default, this is set to true.
    pub affects_lightmapped_mesh_diffuse: bool,
}

impl Default for EnvironmentMapLight {
    fn default() -> Self {
        EnvironmentMapLight {
            diffuse_map: Handle::default(),
            specular_map: Handle::default(),
            intensity: 0.0,
            rotation: Quat::IDENTITY,
            affects_lightmapped_mesh_diffuse: true,
        }
    }
}

impl LightProbeComponent for EnvironmentMapLight {
    type AssetId = EnvironmentMapIds;

    // Information needed to render with the environment map attached to the
    // view.
    type ViewLightProbeInfo = EnvironmentMapViewLightProbeInfo;

    fn id(&self, image_assets: &RenderAssets<GpuImage>) -> Option<Self::AssetId> {
        if image_assets.get(&self.diffuse_map).is_none()
            || image_assets.get(&self.specular_map).is_none()
        {
            None
        } else {
            Some(EnvironmentMapIds {
                diffuse: self.diffuse_map.id(),
                specular: self.specular_map.id(),
            })
        }
    }

    fn intensity(&self) -> f32 {
        self.intensity
    }

    fn affects_lightmapped_mesh_diffuse(&self) -> bool {
        self.affects_lightmapped_mesh_diffuse
    }

    fn create_render_view_light_probes(
        view_component: Option<&EnvironmentMapLight>,
        image_assets: &RenderAssets<GpuImage>,
    ) -> RenderViewLightProbes<Self> {
        let mut render_view_light_probes = RenderViewLightProbes::new();

        // Find the index of the cubemap associated with the view, and determine
        // its smallest mip level.
        if let Some(EnvironmentMapLight {
            diffuse_map: diffuse_map_handle,
            specular_map: specular_map_handle,
            intensity,
            affects_lightmapped_mesh_diffuse,
            ..
        }) = view_component
        {
            if let (Some(_), Some(specular_map)) = (
                image_assets.get(diffuse_map_handle),
                image_assets.get(specular_map_handle),
            ) {
                render_view_light_probes.view_light_probe_info = EnvironmentMapViewLightProbeInfo {
                    cubemap_index: render_view_light_probes.get_or_insert_cubemap(
                        &EnvironmentMapIds {
                            diffuse: diffuse_map_handle.id(),
                            specular: specular_map_handle.id(),
                        },
                    ) as i32,
                    smallest_specular_mip_level: specular_map.mip_level_count - 1,
                    intensity: *intensity,
                    affects_lightmapped_mesh_diffuse: *affects_lightmapped_mesh_diffuse,
                };
            }
        };

        render_view_light_probes
    }
}

impl ClusterAssignable for EnvironmentMapLight {
    type Query<'a> = (Entity, &'a GlobalTransform);
    type Filter = (With<LightProbe>, With<EnvironmentMapLight>);

    fn entity(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Entity {
        query_data.0
    }

    fn transform(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> GlobalTransform {
        *query_data.1
    }

    fn range(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> f32 {
        query_data.1.radius_vec3a(Vec3A::ONE)
    }

    fn shadows_enabled(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        false
    }

    fn volumetric(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        false
    }

    fn render_layers(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Option<bevy_render::view::RenderLayers> {
        None
    }

    fn cull_method(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Option<Box<dyn crate::cluster::plugin::ClusterAssignableCullMethod>> {
        None
    }

    fn visible(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        true
    }
}

/// The component that defines an irradiance volume.
///
/// See [`crate::irradiance_volume`] for detailed information.
#[derive(Clone, Reflect, Component, Debug)]
#[reflect(Component, Default, Debug, Clone)]
pub struct IrradianceVolume {
    /// The 3D texture that represents the ambient cubes, encoded in the format
    /// described in [`crate::irradiance_volume`].
    pub voxels: Handle<Image>,

    /// Scale factor applied to the diffuse and specular light generated by this component.
    ///
    /// After applying this multiplier, the resulting values should
    /// be in units of [cd/m^2](https://en.wikipedia.org/wiki/Candela_per_square_metre).
    ///
    /// See also <https://google.github.io/filament/Filament.html#lighting/imagebasedlights/iblunit>.
    pub intensity: f32,

    /// Whether the light from this irradiance volume has an effect on meshes
    /// with lightmaps.
    ///
    /// Set this to false if your lightmap baking tool bakes the light from this
    /// irradiance volume into the lightmaps in order to avoid counting the
    /// irradiance twice. Frequently, applications use irradiance volumes as a
    /// lower-quality alternative to lightmaps for capturing indirect
    /// illumination on dynamic objects, and such applications will want to set
    /// this value to false.
    ///
    /// By default, this is set to true.
    pub affects_lightmapped_meshes: bool,
}

impl Default for IrradianceVolume {
    #[inline]
    fn default() -> Self {
        IrradianceVolume {
            voxels: default(),
            intensity: 0.0,
            affects_lightmapped_meshes: true,
        }
    }
}

impl LightProbeComponent for IrradianceVolume {
    type AssetId = AssetId<Image>;

    // Irradiance volumes can't be attached to the view, so we store nothing
    // here.
    type ViewLightProbeInfo = ();

    fn id(&self, image_assets: &RenderAssets<GpuImage>) -> Option<Self::AssetId> {
        if image_assets.get(&self.voxels).is_none() {
            None
        } else {
            Some(self.voxels.id())
        }
    }

    fn intensity(&self) -> f32 {
        self.intensity
    }

    fn affects_lightmapped_mesh_diffuse(&self) -> bool {
        self.affects_lightmapped_meshes
    }

    fn create_render_view_light_probes(
        _: Option<&Self>,
        _: &RenderAssets<GpuImage>,
    ) -> RenderViewLightProbes<Self> {
        RenderViewLightProbes::new()
    }
}

impl ClusterAssignable for IrradianceVolume {
    type Query<'a> = (Entity, &'a GlobalTransform);
    type Filter = (With<LightProbe>, With<IrradianceVolume>);

    fn entity(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Entity {
        query_data.0
    }

    fn transform(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> GlobalTransform {
        *query_data.1
    }

    fn range(
        query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> f32 {
        query_data.1.radius_vec3a(Vec3A::ONE)
    }

    fn shadows_enabled(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        false
    }

    fn volumetric(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        false
    }

    fn render_layers(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Option<bevy_render::view::RenderLayers> {
        None
    }

    fn cull_method(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> Option<Box<dyn crate::cluster::plugin::ClusterAssignableCullMethod>> {
        None
    }

    fn visible(
        _query_data: &<<Self::Query<'_> as bevy_ecs::query::QueryData>::ReadOnly as bevy_ecs::query::QueryData>::Item<'_>,
    ) -> bool {
        true
    }
}

/// Adds a diffuse or specular texture view to the `texture_views` list, and
/// populates `sampler` if this is the first such view.
fn add_cubemap_texture_view<'a>(
    texture_views: &mut Vec<&'a <TextureView as Deref>::Target>,
    sampler: &mut Option<&'a Sampler>,
    image_id: AssetId<Image>,
    images: &'a RenderAssets<GpuImage>,
    fallback_image: &'a FallbackImage,
) {
    match images.get(image_id) {
        None => {
            // Use the fallback image if the cubemap isn't loaded yet.
            texture_views.push(&*fallback_image.cube.texture_view);
        }
        Some(image) => {
            // If this is the first texture view, populate `sampler`.
            if sampler.is_none() {
                *sampler = Some(&image.sampler);
            }

            texture_views.push(&*image.texture_view);
        }
    }
}
