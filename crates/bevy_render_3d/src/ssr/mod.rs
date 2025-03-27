//! Screen space reflections implemented via raymarching.

pub mod plugin;
pub(crate) mod render;

use bevy_core_pipeline::{
    core_3d::DEPTH_TEXTURE_SAMPLING_SUPPORTED,
    prepass::{DeferredPrepass, DepthPrepass},
};
use bevy_ecs::{
    component::Component, query::QueryItem, reflect::ReflectComponent, system::lifetimeless::Read,
};
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
use bevy_render::extract_component::ExtractComponent;
use bevy_utils::once;

use render::ScreenSpaceReflectionsUniform;
use tracing::info;

/// Add this component to a camera to enable *screen-space reflections* (SSR).
///
/// Screen-space reflections currently require deferred rendering in order to
/// appear. Therefore, they also need the [`DepthPrepass`] and [`DeferredPrepass`]
/// components, which are inserted automatically.
///
/// SSR currently performs no roughness filtering for glossy reflections, so
/// only very smooth surfaces will reflect objects in screen space. You can
/// adjust the `perceptual_roughness_threshold` in order to tune the threshold
/// below which screen-space reflections will be traced.
///
/// As with all screen-space techniques, SSR can only reflect objects on screen.
/// When objects leave the camera, they will disappear from reflections.
/// An alternative that doesn't suffer from this problem is the combination of
/// a [`LightProbe`](crate::LightProbe) and [`EnvironmentMapLight`]. The advantage of SSR is
/// that it can reflect all objects, not just static ones.
///
/// SSR is an approximation technique and produces artifacts in some situations.
/// Hand-tuning the settings in this component will likely be useful.
///
/// Screen-space reflections are presently unsupported on WebGL 2 because of a
/// bug whereby Naga doesn't generate correct GLSL when sampling depth buffers,
/// which is required for screen-space raymarching.
#[derive(Clone, Copy, Component, Reflect)]
#[reflect(Component, Default, Clone)]
#[require(DepthPrepass, DeferredPrepass)]
#[doc(alias = "Ssr")]
pub struct ScreenSpaceReflections {
    /// The maximum PBR roughness level that will enable screen space
    /// reflections.
    pub perceptual_roughness_threshold: f32,

    /// When marching the depth buffer, we only have 2.5D information and don't
    /// know how thick surfaces are. We shall assume that the depth buffer
    /// fragments are cuboids with a constant thickness defined by this
    /// parameter.
    pub thickness: f32,

    /// The number of steps to be taken at regular intervals to find an initial
    /// intersection. Must not be zero.
    ///
    /// Higher values result in higher-quality reflections, because the
    /// raymarching shader is less likely to miss objects. However, they take
    /// more GPU time.
    pub linear_steps: u32,

    /// Exponent to be applied in the linear part of the march.
    ///
    /// A value of 1.0 will result in equidistant steps, and higher values will
    /// compress the earlier steps, and expand the later ones. This might be
    /// desirable in order to get more detail close to objects.
    ///
    /// For optimal performance, this should be a small unsigned integer, such
    /// as 1 or 2.
    pub linear_march_exponent: f32,

    /// Number of steps in a bisection (binary search) to perform once the
    /// linear search has found an intersection. Helps narrow down the hit,
    /// increasing the chance of the secant method finding an accurate hit
    /// point.
    pub bisection_steps: u32,

    /// Approximate the root position using the secant method—by solving for
    /// line-line intersection between the ray approach rate and the surface
    /// gradient.
    pub use_secant: bool,
}

impl Default for ScreenSpaceReflections {
    // Reasonable default values.
    //
    // These are from
    // <https://gist.github.com/h3r2tic/9c8356bdaefbe80b1a22ae0aaee192db?permalink_comment_id=4552149#gistcomment-4552149>.
    fn default() -> Self {
        Self {
            perceptual_roughness_threshold: 0.1,
            linear_steps: 16,
            bisection_steps: 4,
            use_secant: true,
            thickness: 0.25,
            linear_march_exponent: 1.0,
        }
    }
}

impl ExtractComponent for ScreenSpaceReflections {
    type QueryData = Read<ScreenSpaceReflections>;

    type QueryFilter = ();

    type Out = ScreenSpaceReflectionsUniform;

    fn extract_component(settings: QueryItem<'_, Self::QueryData>) -> Option<Self::Out> {
        if !DEPTH_TEXTURE_SAMPLING_SUPPORTED {
            once!(info!(
                "Disabling screen-space reflections on this platform because depth textures \
                aren't supported correctly"
            ));
            return None;
        }

        Some((*settings).into())
    }
}
