use bevy_ecs::{component::Component, entity::hash_map::EntityHashMap, reflect::ReflectComponent};
use bevy_math::{ops, Mat4};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

use crate::light::VisibleMeshEntities;

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub struct Cascades {
    /// Map from a view to the configuration of each of its [`Cascade`]s.
    pub(crate) cascades: EntityHashMap<Vec<Cascade>>,
}

#[derive(Clone, Debug, Default, Reflect)]
#[reflect(Clone, Default)]
pub struct Cascade {
    /// The transform of the light, i.e. the view to world matrix.
    pub(crate) world_from_cascade: Mat4,
    /// The orthographic projection for this cascade.
    pub(crate) clip_from_cascade: Mat4,
    /// The view-projection matrix for this cascade, converting world space into light clip space.
    /// Importantly, this is derived and stored separately from `view_transform` and `projection` to
    /// ensure shadow stability.
    pub(crate) clip_from_world: Mat4,
    /// Size of each shadow map texel in world units.
    pub(crate) texel_size: f32,
}

/// Controls how cascaded shadow mapping works.
/// Prefer using [`CascadeShadowConfigBuilder`] to construct an instance.
///
/// ```
/// # use bevy_pbr::CascadeShadowConfig;
/// # use bevy_pbr::CascadeShadowConfigBuilder;
/// # use bevy_utils::default;
/// #
/// let config: CascadeShadowConfig = CascadeShadowConfigBuilder {
///   maximum_distance: 100.0,
///   ..default()
/// }.into();
/// ```
#[derive(Component, Clone, Debug, Reflect)]
#[reflect(Component, Default, Debug, Clone)]
pub struct CascadeShadowConfig {
    /// The (positive) distance to the far boundary of each cascade.
    pub bounds: Vec<f32>,
    /// The proportion of overlap each cascade has with the previous cascade.
    pub overlap_proportion: f32,
    /// The (positive) distance to the near boundary of the first cascade.
    pub minimum_distance: f32,
}

impl Default for CascadeShadowConfig {
    fn default() -> Self {
        CascadeShadowConfigBuilder::default().into()
    }
}

/// Builder for [`CascadeShadowConfig`].
pub struct CascadeShadowConfigBuilder {
    /// The number of shadow cascades.
    /// More cascades increases shadow quality by mitigating perspective aliasing - a phenomenon where areas
    /// nearer the camera are covered by fewer shadow map texels than areas further from the camera, causing
    /// blocky looking shadows.
    ///
    /// This does come at the cost increased rendering overhead, however this overhead is still less
    /// than if you were to use fewer cascades and much larger shadow map textures to achieve the
    /// same quality level.
    ///
    /// In case rendered geometry covers a relatively narrow and static depth relative to camera, it may
    /// make more sense to use fewer cascades and a higher resolution shadow map texture as perspective aliasing
    /// is not as much an issue. Be sure to adjust `minimum_distance` and `maximum_distance` appropriately.
    pub num_cascades: usize,
    /// The minimum shadow distance, which can help improve the texel resolution of the first cascade.
    /// Areas nearer to the camera than this will likely receive no shadows.
    ///
    /// NOTE: Due to implementation details, this usually does not impact shadow quality as much as
    /// `first_cascade_far_bound` and `maximum_distance`. At many view frustum field-of-views, the
    /// texel resolution of the first cascade is dominated by the width / height of the view frustum plane
    /// at `first_cascade_far_bound` rather than the depth of the frustum from `minimum_distance` to
    /// `first_cascade_far_bound`.
    pub minimum_distance: f32,
    /// The maximum shadow distance.
    /// Areas further from the camera than this will likely receive no shadows.
    pub maximum_distance: f32,
    /// Sets the far bound of the first cascade, relative to the view origin.
    /// In-between cascades will be exponentially spaced relative to the maximum shadow distance.
    /// NOTE: This is ignored if there is only one cascade, the maximum distance takes precedence.
    pub first_cascade_far_bound: f32,
    /// Sets the overlap proportion between cascades.
    /// The overlap is used to make the transition from one cascade's shadow map to the next
    /// less abrupt by blending between both shadow maps.
    pub overlap_proportion: f32,
}

impl CascadeShadowConfigBuilder {
    /// Returns the cascade config as specified by this builder.
    pub fn build(&self) -> CascadeShadowConfig {
        assert!(
            self.num_cascades > 0,
            "num_cascades must be positive, but was {}",
            self.num_cascades
        );
        assert!(
            self.minimum_distance >= 0.0,
            "maximum_distance must be non-negative, but was {}",
            self.minimum_distance
        );
        assert!(
            self.num_cascades == 1 || self.minimum_distance < self.first_cascade_far_bound,
            "minimum_distance must be less than first_cascade_far_bound, but was {}",
            self.minimum_distance
        );
        assert!(
            self.maximum_distance > self.minimum_distance,
            "maximum_distance must be greater than minimum_distance, but was {}",
            self.maximum_distance
        );
        assert!(
            (0.0..1.0).contains(&self.overlap_proportion),
            "overlap_proportion must be in [0.0, 1.0) but was {}",
            self.overlap_proportion
        );
        CascadeShadowConfig {
            bounds: calculate_cascade_bounds(
                self.num_cascades,
                self.first_cascade_far_bound,
                self.maximum_distance,
            ),
            overlap_proportion: self.overlap_proportion,
            minimum_distance: self.minimum_distance,
        }
    }
}

impl Default for CascadeShadowConfigBuilder {
    fn default() -> Self {
        // The defaults are chosen to be similar to be Unity, Unreal, and Godot.
        // Unity: first cascade far bound = 10.05, maximum distance = 150.0
        // Unreal Engine 5: maximum distance = 200.0
        // Godot: first cascade far bound = 10.0, maximum distance = 100.0
        Self {
            // Currently only support one cascade in WebGL 2.
            num_cascades: if cfg!(all(
                feature = "webgl",
                target_arch = "wasm32",
                not(feature = "webgpu")
            )) {
                1
            } else {
                4
            },
            minimum_distance: 0.1,
            maximum_distance: 150.0,
            first_cascade_far_bound: 10.0,
            overlap_proportion: 0.2,
        }
    }
}

impl From<CascadeShadowConfigBuilder> for CascadeShadowConfig {
    fn from(builder: CascadeShadowConfigBuilder) -> Self {
        builder.build()
    }
}

#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Default, Clone)]
pub struct CascadesVisibleEntities {
    /// Map of view entity to the visible entities for each cascade frustum.
    #[reflect(ignore, clone)]
    pub entities: EntityHashMap<Vec<VisibleMeshEntities>>,
}

fn calculate_cascade_bounds(
    num_cascades: usize,
    nearest_bound: f32,
    shadow_maximum_distance: f32,
) -> Vec<f32> {
    if num_cascades == 1 {
        return vec![shadow_maximum_distance];
    }
    let base = ops::powf(
        shadow_maximum_distance / nearest_bound,
        1.0 / (num_cascades - 1) as f32,
    );
    (0..num_cascades)
        .map(|i| nearest_bound * ops::powf(base, i as f32))
        .collect()
}
