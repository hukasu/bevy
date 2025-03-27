//! Spatial clustering of objects, currently just point and spot lights.

mod cluster;
mod clusterable_objects;
pub mod plugin;
#[cfg(test)]
mod test;

use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_math::{AspectRatio, UVec2, UVec3};
use bevy_reflect::{prelude::ReflectDefault, Reflect};

use tracing::warn;

pub use clusterable_objects::GlobalClusterableObjectMeta;

// Clustered-forward rendering notes
// The main initial reference material used was this rather accessible article:
// http://www.aortiz.me/2018/12/21/CG.html
// Some inspiration was taken from “Practical Clustered Shading” which is part 2 of:
// https://efficientshading.com/2015/01/01/real-time-many-light-management-and-shadows-with-clustered-shading/
// (Also note that Part 3 of the above shows how we could support the shadow mapping for many lights.)
// The z-slicing method mentioned in the aortiz article is originally from Tiago Sousa's Siggraph 2016 talk about Doom 2016:
// http://advances.realtimerendering.com/s2016/Siggraph2016_idTech6.pdf

// NOTE: this must be kept in sync with the same constants in
// `mesh_view_types.wgsl`.
pub const MAX_UNIFORM_BUFFER_CLUSTERABLE_OBJECTS: usize = 204;

// NOTE: Clustered-forward rendering requires 3 storage buffer bindings so check that
// at least that many are supported using this constant and SupportedBindingType::from_device()
pub const CLUSTERED_FORWARD_STORAGE_BUFFER_COUNT: u32 = 3;

/// Configuration of the clustering strategy for clustered forward rendering
#[derive(Debug, Copy, Clone, Component, Reflect)]
#[reflect(Component, Debug, Default, Clone)]
pub enum ClusterConfig {
    /// Disable cluster calculations for this view
    None,
    /// One single cluster. Optimal for low-light complexity scenes or scenes where
    /// most lights affect the entire scene.
    Single,
    /// Explicit `X`, `Y` and `Z` counts (may yield non-square `X/Y` clusters depending on the aspect ratio)
    XYZ {
        dimensions: UVec3,
        z_config: ClusterZConfig,
        /// Specify if clusters should automatically resize in `X/Y` if there is a risk of exceeding
        /// the available cluster-object index limit
        dynamic_resizing: bool,
    },
    /// Fixed number of `Z` slices, `X` and `Y` calculated to give square clusters
    /// with at most total clusters. For top-down games where lights will generally always be within a
    /// short depth range, it may be useful to use this configuration with 1 or few `Z` slices. This
    /// would reduce the number of lights per cluster by distributing more clusters in screen space
    /// `X/Y` which matches how lights are distributed in the scene.
    FixedZ {
        total: u32,
        z_slices: u32,
        z_config: ClusterZConfig,
        /// Specify if clusters should automatically resize in `X/Y` if there is a risk of exceeding
        /// the available clusterable object index limit
        dynamic_resizing: bool,
    },
}

impl ClusterConfig {
    fn dimensions_for_screen_size(&self, screen_size: UVec2) -> UVec3 {
        match &self {
            ClusterConfig::None => UVec3::ZERO,
            ClusterConfig::Single => UVec3::ONE,
            ClusterConfig::XYZ { dimensions, .. } => *dimensions,
            ClusterConfig::FixedZ {
                total, z_slices, ..
            } => {
                let aspect_ratio: f32 = AspectRatio::try_from_pixels(screen_size.x, screen_size.y)
                    .expect("Failed to calculate aspect ratio for Cluster: screen dimensions must be positive, non-zero values")
                    .ratio();
                let mut z_slices = *z_slices;
                if *total < z_slices {
                    warn!("ClusterConfig has more z-slices than total clusters!");
                    z_slices = *total;
                }
                let per_layer = *total as f32 / z_slices as f32;

                let y = f32::sqrt(per_layer / aspect_ratio);

                let mut x = (y * aspect_ratio) as u32;
                let mut y = y as u32;

                // check extremes
                if x == 0 {
                    x = 1;
                    y = per_layer as u32;
                }
                if y == 0 {
                    x = per_layer as u32;
                    y = 1;
                }

                UVec3::new(x, y, z_slices)
            }
        }
    }

    fn first_slice_depth(&self) -> f32 {
        match self {
            ClusterConfig::None | ClusterConfig::Single => 0.0,
            ClusterConfig::XYZ { z_config, .. } | ClusterConfig::FixedZ { z_config, .. } => {
                z_config.first_slice_depth
            }
        }
    }

    fn far_z_mode(&self) -> ClusterFarZMode {
        match self {
            ClusterConfig::None => ClusterFarZMode::Constant(0.0),
            ClusterConfig::Single => ClusterFarZMode::MaxClusterableObjectRange,
            ClusterConfig::XYZ { z_config, .. } | ClusterConfig::FixedZ { z_config, .. } => {
                z_config.far_z_mode
            }
        }
    }

    fn dynamic_resizing(&self) -> bool {
        match self {
            ClusterConfig::None | ClusterConfig::Single => false,
            ClusterConfig::XYZ {
                dynamic_resizing, ..
            }
            | ClusterConfig::FixedZ {
                dynamic_resizing, ..
            } => *dynamic_resizing,
        }
    }
}

impl Default for ClusterConfig {
    fn default() -> Self {
        // 24 depth slices, square clusters with at most 4096 total clusters
        // use max light distance as clusters max `Z`-depth, first slice extends to 5.0
        Self::FixedZ {
            total: 4096,
            z_slices: 24,
            z_config: ClusterZConfig::default(),
            dynamic_resizing: true,
        }
    }
}

/// Configure the depth-slicing strategy for clustered forward rendering
#[derive(Debug, Copy, Clone, Reflect)]
#[reflect(Default, Clone)]
pub struct ClusterZConfig {
    /// Far `Z` plane of the first depth slice
    pub first_slice_depth: f32,
    /// Strategy for how to evaluate the far `Z` plane of the furthest depth slice
    pub far_z_mode: ClusterFarZMode,
}

impl Default for ClusterZConfig {
    fn default() -> Self {
        Self {
            first_slice_depth: 5.0,
            far_z_mode: ClusterFarZMode::MaxClusterableObjectRange,
        }
    }
}

/// Configure the far z-plane mode used for the furthest depth slice for clustered forward
/// rendering
#[derive(Debug, Copy, Clone, Reflect)]
#[reflect(Clone)]
pub enum ClusterFarZMode {
    /// Calculate the required maximum z-depth based on currently visible
    /// clusterable objects.  Makes better use of available clusters, speeding
    /// up GPU lighting operations at the expense of some CPU time and using
    /// more indices in the clusterable object index lists.
    MaxClusterableObjectRange,
    /// Constant max z-depth
    Constant(f32),
}
