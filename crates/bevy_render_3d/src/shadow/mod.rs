pub mod phase_item;
pub mod plugin;
pub(crate) mod render;

use bevy_ecs::{component::Component, reflect::ReflectComponent};
use bevy_reflect::{prelude::ReflectDefault, Reflect};
use bevy_render::extract_component::ExtractComponent;

/// Add this component to a [`Camera3d`](bevy_core_pipeline::core_3d::Camera3d)
/// to control how to anti-alias shadow edges.
///
/// The different modes use different approaches to
/// [Percentage Closer Filtering](https://developer.nvidia.com/gpugems/gpugems/part-ii-lighting-and-shadows/chapter-11-shadow-map-antialiasing).
#[derive(Debug, Component, ExtractComponent, Reflect, Clone, Copy, PartialEq, Eq, Default)]
#[reflect(Component, Default, Debug, PartialEq, Clone)]
pub enum ShadowFilteringMethod {
    /// Hardware 2x2.
    ///
    /// Fast but poor quality.
    Hardware2x2,
    /// Approximates a fixed Gaussian blur, good when TAA isn't in use.
    ///
    /// Good quality, good performance.
    ///
    /// For directional and spot lights, this uses a [method by Ignacio Castaño
    /// for *The Witness*] using 9 samples and smart filtering to achieve the same
    /// as a regular 5x5 filter kernel.
    ///
    /// [method by Ignacio Castaño for *The Witness*]: https://web.archive.org/web/20230210095515/http://the-witness.net/news/2013/09/shadow-mapping-summary-part-1/
    #[default]
    Gaussian,
    /// A randomized filter that varies over time, good when TAA is in use.
    ///
    /// Good quality when used with `TemporalAntiAliasing`
    /// and good performance.
    ///
    /// For directional and spot lights, this uses a [method by Jorge Jimenez for
    /// *Call of Duty: Advanced Warfare*] using 8 samples in spiral pattern,
    /// randomly-rotated by interleaved gradient noise with spatial variation.
    ///
    /// [method by Jorge Jimenez for *Call of Duty: Advanced Warfare*]: https://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare/
    Temporal,
}
