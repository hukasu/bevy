//! Constants for operating with the light units: lumens, and lux.

/// Approximations for converting the wattage of lamps to lumens.
///
/// The **lumen** (symbol: **lm**) is the unit of [luminous flux], a measure
/// of the total quantity of [visible light] emitted by a source per unit of
/// time, in the [International System of Units] (SI).
///
/// For more information, see [wikipedia](https://en.wikipedia.org/wiki/Lumen_(unit))
///
/// [luminous flux]: https://en.wikipedia.org/wiki/Luminous_flux
/// [visible light]: https://en.wikipedia.org/wiki/Visible_light
/// [International System of Units]: https://en.wikipedia.org/wiki/International_System_of_Units
pub mod lumens {
    pub const LUMENS_PER_LED_WATTS: f32 = 90.0;
    pub const LUMENS_PER_INCANDESCENT_WATTS: f32 = 13.8;
    pub const LUMENS_PER_HALOGEN_WATTS: f32 = 19.8;
}

/// Predefined for lux values in several locations.
///
/// The **lux** (symbol: **lx**) is the unit of [illuminance], or [luminous flux] per unit area,
/// in the [International System of Units] (SI). It is equal to one lumen per square meter.
///
/// For more information, see [wikipedia](https://en.wikipedia.org/wiki/Lux)
///
/// [illuminance]: https://en.wikipedia.org/wiki/Illuminance
/// [luminous flux]: https://en.wikipedia.org/wiki/Luminous_flux
/// [International System of Units]: https://en.wikipedia.org/wiki/International_System_of_Units
pub mod lux {
    /// The amount of light (lux) in a moonless, overcast night sky. (starlight)
    pub const MOONLESS_NIGHT: f32 = 0.0001;
    /// The amount of light (lux) during a full moon on a clear night.
    pub const FULL_MOON_NIGHT: f32 = 0.05;
    /// The amount of light (lux) during the dark limit of civil twilight under a clear sky.
    pub const CIVIL_TWILIGHT: f32 = 3.4;
    /// The amount of light (lux) in family living room lights.
    pub const LIVING_ROOM: f32 = 50.;
    /// The amount of light (lux) in an office building's hallway/toilet lighting.
    pub const HALLWAY: f32 = 80.;
    /// The amount of light (lux) in very dark overcast day
    pub const DARK_OVERCAST_DAY: f32 = 100.;
    /// The amount of light (lux) in an office.
    pub const OFFICE: f32 = 320.;
    /// The amount of light (lux) during sunrise or sunset on a clear day.
    pub const CLEAR_SUNRISE: f32 = 400.;
    /// The amount of light (lux) on an overcast day; typical TV studio lighting
    pub const OVERCAST_DAY: f32 = 1000.;
    /// The amount of light (lux) from ambient daylight (not direct sunlight).
    pub const AMBIENT_DAYLIGHT: f32 = 10_000.;
    /// The amount of light (lux) in full daylight (not direct sun).
    pub const FULL_DAYLIGHT: f32 = 20_000.;
    /// The amount of light (lux) in direct sunlight.
    pub const DIRECT_SUNLIGHT: f32 = 100_000.;
    /// The amount of light (lux) of raw sunlight, not filtered by the atmosphere.
    pub const RAW_SUNLIGHT: f32 = 130_000.;
}
