use crate::{
    primitives::{Primitive2d, Primitive3d},
    Quat, Rot2, Vec2, Vec3, Vec3A, Vec4,
};

use core::f32::consts::FRAC_1_SQRT_2;
use core::fmt;
use derive_more::derive::Into;

#[cfg(feature = "bevy_reflect")]
use bevy_reflect::Reflect;

#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

#[cfg(all(debug_assertions, feature = "std"))]
use std::eprintln;

use thiserror::Error;

/// An error indicating that a direction is invalid.
#[derive(Debug, PartialEq, Error)]
pub enum InvalidDirectionError {
    /// The length of the direction vector is zero or very close to zero.
    #[error("The length of the direction vector is zero or very close to zero")]
    Zero,
    /// The length of the direction vector is `std::f32::INFINITY`.
    #[error("The length of the direction vector is `std::f32::INFINITY`")]
    Infinite,
    /// The length of the direction vector is `NaN`.
    #[error("The length of the direction vector is `NaN`")]
    NaN,
}

impl InvalidDirectionError {
    /// Creates an [`InvalidDirectionError`] from the length of an invalid direction vector.
    pub fn from_length(length: f32) -> Self {
        if length.is_nan() {
            InvalidDirectionError::NaN
        } else if !length.is_finite() {
            // If the direction is non-finite but also not NaN, it must be infinite
            InvalidDirectionError::Infinite
        } else {
            // If the direction is invalid but neither NaN nor infinite, it must be zero
            InvalidDirectionError::Zero
        }
    }
}

/// Checks that a vector with the given squared length is normalized.
///
/// Warns for small error with a length threshold of approximately `1e-4`,
/// and panics for large error with a length threshold of approximately `1e-2`.
///
/// The format used for the logged warning is `"Warning: {warning} The length is {length}`,
/// and similarly for the error.
#[cfg(debug_assertions)]
fn assert_is_normalized(message: &str, length_squared: f32) {
    use crate::ops;

    let length_error_squared = ops::abs(length_squared - 1.0);

    // Panic for large error and warn for slight error.
    if length_error_squared > 2e-2 || length_error_squared.is_nan() {
        // Length error is approximately 1e-2 or more.
        panic!(
            "Error: {message} The length is {}.",
            ops::sqrt(length_squared)
        );
    } else if length_error_squared > 2e-4 {
        // Length error is approximately 1e-4 or more.
        #[cfg(feature = "std")]
        #[expect(clippy::print_stderr, reason = "Allowed behind `std` feature gate.")]
        {
            eprintln!(
                "Warning: {message} The length is {}.",
                ops::sqrt(length_squared)
            );
        }
    }
}

/// A normalized vector pointing in a direction in 2D space
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
#[doc(alias = "Direction2d")]
pub struct Dir2(Vec2);
impl Primitive2d for Dir2 {}

impl Dir2 {
    /// A unit vector pointing along the positive X axis.
    pub const X: Self = Self(Vec2::X);
    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self(Vec2::Y);
    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(Vec2::NEG_X);
    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(Vec2::NEG_Y);
    /// The directional axes.
    pub const AXES: [Self; 2] = [Self::X, Self::Y];

    /// The "north" direction, equivalent to [`Dir2::Y`].
    pub const NORTH: Self = Self(Vec2::Y);
    /// The "south" direction, equivalent to [`Dir2::NEG_Y`].
    pub const SOUTH: Self = Self(Vec2::NEG_Y);
    /// The "east" direction, equivalent to [`Dir2::X`].
    pub const EAST: Self = Self(Vec2::X);
    /// The "west" direction, equivalent to [`Dir2::NEG_X`].
    pub const WEST: Self = Self(Vec2::NEG_X);
    /// The "north-east" direction, between [`Dir2::NORTH`] and [`Dir2::EAST`].
    pub const NORTH_EAST: Self = Self(Vec2::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2));
    /// The "north-west" direction, between [`Dir2::NORTH`] and [`Dir2::WEST`].
    pub const NORTH_WEST: Self = Self(Vec2::new(-FRAC_1_SQRT_2, FRAC_1_SQRT_2));
    /// The "south-east" direction, between [`Dir2::SOUTH`] and [`Dir2::EAST`].
    pub const SOUTH_EAST: Self = Self(Vec2::new(FRAC_1_SQRT_2, -FRAC_1_SQRT_2));
    /// The "south-west" direction, between [`Dir2::SOUTH`] and [`Dir2::WEST`].
    pub const SOUTH_WEST: Self = Self(Vec2::new(-FRAC_1_SQRT_2, -FRAC_1_SQRT_2));

    /// Create a direction from a finite, nonzero [`Vec2`], normalizing it.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new(value: Vec2) -> Result<Self, InvalidDirectionError> {
        Self::new_and_length(value).map(|(dir, _)| dir)
    }

    /// Create a [`Dir2`] from a [`Vec2`] that is already normalized.
    ///
    /// # Warning
    ///
    /// `value` must be normalized, i.e its length must be `1.0`.
    pub fn new_unchecked(value: Vec2) -> Self {
        #[cfg(debug_assertions)]
        assert_is_normalized(
            "The vector given to `Dir2::new_unchecked` is not normalized.",
            value.length_squared(),
        );

        Self(value)
    }

    /// Create a direction from a finite, nonzero [`Vec2`], normalizing it and
    /// also returning its original length.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new_and_length(value: Vec2) -> Result<(Self, f32), InvalidDirectionError> {
        let length = value.length();
        let direction = (length.is_finite() && length > 0.0).then_some(value / length);

        direction
            .map(|dir| (Self(dir), length))
            .ok_or(InvalidDirectionError::from_length(length))
    }

    /// Create a direction from its `x` and `y` components.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the vector formed by the components is zero (or very close to zero), infinite, or `NaN`.
    pub fn from_xy(x: f32, y: f32) -> Result<Self, InvalidDirectionError> {
        Self::new(Vec2::new(x, y))
    }

    /// Create a direction from its `x` and `y` components, assuming the resulting vector is normalized.
    ///
    /// # Warning
    ///
    /// The vector produced from `x` and `y` must be normalized, i.e its length must be `1.0`.
    pub fn from_xy_unchecked(x: f32, y: f32) -> Self {
        Self::new_unchecked(Vec2::new(x, y))
    }

    /// Returns the inner [`Vec2`]
    pub const fn as_vec2(&self) -> Vec2 {
        self.0
    }

    /// Performs a spherical linear interpolation between `self` and `rhs`
    /// based on the value `s`.
    ///
    /// This corresponds to interpolating between the two directions at a constant angular velocity.
    ///
    /// When `s == 0.0`, the result will be equal to `self`.
    /// When `s == 1.0`, the result will be equal to `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::Dir2;
    /// # use approx::{assert_relative_eq, RelativeEq};
    /// #
    /// let dir1 = Dir2::X;
    /// let dir2 = Dir2::Y;
    ///
    /// let result1 = dir1.slerp(dir2, 1.0 / 3.0);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(result1, Dir2::from_xy(0.75_f32.sqrt(), 0.5).unwrap());
    ///
    /// let result2 = dir1.slerp(dir2, 0.5);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(result2, Dir2::from_xy(0.5_f32.sqrt(), 0.5_f32.sqrt()).unwrap());
    /// ```
    #[inline]
    pub fn slerp(self, rhs: Self, s: f32) -> Self {
        let angle = self.angle_to(rhs.0);
        Rot2::radians(angle * s) * self
    }

    /// Get the rotation that rotates this direction to `other`.
    #[inline]
    pub fn rotation_to(self, other: Self) -> Rot2 {
        // Rotate `self` to X-axis, then X-axis to `other`:
        other.rotation_from_x() * self.rotation_to_x()
    }

    /// Get the rotation that rotates `other` to this direction.
    #[inline]
    pub fn rotation_from(self, other: Self) -> Rot2 {
        other.rotation_to(self)
    }

    /// Get the rotation that rotates the X-axis to this direction.
    #[inline]
    pub fn rotation_from_x(self) -> Rot2 {
        Rot2::from_sin_cos(self.0.y, self.0.x)
    }

    /// Get the rotation that rotates this direction to the X-axis.
    #[inline]
    pub fn rotation_to_x(self) -> Rot2 {
        // (This is cheap, it just negates one component.)
        self.rotation_from_x().inverse()
    }

    /// Get the rotation that rotates the Y-axis to this direction.
    #[inline]
    pub fn rotation_from_y(self) -> Rot2 {
        // `x <- y`, `y <- -x` correspond to rotating clockwise by pi/2;
        // this transforms the Y-axis into the X-axis, maintaining the relative position
        // of our direction. Then we just use the same technique as `rotation_from_x`.
        Rot2::from_sin_cos(-self.0.x, self.0.y)
    }

    /// Get the rotation that rotates this direction to the Y-axis.
    #[inline]
    pub fn rotation_to_y(self) -> Rot2 {
        self.rotation_from_y().inverse()
    }

    /// Returns `self` after an approximate normalization, assuming the value is already nearly normalized.
    /// Useful for preventing numerical error accumulation.
    /// See [`Dir3::fast_renormalize`] for an example of when such error accumulation might occur.
    #[inline]
    pub fn fast_renormalize(self) -> Self {
        let length_squared = self.0.length_squared();
        // Based on a Taylor approximation of the inverse square root, see [`Dir3::fast_renormalize`] for more details.
        Self(self * (0.5 * (3.0 - length_squared)))
    }
}

impl TryFrom<Vec2> for Dir2 {
    type Error = InvalidDirectionError;

    fn try_from(value: Vec2) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Dir2> for Vec2 {
    fn from(value: Dir2) -> Self {
        value.as_vec2()
    }
}

impl core::ops::Deref for Dir2 {
    type Target = Vec2;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::Neg for Dir2 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl core::ops::Mul<f32> for Dir2 {
    type Output = Vec2;
    fn mul(self, rhs: f32) -> Self::Output {
        self.0 * rhs
    }
}

impl core::ops::Mul<Dir2> for f32 {
    type Output = Vec2;
    fn mul(self, rhs: Dir2) -> Self::Output {
        self * rhs.0
    }
}

impl core::ops::Mul<Dir2> for Rot2 {
    type Output = Dir2;

    /// Rotates the [`Dir2`] using a [`Rot2`].
    fn mul(self, direction: Dir2) -> Self::Output {
        let rotated = self * *direction;

        #[cfg(debug_assertions)]
        assert_is_normalized(
            "`Dir2` is denormalized after rotation.",
            rotated.length_squared(),
        );

        Dir2(rotated)
    }
}

impl fmt::Display for Dir2 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(any(feature = "approx", test))]
impl approx::AbsDiffEq for Dir2 {
    type Epsilon = f32;
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

#[cfg(any(feature = "approx", test))]
impl approx::RelativeEq for Dir2 {
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

#[cfg(any(feature = "approx", test))]
impl approx::UlpsEq for Dir2 {
    fn default_max_ulps() -> u32 {
        4
    }
    fn ulps_eq(&self, other: &Self, epsilon: f32, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}

/// A normalized vector pointing in a direction in 3D space
#[derive(Clone, Copy, Debug, PartialEq, Into)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
#[doc(alias = "Direction3d")]
pub struct Dir3(Vec3);
impl Primitive3d for Dir3 {}

impl Dir3 {
    /// A unit vector pointing along the positive X axis.
    pub const X: Self = Self(Vec3::X);
    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self(Vec3::Y);
    /// A unit vector pointing along the positive Z axis.
    pub const Z: Self = Self(Vec3::Z);
    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(Vec3::NEG_X);
    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(Vec3::NEG_Y);
    /// A unit vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self(Vec3::NEG_Z);
    /// The directional axes.
    pub const AXES: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Create a direction from a finite, nonzero [`Vec3`], normalizing it.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new(value: Vec3) -> Result<Self, InvalidDirectionError> {
        Self::new_and_length(value).map(|(dir, _)| dir)
    }

    /// Create a [`Dir3`] from a [`Vec3`] that is already normalized.
    ///
    /// # Warning
    ///
    /// `value` must be normalized, i.e its length must be `1.0`.
    pub fn new_unchecked(value: Vec3) -> Self {
        #[cfg(debug_assertions)]
        assert_is_normalized(
            "The vector given to `Dir3::new_unchecked` is not normalized.",
            value.length_squared(),
        );

        Self(value)
    }

    /// Create a direction from a finite, nonzero [`Vec3`], normalizing it and
    /// also returning its original length.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new_and_length(value: Vec3) -> Result<(Self, f32), InvalidDirectionError> {
        let length = value.length();
        let direction = (length.is_finite() && length > 0.0).then_some(value / length);

        direction
            .map(|dir| (Self(dir), length))
            .ok_or(InvalidDirectionError::from_length(length))
    }

    /// Create a direction from its `x`, `y`, and `z` components.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the vector formed by the components is zero (or very close to zero), infinite, or `NaN`.
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Result<Self, InvalidDirectionError> {
        Self::new(Vec3::new(x, y, z))
    }

    /// Create a direction from its `x`, `y`, and `z` components, assuming the resulting vector is normalized.
    ///
    /// # Warning
    ///
    /// The vector produced from `x`, `y`, and `z` must be normalized, i.e its length must be `1.0`.
    pub fn from_xyz_unchecked(x: f32, y: f32, z: f32) -> Self {
        Self::new_unchecked(Vec3::new(x, y, z))
    }

    /// Returns the inner [`Vec3`]
    pub const fn as_vec3(&self) -> Vec3 {
        self.0
    }

    /// Performs a spherical linear interpolation between `self` and `rhs`
    /// based on the value `s`.
    ///
    /// This corresponds to interpolating between the two directions at a constant angular velocity.
    ///
    /// When `s == 0.0`, the result will be equal to `self`.
    /// When `s == 1.0`, the result will be equal to `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::Dir3;
    /// # use approx::{assert_relative_eq, RelativeEq};
    /// #
    /// let dir1 = Dir3::X;
    /// let dir2 = Dir3::Y;
    ///
    /// let result1 = dir1.slerp(dir2, 1.0 / 3.0);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(
    ///     result1,
    ///     Dir3::from_xyz(0.75_f32.sqrt(), 0.5, 0.0).unwrap(),
    ///     epsilon = 0.000001
    /// );
    ///
    /// let result2 = dir1.slerp(dir2, 0.5);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(result2, Dir3::from_xyz(0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0).unwrap());
    /// ```
    #[inline]
    pub fn slerp(self, rhs: Self, s: f32) -> Self {
        let quat = Quat::IDENTITY.slerp(Quat::from_rotation_arc(self.0, rhs.0), s);
        Dir3(quat.mul_vec3(self.0))
    }

    /// Returns `self` after an approximate normalization, assuming the value is already nearly normalized.
    /// Useful for preventing numerical error accumulation.
    ///
    /// # Example
    /// The following seemingly benign code would start accumulating errors over time,
    /// leading to `dir` eventually not being normalized anymore.
    /// ```
    /// # use bevy_math::prelude::*;
    /// # let N: usize = 200;
    /// let mut dir = Dir3::X;
    /// let quaternion = Quat::from_euler(EulerRot::XYZ, 1.0, 2.0, 3.0);
    /// for i in 0..N {
    ///     dir = quaternion * dir;
    /// }
    /// ```
    /// Instead, do the following.
    /// ```
    /// # use bevy_math::prelude::*;
    /// # let N: usize = 200;
    /// let mut dir = Dir3::X;
    /// let quaternion = Quat::from_euler(EulerRot::XYZ, 1.0, 2.0, 3.0);
    /// for i in 0..N {
    ///     dir = quaternion * dir;
    ///     dir = dir.fast_renormalize();
    /// }
    /// ```
    #[inline]
    pub fn fast_renormalize(self) -> Self {
        // We numerically approximate the inverse square root by a Taylor series around 1
        // As we expect the error (x := length_squared - 1) to be small
        // inverse_sqrt(length_squared) = (1 + x)^(-1/2) = 1 - 1/2 x + O(x²)
        // inverse_sqrt(length_squared) ≈ 1 - 1/2 (length_squared - 1) = 1/2 (3 - length_squared)

        // Iterative calls to this method quickly converge to a normalized value,
        // so long as the denormalization is not large ~ O(1/10).
        // One iteration can be described as:
        // l_sq <- l_sq * (1 - 1/2 (l_sq - 1))²;
        // Rewriting in terms of the error x:
        // 1 + x <- (1 + x) * (1 - 1/2 x)²
        // 1 + x <- (1 + x) * (1 - x + 1/4 x²)
        // 1 + x <- 1 - x + 1/4 x² + x - x² + 1/4 x³
        // x <- -1/4 x² (3 - x)
        // If the error is small, say in a range of (-1/2, 1/2), then:
        // |-1/4 x² (3 - x)| <= (3/4 + 1/4 * |x|) * x² <= (3/4 + 1/4 * 1/2) * x² < x² < 1/2 x
        // Therefore the sequence of iterates converges to 0 error as a second order method.

        let length_squared = self.0.length_squared();
        Self(self * (0.5 * (3.0 - length_squared)))
    }
}

impl TryFrom<Vec3> for Dir3 {
    type Error = InvalidDirectionError;

    fn try_from(value: Vec3) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl core::ops::Deref for Dir3 {
    type Target = Vec3;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::Neg for Dir3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl core::ops::Mul<f32> for Dir3 {
    type Output = Vec3;
    fn mul(self, rhs: f32) -> Self::Output {
        self.0 * rhs
    }
}

impl core::ops::Mul<Dir3> for f32 {
    type Output = Vec3;
    fn mul(self, rhs: Dir3) -> Self::Output {
        self * rhs.0
    }
}

impl core::ops::Mul<Dir3> for Quat {
    type Output = Dir3;

    /// Rotates the [`Dir3`] using a [`Quat`].
    fn mul(self, direction: Dir3) -> Self::Output {
        let rotated = self * *direction;

        #[cfg(debug_assertions)]
        assert_is_normalized(
            "`Dir3` is denormalized after rotation.",
            rotated.length_squared(),
        );

        Dir3(rotated)
    }
}

impl fmt::Display for Dir3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for Dir3 {
    type Epsilon = f32;
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for Dir3 {
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl approx::UlpsEq for Dir3 {
    fn default_max_ulps() -> u32 {
        4
    }
    fn ulps_eq(&self, other: &Self, epsilon: f32, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}

/// A normalized SIMD vector pointing in a direction in 3D space.
///
/// This type stores a 16 byte aligned [`Vec3A`].
/// This may or may not be faster than [`Dir3`]: make sure to benchmark!
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
#[doc(alias = "Direction3dA")]
pub struct Dir3A(Vec3A);
impl Primitive3d for Dir3A {}

impl Dir3A {
    /// A unit vector pointing along the positive X axis.
    pub const X: Self = Self(Vec3A::X);
    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self(Vec3A::Y);
    /// A unit vector pointing along the positive Z axis.
    pub const Z: Self = Self(Vec3A::Z);
    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(Vec3A::NEG_X);
    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(Vec3A::NEG_Y);
    /// A unit vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self(Vec3A::NEG_Z);
    /// The directional axes.
    pub const AXES: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// Create a direction from a finite, nonzero [`Vec3A`], normalizing it.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new(value: Vec3A) -> Result<Self, InvalidDirectionError> {
        Self::new_and_length(value).map(|(dir, _)| dir)
    }

    /// Create a [`Dir3A`] from a [`Vec3A`] that is already normalized.
    ///
    /// # Warning
    ///
    /// `value` must be normalized, i.e its length must be `1.0`.
    pub fn new_unchecked(value: Vec3A) -> Self {
        #[cfg(debug_assertions)]
        assert_is_normalized(
            "The vector given to `Dir3A::new_unchecked` is not normalized.",
            value.length_squared(),
        );

        Self(value)
    }

    /// Create a direction from a finite, nonzero [`Vec3A`], normalizing it and
    /// also returning its original length.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new_and_length(value: Vec3A) -> Result<(Self, f32), InvalidDirectionError> {
        let length = value.length();
        let direction = (length.is_finite() && length > 0.0).then_some(value / length);

        direction
            .map(|dir| (Self(dir), length))
            .ok_or(InvalidDirectionError::from_length(length))
    }

    /// Create a direction from its `x`, `y`, and `z` components.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the vector formed by the components is zero (or very close to zero), infinite, or `NaN`.
    pub fn from_xyz(x: f32, y: f32, z: f32) -> Result<Self, InvalidDirectionError> {
        Self::new(Vec3A::new(x, y, z))
    }

    /// Create a direction from its `x`, `y`, and `z` components, assuming the resulting vector is normalized.
    ///
    /// # Warning
    ///
    /// The vector produced from `x`, `y`, and `z` must be normalized, i.e its length must be `1.0`.
    pub fn from_xyz_unchecked(x: f32, y: f32, z: f32) -> Self {
        Self::new_unchecked(Vec3A::new(x, y, z))
    }

    /// Returns the inner [`Vec3A`]
    pub const fn as_vec3a(&self) -> Vec3A {
        self.0
    }

    /// Performs a spherical linear interpolation between `self` and `rhs`
    /// based on the value `s`.
    ///
    /// This corresponds to interpolating between the two directions at a constant angular velocity.
    ///
    /// When `s == 0.0`, the result will be equal to `self`.
    /// When `s == 1.0`, the result will be equal to `rhs`.
    ///
    /// # Example
    ///
    /// ```
    /// # use bevy_math::Dir3A;
    /// # use approx::{assert_relative_eq, RelativeEq};
    /// #
    /// let dir1 = Dir3A::X;
    /// let dir2 = Dir3A::Y;
    ///
    /// let result1 = dir1.slerp(dir2, 1.0 / 3.0);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(
    ///     result1,
    ///     Dir3A::from_xyz(0.75_f32.sqrt(), 0.5, 0.0).unwrap(),
    ///     epsilon = 0.000001
    /// );
    ///
    /// let result2 = dir1.slerp(dir2, 0.5);
    /// #[cfg(feature = "approx")]
    /// assert_relative_eq!(result2, Dir3A::from_xyz(0.5_f32.sqrt(), 0.5_f32.sqrt(), 0.0).unwrap());
    /// ```
    #[inline]
    pub fn slerp(self, rhs: Self, s: f32) -> Self {
        let quat = Quat::IDENTITY.slerp(
            Quat::from_rotation_arc(Vec3::from(self.0), Vec3::from(rhs.0)),
            s,
        );
        Dir3A(quat.mul_vec3a(self.0))
    }

    /// Returns `self` after an approximate normalization, assuming the value is already nearly normalized.
    /// Useful for preventing numerical error accumulation.
    ///
    /// See [`Dir3::fast_renormalize`] for an example of when such error accumulation might occur.
    #[inline]
    pub fn fast_renormalize(self) -> Self {
        let length_squared = self.0.length_squared();
        // Based on a Taylor approximation of the inverse square root, see [`Dir3::fast_renormalize`] for more details.
        Self(self * (0.5 * (3.0 - length_squared)))
    }
}

impl From<Dir3> for Dir3A {
    fn from(value: Dir3) -> Self {
        Self(value.0.into())
    }
}

impl From<Dir3A> for Dir3 {
    fn from(value: Dir3A) -> Self {
        Self(value.0.into())
    }
}

impl TryFrom<Vec3A> for Dir3A {
    type Error = InvalidDirectionError;

    fn try_from(value: Vec3A) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<Dir3A> for Vec3A {
    fn from(value: Dir3A) -> Self {
        value.0
    }
}

impl core::ops::Deref for Dir3A {
    type Target = Vec3A;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::Neg for Dir3A {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl core::ops::Mul<f32> for Dir3A {
    type Output = Vec3A;
    fn mul(self, rhs: f32) -> Self::Output {
        self.0 * rhs
    }
}

impl core::ops::Mul<Dir3A> for f32 {
    type Output = Vec3A;
    fn mul(self, rhs: Dir3A) -> Self::Output {
        self * rhs.0
    }
}

impl core::ops::Mul<Dir3A> for Quat {
    type Output = Dir3A;

    /// Rotates the [`Dir3A`] using a [`Quat`].
    fn mul(self, direction: Dir3A) -> Self::Output {
        let rotated = self * *direction;

        #[cfg(debug_assertions)]
        assert_is_normalized(
            "`Dir3A` is denormalized after rotation.",
            rotated.length_squared(),
        );

        Dir3A(rotated)
    }
}

impl fmt::Display for Dir3A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for Dir3A {
    type Epsilon = f32;
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for Dir3A {
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl approx::UlpsEq for Dir3A {
    fn default_max_ulps() -> u32 {
        4
    }
    fn ulps_eq(&self, other: &Self, epsilon: f32, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}

/// A normalized vector pointing in a direction in 4D space
#[derive(Clone, Copy, Debug, PartialEq, Into)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Clone)
)]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
#[doc(alias = "Direction4d")]
pub struct Dir4(Vec4);

impl Dir4 {
    /// A unit vector pointing along the positive X axis
    pub const X: Self = Self(Vec4::X);
    /// A unit vector pointing along the positive Y axis.
    pub const Y: Self = Self(Vec4::Y);
    /// A unit vector pointing along the positive Z axis.
    pub const Z: Self = Self(Vec4::Z);
    /// A unit vector pointing along the positive W axis.
    pub const W: Self = Self(Vec4::W);
    /// A unit vector pointing along the negative X axis.
    pub const NEG_X: Self = Self(Vec4::NEG_X);
    /// A unit vector pointing along the negative Y axis.
    pub const NEG_Y: Self = Self(Vec4::NEG_Y);
    /// A unit vector pointing along the negative Z axis.
    pub const NEG_Z: Self = Self(Vec4::NEG_Z);
    /// A unit vector pointing along the negative W axis.
    pub const NEG_W: Self = Self(Vec4::NEG_W);
    /// The directional axes.
    pub const AXES: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    /// Create a direction from a finite, nonzero [`Vec4`], normalizing it.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new(value: Vec4) -> Result<Self, InvalidDirectionError> {
        Self::new_and_length(value).map(|(dir, _)| dir)
    }

    /// Create a [`Dir4`] from a [`Vec4`] that is already normalized.
    ///
    /// # Warning
    ///
    /// `value` must be normalized, i.e its length must be `1.0`.
    pub fn new_unchecked(value: Vec4) -> Self {
        #[cfg(debug_assertions)]
        assert_is_normalized(
            "The vector given to `Dir4::new_unchecked` is not normalized.",
            value.length_squared(),
        );
        Self(value)
    }

    /// Create a direction from a finite, nonzero [`Vec4`], normalizing it and
    /// also returning its original length.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the given vector is zero (or very close to zero), infinite, or `NaN`.
    pub fn new_and_length(value: Vec4) -> Result<(Self, f32), InvalidDirectionError> {
        let length = value.length();
        let direction = (length.is_finite() && length > 0.0).then_some(value / length);

        direction
            .map(|dir| (Self(dir), length))
            .ok_or(InvalidDirectionError::from_length(length))
    }

    /// Create a direction from its `x`, `y`, `z`, and `w` components.
    ///
    /// Returns [`Err(InvalidDirectionError)`](InvalidDirectionError) if the length
    /// of the vector formed by the components is zero (or very close to zero), infinite, or `NaN`.
    pub fn from_xyzw(x: f32, y: f32, z: f32, w: f32) -> Result<Self, InvalidDirectionError> {
        Self::new(Vec4::new(x, y, z, w))
    }

    /// Create a direction from its `x`, `y`, `z`, and `w` components, assuming the resulting vector is normalized.
    ///
    /// # Warning
    ///
    /// The vector produced from `x`, `y`, `z`, and `w` must be normalized, i.e its length must be `1.0`.
    pub fn from_xyzw_unchecked(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self::new_unchecked(Vec4::new(x, y, z, w))
    }

    /// Returns the inner [`Vec4`]
    pub const fn as_vec4(&self) -> Vec4 {
        self.0
    }

    /// Returns `self` after an approximate normalization, assuming the value is already nearly normalized.
    /// Useful for preventing numerical error accumulation.
    #[inline]
    pub fn fast_renormalize(self) -> Self {
        // We numerically approximate the inverse square root by a Taylor series around 1
        // As we expect the error (x := length_squared - 1) to be small
        // inverse_sqrt(length_squared) = (1 + x)^(-1/2) = 1 - 1/2 x + O(x²)
        // inverse_sqrt(length_squared) ≈ 1 - 1/2 (length_squared - 1) = 1/2 (3 - length_squared)

        // Iterative calls to this method quickly converge to a normalized value,
        // so long as the denormalization is not large ~ O(1/10).
        // One iteration can be described as:
        // l_sq <- l_sq * (1 - 1/2 (l_sq - 1))²;
        // Rewriting in terms of the error x:
        // 1 + x <- (1 + x) * (1 - 1/2 x)²
        // 1 + x <- (1 + x) * (1 - x + 1/4 x²)
        // 1 + x <- 1 - x + 1/4 x² + x - x² + 1/4 x³
        // x <- -1/4 x² (3 - x)
        // If the error is small, say in a range of (-1/2, 1/2), then:
        // |-1/4 x² (3 - x)| <= (3/4 + 1/4 * |x|) * x² <= (3/4 + 1/4 * 1/2) * x² < x² < 1/2 x
        // Therefore the sequence of iterates converges to 0 error as a second order method.

        let length_squared = self.0.length_squared();
        Self(self * (0.5 * (3.0 - length_squared)))
    }
}

impl TryFrom<Vec4> for Dir4 {
    type Error = InvalidDirectionError;

    fn try_from(value: Vec4) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl core::ops::Deref for Dir4 {
    type Target = Vec4;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl core::ops::Neg for Dir4 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl core::ops::Mul<f32> for Dir4 {
    type Output = Vec4;
    fn mul(self, rhs: f32) -> Self::Output {
        self.0 * rhs
    }
}

impl core::ops::Mul<Dir4> for f32 {
    type Output = Vec4;
    fn mul(self, rhs: Dir4) -> Self::Output {
        self * rhs.0
    }
}

impl fmt::Display for Dir4 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(feature = "approx")]
impl approx::AbsDiffEq for Dir4 {
    type Epsilon = f32;
    fn default_epsilon() -> f32 {
        f32::EPSILON
    }
    fn abs_diff_eq(&self, other: &Self, epsilon: f32) -> bool {
        self.as_ref().abs_diff_eq(other.as_ref(), epsilon)
    }
}

#[cfg(feature = "approx")]
impl approx::RelativeEq for Dir4 {
    fn default_max_relative() -> f32 {
        f32::EPSILON
    }
    fn relative_eq(&self, other: &Self, epsilon: f32, max_relative: f32) -> bool {
        self.as_ref()
            .relative_eq(other.as_ref(), epsilon, max_relative)
    }
}

#[cfg(feature = "approx")]
impl approx::UlpsEq for Dir4 {
    fn default_max_ulps() -> u32 {
        4
    }

    fn ulps_eq(&self, other: &Self, epsilon: f32, max_ulps: u32) -> bool {
        self.as_ref().ulps_eq(other.as_ref(), epsilon, max_ulps)
    }
}

#[cfg(test)]
#[cfg(feature = "approx")]
mod tests {
    use crate::ops;

    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn dir2_creation() {
        assert_eq!(Dir2::new(Vec2::X * 12.5), Ok(Dir2::X));
        assert_eq!(
            Dir2::new(Vec2::new(0.0, 0.0)),
            Err(InvalidDirectionError::Zero)
        );
        assert_eq!(
            Dir2::new(Vec2::new(f32::INFINITY, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir2::new(Vec2::new(f32::NEG_INFINITY, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir2::new(Vec2::new(f32::NAN, 0.0)),
            Err(InvalidDirectionError::NaN)
        );
        assert_eq!(Dir2::new_and_length(Vec2::X * 6.5), Ok((Dir2::X, 6.5)));
    }

    #[test]
    fn dir2_slerp() {
        assert_relative_eq!(
            Dir2::X.slerp(Dir2::Y, 0.5),
            Dir2::from_xy(ops::sqrt(0.5_f32), ops::sqrt(0.5_f32)).unwrap()
        );
        assert_eq!(Dir2::Y.slerp(Dir2::X, 0.0), Dir2::Y);
        assert_relative_eq!(Dir2::X.slerp(Dir2::Y, 1.0), Dir2::Y);
        assert_relative_eq!(
            Dir2::Y.slerp(Dir2::X, 1.0 / 3.0),
            Dir2::from_xy(0.5, ops::sqrt(0.75_f32)).unwrap()
        );
        assert_relative_eq!(
            Dir2::X.slerp(Dir2::Y, 2.0 / 3.0),
            Dir2::from_xy(0.5, ops::sqrt(0.75_f32)).unwrap()
        );
    }

    #[test]
    fn dir2_to_rotation2d() {
        assert_relative_eq!(Dir2::EAST.rotation_to(Dir2::NORTH_EAST), Rot2::FRAC_PI_4);
        assert_relative_eq!(Dir2::NORTH.rotation_from(Dir2::NORTH_EAST), Rot2::FRAC_PI_4);
        assert_relative_eq!(Dir2::SOUTH.rotation_to_x(), Rot2::FRAC_PI_2);
        assert_relative_eq!(Dir2::SOUTH.rotation_to_y(), Rot2::PI);
        assert_relative_eq!(Dir2::NORTH_WEST.rotation_from_x(), Rot2::degrees(135.0));
        assert_relative_eq!(Dir2::NORTH_WEST.rotation_from_y(), Rot2::FRAC_PI_4);
    }

    #[test]
    fn dir2_renorm() {
        // Evil denormalized Rot2
        let (sin, cos) = ops::sin_cos(1.0_f32);
        let rot2 = Rot2::from_sin_cos(sin * (1.0 + 1e-5), cos * (1.0 + 1e-5));
        let mut dir_a = Dir2::X;
        let mut dir_b = Dir2::X;

        // We test that renormalizing an already normalized dir doesn't do anything
        assert_relative_eq!(dir_b, dir_b.fast_renormalize(), epsilon = 0.000001);

        for _ in 0..50 {
            dir_a = rot2 * dir_a;
            dir_b = rot2 * dir_b;
            dir_b = dir_b.fast_renormalize();
        }

        // `dir_a` should've gotten denormalized, meanwhile `dir_b` should stay normalized.
        assert!(
            !dir_a.is_normalized(),
            "Denormalization doesn't work, test is faulty"
        );
        assert!(dir_b.is_normalized(), "Renormalisation did not work.");
    }

    #[test]
    fn dir3_creation() {
        assert_eq!(Dir3::new(Vec3::X * 12.5), Ok(Dir3::X));
        assert_eq!(
            Dir3::new(Vec3::new(0.0, 0.0, 0.0)),
            Err(InvalidDirectionError::Zero)
        );
        assert_eq!(
            Dir3::new(Vec3::new(f32::INFINITY, 0.0, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir3::new(Vec3::new(f32::NEG_INFINITY, 0.0, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir3::new(Vec3::new(f32::NAN, 0.0, 0.0)),
            Err(InvalidDirectionError::NaN)
        );
        assert_eq!(Dir3::new_and_length(Vec3::X * 6.5), Ok((Dir3::X, 6.5)));

        // Test rotation
        assert!(
            (Quat::from_rotation_z(core::f32::consts::FRAC_PI_2) * Dir3::X)
                .abs_diff_eq(Vec3::Y, 10e-6)
        );
    }

    #[test]
    fn dir3_slerp() {
        assert_relative_eq!(
            Dir3::X.slerp(Dir3::Y, 0.5),
            Dir3::from_xyz(ops::sqrt(0.5f32), ops::sqrt(0.5f32), 0.0).unwrap()
        );
        assert_relative_eq!(Dir3::Y.slerp(Dir3::Z, 0.0), Dir3::Y);
        assert_relative_eq!(Dir3::Z.slerp(Dir3::X, 1.0), Dir3::X, epsilon = 0.000001);
        assert_relative_eq!(
            Dir3::X.slerp(Dir3::Z, 1.0 / 3.0),
            Dir3::from_xyz(ops::sqrt(0.75f32), 0.0, 0.5).unwrap(),
            epsilon = 0.000001
        );
        assert_relative_eq!(
            Dir3::Z.slerp(Dir3::Y, 2.0 / 3.0),
            Dir3::from_xyz(0.0, ops::sqrt(0.75f32), 0.5).unwrap()
        );
    }

    #[test]
    fn dir3_renorm() {
        // Evil denormalized quaternion
        let rot3 = Quat::from_euler(glam::EulerRot::XYZ, 1.0, 2.0, 3.0) * (1.0 + 1e-5);
        let mut dir_a = Dir3::X;
        let mut dir_b = Dir3::X;

        // We test that renormalizing an already normalized dir doesn't do anything
        assert_relative_eq!(dir_b, dir_b.fast_renormalize(), epsilon = 0.000001);

        for _ in 0..50 {
            dir_a = rot3 * dir_a;
            dir_b = rot3 * dir_b;
            dir_b = dir_b.fast_renormalize();
        }

        // `dir_a` should've gotten denormalized, meanwhile `dir_b` should stay normalized.
        assert!(
            !dir_a.is_normalized(),
            "Denormalization doesn't work, test is faulty"
        );
        assert!(dir_b.is_normalized(), "Renormalisation did not work.");
    }

    #[test]
    fn dir3a_creation() {
        assert_eq!(Dir3A::new(Vec3A::X * 12.5), Ok(Dir3A::X));
        assert_eq!(
            Dir3A::new(Vec3A::new(0.0, 0.0, 0.0)),
            Err(InvalidDirectionError::Zero)
        );
        assert_eq!(
            Dir3A::new(Vec3A::new(f32::INFINITY, 0.0, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir3A::new(Vec3A::new(f32::NEG_INFINITY, 0.0, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir3A::new(Vec3A::new(f32::NAN, 0.0, 0.0)),
            Err(InvalidDirectionError::NaN)
        );
        assert_eq!(Dir3A::new_and_length(Vec3A::X * 6.5), Ok((Dir3A::X, 6.5)));

        // Test rotation
        assert!(
            (Quat::from_rotation_z(core::f32::consts::FRAC_PI_2) * Dir3A::X)
                .abs_diff_eq(Vec3A::Y, 10e-6)
        );
    }

    #[test]
    fn dir3a_slerp() {
        assert_relative_eq!(
            Dir3A::X.slerp(Dir3A::Y, 0.5),
            Dir3A::from_xyz(ops::sqrt(0.5f32), ops::sqrt(0.5f32), 0.0).unwrap()
        );
        assert_relative_eq!(Dir3A::Y.slerp(Dir3A::Z, 0.0), Dir3A::Y);
        assert_relative_eq!(Dir3A::Z.slerp(Dir3A::X, 1.0), Dir3A::X, epsilon = 0.000001);
        assert_relative_eq!(
            Dir3A::X.slerp(Dir3A::Z, 1.0 / 3.0),
            Dir3A::from_xyz(ops::sqrt(0.75f32), 0.0, 0.5).unwrap(),
            epsilon = 0.000001
        );
        assert_relative_eq!(
            Dir3A::Z.slerp(Dir3A::Y, 2.0 / 3.0),
            Dir3A::from_xyz(0.0, ops::sqrt(0.75f32), 0.5).unwrap()
        );
    }

    #[test]
    fn dir3a_renorm() {
        // Evil denormalized quaternion
        let rot3 = Quat::from_euler(glam::EulerRot::XYZ, 1.0, 2.0, 3.0) * (1.0 + 1e-5);
        let mut dir_a = Dir3A::X;
        let mut dir_b = Dir3A::X;

        // We test that renormalizing an already normalized dir doesn't do anything
        assert_relative_eq!(dir_b, dir_b.fast_renormalize(), epsilon = 0.000001);

        for _ in 0..50 {
            dir_a = rot3 * dir_a;
            dir_b = rot3 * dir_b;
            dir_b = dir_b.fast_renormalize();
        }

        // `dir_a` should've gotten denormalized, meanwhile `dir_b` should stay normalized.
        assert!(
            !dir_a.is_normalized(),
            "Denormalization doesn't work, test is faulty"
        );
        assert!(dir_b.is_normalized(), "Renormalisation did not work.");
    }

    #[test]
    fn dir4_creation() {
        assert_eq!(Dir4::new(Vec4::X * 12.5), Ok(Dir4::X));
        assert_eq!(
            Dir4::new(Vec4::new(0.0, 0.0, 0.0, 0.0)),
            Err(InvalidDirectionError::Zero)
        );
        assert_eq!(
            Dir4::new(Vec4::new(f32::INFINITY, 0.0, 0.0, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir4::new(Vec4::new(f32::NEG_INFINITY, 0.0, 0.0, 0.0)),
            Err(InvalidDirectionError::Infinite)
        );
        assert_eq!(
            Dir4::new(Vec4::new(f32::NAN, 0.0, 0.0, 0.0)),
            Err(InvalidDirectionError::NaN)
        );
        assert_eq!(Dir4::new_and_length(Vec4::X * 6.5), Ok((Dir4::X, 6.5)));
    }

    #[test]
    fn dir4_renorm() {
        // Evil denormalized matrix
        let mat4 = bevy_math::Mat4::from_quat(Quat::from_euler(glam::EulerRot::XYZ, 1.0, 2.0, 3.0))
            * (1.0 + 1e-5);
        let mut dir_a = Dir4::from_xyzw(1., 1., 0., 0.).unwrap();
        let mut dir_b = Dir4::from_xyzw(1., 1., 0., 0.).unwrap();
        // We test that renormalizing an already normalized dir doesn't do anything
        assert_relative_eq!(dir_b, dir_b.fast_renormalize(), epsilon = 0.000001);
        for _ in 0..50 {
            dir_a = Dir4(mat4 * *dir_a);
            dir_b = Dir4(mat4 * *dir_b);
            dir_b = dir_b.fast_renormalize();
        }
        // `dir_a` should've gotten denormalized, meanwhile `dir_b` should stay normalized.
        assert!(
            !dir_a.is_normalized(),
            "Denormalization doesn't work, test is faulty"
        );
        assert!(dir_b.is_normalized(), "Renormalisation did not work.");
    }
}
