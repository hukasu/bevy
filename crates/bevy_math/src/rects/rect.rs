use crate::{IRect, URect, Vec2};

#[cfg(feature = "bevy_reflect")]
use bevy_reflect::{std_traits::ReflectDefault, Reflect};
#[cfg(all(feature = "serialize", feature = "bevy_reflect"))]
use bevy_reflect::{ReflectDeserialize, ReflectSerialize};

/// A rectangle defined by two opposite corners.
///
/// The rectangle is axis aligned, and defined by its minimum and maximum coordinates,
/// stored in `Rect::min` and `Rect::max`, respectively. The minimum/maximum invariant
/// must be upheld by the user when directly assigning the fields, otherwise some methods
/// produce invalid results. It is generally recommended to use one of the constructor
/// methods instead, which will ensure this invariant is met, unless you already have
/// the minimum and maximum corners.
#[repr(C)]
#[derive(Default, Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bevy_reflect",
    derive(Reflect),
    reflect(Debug, PartialEq, Default, Clone)
)]
#[cfg_attr(
    all(feature = "serialize", feature = "bevy_reflect"),
    reflect(Serialize, Deserialize)
)]
pub struct Rect {
    /// The minimum corner point of the rect.
    pub min: Vec2,
    /// The maximum corner point of the rect.
    pub max: Vec2,
}

impl Rect {
    /// An empty `Rect`, represented by maximum and minimum corner points
    /// at `Vec2::NEG_INFINITY` and `Vec2::INFINITY`, respectively.
    /// This is so the `Rect` has a infinitely negative size.
    /// This is useful, because when taking a union B of a non-empty `Rect` A and
    /// this empty `Rect`, B will simply equal A.
    pub const EMPTY: Self = Self {
        max: Vec2::NEG_INFINITY,
        min: Vec2::INFINITY,
    };
    /// Create a new rectangle from two corner points.
    ///
    /// The two points do not need to be the minimum and/or maximum corners.
    /// They only need to be two opposite corners.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Rect;
    /// let r = Rect::new(0., 4., 10., 6.); // w=10 h=2
    /// let r = Rect::new(2., 3., 5., -1.); // w=3 h=4
    /// ```
    #[inline]
    pub fn new(x0: f32, y0: f32, x1: f32, y1: f32) -> Self {
        Self::from_corners(Vec2::new(x0, y0), Vec2::new(x1, y1))
    }

    /// Create a new rectangle from two corner points.
    ///
    /// The two points do not need to be the minimum and/or maximum corners.
    /// They only need to be two opposite corners.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// // Unit rect from [0,0] to [1,1]
    /// let r = Rect::from_corners(Vec2::ZERO, Vec2::ONE); // w=1 h=1
    /// // Same; the points do not need to be ordered
    /// let r = Rect::from_corners(Vec2::ONE, Vec2::ZERO); // w=1 h=1
    /// ```
    #[inline]
    pub fn from_corners(p0: Vec2, p1: Vec2) -> Self {
        Self {
            min: p0.min(p1),
            max: p0.max(p1),
        }
    }

    /// Create a new rectangle from its center and size.
    ///
    /// # Panics
    ///
    /// This method panics if any of the components of the size is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::from_center_size(Vec2::ZERO, Vec2::ONE); // w=1 h=1
    /// assert!(r.min.abs_diff_eq(Vec2::splat(-0.5), 1e-5));
    /// assert!(r.max.abs_diff_eq(Vec2::splat(0.5), 1e-5));
    /// ```
    #[inline]
    pub fn from_center_size(origin: Vec2, size: Vec2) -> Self {
        assert!(size.cmpge(Vec2::ZERO).all(), "Rect size must be positive");
        let half_size = size / 2.;
        Self::from_center_half_size(origin, half_size)
    }

    /// Create a new rectangle from its center and half-size.
    ///
    /// # Panics
    ///
    /// This method panics if any of the components of the half-size is negative.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::from_center_half_size(Vec2::ZERO, Vec2::ONE); // w=2 h=2
    /// assert!(r.min.abs_diff_eq(Vec2::splat(-1.), 1e-5));
    /// assert!(r.max.abs_diff_eq(Vec2::splat(1.), 1e-5));
    /// ```
    #[inline]
    pub fn from_center_half_size(origin: Vec2, half_size: Vec2) -> Self {
        assert!(
            half_size.cmpge(Vec2::ZERO).all(),
            "Rect half_size must be positive"
        );
        Self {
            min: origin - half_size,
            max: origin + half_size,
        }
    }

    /// Check if the rectangle is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::from_corners(Vec2::ZERO, Vec2::new(0., 1.)); // w=0 h=1
    /// assert!(r.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.cmpge(self.max).any()
    }

    /// Rectangle width (max.x - min.x).
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Rect;
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// assert!((r.width() - 5.).abs() <= 1e-5);
    /// ```
    #[inline]
    pub fn width(&self) -> f32 {
        self.max.x - self.min.x
    }

    /// Rectangle height (max.y - min.y).
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Rect;
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// assert!((r.height() - 1.).abs() <= 1e-5);
    /// ```
    #[inline]
    pub fn height(&self) -> f32 {
        self.max.y - self.min.y
    }

    /// Rectangle size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// assert!(r.size().abs_diff_eq(Vec2::new(5., 1.), 1e-5));
    /// ```
    #[inline]
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// Rectangle half-size.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// assert!(r.half_size().abs_diff_eq(Vec2::new(2.5, 0.5), 1e-5));
    /// ```
    #[inline]
    pub fn half_size(&self) -> Vec2 {
        self.size() * 0.5
    }

    /// The center point of the rectangle.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// assert!(r.center().abs_diff_eq(Vec2::new(2.5, 0.5), 1e-5));
    /// ```
    #[inline]
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Check if a point lies within this rectangle, inclusive of its edges.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::Rect;
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// assert!(r.contains(r.center()));
    /// assert!(r.contains(r.min));
    /// assert!(r.contains(r.max));
    /// ```
    #[inline]
    pub fn contains(&self, point: Vec2) -> bool {
        (point.cmpge(self.min) & point.cmple(self.max)).all()
    }

    /// Build a new rectangle formed of the union of this rectangle and another rectangle.
    ///
    /// The union is the smallest rectangle enclosing both rectangles.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r1 = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// let r2 = Rect::new(1., -1., 3., 3.); // w=2 h=4
    /// let r = r1.union(r2);
    /// assert!(r.min.abs_diff_eq(Vec2::new(0., -1.), 1e-5));
    /// assert!(r.max.abs_diff_eq(Vec2::new(5., 3.), 1e-5));
    /// ```
    #[inline]
    pub fn union(&self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Build a new rectangle formed of the union of this rectangle and a point.
    ///
    /// The union is the smallest rectangle enclosing both the rectangle and the point. If the
    /// point is already inside the rectangle, this method returns a copy of the rectangle.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// let u = r.union_point(Vec2::new(3., 6.));
    /// assert!(u.min.abs_diff_eq(Vec2::ZERO, 1e-5));
    /// assert!(u.max.abs_diff_eq(Vec2::new(5., 6.), 1e-5));
    /// ```
    #[inline]
    pub fn union_point(&self, other: Vec2) -> Self {
        Self {
            min: self.min.min(other),
            max: self.max.max(other),
        }
    }

    /// Build a new rectangle formed of the intersection of this rectangle and another rectangle.
    ///
    /// The intersection is the largest rectangle enclosed in both rectangles. If the intersection
    /// is empty, this method returns an empty rectangle ([`Rect::is_empty()`] returns `true`), but
    /// the actual values of [`Rect::min`] and [`Rect::max`] are implementation-dependent.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r1 = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// let r2 = Rect::new(1., -1., 3., 3.); // w=2 h=4
    /// let r = r1.intersect(r2);
    /// assert!(r.min.abs_diff_eq(Vec2::new(1., 0.), 1e-5));
    /// assert!(r.max.abs_diff_eq(Vec2::new(3., 1.), 1e-5));
    /// ```
    #[inline]
    pub fn intersect(&self, other: Self) -> Self {
        let mut r = Self {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        };
        // Collapse min over max to enforce invariants and ensure e.g. width() or
        // height() never return a negative value.
        r.min = r.min.min(r.max);
        r
    }

    /// Create a new rectangle by expanding it evenly on all sides.
    ///
    /// A positive expansion value produces a larger rectangle,
    /// while a negative expansion value produces a smaller rectangle.
    /// If this would result in zero or negative width or height, [`Rect::EMPTY`] is returned instead.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::new(0., 0., 5., 1.); // w=5 h=1
    /// let r2 = r.inflate(3.); // w=11 h=7
    /// assert!(r2.min.abs_diff_eq(Vec2::splat(-3.), 1e-5));
    /// assert!(r2.max.abs_diff_eq(Vec2::new(8., 4.), 1e-5));
    ///
    /// let r = Rect::new(0., -1., 6., 7.); // w=6 h=8
    /// let r2 = r.inflate(-2.); // w=11 h=7
    /// assert!(r2.min.abs_diff_eq(Vec2::new(2., 1.), 1e-5));
    /// assert!(r2.max.abs_diff_eq(Vec2::new(4., 5.), 1e-5));
    /// ```
    #[inline]
    pub fn inflate(&self, expansion: f32) -> Self {
        let mut r = Self {
            min: self.min - expansion,
            max: self.max + expansion,
        };
        // Collapse min over max to enforce invariants and ensure e.g. width() or
        // height() never return a negative value.
        r.min = r.min.min(r.max);
        r
    }

    /// Build a new rectangle from this one with its coordinates expressed
    /// relative to `other` in a normalized ([0..1] x [0..1]) coordinate system.
    ///
    /// # Examples
    ///
    /// ```
    /// # use bevy_math::{Rect, Vec2};
    /// let r = Rect::new(2., 3., 4., 6.);
    /// let s = Rect::new(0., 0., 10., 10.);
    /// let n = r.normalize(s);
    ///
    /// assert_eq!(n.min.x, 0.2);
    /// assert_eq!(n.min.y, 0.3);
    /// assert_eq!(n.max.x, 0.4);
    /// assert_eq!(n.max.y, 0.6);
    /// ```
    pub fn normalize(&self, other: Self) -> Self {
        let outer_size = other.size();
        Self {
            min: (self.min - other.min) / outer_size,
            max: (self.max - other.min) / outer_size,
        }
    }

    /// Returns self as [`IRect`] (i32)
    #[inline]
    pub fn as_irect(&self) -> IRect {
        IRect::from_corners(self.min.as_ivec2(), self.max.as_ivec2())
    }

    /// Returns self as [`URect`] (u32)
    #[inline]
    pub fn as_urect(&self) -> URect {
        URect::from_corners(self.min.as_uvec2(), self.max.as_uvec2())
    }
}

#[cfg(test)]
mod tests {
    use crate::ops;

    use super::*;

    #[test]
    fn well_formed() {
        let r = Rect::from_center_size(Vec2::new(3., -5.), Vec2::new(8., 11.));

        assert!(r.min.abs_diff_eq(Vec2::new(-1., -10.5), 1e-5));
        assert!(r.max.abs_diff_eq(Vec2::new(7., 0.5), 1e-5));

        assert!(r.center().abs_diff_eq(Vec2::new(3., -5.), 1e-5));

        assert!(ops::abs(r.width() - 8.) <= 1e-5);
        assert!(ops::abs(r.height() - 11.) <= 1e-5);
        assert!(r.size().abs_diff_eq(Vec2::new(8., 11.), 1e-5));
        assert!(r.half_size().abs_diff_eq(Vec2::new(4., 5.5), 1e-5));

        assert!(r.contains(Vec2::new(3., -5.)));
        assert!(r.contains(Vec2::new(-1., -10.5)));
        assert!(r.contains(Vec2::new(-1., 0.5)));
        assert!(r.contains(Vec2::new(7., -10.5)));
        assert!(r.contains(Vec2::new(7., 0.5)));
        assert!(!r.contains(Vec2::new(50., -5.)));
    }

    #[test]
    fn rect_union() {
        let r = Rect::from_center_size(Vec2::ZERO, Vec2::ONE); // [-0.5,-0.5] - [0.5,0.5]

        // overlapping
        let r2 = Rect {
            min: Vec2::new(-0.8, 0.3),
            max: Vec2::new(0.1, 0.7),
        };
        let u = r.union(r2);
        assert!(u.min.abs_diff_eq(Vec2::new(-0.8, -0.5), 1e-5));
        assert!(u.max.abs_diff_eq(Vec2::new(0.5, 0.7), 1e-5));

        // disjoint
        let r2 = Rect {
            min: Vec2::new(-1.8, -0.5),
            max: Vec2::new(-1.5, 0.3),
        };
        let u = r.union(r2);
        assert!(u.min.abs_diff_eq(Vec2::new(-1.8, -0.5), 1e-5));
        assert!(u.max.abs_diff_eq(Vec2::new(0.5, 0.5), 1e-5));

        // included
        let r2 = Rect::from_center_size(Vec2::ZERO, Vec2::splat(0.5));
        let u = r.union(r2);
        assert!(u.min.abs_diff_eq(r.min, 1e-5));
        assert!(u.max.abs_diff_eq(r.max, 1e-5));

        // including
        let r2 = Rect::from_center_size(Vec2::ZERO, Vec2::splat(1.5));
        let u = r.union(r2);
        assert!(u.min.abs_diff_eq(r2.min, 1e-5));
        assert!(u.max.abs_diff_eq(r2.max, 1e-5));
    }

    #[test]
    fn rect_union_pt() {
        let r = Rect::from_center_size(Vec2::ZERO, Vec2::ONE); // [-0.5,-0.5] - [0.5,0.5]

        // inside
        let v = Vec2::new(0.3, -0.2);
        let u = r.union_point(v);
        assert!(u.min.abs_diff_eq(r.min, 1e-5));
        assert!(u.max.abs_diff_eq(r.max, 1e-5));

        // outside
        let v = Vec2::new(10., -3.);
        let u = r.union_point(v);
        assert!(u.min.abs_diff_eq(Vec2::new(-0.5, -3.), 1e-5));
        assert!(u.max.abs_diff_eq(Vec2::new(10., 0.5), 1e-5));
    }

    #[test]
    fn rect_intersect() {
        let r = Rect::from_center_size(Vec2::ZERO, Vec2::ONE); // [-0.5,-0.5] - [0.5,0.5]

        // overlapping
        let r2 = Rect {
            min: Vec2::new(-0.8, 0.3),
            max: Vec2::new(0.1, 0.7),
        };
        let u = r.intersect(r2);
        assert!(u.min.abs_diff_eq(Vec2::new(-0.5, 0.3), 1e-5));
        assert!(u.max.abs_diff_eq(Vec2::new(0.1, 0.5), 1e-5));

        // disjoint
        let r2 = Rect {
            min: Vec2::new(-1.8, -0.5),
            max: Vec2::new(-1.5, 0.3),
        };
        let u = r.intersect(r2);
        assert!(u.is_empty());
        assert!(u.width() <= 1e-5);

        // included
        let r2 = Rect::from_center_size(Vec2::ZERO, Vec2::splat(0.5));
        let u = r.intersect(r2);
        assert!(u.min.abs_diff_eq(r2.min, 1e-5));
        assert!(u.max.abs_diff_eq(r2.max, 1e-5));

        // including
        let r2 = Rect::from_center_size(Vec2::ZERO, Vec2::splat(1.5));
        let u = r.intersect(r2);
        assert!(u.min.abs_diff_eq(r.min, 1e-5));
        assert!(u.max.abs_diff_eq(r.max, 1e-5));
    }

    #[test]
    fn rect_inflate() {
        let r = Rect::from_center_size(Vec2::ZERO, Vec2::ONE); // [-0.5,-0.5] - [0.5,0.5]

        let r2 = r.inflate(0.3);
        assert!(r2.min.abs_diff_eq(Vec2::new(-0.8, -0.8), 1e-5));
        assert!(r2.max.abs_diff_eq(Vec2::new(0.8, 0.8), 1e-5));
    }
}
