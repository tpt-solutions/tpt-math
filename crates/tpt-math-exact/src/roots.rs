//! Error-bounded root isolation for scalar functions, on exact intervals.
//!
//! A root finder over `f64` returns one approximate root and no guarantee that
//! it is the only one. Interval bisection works differently: it returns boxes
//! that provably contain every root of the function in the starting interval,
//! and it discards boxes that provably contain none. The guarantee rests on an
//! *interval extension* `F` of the function: a map from an interval `X` to an
//! interval that contains `{ f(x) : x ∈ X }`. If `F(X)` excludes zero, `X` holds
//! no root.
//!
//! The boxes returned by [`isolate_roots`] are closed intervals that may share
//! endpoints, so a root that sits exactly on a split point appears in both.
//!
//! [`poly_enclosure`] builds a valid extension for any polynomial, by Horner's
//! rule in interval arithmetic. [`has_sign_change`] gives a separate certificate
//! that a continuous function has a root in an interval.

use alloc::vec;
use alloc::vec::Vec;

use crate::{Interval, Rational};

/// Isolate the roots of a function in `[lo, hi]` to within `min_width`.
///
/// `enclosure` must be an interval extension of the function. The search splits
/// a box in half while its enclosure contains zero and its width exceeds
/// `min_width`. It stops early at `max_depth` splits, returning any box still
/// undecided at that point.
///
/// Returns the undecided boxes, sorted by their lower endpoint. Every root of the
/// function in `[lo, hi]` lies in at least one returned box. A returned box may
/// contain no root, because the enclosure is only an over-estimate.
///
/// # Panics
///
/// Panics if `lo > hi`.
///
/// # Examples
///
/// ```
/// use tpt_math_exact::roots::{isolate_roots, poly_enclosure};
/// use tpt_math_exact::{BigInt, BigRational};
///
/// let r = |n: i64| BigRational::from_integer(BigInt::from(n));
/// // f(x) = x² − 1 has roots −1 and 1 in [−2, 2].
/// let f = poly_enclosure(vec![r(-1), r(0), r(1)]);
/// let boxes = isolate_roots(f, r(-2), r(2), BigRational::new(1.into(), 1000.into()), 64);
/// assert!(boxes.len() >= 2);
/// ```
pub fn isolate_roots<F>(
    enclosure: F,
    lo: Rational,
    hi: Rational,
    min_width: Rational,
    max_depth: usize,
) -> Vec<Interval<Rational>>
where
    F: Fn(&Interval<Rational>) -> Interval<Rational>,
{
    let zero = Rational::from_integer(0.into());
    let mut stack: Vec<(Interval<Rational>, usize)> = vec![(Interval::new(lo, hi), 0)];
    let mut boxes = Vec::new();

    while let Some((current, depth)) = stack.pop() {
        let value = enclosure(&current);
        // Certainly no root: the enclosure lies strictly on one side of zero.
        if *value.lo() > zero || *value.hi() < zero {
            continue;
        }
        let width = current.width();
        if width <= min_width || depth >= max_depth {
            boxes.push(current);
            continue;
        }
        let mid = current.midpoint();
        stack.push((Interval::new(mid.clone(), current.hi().clone()), depth + 1));
        stack.push((Interval::new(current.lo().clone(), mid), depth + 1));
    }

    boxes.sort_by(|a, b| a.lo().cmp(b.lo()));
    boxes
}

/// Interval extension of the polynomial `Σ cᵢ xⁱ`, evaluated by Horner's rule.
///
/// `coeffs` lists the coefficients from the constant term upwards. An empty
/// vector gives the zero polynomial. The closure owns the coefficients. It is a
/// valid interval extension for [`isolate_roots`]: each Horner step uses exact
/// interval addition and multiplication, so the result always contains the true
/// range.
pub fn poly_enclosure(coeffs: Vec<Rational>) -> impl Fn(&Interval<Rational>) -> Interval<Rational> {
    move |x: &Interval<Rational>| {
        let Some((leading, rest)) = coeffs.split_last() else {
            let zero = Rational::from_integer(0.into());
            return Interval::new(zero.clone(), zero);
        };
        let mut acc = Interval::new(leading.clone(), leading.clone());
        for c in rest.iter().rev() {
            acc = acc * x.clone() + Interval::new(c.clone(), c.clone());
        }
        acc
    }
}

/// Evaluate the polynomial `Σ cᵢ xⁱ` at a point, by Horner's rule.
///
/// Empty `coeffs` give zero.
pub fn poly_eval(coeffs: &[Rational], x: &Rational) -> Rational {
    coeffs
        .iter()
        .rev()
        .fold(Rational::from_integer(0.into()), |acc, c| acc * x + c)
}

/// `true` if `f` takes strictly opposite signs at the ends of `[lo, hi]`.
///
/// By the intermediate value theorem, this certifies a root in the interval for
/// a continuous `f`. It does not rule out roots when it returns `false`, and it
/// says nothing for a discontinuous `f`.
///
/// # Examples
///
/// ```
/// use tpt_math_exact::roots::{has_sign_change, poly_eval};
/// use tpt_math_exact::{BigInt, BigRational};
///
/// let r = |n: i64| BigRational::from_integer(BigInt::from(n));
/// let coeffs = [r(-2), r(0), r(1)]; // x² − 2
/// let f = |x: &BigRational| poly_eval(&coeffs, x);
/// assert!(has_sign_change(f, &r(1), &r(2)));
/// assert!(!has_sign_change(f, &r(2), &r(3)));
/// ```
pub fn has_sign_change<F>(f: F, lo: &Rational, hi: &Rational) -> bool
where
    F: Fn(&Rational) -> Rational,
{
    let zero = Rational::from_integer(0.into());
    let (a, b) = (f(lo), f(hi));
    (a < zero && b > zero) || (a > zero && b < zero)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BigInt, BigRational};

    fn r(n: i64) -> Rational {
        BigRational::from_integer(BigInt::from(n))
    }

    fn frac(n: i64, d: i64) -> Rational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    fn encloses(boxes: &[Interval<Rational>], x: f64) -> bool {
        boxes.iter().any(|b| {
            let lo = num_traits::ToPrimitive::to_f64(b.lo()).unwrap();
            let hi = num_traits::ToPrimitive::to_f64(b.hi()).unwrap();
            lo <= x && x <= hi
        })
    }

    #[test]
    fn finds_both_roots_of_a_quadratic() {
        let coeffs = [r(-1), r(0), r(1)]; // x² − 1
        let boxes = isolate_roots(
            poly_enclosure(coeffs.to_vec()),
            r(-2),
            r(2),
            frac(1, 1000),
            64,
        );
        assert!(encloses(&boxes, -1.0));
        assert!(encloses(&boxes, 1.0));
        // The box count stays small: the search does not sweep the whole interval.
        assert!(boxes.len() < 20, "{} boxes", boxes.len());
    }

    #[test]
    fn irrational_root_is_enclosed_to_requested_width() {
        let coeffs = [r(-2), r(0), r(1)]; // x² − 2
        let boxes = isolate_roots(
            poly_enclosure(coeffs.to_vec()),
            r(0),
            r(2),
            frac(1, 10_000),
            64,
        );
        assert!(encloses(&boxes, core::f64::consts::SQRT_2));
        for b in &boxes {
            assert!(b.width() <= frac(1, 10_000));
        }
    }

    #[test]
    fn proves_absence_of_roots() {
        // x² + 1 has no real root. The enclosure over [−2, 2] is [−3, 5], which
        // contains zero, but both halves exclude it after one split.
        let coeffs = [r(1), r(0), r(1)];
        let boxes = isolate_roots(
            poly_enclosure(coeffs.to_vec()),
            r(-2),
            r(2),
            frac(1, 100),
            64,
        );
        assert!(boxes.is_empty());
    }

    #[test]
    fn depth_limit_returns_undecided_boxes_instead_of_splitting_further() {
        let coeffs = [r(-1), r(0), r(1)];
        let boxes = isolate_roots(
            poly_enclosure(coeffs.to_vec()),
            r(0),
            r(2),
            frac(1, 1_000_000),
            0,
        );
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].lo(), &r(0));
        assert_eq!(boxes[0].hi(), &r(2));
    }

    #[test]
    fn enclosure_contains_point_values() {
        // The Horner extension must enclose f(x) for every x in the box.
        let coeffs = [r(3), r(-2), r(0), r(1)]; // x³ − 2x + 3
        let f = poly_enclosure(coeffs.to_vec());
        let x = Interval::new(frac(-1, 2), frac(3, 2));
        let enclosure = f(&x);
        for k in 0..=20 {
            let p = frac(-1, 2) + frac(k, 20) * frac(2, 1);
            let v = poly_eval(&coeffs, &p);
            assert!(enclosure.lo() <= &v && &v <= enclosure.hi());
        }
    }

    #[test]
    fn empty_polynomial_is_zero_everywhere() {
        let f = poly_enclosure(Vec::new());
        let boxes = isolate_roots(f, r(-1), r(1), frac(1, 10), 8);
        assert!(!boxes.is_empty());
        assert_eq!(poly_eval(&[], &r(5)), r(0));
    }

    #[test]
    fn sign_change_certifies_root_in_interval() {
        let coeffs = [r(-2), r(0), r(1)];
        let f = |x: &Rational| poly_eval(&coeffs, x);
        assert!(has_sign_change(f, &r(1), &r(2)));
        assert!(!has_sign_change(f, &r(2), &r(3)));
        // A root exactly at an endpoint is not a strict sign change.
        assert!(!has_sign_change(|x: &Rational| x - &r(1), &r(1), &r(3)));
    }

    #[test]
    #[should_panic]
    fn inverted_search_interval_panics() {
        let _ = isolate_roots(poly_enclosure(vec![r(1)]), r(2), r(1), frac(1, 10), 4);
    }
}
