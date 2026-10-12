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

use crate::{BigInt, Interval, Rational};

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

/// Refine an enclosure of a root of `f` in `[lo, hi]` by interval Newton steps.
///
/// Each step computes `N(X) = m − f(m) / F′(X)` at the midpoint `m` of the
/// current box `X` and intersects it with `X`. The derivative enclosure `df`
/// must enclose `f′` on the box. If `df(X)` contains zero, the step is skipped
/// and the box is returned unchanged.
///
/// Returns `None` when a step proves that the box holds no root. Otherwise
/// returns the refined box, which contains every root in the starting interval
/// and has width at most `tol`, or the best box reached within `max_iter` steps.
///
/// # Examples
///
/// ```
/// use tpt_math_exact::roots::{newton_refine, poly_enclosure, poly_eval};
/// use tpt_math_exact::{BigInt, BigRational};
///
/// let r = |n: i64| BigRational::from_integer(BigInt::from(n));
/// let f_coeffs = vec![r(-2), r(0), r(1)]; // x² − 2
/// let df = poly_enclosure(vec![r(0), r(2)]); // 2x
/// let root = newton_refine(
///     |x: &BigRational| poly_eval(&f_coeffs, x),
///     df,
///     r(1),
///     r(2),
///     BigRational::new(1.into(), 1_000_000.into()),
///     32,
/// )
/// .unwrap();
/// assert!(root.width() <= BigRational::new(1.into(), 1_000_000.into()));
/// ```
pub fn newton_refine<F, D>(
    f: F,
    df: D,
    lo: Rational,
    hi: Rational,
    tol: Rational,
    max_iter: usize,
) -> Option<Interval<Rational>>
where
    F: Fn(&Rational) -> Rational,
    D: Fn(&Interval<Rational>) -> Interval<Rational>,
{
    let mut x = Interval::new(lo, hi);
    for _ in 0..max_iter {
        if x.width() <= tol {
            break;
        }
        let mid = x.midpoint();
        let fm = f(&mid);
        let slope = df(&x);
        let quotient = match Interval::new(fm.clone(), fm).div_nonzero(&slope) {
            Some(q) => q,
            // The slope contains zero, so Newton's step is not available here.
            None => return Some(x),
        };
        let newton = &point(mid) - &quotient;
        x = x.intersect(&newton)?;
    }
    Some(x)
}

/// Enclosure of the solution of `y′ = f(t, y)`, `y(t0) = y0`, at `t1`, by a
/// validated Euler method.
///
/// `f` takes an enclosure of `t` and an enclosure of `y`, and must return an
/// enclosure of `f` over both. Each step of size `h = (t1 − t0)/steps` first
/// finds an a-priori box `B` with `y + h·f(T, B) ⊆ B` (the Picard–Lindelöf
/// condition). The solution stays in `B` for the whole step, so the step's
/// endpoint enclosure `y + h·f(T, B)` is rigorous.
///
/// `inflation` widens the first candidate box so that the containment test can
/// succeed. It should be small compared with the step's variation. Returns
/// `None` if no a-priori box is found, which usually means the step is too large
/// or `f` is not Lipschitz on the box.
///
/// Width grows roughly linearly with `t1 − t0` times `inflation`, so the bound
/// is only as tight as the chosen `inflation` allows.
///
/// # Panics
///
/// Panics if `t1 ≤ t0` or `steps` is zero.
///
/// # Examples
///
/// ```
/// use tpt_math_exact::roots::validated_euler;
/// use tpt_math_exact::{BigInt, BigRational, Interval};
///
/// let r = |n: i64| BigRational::from_integer(BigInt::from(n));
/// let frac = |n: i64, d: i64| BigRational::new(BigInt::from(n), BigInt::from(d));
/// // y′ = y, y(0) = 1: the solution at t = 1 is e.
/// let f = |_t: &Interval<BigRational>, y: &Interval<BigRational>| y.clone();
/// let end = validated_euler(f, r(0), r(1), r(1), 20, frac(1, 1_000_000), 50).unwrap();
/// let e = core::f64::consts::E;
/// let lo = num_traits::ToPrimitive::to_f64(end.lo()).unwrap();
/// let hi = num_traits::ToPrimitive::to_f64(end.hi()).unwrap();
/// assert!(lo <= e && e <= hi);
/// ```
pub fn validated_euler<F>(
    f: F,
    t0: Rational,
    t1: Rational,
    y0: Rational,
    steps: usize,
    inflation: Rational,
    max_picard: usize,
) -> Option<Interval<Rational>>
where
    F: Fn(&Interval<Rational>, &Interval<Rational>) -> Interval<Rational>,
{
    assert!(t1 > t0, "validated_euler: t1 must exceed t0");
    assert!(steps > 0, "validated_euler: steps must be positive");
    let h = (t1.clone() - t0.clone()) / Rational::from_integer(BigInt::from(steps));
    let h_box = point(h.clone());
    let mut y = point(y0);
    for step in 0..steps {
        let t_lo = t0.clone() + h.clone() * Rational::from_integer(BigInt::from(step));
        let t_box = Interval::new(t_lo.clone(), t_lo + h.clone());

        // A-priori enclosure: start from the hull of y and its Euler image, widen
        // it, and iterate until the step maps the box into itself.
        let euler = y.clone() + h_box.clone() * f(&t_box, &y);
        let mut b = y.hull(&euler);
        b = Interval::new(
            b.lo().clone() - inflation.clone(),
            b.hi().clone() + inflation.clone(),
        );
        let mut found = false;
        let mut margin = inflation.clone();
        for _ in 0..max_picard {
            let candidate = y.clone() + h_box.clone() * f(&t_box, &b);
            if b.lo() <= candidate.lo() && candidate.hi() <= b.hi() {
                found = true;
                break;
            }
            // Not yet invariant: grow the box to cover the candidate, with a
            // margin that doubles on each attempt.
            let grown = candidate.hull(&y);
            b = Interval::new(
                grown.lo().clone() - margin.clone(),
                grown.hi().clone() + margin.clone(),
            );
            margin = margin.clone() + margin;
        }
        if !found {
            return None;
        }
        y = y + h_box.clone() * f(&t_box, &b);
    }
    Some(y)
}

/// Degenerate interval `[x, x]`.
fn point(x: Rational) -> Interval<Rational> {
    Interval::new(x.clone(), x)
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

#[cfg(test)]
mod newton_euler_tests {
    use super::*;
    use crate::{BigInt, BigRational};

    fn r(n: i64) -> Rational {
        BigRational::from_integer(BigInt::from(n))
    }

    fn frac(n: i64, d: i64) -> Rational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    #[test]
    fn newton_encloses_sqrt_two_to_requested_width() {
        let coeffs = vec![r(-2), r(0), r(1)];
        let df = poly_enclosure(vec![r(0), r(2)]);
        let root = newton_refine(
            |x: &Rational| poly_eval(&coeffs, x),
            df,
            r(1),
            r(2),
            frac(1, 1_000_000),
            64,
        )
        .unwrap();
        assert!(root.width() <= frac(1, 1_000_000));
        let s = core::f64::consts::SQRT_2;
        let lo = num_traits::ToPrimitive::to_f64(root.lo()).unwrap();
        let hi = num_traits::ToPrimitive::to_f64(root.hi()).unwrap();
        assert!(lo <= s && s <= hi);
    }

    #[test]
    fn newton_proves_absence_of_root() {
        // x² + 1 on [1, 2]: the Newton image of the box falls outside it.
        let coeffs = vec![r(1), r(0), r(1)];
        let df = poly_enclosure(vec![r(0), r(2)]);
        let out = newton_refine(
            |x: &Rational| poly_eval(&coeffs, x),
            df,
            r(1),
            r(2),
            frac(1, 1000),
            64,
        );
        assert!(out.is_none());
    }

    #[test]
    fn newton_skips_steps_when_slope_contains_zero() {
        // f′ = 2x contains zero on [−1, 1], so the box is returned unchanged.
        let coeffs = vec![r(-2), r(0), r(1)];
        let df = poly_enclosure(vec![r(0), r(2)]);
        let out = newton_refine(
            |x: &Rational| poly_eval(&coeffs, x),
            df,
            r(-1),
            r(1),
            frac(1, 1000),
            8,
        )
        .unwrap();
        assert_eq!(out.lo(), &r(-1));
        assert_eq!(out.hi(), &r(1));
    }

    #[test]
    fn euler_encloses_exponential_and_tightens_with_smaller_inflation() {
        let f = |_t: &Interval<Rational>, y: &Interval<Rational>| y.clone();
        let loose = validated_euler(f, r(0), r(1), r(1), 10, frac(1, 1000), 80).unwrap();
        let tight = validated_euler(f, r(0), r(1), r(1), 10, frac(1, 1_000_000), 80).unwrap();
        let e = core::f64::consts::E;
        let enc = |iv: &Interval<Rational>| {
            (
                num_traits::ToPrimitive::to_f64(iv.lo()).unwrap(),
                num_traits::ToPrimitive::to_f64(iv.hi()).unwrap(),
            )
        };
        let (lo, hi) = enc(&tight);
        assert!(lo <= e && e <= hi, "[{lo}, {hi}]");
        assert!(tight.width() < loose.width());
    }

    #[test]
    fn euler_reports_failure_when_no_a_priori_box_exists() {
        // With no room to grow (max_picard = 0) no box can be certified.
        let f = |_t: &Interval<Rational>, y: &Interval<Rational>| y.clone();
        assert!(validated_euler(f, r(0), r(1), r(1), 4, frac(1, 1000), 0).is_none());
    }

    #[test]
    #[should_panic(expected = "t1 must exceed t0")]
    fn euler_rejects_backwards_interval() {
        let f = |_t: &Interval<Rational>, y: &Interval<Rational>| y.clone();
        let _ = validated_euler(f, r(1), r(0), r(1), 4, frac(1, 1000), 8);
    }
}

/// Validated Euler for a system `y′ = f(t, y)` of `n` scalar equations.
///
/// The scalar [`validated_euler`] generalised to vectors. `f` takes an enclosure
/// of `t` and a slice of enclosures of the components `y`, and returns one
/// enclosure per component. The a-priori box is checked component by component,
/// so the containment test covers the whole vector. The other arguments and the
/// rigour argument are as for [`validated_euler`].
///
/// # Panics
///
/// Panics if `t1 ≤ t0`, if `steps` is zero, or if `f` returns a vector of the
/// wrong length.
///
/// # Examples
///
/// ```
/// use tpt_math_exact::roots::validated_euler_system;
/// use tpt_math_exact::{BigInt, BigRational, Interval};
///
/// let r = |n: i64| BigRational::from_integer(BigInt::from(n));
/// let frac = |n: i64, d: i64| BigRational::new(BigInt::from(n), BigInt::from(d));
/// // Harmonic oscillator: y₀′ = y₁, y₁′ = −y₀, with y(0) = (1, 0), so y₀(1) = cos 1.
/// let f = |_t: &Interval<BigRational>, y: &[Interval<BigRational>]| {
///     vec![y[1].clone(), Interval::new(r(-1), r(-1)) * y[0].clone()]
/// };
/// let end = validated_euler_system(f, r(0), r(1), &[r(1), r(0)], 20, frac(1, 1_000_000), 80)
///     .unwrap();
/// let c = 1.0_f64.cos();
/// let lo = num_traits::ToPrimitive::to_f64(end[0].lo()).unwrap();
/// let hi = num_traits::ToPrimitive::to_f64(end[0].hi()).unwrap();
/// assert!(lo <= c && c <= hi);
/// ```
pub fn validated_euler_system<F>(
    f: F,
    t0: Rational,
    t1: Rational,
    y0: &[Rational],
    steps: usize,
    inflation: Rational,
    max_picard: usize,
) -> Option<Vec<Interval<Rational>>>
where
    F: Fn(&Interval<Rational>, &[Interval<Rational>]) -> Vec<Interval<Rational>>,
{
    assert!(t1 > t0, "validated_euler_system: t1 must exceed t0");
    assert!(steps > 0, "validated_euler_system: steps must be positive");
    let n = y0.len();
    let h = (t1.clone() - t0.clone()) / Rational::from_integer(BigInt::from(steps));
    let h_box = point(h.clone());
    let mut y: Vec<Interval<Rational>> = y0.iter().cloned().map(point).collect();

    // Component-wise Euler step: y + h·g where g is an enclosure of f.
    let step_with =
        |g: &[Interval<Rational>], y: &[Interval<Rational>]| -> Vec<Interval<Rational>> {
            assert_eq!(
                g.len(),
                n,
                "validated_euler_system: f returned the wrong length"
            );
            y.iter()
                .zip(g)
                .map(|(yi, gi)| yi.clone() + h_box.clone() * gi.clone())
                .collect()
        };

    for step in 0..steps {
        let t_lo = t0.clone() + h.clone() * Rational::from_integer(BigInt::from(step));
        let t_box = Interval::new(t_lo.clone(), t_lo + h.clone());

        // A-priori box, one component at a time, from the Euler image of y.
        let first = step_with(&f(&t_box, &y), &y);
        let mut b: Vec<Interval<Rational>> = y
            .iter()
            .zip(&first)
            .map(|(yi, fi)| {
                let hull = yi.hull(fi);
                Interval::new(
                    hull.lo().clone() - inflation.clone(),
                    hull.hi().clone() + inflation.clone(),
                )
            })
            .collect();

        let mut found = false;
        let mut margin = inflation.clone();
        for _ in 0..max_picard {
            let candidate = step_with(&f(&t_box, &b), &y);
            let contained = b
                .iter()
                .zip(&candidate)
                .all(|(bi, ci)| bi.lo() <= ci.lo() && ci.hi() <= bi.hi());
            if contained {
                found = true;
                break;
            }
            b = y
                .iter()
                .zip(&candidate)
                .map(|(yi, ci)| {
                    let grown = ci.hull(yi);
                    Interval::new(
                        grown.lo().clone() - margin.clone(),
                        grown.hi().clone() + margin.clone(),
                    )
                })
                .collect();
            margin = margin.clone() + margin;
        }
        if !found {
            return None;
        }
        y = step_with(&f(&t_box, &b), &y);
    }
    Some(y)
}

#[cfg(test)]
mod system_tests {
    use super::*;
    use crate::{BigInt, BigRational};

    fn r(n: i64) -> Rational {
        BigRational::from_integer(BigInt::from(n))
    }

    fn frac(n: i64, d: i64) -> Rational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    #[test]
    fn oscillator_enclosure_contains_cosine_and_sine() {
        let f = |_t: &Interval<Rational>, y: &[Interval<Rational>]| {
            vec![y[1].clone(), Interval::new(r(-1), r(-1)) * y[0].clone()]
        };
        let end = validated_euler_system(f, r(0), r(1), &[r(1), r(0)], 20, frac(1, 1_000_000), 80)
            .unwrap();
        let enc = |iv: &Interval<Rational>| {
            (
                num_traits::ToPrimitive::to_f64(iv.lo()).unwrap(),
                num_traits::ToPrimitive::to_f64(iv.hi()).unwrap(),
            )
        };
        let (lo0, hi0) = enc(&end[0]);
        let (lo1, hi1) = enc(&end[1]);
        assert!(lo0 <= 1.0_f64.cos() && 1.0_f64.cos() <= hi0);
        // y₁ = y₀′ = −sin t.
        assert!(lo1 <= -1.0_f64.sin() && -1.0_f64.sin() <= hi1);
    }

    #[test]
    fn system_with_one_component_matches_scalar_solver() {
        let scalar = |_t: &Interval<Rational>, y: &Interval<Rational>| y.clone();
        let vector = |_t: &Interval<Rational>, y: &[Interval<Rational>]| vec![y[0].clone()];
        let a = validated_euler(scalar, r(0), r(1), r(1), 10, frac(1, 1000), 80).unwrap();
        let b = validated_euler_system(vector, r(0), r(1), &[r(1)], 10, frac(1, 1000), 80).unwrap();
        assert_eq!(&a, &b[0]);
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn wrong_length_from_f_panics() {
        let bad = |_t: &Interval<Rational>, _y: &[Interval<Rational>]| vec![];
        let _ = validated_euler_system(bad, r(0), r(1), &[r(1)], 2, frac(1, 1000), 8);
    }
}
