//! Bracketing root finders for scalar functions.

use crate::Scalar;

/// Errors from the bracketing root finders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RootError {
    /// `f(a)` and `f(b)` have the same sign, so the interval brackets no root.
    NoSignChange,
    /// The iteration budget was exhausted before the bracket shrank below the
    /// tolerance.
    NotConverged {
        /// Iterations performed.
        iterations: usize,
    },
}

impl core::fmt::Display for RootError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            RootError::NoSignChange => write!(f, "f(a) and f(b) have the same sign"),
            RootError::NotConverged { iterations } => {
                write!(f, "root finder did not converge in {iterations} iterations")
            }
        }
    }
}

/// Find a root of `f` in `[a, b]` by bisection.
///
/// The interval must bracket a sign change. Stops once the half-width of the
/// bracket is at most `tol`.
///
/// # Errors
///
/// [`RootError::NoSignChange`] if `f(a)` and `f(b)` have the same sign, and
/// [`RootError::NotConverged`] if `max_iter` halvings are not enough.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::roots::bisect;
///
/// let r = bisect(|x: f64| x * x - 2.0, 0.0, 2.0, 1e-12, 200).unwrap();
/// assert!((r - 2f64.sqrt()).abs() < 1e-11);
/// ```
pub fn bisect<T, F>(f: F, a: T, b: T, tol: T, max_iter: usize) -> Result<T, RootError>
where
    T: Scalar,
    F: Fn(T) -> T,
{
    let (mut lo, mut hi) = (a, b);
    let mut flo = f(lo);
    let fhi = f(hi);
    if flo == T::zero() {
        return Ok(lo);
    }
    if fhi == T::zero() {
        return Ok(hi);
    }
    if (flo > T::zero()) == (fhi > T::zero()) {
        return Err(RootError::NoSignChange);
    }
    let two = T::one() + T::one();
    for _ in 0..max_iter {
        let mid = lo + (hi - lo) / two;
        if (hi - lo) / two <= tol {
            return Ok(mid);
        }
        let fmid = f(mid);
        if fmid == T::zero() {
            return Ok(mid);
        }
        if (fmid > T::zero()) == (flo > T::zero()) {
            lo = mid;
            flo = fmid;
        } else {
            hi = mid;
        }
    }
    Err(RootError::NotConverged {
        iterations: max_iter,
    })
}

/// Find a root of `f` in `[a, b]` by Brent's method (inverse quadratic
/// interpolation and secant steps, with bisection as the safeguard).
///
/// The bracket must contain a sign change. Converges superlinearly on smooth
/// functions while never doing worse than bisection.
///
/// # Errors
///
/// As [`bisect`].
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::roots::brent;
///
/// let r = brent(|x: f64| x.cos() - x, 0.0, 1.0, 1e-14, 100).unwrap();
/// assert!((r - 0.739_085_133_215_160_6).abs() < 1e-12);
/// ```
pub fn brent<T, F>(f: F, a: T, b: T, tol: T, max_iter: usize) -> Result<T, RootError>
where
    T: Scalar,
    F: Fn(T) -> T,
{
    let two = T::one() + T::one();
    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    if fa == T::zero() {
        return Ok(a);
    }
    if fb == T::zero() {
        return Ok(b);
    }
    if (fa > T::zero()) == (fb > T::zero()) {
        return Err(RootError::NoSignChange);
    }
    // `b` is always the best estimate: |f(b)| <= |f(a)|.
    if fa.abs() < fb.abs() {
        core::mem::swap(&mut a, &mut b);
        core::mem::swap(&mut fa, &mut fb);
    }
    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut used_bisection = true;
    for _ in 0..max_iter {
        if fb == T::zero() || (b - a).abs() <= tol {
            return Ok(b);
        }
        let mut s = if fa != fc && fb != fc {
            // Inverse quadratic interpolation.
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            // Secant step.
            b - fb * (b - a) / (fb - fa)
        };
        let three = two + T::one();
        let quarter_point = (three * a + b) / (two + two);
        let (min_ab, max_ab) = if quarter_point < b {
            (quarter_point, b)
        } else {
            (b, quarter_point)
        };
        let bisect_needed = !(s > min_ab && s < max_ab)
            || (used_bisection && (s - b).abs() >= (b - c).abs() / two)
            || (!used_bisection && (s - b).abs() >= (c - d).abs() / two)
            || (used_bisection && (b - c).abs() < tol)
            || (!used_bisection && (c - d).abs() < tol);
        if bisect_needed {
            s = (a + b) / two;
            used_bisection = true;
        } else {
            used_bisection = false;
        }
        let fs = f(s);
        d = c;
        c = b;
        fc = fb;
        if (fa > T::zero()) != (fs > T::zero()) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            core::mem::swap(&mut a, &mut b);
            core::mem::swap(&mut fa, &mut fb);
        }
    }
    Err(RootError::NotConverged {
        iterations: max_iter,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisect_and_brent_find_known_roots() {
        let root = 2f64.sqrt();
        let b = bisect(|x: f64| x * x - 2.0, 0.0, 2.0, 1e-13, 200).unwrap();
        assert!((b - root).abs() < 1e-11);
        let r = brent(|x: f64| x * x - 2.0, 0.0, 2.0, 1e-14, 100).unwrap();
        assert!((r - root).abs() < 1e-12);
    }

    #[test]
    fn brent_handles_a_transcendental_root() {
        // x e^x − 1 = 0 has root W(1) ≈ 0.5671432904097838.
        let r = brent(|x: f64| x * x.exp() - 1.0, 0.0, 1.0, 1e-14, 100).unwrap();
        assert!((r - 0.567_143_290_409_783_8).abs() < 1e-12);
    }

    #[test]
    fn unbracketed_intervals_are_rejected() {
        assert_eq!(
            bisect(|x: f64| x * x + 1.0, -1.0, 1.0, 1e-9, 100),
            Err(RootError::NoSignChange)
        );
        assert_eq!(
            brent(|x: f64| x * x + 1.0, -1.0, 1.0, 1e-9, 100),
            Err(RootError::NoSignChange)
        );
    }

    #[test]
    fn exact_endpoint_roots_are_returned() {
        assert_eq!(brent(|x: f64| x, 0.0, 1.0, 1e-9, 10), Ok(0.0));
        assert_eq!(bisect(|x: f64| x - 1.0, 0.0, 1.0, 1e-9, 10), Ok(1.0));
    }
}
