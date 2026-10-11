//! Numerical integration of scalar functions.

use crate::Scalar;

/// Composite Simpson's rule on `[a, b]` with `n` subintervals.
///
/// `n` is rounded up to the next even number (Simpson needs pairs of
/// subintervals), and is at least 2. The error is `O(h⁴)`; the rule is exact
/// for cubic polynomials.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::quadrature::simpson;
///
/// // ∫₀² x³ dx = 4, exactly.
/// assert!((simpson(|x: f64| x * x * x, 0.0, 2.0, 2) - 4.0).abs() < 1e-14);
/// ```
pub fn simpson<T, F>(f: F, a: T, b: T, n: usize) -> T
where
    T: Scalar,
    F: Fn(T) -> T,
{
    let n = if n < 2 {
        2
    } else if n % 2 == 1 {
        n + 1
    } else {
        n
    };
    let two = T::one() + T::one();
    let three = two + T::one();
    let four = two + two;
    let h = (b - a) / T::from(n).unwrap_or(T::one());
    let mut sum = f(a) + f(b);
    for k in 1..n {
        let x = a + h * T::from(k).unwrap_or(T::zero());
        let weight = if k % 2 == 1 { four } else { two };
        sum = sum + weight * f(x);
    }
    sum * h / three
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpson_integrates_sine_over_a_half_period() {
        // The error is O(h⁴): about 1.7e-8 at 100 panels, so use 200.
        let v = simpson(|x: f64| x.sin(), 0.0, core::f64::consts::PI, 200);
        assert!((v - 2.0).abs() < 1e-8);
    }

    #[test]
    fn simpson_rounds_odd_panel_counts_up() {
        let odd = simpson(|x: f64| x * x, 0.0, 3.0, 5);
        let even = simpson(|x: f64| x * x, 0.0, 3.0, 6);
        assert_eq!(odd, even);
        assert!((even - 9.0).abs() < 1e-12);
    }
}
