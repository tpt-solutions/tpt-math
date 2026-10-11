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

/// Nodes and weights of the `N`-point Gauss–Legendre rule on `[−1, 1]`.
///
/// Each node is a root of the Legendre polynomial `P_N`, found by Newton's
/// method from the standard Chebyshev-style starting guess. The weights follow
/// from the derivative at the node.
fn gauss_legendre_nodes<const N: usize>() -> ([f64; N], [f64; N]) {
    let mut nodes = [0.0; N];
    let mut weights = [0.0; N];
    for i in 0..N {
        // Explicit trait call: the inherent f64::cos is unavailable without std.
        let mut x =
            num_traits::Float::cos(core::f64::consts::PI * (i as f64 + 0.75) / (N as f64 + 0.5));
        let mut dp = 0.0;
        for _ in 0..100 {
            // Three-term recurrence: p1 = P_N(x), p0 = P_{N-1}(x).
            let (mut p0, mut p1) = (1.0, x);
            for k in 2..=N {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            dp = N as f64 * (x * p1 - p0) / (x * x - 1.0);
            let dx = p1 / dp;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        nodes[i] = x;
        weights[i] = 2.0 / ((1.0 - x * x) * dp * dp);
    }
    (nodes, weights)
}

/// The `N`-point Gauss–Legendre rule on `[a, b]`.
///
/// Exact for polynomials of degree `2N − 1` or less. For smooth integrands it
/// converges much faster than Simpson's rule, which is why it is the building
/// block for [`adaptive_gauss_legendre`]. `N = 0` integrates to zero.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::quadrature::gauss_legendre;
///
/// // A 5-point rule is exact through degree 9: ∫₀¹ x⁹ dx = 1/10.
/// let v = gauss_legendre::<_, 5>(|x: f64| x.powi(9), 0.0, 1.0);
/// assert!((v - 0.1).abs() < 1e-14);
/// ```
pub fn gauss_legendre<F, const N: usize>(f: F, a: f64, b: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = gauss_legendre_nodes::<N>();
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    half * nodes
        .iter()
        .zip(&weights)
        .map(|(&x, &w)| w * f(mid + half * x))
        .sum::<f64>()
}

/// Failure mode of [`adaptive_gauss_legendre`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuadratureError {
    /// A panel still failed its error test at the maximum bisection depth. The
    /// integrand is probably non-smooth or singular there.
    MaxDepth,
}

impl core::fmt::Display for QuadratureError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            QuadratureError::MaxDepth => write!(f, "panel still inaccurate at maximum depth"),
        }
    }
}

impl core::error::Error for QuadratureError {}

/// Adaptive `N`-point Gauss–Legendre quadrature on `[a, b]`.
///
/// Each panel is compared with its two halves. If the difference is at most
/// `tol`, the two halves are accepted. Otherwise the panel is bisected, to at
/// most `max_depth` levels. `tol` is an absolute tolerance applied to each
/// accepted panel, so the total error can be up to `tol` times the number of
/// panels. The difference is only an estimate of the error, so keep `tol`
/// comfortably above the rounding floor of the integrand.
///
/// # Errors
///
/// Returns [`QuadratureError::MaxDepth`] if a panel is still inaccurate at
/// `max_depth` levels.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::quadrature::adaptive_gauss_legendre;
///
/// // A sharp peak that a fixed-order rule would resolve poorly.
/// let f = |x: f64| 1.0 / (1.0 + 100.0 * x * x);
/// let v = adaptive_gauss_legendre::<_, 10>(f, -1.0, 1.0, 1e-10, 30).unwrap();
/// assert!((v - 2.0 * (10.0f64).atan() / 10.0).abs() < 1e-9);
/// ```
pub fn adaptive_gauss_legendre<F, const N: usize>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> Result<f64, QuadratureError>
where
    F: Fn(f64) -> f64,
{
    let whole = gauss_legendre::<_, N>(&f, a, b);
    adaptive_panel::<_, N>(&f, a, b, whole, tol, max_depth)
}

fn adaptive_panel<F, const N: usize>(
    f: &F,
    a: f64,
    b: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> Result<f64, QuadratureError>
where
    F: Fn(f64) -> f64,
{
    let mid = 0.5 * (a + b);
    let left = gauss_legendre::<_, N>(f, a, mid);
    let right = gauss_legendre::<_, N>(f, mid, b);
    if (left + right - whole).abs() <= tol {
        return Ok(left + right);
    }
    if depth == 0 {
        return Err(QuadratureError::MaxDepth);
    }
    let l = adaptive_panel::<_, N>(f, a, mid, left, tol, depth - 1)?;
    let r = adaptive_panel::<_, N>(f, mid, b, right, tol, depth - 1)?;
    Ok(l + r)
}

#[cfg(test)]
mod gauss_tests {
    use super::*;

    #[test]
    fn nodes_and_weights_match_known_five_point_rule() {
        let (nodes, weights) = gauss_legendre_nodes::<5>();
        // Nodes: 0 and ±√(5 − 2√(10/7))/3 and ±√(5 + 2√(10/7))/3.
        let inner = ((5.0 - 2.0 * (10.0f64 / 7.0).sqrt()) / 9.0).sqrt();
        let outer = ((5.0 + 2.0 * (10.0f64 / 7.0).sqrt()) / 9.0).sqrt();
        let mut sorted = nodes;
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] + outer).abs() < 1e-14);
        assert!((sorted[1] + inner).abs() < 1e-14);
        assert!(sorted[2].abs() < 1e-14);
        assert!((sorted[3] - inner).abs() < 1e-14);
        assert!((sorted[4] - outer).abs() < 1e-14);
        assert!((weights.iter().sum::<f64>() - 2.0).abs() < 1e-14);
    }

    #[test]
    fn exact_for_polynomials_up_to_degree_2n_minus_1() {
        // ∫₋₁¹ x⁶ dx = 2/7 is exact for a 4-point rule (degree 7).
        let v = gauss_legendre::<_, 4>(|x: f64| x.powi(6), -1.0, 1.0);
        assert!((v - 2.0 / 7.0).abs() < 1e-14);
    }

    #[test]
    fn smooth_integrand_converges_fast() {
        let v = gauss_legendre::<_, 10>(f64::sin, 0.0, core::f64::consts::PI);
        assert!((v - 2.0).abs() < 1e-14);
    }

    #[test]
    fn adaptive_handles_peaked_integrand() {
        let f = |x: f64| 1.0 / (1.0 + 100.0 * x * x);
        let v = adaptive_gauss_legendre::<_, 10>(f, -1.0, 1.0, 1e-10, 30).unwrap();
        assert!((v - 2.0 * 10.0f64.atan() / 10.0).abs() < 1e-9, "{v}");
    }

    #[test]
    fn adaptive_reports_max_depth_for_singular_integrand() {
        // 1/√|x| is integrable on [−1, 1] but singular at 0, so with no depth
        // budget the error test must fail there.
        let result =
            adaptive_gauss_legendre::<_, 5>(|x: f64| 1.0 / x.abs().sqrt(), -1.0, 1.0, 1e-14, 0);
        assert_eq!(result, Err(QuadratureError::MaxDepth));
    }
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
