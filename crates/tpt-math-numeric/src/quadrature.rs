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

/// Kronrod extension nodes of the 15-point rule on `[−1, 1]` (QUADPACK
/// `qk15`). Odd indices are the embedded 7-point Gauss nodes; index 7 is the
/// centre.
const GK15_NODES: [f64; 8] = [
    0.9914553711208126,
    0.9491079123427585,
    0.8648644233597691,
    0.7415311855993945,
    0.5860872354676911,
    0.4058451513773972,
    0.20778495500789848,
    0.0,
];

/// Kronrod weights matching [`GK15_NODES`].
const GK15_WEIGHTS: [f64; 8] = [
    0.022935322010529224,
    0.06309209262997856,
    0.10479001032225019,
    0.14065325971552592,
    0.1690047266392679,
    0.19035057806478542,
    0.20443294007529889,
    0.20948214108472782,
];

/// Gauss weights for the embedded 7-point rule, at the odd Kronrod indices
/// `1, 3, 5` and the centre `7`.
const G7_WEIGHTS: [f64; 4] = [
    0.1294849661688697,
    0.27970539148927664,
    0.3818300505051189,
    0.4179591836734694,
];

/// The 15-point Gauss–Kronrod pair on `[a, b]`.
///
/// Returns `(kronrod, error)`. `kronrod` is the 15-point Kronrod estimate, which
/// is exact for polynomials of degree 22 or less. `error` is `|K₁₅ − G₇|`, the
/// difference from the embedded 7-point Gauss rule, and serves as the usual
/// error estimate for an adaptive scheme.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::quadrature::gauss_kronrod_15;
///
/// // ∫₀¹ eˣ dx = e − 1.
/// let (v, err) = gauss_kronrod_15(f64::exp, 0.0, 1.0);
/// assert!((v - (core::f64::consts::E - 1.0)).abs() < 1e-14);
/// assert!(err < 1e-10);
/// ```
pub fn gauss_kronrod_15<F>(f: F, a: f64, b: f64) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let mid = 0.5 * (a + b);
    let half = 0.5 * (b - a);
    let centre = f(mid);
    let mut kronrod = GK15_WEIGHTS[7] * centre;
    let mut gauss = G7_WEIGHTS[3] * centre;
    for j in 0..7 {
        let x = half * GK15_NODES[j];
        let pair = f(mid - x) + f(mid + x);
        kronrod += GK15_WEIGHTS[j] * pair;
        // Gauss nodes are the odd indices 1, 3, 5 (index 6 is Kronrod-only).
        if j % 2 == 1 {
            gauss += G7_WEIGHTS[j / 2] * pair;
        }
    }
    (half * kronrod, (half * (kronrod - gauss)).abs())
}

/// Adaptive 15-point Gauss–Kronrod quadrature on `[a, b]`.
///
/// Each panel is integrated with [`gauss_kronrod_15`]. A panel is accepted when
/// its error estimate is at most `tol`; otherwise it is bisected, to at most
/// `max_depth` levels. As with [`adaptive_gauss_legendre`], `tol` applies per
/// accepted panel.
///
/// # Errors
///
/// Returns [`QuadratureError::MaxDepth`] if a panel is still inaccurate at
/// `max_depth` levels.
///
/// # Examples
///
/// ```
/// use tpt_math_numeric::quadrature::adaptive_gauss_kronrod;
///
/// // ∫₀¹ 1/√x dx = 2, despite the integrable singularity at 0.
/// let v = adaptive_gauss_kronrod(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10, 60).unwrap();
/// assert!((v - 2.0).abs() < 1e-8);
/// ```
pub fn adaptive_gauss_kronrod<F>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_depth: usize,
) -> Result<f64, QuadratureError>
where
    F: Fn(f64) -> f64,
{
    adaptive_gk_panel(&f, a, b, tol, max_depth)
}

fn adaptive_gk_panel<F>(
    f: &F,
    a: f64,
    b: f64,
    tol: f64,
    depth: usize,
) -> Result<f64, QuadratureError>
where
    F: Fn(f64) -> f64,
{
    let (value, err) = gauss_kronrod_15(f, a, b);
    if err <= tol {
        return Ok(value);
    }
    if depth == 0 {
        return Err(QuadratureError::MaxDepth);
    }
    let mid = 0.5 * (a + b);
    let l = adaptive_gk_panel(f, a, mid, tol, depth - 1)?;
    let r = adaptive_gk_panel(f, mid, b, tol, depth - 1)?;
    Ok(l + r)
}

#[cfg(test)]
mod gauss_kronrod_tests {
    use super::*;

    #[test]
    fn kronrod_weights_sum_to_two() {
        // Weights are on [−1, 1] over the full symmetric node set.
        let total = GK15_WEIGHTS[7] + 2.0 * GK15_WEIGHTS[..7].iter().sum::<f64>();
        assert!((total - 2.0).abs() < 1e-14, "{total}");
        let gauss = G7_WEIGHTS[3] + 2.0 * G7_WEIGHTS[..3].iter().sum::<f64>();
        assert!((gauss - 2.0).abs() < 1e-14, "{gauss}");
    }

    #[test]
    fn exact_for_polynomials_up_to_degree_22() {
        // ∫₋₁¹ xᵏ dx is 2/(k+1) for even k and 0 for odd k.
        for k in 0..=22 {
            let exact = if k % 2 == 0 {
                2.0 / (k as f64 + 1.0)
            } else {
                0.0
            };
            let (v, _) = gauss_kronrod_15(|x: f64| x.powi(k), -1.0, 1.0);
            assert!((v - exact).abs() < 1e-13, "degree {k}: {v} vs {exact}");
        }
    }

    #[test]
    fn error_estimate_vanishes_for_low_degree() {
        // Both rules are exact through degree 13, so the estimate is ~0.
        let (_, err) = gauss_kronrod_15(|x: f64| x.powi(12), -1.0, 1.0);
        assert!(err < 1e-13, "{err}");
    }

    #[test]
    fn adaptive_handles_integrable_endpoint_singularity() {
        let v = adaptive_gauss_kronrod(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-10, 60).unwrap();
        assert!((v - 2.0).abs() < 1e-8, "{v}");
    }

    #[test]
    fn adaptive_reports_max_depth_when_budget_is_zero() {
        let result = adaptive_gauss_kronrod(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-14, 0);
        assert_eq!(result, Err(QuadratureError::MaxDepth));
    }
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
