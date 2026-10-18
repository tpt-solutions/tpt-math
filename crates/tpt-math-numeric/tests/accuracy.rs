//! Accuracy and behaviour checks for the public numerical routines, against
//! closed forms. They target the arithmetic and branch choices that the unit
//! tests leave unconstrained (time arguments, interval endpoints, step sizes,
//! error messages).

use tpt_math_numeric::ode::{
    dopri5_dense, dopri5_events, dopri5_integrate, rk4_integrate, rk4_step, Crossing, OdeError,
};
use tpt_math_numeric::quadrature::{
    adaptive_gauss_kronrod, adaptive_gauss_legendre, gauss_legendre, simpson, QuadratureError,
};
use tpt_math_numeric::roots::{bisect, brent, RootError};
use tpt_math_numeric::{SolveReport, SolveStatus};

#[test]
fn simpson_is_exact_for_cubics_and_rounds_odd_panel_counts_up() {
    // ∫₀¹ x³ = 1/4, exact for Simpson's rule.
    let cubic = simpson(|x: f64| x * x * x, 0.0, 1.0, 2);
    assert!((cubic - 0.25).abs() < 1e-14, "{cubic}");
    // ∫₀² x² = 8/3 with n = 2: the weights 1, 4, 1 are checked directly.
    let quad = simpson(|x: f64| x * x, 0.0, 2.0, 2);
    assert!((quad - 8.0 / 3.0).abs() < 1e-14, "{quad}");
    // n = 1 and n = 3 are rounded up to 2 and 4 panels respectively.
    assert_eq!(simpson(|x: f64| x * x, 0.0, 2.0, 1), quad);
    assert_eq!(
        simpson(|x: f64| x.sin(), 0.0, 1.0, 3),
        simpson(|x: f64| x.sin(), 0.0, 1.0, 4)
    );
}

#[test]
fn gauss_legendre_is_exact_to_its_degree_on_shifted_intervals() {
    // A 2-point rule is exact to degree 3: ∫₁³ x³ = 20.
    let v = gauss_legendre::<_, 2>(|x| x * x * x, 1.0, 3.0);
    assert!((v - 20.0).abs() < 1e-12, "{v}");
    // A 5-point rule is exact to degree 9: ∫₋₁¹ x⁸ = 2/9.
    let w = gauss_legendre::<_, 5>(|x| x.powi(8), -1.0, 1.0);
    assert!((w - 2.0 / 9.0).abs() < 1e-13, "{w}");
    // Reversed limits flip the sign.
    let r = gauss_legendre::<_, 3>(|x| x * x, 2.0, 0.0);
    assert!((r + 8.0 / 3.0).abs() < 1e-12, "{r}");
}

#[test]
fn adaptive_rules_meet_the_tolerance_on_a_peaked_integrand() {
    // ∫₀¹ 1/(1 + 100x²) dx = atan(10) / 10.
    let exact = 10f64.atan() / 10.0;
    let a = adaptive_gauss_legendre::<_, 5>(|x| 1.0 / (1.0 + 100.0 * x * x), 0.0, 1.0, 1e-10, 40)
        .unwrap();
    assert!((a - exact).abs() < 1e-9, "legendre {a}");
    let k = adaptive_gauss_kronrod(|x| 1.0 / (1.0 + 100.0 * x * x), 0.0, 1.0, 1e-10, 40).unwrap();
    assert!((k - exact).abs() < 1e-9, "kronrod {k}");
}

#[test]
fn quadrature_error_reports_max_depth() {
    let err = adaptive_gauss_kronrod(|x: f64| 1.0 / x.abs().sqrt(), -1.0, 1.0, 1e-14, 2);
    assert_eq!(err, Err(QuadratureError::MaxDepth));
    assert_eq!(
        QuadratureError::MaxDepth.to_string(),
        "panel still inaccurate at maximum depth"
    );
}

#[test]
fn rk4_step_reproduces_the_linear_stability_polynomial() {
    // For y' = λy, one RK4 step multiplies y by 1 + z + z²/2 + z³/6 + z⁴/24 with
    // z = λh, exactly. Here λ = −2, h = 0.5, so z = −1.
    let y = rk4_step(|_t: f64, y: &[f64; 1]| [-2.0 * y[0]], 0.0, [1.0], 0.5);
    let want = 1.0 - 1.0 + 0.5 - 1.0 / 6.0 + 1.0 / 24.0;
    assert!((y[0] - want).abs() < 1e-14, "{} vs {want}", y[0]);
}

#[test]
fn rk4_integrates_backwards_and_uses_the_time_argument() {
    // y' = 2t, y(0) = 1: y(t) = 1 + t². Exact for RK4 (degree-2 polynomial).
    let f = |t: f64, _y: &[f64; 1]| [2.0 * t];
    let fwd = rk4_integrate(f, 0.0, 3.0, [1.0], 3);
    assert!((fwd[0] - 10.0).abs() < 1e-12, "{}", fwd[0]);
    // Backwards from t = 3 to 0 returns y(0) = 1 (and never takes the wrong step sign).
    let back = rk4_integrate(f, 3.0, 0.0, [10.0], 3);
    assert!((back[0] - 1.0).abs() < 1e-12, "{}", back[0]);
}

#[test]
fn dopri5_handles_a_time_dependent_nonlinear_problem() {
    // y' = t·y, y(0) = 1: y(t) = exp(t²/2). The time arguments of the stages
    // matter here, because the right-hand side depends on t.
    let y = dopri5_integrate(
        |t: f64, y: &[f64; 1]| [t * y[0]],
        0.0,
        2.0,
        [1.0],
        1e-11,
        1e-13,
        100_000,
    )
    .unwrap();
    assert!((y[0] - 2.0f64.exp()).abs() < 1e-8, "{}", y[0]);
}

#[test]
fn ode_errors_have_their_messages() {
    assert_eq!(
        OdeError::MaxStepsExceeded.to_string(),
        "step budget exhausted before reaching t1"
    );
    assert_eq!(
        OdeError::StepSizeUnderflow.to_string(),
        "step size underflow"
    );
}

#[test]
fn dense_output_accessors_and_interval_edges() {
    let sol = dopri5_dense(
        |_t: f64, y: &[f64; 1]| [y[0]],
        1.0,
        4.0,
        [1.0],
        1e-10,
        1e-12,
        100_000,
    )
    .unwrap();
    assert_eq!(sol.t_start(), 1.0);
    assert_eq!(sol.t_end(), 4.0);
    assert!(sol.steps() >= 1);
    let times: Vec<f64> = sol.times().collect();
    assert_eq!(times.len(), sol.steps() + 1);
    assert_eq!(times[0], 1.0);
    assert_eq!(*times.last().unwrap(), 4.0);
    // Both endpoints are inside the span; points just outside are not.
    assert!((sol.eval(1.0).unwrap()[0] - 1.0).abs() < 1e-12);
    assert!((sol.eval(4.0).unwrap()[0] - 3f64.exp()).abs() < 1e-6);
    assert_eq!(sol.eval(0.999_999), None);
    assert_eq!(sol.eval(4.000_001), None);
}

#[test]
fn event_location_finds_both_directions_and_the_exact_time() {
    // y'' = −y with y(0) = 0, y'(0) = 1: y = sin t. Events at y = ½ are crossed
    // rising at π/6 + 2πk-type times; with the start at 0 we see the first
    // rising crossing of ½ at t = π/6 and the falling one at 5π/6.
    let run = dopri5_events(
        |_t: f64, s: &[f64; 2]| [s[1], -s[0]],
        |_t: f64, s: &[f64; 2]| s[0] - 0.5,
        0.0,
        2.0,
        [0.0, 1.0],
        1e-11,
        1e-13,
        100_000,
        false,
    )
    .unwrap();
    assert_eq!(run.events.len(), 1, "{:?}", run.events);
    let ev = &run.events[0];
    assert_eq!(ev.crossing, Crossing::Rising);
    assert!(
        (ev.t - std::f64::consts::FRAC_PI_6).abs() < 1e-8,
        "{}",
        ev.t
    );
    assert!((ev.y[0] - 0.5).abs() < 1e-8);
    // A run that spans the falling crossing at 5π/6 ≈ 2.618 reports it too.
    let run = dopri5_events(
        |_t: f64, s: &[f64; 2]| [s[1], -s[0]],
        |_t: f64, s: &[f64; 2]| s[0] - 0.5,
        0.0,
        3.0,
        [0.0, 1.0],
        1e-11,
        1e-13,
        100_000,
        false,
    )
    .unwrap();
    let falling: Vec<_> = run
        .events
        .iter()
        .filter(|e| e.crossing == Crossing::Falling)
        .collect();
    assert_eq!(falling.len(), 1);
    assert!((falling[0].t - 5.0 * std::f64::consts::FRAC_PI_6).abs() < 1e-8);
}

#[test]
fn terminal_event_ends_the_run_at_the_event_not_the_step_end() {
    let run = dopri5_events(
        |_t: f64, s: &[f64; 1]| [-s[0]],
        |_t: f64, s: &[f64; 1]| s[0] - 0.25,
        0.0,
        10.0,
        [1.0],
        1e-11,
        1e-13,
        100_000,
        true,
    )
    .unwrap();
    assert!(run.terminated);
    let want = 4f64.ln();
    assert!((run.t_end - want).abs() < 1e-8, "{} vs {want}", run.t_end);
    assert!((run.y_end[0] - 0.25).abs() < 1e-8);
}

#[test]
fn bisection_returns_an_endpoint_that_is_already_a_root() {
    assert_eq!(bisect(|x: f64| x - 1.0, 1.0, 3.0, 1e-12, 200), Ok(1.0));
    assert_eq!(bisect(|x: f64| x - 3.0, 1.0, 3.0, 1e-12, 200), Ok(3.0));
    assert_eq!(
        bisect(|x: f64| x * x + 1.0, -1.0, 1.0, 1e-12, 200),
        Err(RootError::NoSignChange)
    );
}

#[test]
fn bisection_and_brent_converge_to_sqrt_two() {
    let b = bisect(|x: f64| x * x - 2.0, 0.0, 2.0, 1e-13, 200).unwrap();
    assert!((b - 2f64.sqrt()).abs() < 1e-12, "{b}");
    let r = brent(|x: f64| x * x - 2.0, 0.0, 2.0, 1e-13, 200).unwrap();
    assert!((r - 2f64.sqrt()).abs() < 1e-12, "{r}");
}

#[test]
fn root_errors_have_their_messages_and_budget_is_enforced() {
    assert_eq!(
        RootError::NoSignChange.to_string(),
        "f(a) and f(b) have the same sign"
    );
    let exhausted = bisect(|x: f64| x - 0.3, 0.0, 1.0, 1e-300, 3);
    assert!(
        matches!(exhausted, Err(RootError::NotConverged { .. })),
        "{exhausted:?}"
    );
}

#[test]
fn solve_report_and_status_display_their_text() {
    assert_eq!(SolveStatus::Converged.to_string(), "converged");
    assert_eq!(SolveStatus::NotConverged.to_string(), "not converged");
    assert_eq!(SolveStatus::Breakdown.to_string(), "breakdown");
    let report = SolveReport {
        status: SolveStatus::Converged,
        iterations: 4,
        residual: 1e-9,
    };
    assert!(report.status.is_converged());
    assert_eq!(report.iterations, 4);
}
