//! Cross-crate example: `tpt-math-prob` + `tpt-math-stats`.
//!
//! Samples from a wrapped `rand_distr` normal distribution using the shared
//! `tpt-math-prob-core` RNG traits, then feeds the sample to `tpt-math-stats`
//! for descriptive statistics and a one-sample t-test.

use tpt_math_prob::dist::{normal, Dist, Distribution, SplitMix64};
use tpt_math_stats::{mean, one_sample_t_test};

fn main() {
    let mut rng = SplitMix64::seed_from_u64(0);
    let gauss = Dist::new(normal((0.0, 1.0)).unwrap());

    let n = 20_000usize;
    let sample: Vec<f64> = (0..n).map(|_| gauss.sample(&mut rng)).collect();

    let m = mean(&sample);
    let (t, p) = one_sample_t_test(&sample, 0.0);

    // A standard normal sample should have mean ~0 and not reject H0 (mu = 0).
    assert!(m.abs() < 0.1, "sample mean too far from 0: {m}");
    assert!(p > 0.01, " spuriously rejected H0: p = {p}");

    println!("prob+stats: n = {n}, mean = {m:.4}, t = {t:.4}, p = {p:.4}");
}
