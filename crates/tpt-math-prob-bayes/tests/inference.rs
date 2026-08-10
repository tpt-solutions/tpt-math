//! End-to-end checks of the public inference surface: conjugate updates,
//! MCMC, and model comparison all working through `tpt-math-prob-core`'s
//! deterministic RNG.

use tpt_math_prob_bayes::{
    binomial_log_likelihood, log_sum_exp, Beta, Distribution, Gamma, Gaussian, Metropolis, Normal,
};
use tpt_math_prob_core::{Rng, SplitMix64};

fn mean_and_std(xs: &[f64]) -> (f64, f64) {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

#[test]
fn sequential_coin_flipping_recovers_the_true_bias() {
    let truth = 0.35;
    let mut rng = SplitMix64::seed_from_u64(1_234_567);

    let mut belief = Beta::uniform();
    for _ in 0..2_000 {
        belief = belief.update_one(rng.next_f64() < truth);
    }

    assert!(
        (belief.mean() - truth).abs() < 0.03,
        "mean = {}",
        belief.mean()
    );
    let (lo, hi) = belief.credible_interval(0.99);
    assert!(lo < truth && truth < hi, "interval = ({lo}, {hi})");
    assert!(belief.std() < 0.02);
    // The posterior predictive is just the posterior mean.
    assert_eq!(belief.predictive_success_probability(), belief.mean());
}

#[test]
fn conjugate_and_mcmc_beta_posteriors_agree() {
    let prior = Beta::new(2.0, 2.0);
    let (successes, failures) = (37u64, 63u64);
    let exact = prior.update(successes, failures);

    let target = move |p: f64| prior.log_unnormalized_posterior(p, successes, failures);
    let mut sampler = Metropolis::with_gaussian_proposal(target, 0.12);
    let mut rng = SplitMix64::seed_from_u64(20_260_810);
    let trace = sampler.run_with_burn_in(&mut rng, 0.5, 5_000, 80_000);

    let (mean, std) = mean_and_std(&trace);
    assert!(
        (mean - exact.mean()).abs() < 0.005,
        "mean {mean} vs {}",
        exact.mean()
    );
    assert!(
        (std - exact.std()).abs() < 0.003,
        "std {std} vs {}",
        exact.std()
    );

    // The MCMC trace should also reproduce the analytic quantiles.
    for &q in &[0.1, 0.5, 0.9] {
        let threshold = exact.quantile(q);
        let empirical =
            trace.iter().filter(|x| **x <= threshold).count() as f64 / trace.len() as f64;
        assert!(
            (empirical - q).abs() < 0.02,
            "q = {q}, empirical = {empirical}"
        );
    }
}

#[test]
fn conjugate_and_mcmc_normal_posteriors_agree() {
    // Generate data from N(2.0, 1.0) and infer the mean with a known variance.
    let known_variance = 1.0f64;
    let mut rng = SplitMix64::seed_from_u64(555_444);
    let generator = Gaussian::new(2.0, known_variance.sqrt());
    let data: Vec<f64> = (0..40).map(|_| generator.sample(&mut rng)).collect();

    let prior = Normal::new(0.0, 3.0);
    let exact = prior.update(&data, known_variance);
    assert!((exact.mean() - 2.0).abs() < 0.4);
    assert!(exact.variance() < prior.variance());

    let target = {
        let data = data.clone();
        move |mu: f64| prior.log_pdf(mu) + Gaussian::new(mu, 1.0).log_likelihood(&data)
    };
    let mut sampler = Metropolis::with_gaussian_proposal(target, 0.5);
    let trace = sampler.run_with_burn_in(&mut rng, 0.0, 5_000, 80_000);

    let (mean, std) = mean_and_std(&trace);
    assert!(
        (mean - exact.mean()).abs() < 0.01,
        "mean {mean} vs {}",
        exact.mean()
    );
    assert!(
        (std - exact.std()).abs() < 0.01,
        "std {std} vs {}",
        exact.std()
    );
}

#[test]
fn posterior_sampling_matches_the_conjugate_posterior() {
    // Draw from the closed-form posterior and compare empirical moments.
    let posterior = Beta::uniform().update(120, 80);
    let mut rng = SplitMix64::seed_from_u64(31);
    let draws: Vec<f64> = (0..50_000).map(|_| posterior.sample(&mut rng)).collect();
    let (mean, std) = mean_and_std(&draws);
    assert!((mean - posterior.mean()).abs() < 0.003);
    assert!((std - posterior.std()).abs() < 0.002);

    // Monte Carlo answer to "is the rate above 0.55?" versus the exact CDF.
    let empirical = draws.iter().filter(|x| **x > 0.55).count() as f64 / draws.len() as f64;
    let exact = 1.0 - posterior.cdf(0.55);
    assert!((empirical - exact).abs() < 0.01, "{empirical} vs {exact}");
}

#[test]
fn model_comparison_via_marginal_likelihood() {
    // 90 heads out of 100 strongly favours the free-rate model.
    let free_rate = Beta::uniform().log_marginal_likelihood(90, 10);
    let fair_coin = binomial_log_likelihood(0.5, 90, 100);
    assert!(
        free_rate - fair_coin > 30.0,
        "log Bayes factor = {}",
        free_rate - fair_coin
    );

    // A balanced sample favours the simpler fair-coin model.
    let free_rate = Beta::uniform().log_marginal_likelihood(50, 50);
    let fair_coin = binomial_log_likelihood(0.5, 50, 100);
    assert!(fair_coin > free_rate);

    // Posterior model probabilities normalise with log-sum-exp.
    let posterior_odds = [free_rate, fair_coin];
    let total = log_sum_exp(&posterior_odds);
    let p_fair = (fair_coin - total).exp();
    assert!((0.0..=1.0).contains(&p_fair) && p_fair > 0.5);
}

#[test]
fn gamma_poisson_workflow() {
    // Simulate Poisson(3.0) counts via inversion, then update a Gamma prior.
    let mut rng = SplitMix64::seed_from_u64(8_675_309);
    let rate = 3.0f64;
    let counts: Vec<u64> = (0..200)
        .map(|_| {
            let mut k = 0u64;
            let mut product = rng.next_f64();
            let threshold = (-rate).exp();
            while product > threshold {
                product *= rng.next_f64();
                k += 1;
            }
            k
        })
        .collect();

    let posterior = Gamma::new(1.0, 1.0).update_poisson(&counts);
    assert!(
        (posterior.mean() - rate).abs() < 0.3,
        "mean = {}",
        posterior.mean()
    );
    assert!(posterior.variance() < 0.05);

    // Sampling the posterior agrees with its analytic mean.
    let draws: Vec<f64> = (0..20_000).map(|_| posterior.sample(&mut rng)).collect();
    let (mean, _) = mean_and_std(&draws);
    assert!((mean - posterior.mean()).abs() < 0.02);
}

#[test]
fn distributions_work_through_the_core_traits() {
    // Anything in this crate can be used generically as a `Distribution<f64>`.
    fn average<D: Distribution<f64>, R: Rng>(dist: &D, rng: &mut R, n: usize) -> f64 {
        (0..n).map(|_| dist.sample(rng)).sum::<f64>() / n as f64
    }

    let mut rng = SplitMix64::seed_from_u64(4_242);
    assert!((average(&Beta::new(2.0, 2.0), &mut rng, 40_000) - 0.5).abs() < 0.01);
    assert!((average(&Gaussian::new(-1.0, 0.5), &mut rng, 40_000) + 1.0).abs() < 0.01);
    assert!((average(&Gamma::new(4.0, 2.0), &mut rng, 40_000) - 2.0).abs() < 0.05);
}

#[test]
fn whole_pipeline_is_reproducible() {
    let run = |seed: u64| {
        let mut rng = SplitMix64::seed_from_u64(seed);
        let mut belief = Beta::jeffreys();
        for _ in 0..500 {
            belief = belief.update_one(rng.next_f64() < 0.6);
        }
        let target = move |p: f64| belief.log_pdf(p);
        let mut sampler = Metropolis::with_gaussian_proposal(target, 0.05);
        let trace = sampler.run_with_burn_in(&mut rng, belief.mean(), 500, 5_000);
        (belief, trace, sampler.acceptance_rate())
    };

    let (belief_a, trace_a, rate_a) = run(77);
    let (belief_b, trace_b, rate_b) = run(77);
    assert_eq!(belief_a, belief_b);
    assert_eq!(trace_a, trace_b);
    assert_eq!(rate_a, rate_b);

    let (belief_c, trace_c, _) = run(78);
    assert_ne!(belief_a, belief_c);
    assert_ne!(trace_a, trace_c);
}
