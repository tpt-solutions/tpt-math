#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs, missing_debug_implementations)]
//! Standard probability distributions, bridging [`rand_distr`] into the
//! [`tpt_math_prob_core`] trait ecosystem.
//!
//! This crate wraps the [`rand_distr`] distributions and makes them sampleable
//! through the minimal [`tpt_math_prob_core::Rng`]/[`Distribution`](`tpt_math_prob_core::Distribution`)
//! traits used across the `tpt-math-prob-*` family. Because our [`Rng`] is far
//! smaller than [`rand`]'s [`RngCore`], we provide [`CoreRng`]: a thin adapter
//! that turns any of our RNGs into a [`rand::RngCore`] so that every
//! [`rand_distr`] distribution can be sampled without pulling in `rand`'s own
//! generators.
//!
//! The distributions themselves are *not* directly re-implemented. Instead the
//! local [`Dist`] wrapper (and the small [`AsU64`] helper for counts) adapts a
//! `rand_distr` distribution to our [`Distribution`](`tpt_math_prob_core::Distribution`)
//! trait. Constructor functions ([`normal`], [`uniform`], [`poisson`], …) give
//! ergonomic, documented entry points, and the raw [`rand_distr`] module is
//! re-exported so advanced users can reach under the bridge.
//!
//! The crate is `no_std`-compatible; all float math is delegated to
//! `rand_distr` (which uses `num-traits`'s `libm` fallback), and `std`-only
//! conveniences are gated behind the default `std` feature.
//!
//! # Examples
//!
//! ```
//! use tpt_math_prob_core::{Distribution, Rng, SplitMix64};
//! use tpt_math_prob_dist::{normal, Dist};
//!
//! let mut rng = SplitMix64::seed_from_u64(0);
//! let gauss = normal((0.0, 1.0)).unwrap();
//! let x: f64 = Dist::new(gauss).sample(&mut rng);
//! // or, equivalently, with the explicit bridge type:
//! let y: f64 = Dist::new(normal((2.0, 3.0)).unwrap()).sample(&mut rng);
//! let _ = x + y;
//! ```

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Re-export the core randomness traits/types so consumers can depend on this
// crate alone. We deliberately re-export `rand_distr` verbatim as well.
pub use tpt_math_prob_core::{Distribution, Rng, Sampler, SplitMix64, Standard};

/// The raw [`rand_distr`] distributions, re-exported for direct use.
///
/// Everything in this module is the unmodified `rand_distr` API, which samples
/// through [`rand::RngCore`]. To sample through our [`Rng`], wrap a value in
/// [`Dist`] (or use the constructor functions like [`normal`]).
pub use rand_distr;

use rand::distr::Distribution as RandDistribution;
use rand::rand_core::RngCore;

/// Adapter turning one of our [`Rng`]s into a [`rand::RngCore`].
///
/// `rand_distr` distributions are generic over [`rand::RngCore`]; our
/// [`tpt_math_prob_core::Rng`] is deliberately much smaller (it only promises
/// [`next_u64`](`Rng::next_u64`)). `CoreRng` supplies the missing methods in
/// terms of `next_u64`, so a distribution can be sampled as
/// `dist.sample(&mut CoreRng(&mut our_rng))`.
///
/// The adapter is zero-cost and owns no state of its own.
///
/// # Examples
///
/// ```
/// use rand::RngCore;
/// use tpt_math_prob_core::SplitMix64;
/// use tpt_math_prob_dist::CoreRng;
///
/// let mut rng = SplitMix64::seed_from_u64(1);
/// let mut core = CoreRng(&mut rng);
/// let word: u32 = core.next_u32();
/// let quad: u64 = core.next_u64();
/// let _ = (word, quad);
/// ```
#[derive(Debug)]
pub struct CoreRng<R: ?Sized>(pub R);
impl<R: Rng + ?Sized> RngCore for CoreRng<R> {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        (self.0.next_u64() >> 32) as u32
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    #[inline]
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut chunks = dst.chunks_exact_mut(8);
        for chunk in chunks.by_ref() {
            chunk.copy_from_slice(&self.next_u64().to_le_bytes());
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let bits = self.next_u64().to_le_bytes();
            rem.copy_from_slice(&bits[..rem.len()]);
        }
    }
}

/// A `rand_distr` distribution adapted to our [`Distribution`](`tpt_math_prob_core::Distribution`)
/// trait.
///
/// Given a `rand_distr` distribution `D` that produces values of type `T`
/// (i.e. `rand_distr::Distribution<T>` is implemented for `D`), [`Dist::new`]
/// wraps it so it can be sampled through any of our [`Rng`]s:
///
/// ```
/// use tpt_math_prob_core::{Distribution, SplitMix64};
/// use tpt_math_prob_dist::{normal, Dist};
///
/// let mut rng = SplitMix64::seed_from_u64(5);
/// let dist = Dist::new(normal((0.0, 1.0)).unwrap());
/// let x: f64 = dist.sample(&mut rng);
/// assert!(x.is_finite());
/// ```
///
/// The blanket [`Distribution`](`tpt_math_prob_core::Distribution) impl below
/// means [`Dist`] works for *every* `rand_distr` distribution, not just the
/// ones with dedicated constructors in this crate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dist<D>(pub D);

impl<D> Dist<D> {
    /// Wrap a `rand_distr` distribution so it can be sampled through our [`Rng`].
    #[inline]
    pub fn new(dist: D) -> Self {
        Dist(dist)
    }

    /// Borrow the wrapped distribution.
    #[inline]
    pub fn inner(&self) -> &D {
        &self.0
    }

    /// Mutably borrow the wrapped distribution.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.0
    }

    /// Consume the adapter and return the wrapped distribution.
    #[inline]
    pub fn into_inner(self) -> D {
        self.0
    }
}

/// Bridge any `rand_distr` distribution into our [`Distribution`](`tpt_math_prob_core::Distribution`)
/// trait.
///
/// Because [`Dist`] is a local type, this `impl` is allowed (the orphan rule
/// forbids implementing our foreign [`Distribution`](`tpt_math_prob_core::Distribution)
/// directly for the foreign `rand_distr` distributions). Sampling simply
/// forwards to the `rand_distr` `Distribution::sample` through [`CoreRng`].
impl<D, T> Distribution<T> for Dist<D>
where
    D: RandDistribution<T>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        let mut r = CoreRng(rng);
        self.0.sample(&mut r)
    }
}

/// Wrap a *float*-producing `rand_distr` distribution and emit integer counts.
///
/// Some `rand_distr` distributions are defined to return an `f64` even though
/// they model integer quantities (most notably [`Poisson`](`rand_distr::Poisson`),
/// which yields a `f64` number of occurrences). [`AsU64`] adapts such a
/// distribution to our [`Distribution<u64>`](`tpt_math_prob_core::Distribution),
/// truncating the (non-negative, whole) `f64` value to `u64`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::{Distribution, SplitMix64};
/// use tpt_math_prob_dist::{poisson, AsU64};
///
/// let mut rng = SplitMix64::seed_from_u64(9);
/// let dist = AsU64(poisson(4.0).unwrap());
/// let n: u64 = dist.sample(&mut rng);
/// assert_eq!(n, n as f64 as u64);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct AsU64<D>(pub D);

impl<D> AsU64<D> {
    /// Wrap a float-producing `rand_distr` distribution as a `u64` source.
    #[inline]
    pub fn new(dist: D) -> Self {
        AsU64(dist)
    }

    /// Consume the adapter and return the wrapped distribution.
    #[inline]
    pub fn into_inner(self) -> D {
        self.0
    }
}

impl<D> Distribution<u64> for AsU64<D>
where
    D: RandDistribution<f64>,
{
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        let mut r = CoreRng(rng);
        let x: f64 = self.0.sample(&mut r);
        if x <= 0.0 {
            0
        } else if x >= u64::MAX as f64 {
            u64::MAX
        } else {
            x as u64
        }
    }
}

/// Generate a re-exported `rand_distr` distribution type alias plus an
/// ergonomic constructor function.
///
/// The alias keeps the canonical `rand_distr` name (so `use` sites read
/// naturally), and the constructor forwards to `rand_distr`'s `new`, mapping
/// the `rand_distr` error to a plain `&'static str` so the public surface is
/// allocation-free and `no_std`-friendly. A `const`-assertion documents the
/// `Output` type the distribution actually produces.
macro_rules! dist {
    (
        $(#[$meta:meta])*
        $func:ident, $name:ident, $inner:ident, $args:ty, $output:ty
    ) => {
        $(#[$meta])*
        #[allow(missing_docs)]
        pub type $name<F = f64> = rand_distr::$inner<F>;

        $(#[$meta])*
        ///
        /// # Errors
        ///
        /// Returns `Err(&str)` if the parameters are invalid for this
        /// distribution (e.g. non-positive scale, zero variance).
        pub fn $func(args: $args) -> Result<$name<f64>, &'static str> {
            const _: () = {
                // Verify the output type at compile time.
                fn _assert(_: $output) {}
            };
            <rand_distr::$inner<f64> as DistConstructable>::construct(args)
                .map_err(|_| "invalid parameters for distribution")
        }
    };
}

/// Helper trait used solely to give the [`dist!`] macro a uniform constructor
/// signature across the various `rand_distr` `new` argument shapes.
trait DistConstructable: Sized {
    /// Constructor argument tuple type (mirrors `rand_distr::*::new`).
    type Args;
    /// Error type returned by the underlying `rand_distr` constructor.
    type Error;
    /// Build the distribution, returning the `rand_distr` error on bad input.
    fn construct(args: Self::Args) -> Result<Self, Self::Error>;
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::Normal<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
{
    type Args = (F, F);
    type Error = rand_distr::NormalError;
    fn construct((m, s): (F, F)) -> Result<Self, Self::Error> {
        rand_distr::Normal::new(m, s)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::LogNormal<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
{
    type Args = (F, F);
    type Error = rand_distr::NormalError;
    fn construct((m, s): (F, F)) -> Result<Self, Self::Error> {
        rand_distr::LogNormal::new(m, s)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::Exp<F>
where
    rand_distr::Exp1: RandDistribution<F>,
{
    type Args = F;
    type Error = rand_distr::ExpError;
    fn construct(lambda: F) -> Result<Self, Self::Error> {
        rand_distr::Exp::new(lambda)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::Gamma<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
    rand_distr::Open01: RandDistribution<F>,
{
    type Args = (F, F);
    type Error = rand_distr::GammaError;
    fn construct((shape, scale): (F, F)) -> Result<Self, Self::Error> {
        rand_distr::Gamma::new(shape, scale)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::Beta<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
    rand_distr::Open01: RandDistribution<F>,
{
    type Args = (F, F);
    type Error = rand_distr::BetaError;
    fn construct((a, b): (F, F)) -> Result<Self, Self::Error> {
        rand_distr::Beta::new(a, b)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::ChiSquared<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
    rand_distr::Open01: RandDistribution<F>,
{
    type Args = F;
    type Error = rand_distr::ChiSquaredError;
    fn construct(k: F) -> Result<Self, Self::Error> {
        rand_distr::ChiSquared::new(k)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::StudentT<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
    rand_distr::Open01: RandDistribution<F>,
    rand_distr::ChiSquared<F>: RandDistribution<F>,
{
    type Args = F;
    type Error = rand_distr::ChiSquaredError;
    fn construct(nu: F) -> Result<Self, Self::Error> {
        rand_distr::StudentT::new(nu)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::Cauchy<F>
where
    rand_distr::StandardNormal: RandDistribution<F>,
    rand_distr::Exp1: RandDistribution<F>,
    rand_distr::Open01: RandDistribution<F>,
    rand_distr::StandardUniform: RandDistribution<F>,
    F: rand_distr::num_traits::FloatConst,
{
    type Args = (F, F);
    type Error = rand_distr::CauchyError;
    fn construct((median, scale): (F, F)) -> Result<Self, Self::Error> {
        rand_distr::Cauchy::new(median, scale)
    }
}

impl<F: rand_distr::num_traits::Float> DistConstructable for rand_distr::Poisson<F>
where
    rand_distr::StandardUniform: RandDistribution<F>,
    F: rand_distr::num_traits::FloatConst,
{
    type Args = F;
    type Error = rand_distr::PoissonError;
    fn construct(lambda: F) -> Result<Self, Self::Error> {
        rand_distr::Poisson::new(lambda)
    }
}

dist!(
    /// `Normal` (Gaussian) distribution, `N(mean, std_dev)`.
    normal, Normal, Normal, (f64, f64), f64
);
dist!(
    /// `LogNormal` distribution.
    lognormal, LogNormal, LogNormal, (f64, f64), f64
);
dist!(
    /// `Exp` (exponential) distribution with rate `lambda`.
    exp, Exp, Exp, f64, f64
);
dist!(
    /// `Gamma` distribution with shape and scale.
    gamma, Gamma, Gamma, (f64, f64), f64
);
dist!(
    /// `Beta` distribution with shape parameters `alpha`, `beta`.
    beta, Beta, Beta, (f64, f64), f64
);
dist!(
    /// `ChiSquared` distribution with `k` degrees of freedom.
    chi_squared, ChiSquared, ChiSquared, f64, f64
);
dist!(
    /// Student's `t` distribution with `nu` degrees of freedom.
    student_t, StudentT, StudentT, f64, f64
);
dist!(
    /// `Cauchy` distribution with `median` and `scale`.
    cauchy, Cauchy, Cauchy, (f64, f64), f64
);
dist!(
    /// `Poisson` distribution with rate `lambda` (samples returned as `u64`).
    poisson, Poisson, Poisson, f64, u64
);

/// `Uniform` distribution over an inclusive-exclusive range `[low, high)`.
///
/// Produces values of the sampled type `T` (e.g. `f64` or `u64`). The
/// constructor uses `rand_distr`'s `Uniform::new`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_core::{Distribution, SplitMix64};
/// use tpt_math_prob_dist::{uniform, Dist};
///
/// let mut rng = SplitMix64::seed_from_u64(2);
/// let dist = Dist::new(uniform(0.0_f64, 1.0).unwrap());
/// let x: f64 = dist.sample(&mut rng);
/// assert!((0.0..1.0).contains(&x));
/// ```
///
/// # Errors
///
/// Returns `Err(&str)` if `low >= high`.
pub fn uniform<T>(low: T, high: T) -> Result<rand_distr::Uniform<T>, &'static str>
where
    T: rand_distr::uniform::SampleUniform,
{
    rand_distr::Uniform::new(low, high)
        .map_err(|_| "invalid parameters for Uniform (require low < high)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_prob_core::{Distribution as CoreDist, Rng as CoreRngTrait, SplitMix64};

    /// Sample `n` values from `dist` via our [`Rng`] and return the mean.
    fn mean_f64<D: CoreDist<f64>>(dist: &D, n: u64) -> f64 {
        let mut rng = SplitMix64::seed_from_u64(0xDEAD_BEEF);
        let mut sum = 0.0;
        for _ in 0..n {
            sum += dist.sample(&mut rng);
        }
        sum / (n as f64)
    }

    /// Sample `n` values from `dist` via our [`Rng`] and return the mean.
    fn mean_u64<D: CoreDist<u64>>(dist: &D, n: u64) -> f64 {
        let mut rng = SplitMix64::seed_from_u64(0xDEAD_BEEF);
        let mut sum = 0.0;
        for _ in 0..n {
            sum += dist.sample(&mut rng) as f64;
        }
        sum / (n as f64)
    }

    #[test]
    fn normal_mean_and_variance() {
        let n = 200_000u64;
        let dist = Dist::new(normal((3.0, 2.0)).unwrap());
        let mut rng = SplitMix64::seed_from_u64(1234);
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        for _ in 0..n {
            let x: f64 = dist.sample(&mut rng);
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / (n as f64);
        let var = sum_sq / (n as f64) - mean * mean;
        assert!((mean - 3.0).abs() < 0.05, "mean={mean}");
        assert!((var - 4.0).abs() < 0.1, "var={var}");
    }

    #[test]
    fn uniform_float_mean() {
        let dist = Dist::new(uniform(0.0_f64, 1.0).unwrap());
        let m = mean_f64(&dist, 100_000);
        assert!((m - 0.5).abs() < 0.01, "mean={m}");
    }

    #[test]
    fn uniform_u64_range() {
        let dist = Dist::new(uniform(10u64, 20u64).unwrap());
        let mut rng = SplitMix64::seed_from_u64(7);
        for _ in 0..1000 {
            let x: u64 = dist.sample(&mut rng);
            assert!((10..20).contains(&x), "x={x}");
        }
    }

    #[test]
    fn exp_mean() {
        // exp(rate=2) has mean 0.5.
        let dist = Dist::new(exp(2.0).unwrap());
        let m = mean_f64(&dist, 100_000);
        assert!((m - 0.5).abs() < 0.02, "mean={m}");
    }

    #[test]
    fn poisson_mean() {
        // poisson(lambda=4) has mean lambda=4.
        let dist = AsU64(poisson(4.0).unwrap());
        let m = mean_u64(&dist, 100_000);
        assert!((m - 4.0).abs() < 0.05, "mean={m}");
    }

    #[test]
    fn lognormal_positive() {
        let dist = Dist::new(lognormal((0.0, 1.0)).unwrap());
        let mut rng = SplitMix64::seed_from_u64(11);
        for _ in 0..1000 {
            let x: f64 = dist.sample(&mut rng);
            assert!(x > 0.0);
        }
    }

    #[test]
    fn gamma_beta_positive() {
        let g = Dist::new(gamma((2.0, 1.0)).unwrap());
        let b = Dist::new(beta((2.0, 3.0)).unwrap());
        let mut rng = SplitMix64::seed_from_u64(13);
        for _ in 0..1000 {
            assert!(g.sample(&mut rng) >= 0.0);
            let x: f64 = b.sample(&mut rng);
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    fn chi_squared_studentt_cauchy_finite() {
        let chi = Dist::new(chi_squared(3.0).unwrap());
        let st = Dist::new(student_t(5.0).unwrap());
        let ca = Dist::new(cauchy((0.0, 1.0)).unwrap());
        let mut rng = SplitMix64::seed_from_u64(17);
        for _ in 0..1000 {
            assert!(chi.sample(&mut rng).is_finite());
            assert!(st.sample(&mut rng).is_finite());
            assert!(ca.sample(&mut rng).is_finite());
        }
    }

    #[test]
    fn core_rng_adapter_forwards_bits() {
        // Sampling through `CoreRng` must match sampling `rand_distr` directly
        // with the same seed, proving the adapter forwards bits faithfully.
        let gauss = rand_distr::Normal::new(0.0, 1.0).unwrap();
        let mut a = SplitMix64::seed_from_u64(42);
        let mut b = SplitMix64::seed_from_u64(42);
        for _ in 0..1000 {
            let via_adapter: f64 = Dist::new(&gauss).sample(&mut a);
            let direct: f64 = {
                let mut r = CoreRng(&mut b);
                gauss.sample(&mut r)
            };
            assert_eq!(via_adapter, direct);
        }
    }

    #[test]
    fn core_rng_next_u64_matches_inner() {
        // `CoreRng` must delegate `next_u64`/`next_u32` to the wrapped `Rng`.
        let mut a = SplitMix64::seed_from_u64(7);
        let mut b = SplitMix64::seed_from_u64(7);
        let mut core = CoreRng(&mut a);
        for _ in 0..1000 {
            assert_eq!(core.next_u64(), b.next_u64());
            assert_eq!(core.next_u32(), (b.next_u64() >> 32) as u32);
        }
    }

    #[test]
    fn direct_rand_distr_usage() {
        // A plain `rand_distr` usage example, bridged via `CoreRng`.
        let gauss = rand_distr::Normal::new(1.0, 0.5).unwrap();
        let mut rng = SplitMix64::seed_from_u64(2024);
        let mut core = CoreRng(&mut rng);
        let x: f64 = gauss.sample(&mut core);
        assert!(x.is_finite());
    }
}
