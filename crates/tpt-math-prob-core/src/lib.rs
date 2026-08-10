#![no_std]
//! Shared randomness traits for the `tpt-math-prob-*` family.
//!
//! This crate defines the traits the probability crates implement against:
//!
//! * [`Rng`] — a minimal source of pseudo-random bits (no `std`, no `alloc`).
//! * [`Distribution<T>`] — a thing that can draw a value of type `T`.
//! * [`Sampler<T>`] — blanket-implemented for every [`Distribution`], adding
//!   bulk-sampling helpers that write into a caller-provided slice.
//!
//! It also ships [`SplitMix64`], a small deterministic PRNG, so consumers have
//! a working `Rng` even in `no_std` environments where `rand` is unavailable.
//!
//! # Examples
//!
//! ```
//! use tpt_math_prob_core::{Rng, Distribution, Standard, SplitMix64};
//!
//! let mut rng = SplitMix64::seed_from_u64(42);
//! let x: f64 = Standard.sample(&mut rng);
//! assert!((0.0..=1.0).contains(&x));
//! ```
//!
//! [`Rng`]: Rng
//! [`Distribution<T>`]: Distribution
//! [`Sampler<T>`]: Sampler

pub use tpt_math_numeric as numeric;

/// A source of pseudo-random bits.
///
/// Implementors must provide [`next_u64`](Rng::next_u64); [`next_f64`] is
/// provided with a sensible default.
pub trait Rng {
    /// Return the next 64 bits of pseudo-random output.
    fn next_u64(&mut self) -> u64;

    /// Return the next `f64` uniformly distributed in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }
}

/// A distribution that can produce samples of type `T`.
pub trait Distribution<T> {
    /// Draw a single sample using `rng`.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T;
}

/// Bulk sampling for any [`Distribution`].
///
/// Every [`Distribution`] automatically implements `Sampler` via the blanket
/// impl below, gaining [`sample_slice`](Sampler::sample_slice).
pub trait Sampler<T>: Distribution<T> {
    /// Draw a single sample (default: delegate to [`Distribution::sample`]).
    fn sample_one<R: Rng + ?Sized>(&self, rng: &mut R) -> T
    where
        T: Sized,
    {
        self.sample(rng)
    }

    /// Fill `buf` with independent samples.
    fn sample_slice<R: Rng + ?Sized>(&self, rng: &mut R, buf: &mut [T])
    where
        T: Sized,
    {
        for slot in buf.iter_mut() {
            *slot = self.sample(rng);
        }
    }
}

impl<D, T> Sampler<T> for D where D: Distribution<T> {}

/// The standard uniform distribution over the type's natural range.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Standard;

impl Distribution<f64> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        rng.next_f64()
    }
}

impl Distribution<f32> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f32 {
        rng.next_f64() as f32
    }
}

impl Distribution<u64> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        rng.next_u64()
    }
}

impl Distribution<u32> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u32 {
        (rng.next_u64() >> 32) as u32
    }
}

impl Distribution<i64> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        rng.next_u64() as i64
    }
}

impl Distribution<bool> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        (rng.next_u64() & 1) == 1
    }
}

/// `SplitMix64`: a tiny, fast, deterministic 64-bit PRNG.
///
/// Not cryptographically secure, but excellent for reproducible tests,
/// simulations, and `no_std` use where `rand`'s generators are unavailable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Create a generator from an explicit 64-bit state.
    pub fn new(state: u64) -> Self {
        SplitMix64 { state }
    }

    /// Create a generator seeded from a `u64` (the canonical entry point).
    pub fn seed_from_u64(seed: u64) -> Self {
        SplitMix64::new(seed.wrapping_add(0x9E37_79B9_7F4A_7C15))
    }
}

impl Rng for SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Any mutable reference to an [`Rng`] is itself an [`Rng`].
///
/// This lets a borrowed generator be passed to APIs that take `&mut impl Rng`
/// (e.g. the sampler helpers) without cloning the underlying state.
impl<R: Rng + ?Sized> Rng for &mut R {
    fn next_u64(&mut self) -> u64 {
        (**self).next_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_is_deterministic() {
        let mut a = SplitMix64::seed_from_u64(1);
        let mut b = SplitMix64::seed_from_u64(1);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn standard_f64_is_in_unit_interval() {
        let mut rng = SplitMix64::seed_from_u64(7);
        for _ in 0..1000 {
            let x: f64 = Standard.sample(&mut rng);
            assert!((0.0..=1.0).contains(&x));
        }
    }

    #[test]
    fn sampler_fills_slice() {
        let mut rng = SplitMix64::seed_from_u64(99);
        let mut buf = [0u64; 8];
        Standard.sample_slice(&mut rng, &mut buf);
        assert_ne!(buf, [0u64; 8]);
    }
}
