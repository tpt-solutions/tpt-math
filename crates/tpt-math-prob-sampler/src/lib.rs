#![no_std]
//! Sampling strategies for the `tpt-math-prob-*` family.
//!
//! This crate consolidates reusable sampling *strategies* on top of the shared
//! [`tpt_math_prob_core`] traits ([`Distribution`], [`Rng`], [`Sampler`]). It is
//! `no_std`-compatible; the bulk of the API is allocation-free and works with
//! any slice-backed storage, while the convenience constructors that own a
//! [`alloc::vec::Vec`] are gated behind the `std` (default) or `alloc`
//! features.
//!
//! The crate provides:
//!
//! * [`InverseCdfSampler`] — inverse-transform (alias) sampling over a weighted
//!   categorical distribution, generic over any CDF storage.
//! * [`SystematicResampler`] (with multinomial, systematic and stratified
//!   resampling schemes) — resample `n` particles from weighted possibilities,
//!   writing into a caller-provided index buffer.
//! * [`ReservoirSampler`] — streaming `n`-from-`k` (with `k` unknown) uniform
//!   reservoir sampling.
//! * [`RejectionSampler`] — standard rejection sampling that wraps a target
//!   density and a proposal [`Distribution`] + [`Density`] with a bounding
//!   constant `m`.
//! * Free helpers: [`categorical`], [`sample_categorical`], [`uniform_index`],
//!   [`shuffle`], and the small [`Uniform`] / [`Bernoulli`] building blocks.
//!
//! # Examples
//!
//! ```
//! use tpt_math_prob_sampler::{categorical, SystematicResampler, ResampleScheme};
//! use tpt_math_prob_core::{Distribution, Rng, Standard, SplitMix64};
//!
//! let mut rng = SplitMix64::seed_from_u64(0);
//! let cat = categorical(&[1.0, 2.0, 3.0]).unwrap();
//! let x = cat.sample(&mut rng);
//! assert!(x < 3);
//!
//! let resampler = SystematicResampler::new(cat.cdf().to_vec()).unwrap();
//! let mut out = [0usize; 5];
//! resampler.sample_indices(&mut rng, ResampleScheme::Systematic, &mut out);
//! assert_eq!(out.len(), 5);
//! ```

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

// Re-export the core traits/types so consumers only depend on this crate.
pub use tpt_math_prob_core::{Distribution, Rng, Sampler, SplitMix64, Standard};

use core::fmt;

#[cfg(any(feature = "std", feature = "alloc"))]
use alloc::vec::Vec;

/// Absolute value without relying on `f64::abs` (kept `no_std`-portable).
#[inline]
fn fabs(x: f64) -> f64 {
    if x < 0.0 {
        -x
    } else {
        x
    }
}

/// Draw a single `f64` from [`Standard`] (avoids ambiguous-trait inference).
#[inline]
fn uniform_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    Distribution::<f64>::sample(&Standard, rng)
}

/// Errors produced while constructing or validating a sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SamplerError {
    /// No weights / empty distribution.
    Empty,
    /// A weight was negative.
    NegativeWeight,
    /// A weight was not finite (`NaN` or `inf`).
    NonFiniteWeight,
    /// The total weight was zero.
    ZeroTotal,
    /// A `from_cdf` CDF was not monotone non-decreasing or did not end in `1`.
    InvalidCdf,
    /// A probability parameter was outside `[0, 1]`.
    InvalidProbability,
    /// A rejection-sampling envelope constant `m` was not strictly positive.
    InvalidBound,
    /// A uniform interval was empty or invalid (`lo >= hi`).
    InvalidInterval,
    /// Requested `n` exceeds the number of distinct items.
    TooManySamples,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SamplerError::Empty => "empty distribution (no weights)",
            SamplerError::NegativeWeight => "negative weight encountered",
            SamplerError::NonFiniteWeight => "non-finite (NaN/inf) weight encountered",
            SamplerError::ZeroTotal => "total weight is zero",
            SamplerError::InvalidCdf => "CDF is not monotone non-decreasing ending in 1",
            SamplerError::InvalidProbability => "probability not in [0, 1]",
            SamplerError::InvalidBound => "envelope constant m must be strictly positive",
            SamplerError::InvalidInterval => "uniform interval lo >= hi",
            SamplerError::TooManySamples => "requested sample count exceeds the population",
        };
        f.write_str(s)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SamplerError {}

/// Validate a slice of unnormalised weights and return their (positive) sum.
fn validate_weights(weights: &[f64]) -> Result<f64, SamplerError> {
    if weights.is_empty() {
        return Err(SamplerError::Empty);
    }
    let mut total = 0.0;
    for &w in weights {
        if !w.is_finite() {
            return Err(SamplerError::NonFiniteWeight);
        }
        if w < 0.0 {
            return Err(SamplerError::NegativeWeight);
        }
        total += w;
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(SamplerError::ZeroTotal);
    }
    Ok(total)
}

/// Locate the first index `i` with `cdf[i] > u` (inverse-transform lookup).
///
/// `cdf` is assumed normalised so that `cdf.last() == 1` and is non-decreasing.
/// `u` must be in `[0, 1)`.
#[inline]
fn inverse_cdf_index(cdf: &[f64], u: f64) -> usize {
    let mut idx = cdf.partition_point(|&c| c <= u);
    if idx >= cdf.len() {
        // Defensive: u could be exactly 1.0 due to rounding; clamp into the
        // last bin that actually has positive probability.
        idx = cdf.len() - 1;
        while idx > 0 && cdf[idx] == cdf[idx - 1] {
            idx -= 1;
        }
    }
    idx
}

// =====================================================================
// Inverse-transform (alias) sampler
// =====================================================================

/// Inverse-transform sampler for a categorical (discrete) distribution.
///
/// Given a cumulative distribution function `cdf` (non-decreasing, ending in
/// `1.0`) stored in any type implementing [`AsRef<[f64]>`][core::convert::AsRef],
/// it draws an `usize` index by the alias / inverse-CDF method. The storage is
/// generic so the same type works with a borrowed `&[f64]`, an array, or an
/// owned [`alloc::vec::Vec`].
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::{InverseCdfSampler, categorical};
/// use tpt_math_prob_core::{Distribution, Rng, SplitMix64};
///
/// let cdf = categorical(&[1.0, 1.0, 2.0]).unwrap();
/// let s = InverseCdfSampler::from_cdf(cdf.cdf().to_vec()).unwrap();
/// let mut rng = SplitMix64::seed_from_u64(3);
/// let i = s.sample(&mut rng);
/// assert!(i < 3);
/// ```
pub struct InverseCdfSampler<S> {
    cdf: S,
}

impl<S: AsRef<[f64]>> Clone for InverseCdfSampler<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        InverseCdfSampler {
            cdf: self.cdf.clone(),
        }
    }
}
impl<S: AsRef<[f64]>> Copy for InverseCdfSampler<S> where S: Copy {}
impl<S: AsRef<[f64]>> fmt::Debug for InverseCdfSampler<S>
where
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InverseCdfSampler")
            .field("cdf", &self.cdf)
            .finish()
    }
}
impl<S: AsRef<[f64]>> PartialEq for InverseCdfSampler<S>
where
    S: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.cdf == other.cdf
    }
}

/// Convenience alias for the owned, [`alloc`] backed inverse-CDF sampler.
#[cfg(any(feature = "std", feature = "alloc"))]
pub type WeightedIndex = InverseCdfSampler<Vec<f64>>;

impl<S: AsRef<[f64]>> InverseCdfSampler<S> {
    /// Build a sampler from an existing, already-normalised cumulative
    /// distribution function.
    ///
    /// The CDF must be non-decreasing, all entries in `[0, 1]`, and end in
    /// exactly `1.0` (within `1e-12`). Returns [`SamplerError::InvalidCdf`]
    /// otherwise.
    pub fn from_cdf(cdf: S) -> Result<Self, SamplerError> {
        let slice = cdf.as_ref();
        if slice.is_empty() {
            return Err(SamplerError::Empty);
        }
        let mut prev = 0.0;
        for &c in slice {
            if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                return Err(SamplerError::InvalidCdf);
            }
            if c < prev {
                return Err(SamplerError::InvalidCdf);
            }
            prev = c;
        }
        if fabs(prev - 1.0) > 1e-12 {
            return Err(SamplerError::InvalidCdf);
        }
        Ok(InverseCdfSampler { cdf })
    }

    /// The number of categories.
    pub fn len(&self) -> usize {
        self.cdf.as_ref().len()
    }

    /// True when there are no categories (constructed from an empty CDF).
    pub fn is_empty(&self) -> bool {
        self.cdf.as_ref().is_empty()
    }

    /// Borrow the underlying cumulative distribution function.
    pub fn cdf(&self) -> &[f64] {
        self.cdf.as_ref()
    }

    /// Probability of category `i` (difference of adjacent CDF entries).
    ///
    /// Returns `0.0` for out-of-range `i`.
    pub fn probability(&self, i: usize) -> f64 {
        let cdf = self.cdf.as_ref();
        if i >= cdf.len() {
            return 0.0;
        }
        if i == 0 {
            cdf[0]
        } else {
            cdf[i] - cdf[i - 1]
        }
    }

    /// Draw an index given a uniform variate `u in [0, 1)`.
    ///
    /// Useful for deterministic / systematic resampling.
    pub fn sample_index_from_uniform(&self, u: f64) -> usize {
        inverse_cdf_index(self.cdf.as_ref(), u)
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl InverseCdfSampler<Vec<f64>> {
    /// Build a sampler from a slice of non-negative unnormalised `weights`.
    ///
    /// The weights are normalised and turned into a cumulative distribution
    /// function. Returns an error if the slice is empty, any weight is
    /// negative / non-finite, or the total weight is zero.
    pub fn from_weights(weights: &[f64]) -> Result<Self, SamplerError> {
        let total = validate_weights(weights)?;
        let mut cdf = Vec::with_capacity(weights.len());
        let mut acc = 0.0;
        for &w in weights {
            acc += w / total;
            cdf.push(acc);
        }
        // Guard against floating-point drift: force the final entry to 1.
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        Ok(InverseCdfSampler { cdf })
    }
}

impl<S: AsRef<[f64]>> Distribution<usize> for InverseCdfSampler<S> {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let u: f64 = Standard.sample(rng);
        self.sample_index_from_uniform(u)
    }
}

/// Build a [`WeightedIndex`] (owned, [`alloc`]-backed inverse-CDF sampler) from
/// unnormalised `weights`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::categorical;
/// let cat = categorical(&[0.2, 0.8]).unwrap();
/// assert_eq!(cat.len(), 2);
/// ```
#[cfg(any(feature = "std", feature = "alloc"))]
pub fn categorical(weights: &[f64]) -> Result<WeightedIndex, SamplerError> {
    WeightedIndex::from_weights(weights)
}

/// Draw a categorical index from `weights` using allocation-free linear scan
/// (inverse-transform without a normalised CDF table).
///
/// `weights` are non-negative unnormalised probabilities. Returns `None` if the
/// weights are invalid (empty, negative, non-finite or summing to zero).
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::sample_categorical;
/// use tpt_math_prob_core::{Rng, SplitMix64};
/// let mut rng = SplitMix64::seed_from_u64(1);
/// let i = sample_categorical(&[1.0, 1.0, 1.0], &mut rng).unwrap();
/// assert!(i < 3);
/// ```
pub fn sample_categorical<R: Rng + ?Sized>(weights: &[f64], rng: &mut R) -> Option<usize> {
    let total = validate_weights(weights).ok()?;
    let mut u = uniform_f64(rng) * total;
    for (i, &w) in weights.iter().enumerate() {
        if u < w {
            return Some(i);
        }
        u -= w;
    }
    // Rounding can land exactly on the total; return the last index.
    Some(weights.len() - 1)
}

// =====================================================================
// Systematic / multinomial / stratified resampling
// =====================================================================

/// The resampling scheme used by [`SystematicResampler::sample_indices`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResampleScheme {
    /// Independent inverse-CDF draws (a plain multinomial resample).
    Multinomial,
    /// One random offset; evenly spaced "ladders" through the CDF
    /// (low variance, deterministic given the seed).
    Systematic,
    /// One random offset per slot, jittered inside its own stratum.
    Stratified,
}

/// Resamples `n` particle *indices* from a weighted categorical distribution.
///
/// Unlike [`InverseCdfSampler`] this writes `n` indices into a caller-provided
/// buffer ([`sample_indices`](SystematicResampler::sample_indices)) and supports
/// the low-variance [systematic](ResampleScheme::Systematic) and
/// [stratified](ResampleScheme::Stratified) schemes in addition to a plain
/// [multinomial](ResampleScheme::Multinomial) resample.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::{categorical, SystematicResampler, ResampleScheme};
/// use tpt_math_prob_core::{Rng, SplitMix64};
/// let cat = categorical(&[1.0, 2.0, 3.0]).unwrap();
/// let resampler = SystematicResampler::new(cat.cdf().to_vec()).unwrap();
/// let mut rng = SplitMix64::seed_from_u64(0);
/// let mut out = [0usize; 4];
/// resampler.sample_indices(&mut rng, ResampleScheme::Systematic, &mut out);
/// assert!(out.iter().all(|&i| i < 3));
/// ```
pub struct SystematicResampler<S> {
    sampler: InverseCdfSampler<S>,
}

impl<S: AsRef<[f64]>> Clone for SystematicResampler<S>
where
    S: Clone,
{
    fn clone(&self) -> Self {
        SystematicResampler {
            sampler: self.sampler.clone(),
        }
    }
}
impl<S: AsRef<[f64]>> Copy for SystematicResampler<S> where S: Copy {}
impl<S: AsRef<[f64]>> fmt::Debug for SystematicResampler<S>
where
    S: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SystematicResampler")
            .field("sampler", &self.sampler)
            .finish()
    }
}
impl<S: AsRef<[f64]>> PartialEq for SystematicResampler<S>
where
    S: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.sampler == other.sampler
    }
}

#[cfg(any(feature = "std", feature = "alloc"))]
impl SystematicResampler<Vec<f64>> {
    /// Build from unnormalised `weights`.
    pub fn from_weights(weights: &[f64]) -> Result<Self, SamplerError> {
        Ok(SystematicResampler {
            sampler: InverseCdfSampler::from_weights(weights)?,
        })
    }
}

impl<S: AsRef<[f64]>> SystematicResampler<S> {
    /// Build from an existing normalised CDF (`InverseCdfSampler::from_cdf`).
    pub fn new(cdf: S) -> Result<Self, SamplerError> {
        Ok(SystematicResampler {
            sampler: InverseCdfSampler::from_cdf(cdf)?,
        })
    }

    /// Number of source categories.
    pub fn len(&self) -> usize {
        self.sampler.len()
    }

    /// True when there are no categories.
    pub fn is_empty(&self) -> bool {
        self.sampler.is_empty()
    }

    /// Borrow the underlying cumulative distribution function.
    pub fn cdf(&self) -> &[f64] {
        self.sampler.cdf()
    }

    /// Resample `n` indices into `out` using the chosen [`ResampleScheme`].
    ///
    /// `out.len()` determines `n`; the resampled indices are in `0..len()`.
    pub fn sample_indices<R: Rng + ?Sized>(
        &self,
        rng: &mut R,
        scheme: ResampleScheme,
        out: &mut [usize],
    ) {
        let cdf = self.sampler.cdf();
        let n = out.len();
        if n == 0 {
            return;
        }
        match scheme {
            ResampleScheme::Multinomial => {
                for slot in out.iter_mut() {
                    let u = uniform_f64(rng);
                    *slot = inverse_cdf_index(cdf, u);
                }
            }
            ResampleScheme::Systematic => {
                let u0: f64 = uniform_f64(rng) / (n as f64);
                let mut cursor = 0usize;
                for (j, slot) in out.iter_mut().enumerate() {
                    let u = u0 + (j as f64) / (n as f64);
                    while cursor + 1 < cdf.len() && cdf[cursor] <= u {
                        cursor += 1;
                    }
                    *slot = cursor;
                }
            }
            ResampleScheme::Stratified => {
                for (j, slot) in out.iter_mut().enumerate() {
                    let u: f64 = (uniform_f64(rng) + j as f64) / (n as f64);
                    *slot = inverse_cdf_index(cdf, u);
                }
            }
        }
    }
}

// =====================================================================
// Reservoir sampling (streaming n-of-k, k unknown)
// =====================================================================

/// Streaming reservoir sampler: keep a uniformly random sample of size `cap`
/// from a stream of `k` items of unknown length.
///
/// Fill it with [`offer`](ReservoirSampler::offer) for every incoming item.
/// Once `cap` items have been seen, each subsequent item replaces a uniformly
/// chosen reservoir slot with probability `cap / seen`, so every item seen so
/// far is equally likely to be in the reservoir.
///
/// The sampler borrows a caller-owned buffer `buf` of length `cap`; it never
/// allocates.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::ReservoirSampler;
/// use tpt_math_prob_core::{Rng, SplitMix64};
/// let mut rng = SplitMix64::seed_from_u64(0);
/// let mut slots = [0u32; 3];
/// let mut rs = ReservoirSampler::new(&mut slots);
/// for x in 0u32..100 { rs.offer(x, &mut rng); }
/// // every value 0..100 was equally likely to survive
/// assert!(slots.iter().all(|&v| v < 100));
/// ```
pub struct ReservoirSampler<'a, T> {
    buf: &'a mut [T],
    seen: u64,
}

impl<'a, T> ReservoirSampler<'a, T> {
    /// Create a reservoir backed by `buf` (length = capacity `cap`).
    pub fn new(buf: &'a mut [T]) -> Self {
        ReservoirSampler { buf, seen: 0 }
    }

    /// Capacity `cap` of the reservoir.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// How many items have been offered so far.
    pub fn seen(&self) -> u64 {
        self.seen
    }

    /// Current contents (the first `min(seen, cap)` slots are live).
    pub fn sample(&self) -> &[T] {
        let live = self.seen as usize;
        let n = if live < self.buf.len() {
            live
        } else {
            self.buf.len()
        };
        &self.buf[..n]
    }

    /// Number of live (filled) slots.
    pub fn len(&self) -> usize {
        self.sample().len()
    }

    /// True when no items have been offered yet.
    pub fn is_empty(&self) -> bool {
        self.seen == 0
    }

    /// True when the reservoir is full (at capacity).
    pub fn is_full(&self) -> bool {
        self.seen as usize >= self.buf.len()
    }

    /// Reset the stream (drop all offered items) without clearing values.
    pub fn reset(&mut self) {
        self.seen = 0;
    }

    /// Offer one `item` from the stream.
    ///
    /// While the reservoir is filling, `item` is appended. Once full, `item`
    /// replaces a uniformly chosen slot with probability `cap / seen`; the
    /// displaced item is returned (or `None` if `item` was itself rejected).
    pub fn offer<R: Rng + ?Sized>(&mut self, item: T, rng: &mut R) -> Option<T> {
        let cap = self.buf.len();
        let idx = self.seen as usize;
        self.seen = self.seen.wrapping_add(1);
        if idx < cap {
            // Filling phase: append, displacing the placeholder.
            let _ = core::mem::replace(&mut self.buf[idx], item);
            return None;
        }
        // Full phase: accept with probability cap / seen. Pick a slot in
        // `0..seen`; if it falls in the reservoir (cap slots) we replace it.
        let j = uniform_index(rng, self.seen as usize).expect("seen > 0");
        if j < cap {
            Some(core::mem::replace(&mut self.buf[j], item))
        } else {
            Some(item)
        }
    }
}

/// Draw a uniform `usize` in `0..n` (unbiased, rejection-based).
///
/// Returns `None` if `n == 0`. Operates only on the `Rng` interface (no
/// allocation), and is the building block for [`shuffle`] and
/// [`ReservoirSampler`].
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::uniform_index;
/// use tpt_math_prob_core::{Rng, SplitMix64};
/// let mut rng = SplitMix64::seed_from_u64(0);
/// let i = uniform_index(&mut rng, 6).unwrap();
/// assert!(i < 6);
/// ```
pub fn uniform_index<R: Rng + ?Sized>(rng: &mut R, n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(0);
    }
    let span = n as u64;
    let limit = if span.is_power_of_two() {
        u64::MAX
    } else {
        // Largest multiple of `n` <= u64::MAX; values >= limit are rejected.
        (u64::MAX / span) * span
    };
    loop {
        let x = rng.next_u64();
        if x < limit {
            return Some((x % span) as usize);
        }
    }
}

/// In-place Fisher–Yates shuffle of `slice`.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::shuffle;
/// use tpt_math_prob_core::{Rng, SplitMix64};
/// let mut v = [1, 2, 3, 4, 5];
/// let mut rng = SplitMix64::seed_from_u64(0);
/// shuffle(&mut v, &mut rng);
/// v.sort_unstable();
/// assert_eq!(v, [1, 2, 3, 4, 5]);
/// ```
pub fn shuffle<R: Rng + ?Sized, T>(slice: &mut [T], rng: &mut R) {
    let n = slice.len();
    for i in (1..n).rev() {
        let j = uniform_index(rng, i + 1).expect("i + 1 > 0");
        slice.swap(i, j);
    }
}

// =====================================================================
// Densities and rejection sampling
// =====================================================================

/// A probability density (evaluated on the real line) for rejection sampling.
///
/// A [`Distribution<f64>`] only knows how to *draw*; a `Density` additionally
/// knows the value of its probability density function, which is required for
/// the acceptance test of [`RejectionSampler`].
pub trait Density {
    /// Density at `x`. Must be `>= 0` everywhere and finite at `x`.
    fn density(&self, x: f64) -> f64;
}

/// Adapter turning a closure `Fn(f64) -> f64` into a [`Density`].
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::{DensityFn, Density};
/// let d = DensityFn(|x: f64| if x >= 0.0 && x <= 1.0 { 1.0 } else { 0.0 });
/// assert_eq!(d.density(0.5), 1.0);
/// ```
pub struct DensityFn<F>(pub F);

impl<F: Fn(f64) -> f64> Density for DensityFn<F> {
    fn density(&self, x: f64) -> f64 {
        (self.0)(x)
    }
}

/// Continuous uniform distribution over `[lo, hi)`.
///
/// Implements [`Distribution<f64>`] (drawing) and [`Density`] (evaluation), so
/// it can serve as a proposal for [`RejectionSampler`].
pub struct Uniform {
    lo: f64,
    hi: f64,
    width: f64,
}

impl Uniform {
    /// Construct a uniform distribution on `[lo, hi)`.
    ///
    /// Returns [`SamplerError::InvalidInterval`] when `lo >= hi` or either bound
    /// is non-finite.
    pub fn new(lo: f64, hi: f64) -> Result<Self, SamplerError> {
        if !lo.is_finite() || !hi.is_finite() || lo >= hi {
            return Err(SamplerError::InvalidInterval);
        }
        Ok(Uniform {
            lo,
            hi,
            width: hi - lo,
        })
    }

    /// Lower bound.
    pub fn lo(&self) -> f64 {
        self.lo
    }

    /// Upper bound.
    pub fn hi(&self) -> f64 {
        self.hi
    }
}

impl Distribution<f64> for Uniform {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u: f64 = Standard.sample(rng);
        self.lo + u * self.width
    }
}

impl Density for Uniform {
    fn density(&self, x: f64) -> f64 {
        if x >= self.lo && x < self.hi {
            1.0 / self.width
        } else {
            0.0
        }
    }
}

/// Bernoulli (`p` success) distribution returning `bool`.
pub struct Bernoulli {
    p: f64,
}

impl Bernoulli {
    /// Construct a Bernoulli distribution with success probability `p`.
    ///
    /// Returns [`SamplerError::InvalidProbability`] if `p` is outside `[0, 1]`.
    pub fn new(p: f64) -> Result<Self, SamplerError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(SamplerError::InvalidProbability);
        }
        Ok(Bernoulli { p })
    }

    /// The success probability `p`.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution<bool> for Bernoulli {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> bool {
        let u: f64 = Standard.sample(rng);
        u < self.p
    }
}

/// Standard rejection sampler.
///
/// Draws from a target [`Density`] `target` by proposing from `proposal`
/// (any [`Distribution<f64>`] + [`Density`]) and accepting with probability
///
/// ```text
/// accept = target.density(x) / (m * proposal.density(x))
/// ```
///
/// for an envelope constant `m` satisfying `m * proposal.density(x) >=
/// target.density(x)` for all `x`. Exposure via [`Distribution<f64>`] is
/// guaranteed to terminate by giving up after [`max_iterations`](RejectionSampler::max_iterations)
/// (default `1024`) rejections.
///
/// # Examples
///
/// ```
/// use tpt_math_prob_sampler::{RejectionSampler, Uniform, DensityFn};
/// use tpt_math_prob_core::{Rng, SplitMix64, Distribution};
/// // Sample the triangular density 2x on [0,1] using a Uniform(0,1) proposal.
/// let target = DensityFn(|x: f64| if x >= 0.0 && x <= 1.0 { 2.0 * x } else { 0.0 });
/// let proposal = Uniform::new(0.0, 1.0).unwrap();
/// let rs = RejectionSampler::new(target, proposal, 2.0).unwrap();
/// let mut rng = SplitMix64::seed_from_u64(0);
/// let x = rs.sample(&mut rng);
/// assert!(x >= 0.0 && x <= 1.0);
/// ```
pub struct RejectionSampler<T, P> {
    target: T,
    proposal: P,
    m: f64,
    max_iterations: usize,
}

impl<T, P> RejectionSampler<T, P>
where
    T: Density,
    P: Distribution<f64> + Density,
{
    /// Construct a rejection sampler with envelope constant `m > 0`.
    ///
    /// `m` must satisfy `m * proposal.density(x) >= target.density(x)` for all
    /// `x`; it is validated to be finite and strictly positive.
    pub fn new(target: T, proposal: P, m: f64) -> Result<Self, SamplerError> {
        if !m.is_finite() || m <= 0.0 {
            return Err(SamplerError::InvalidBound);
        }
        Ok(RejectionSampler {
            target,
            proposal,
            m,
            max_iterations: 1024,
        })
    }

    /// Borrow the target density.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Borrow the proposal distribution.
    pub fn proposal(&self) -> &P {
        &self.proposal
    }

    /// The envelope constant `m`.
    pub fn m(&self) -> f64 {
        self.m
    }

    /// Set the maximum number of rejection attempts (default `1024`).
    pub fn max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Best-case acceptance probability `1 / m` (the proposal is optimal when
    /// the envelope is tight).
    pub fn acceptance_probability(&self) -> f64 {
        1.0 / self.m
    }

    /// Attempt to draw a sample, returning `None` if `max_iterations`
    /// rejections occur first.
    pub fn try_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<f64> {
        for _ in 0..self.max_iterations {
            let x = self.proposal.sample(rng);
            let pd = self.proposal.density(x);
            if pd <= 0.0 {
                continue;
            }
            let td = self.target.density(x);
            let u: f64 = Standard.sample(rng);
            if u * self.m * pd <= td {
                return Some(x);
            }
        }
        None
    }
}

impl<T, P> Distribution<f64> for RejectionSampler<T, P>
where
    T: Density,
    P: Distribution<f64> + Density,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        self.try_sample(rng)
            .unwrap_or_else(|| self.proposal.sample(rng))
    }
}

// =====================================================================
// Prelude
// =====================================================================

/// Convenient re-exports for sampling work.
pub mod prelude {
    pub use crate::{
        Bernoulli, Density, DensityFn, Distribution, InverseCdfSampler, RejectionSampler,
        ReservoirSampler, Rng, Sampler, SamplerError, SystematicResampler, Uniform,
    };
    pub use tpt_math_prob_core::{SplitMix64, Standard};

    #[cfg(any(feature = "std", feature = "alloc"))]
    pub use crate::{categorical, WeightedIndex};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(feature = "std", feature = "alloc"))]
    use alloc::vec;
    #[cfg(any(feature = "std", feature = "alloc"))]
    use alloc::vec::Vec;

    const SEED: u64 = 0x1234_5678;

    #[test]
    fn cdf_from_weights_is_normalised() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let s = InverseCdfSampler::from_weights(&[1.0, 2.0, 3.0]).unwrap();
            let cdf = s.cdf();
            assert!((cdf.last().unwrap() - 1.0).abs() < 1e-12);
            assert!((s.probability(0) - 1.0 / 6.0).abs() < 1e-12);
            assert!((s.probability(1) - 2.0 / 6.0).abs() < 1e-12);
            assert!((s.probability(2) - 3.0 / 6.0).abs() < 1e-12);
        }
    }

    #[test]
    fn from_cdf_rejects_bad_input() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            assert_eq!(
                InverseCdfSampler::from_cdf(vec![0.5, 0.4, 1.0]),
                Err(SamplerError::InvalidCdf)
            );
            assert_eq!(
                InverseCdfSampler::from_cdf(vec![0.5, 0.9]),
                Err(SamplerError::InvalidCdf)
            );
        }
        let _ = SEED;
    }

    #[test]
    fn weights_validation_errors() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            assert_eq!(
                InverseCdfSampler::from_weights(&[]),
                Err(SamplerError::Empty)
            );
            assert_eq!(
                InverseCdfSampler::from_weights(&[-1.0]),
                Err(SamplerError::NegativeWeight)
            );
            assert_eq!(
                InverseCdfSampler::from_weights(&[f64::NAN]),
                Err(SamplerError::NonFiniteWeight)
            );
            assert_eq!(
                InverseCdfSampler::from_weights(&[0.0, 0.0]),
                Err(SamplerError::ZeroTotal)
            );
        }
    }

    #[test]
    fn zero_weight_never_selected() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let s = InverseCdfSampler::from_weights(&[1.0, 0.0, 0.0, 1.0]).unwrap();
            let mut rng = SplitMix64::seed_from_u64(SEED);
            for _ in 0..5000 {
                let i = s.sample(&mut rng);
                assert!(i == 0 || i == 3);
            }
        }
    }

    #[test]
    fn sample_index_from_uniform_boundaries() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let s = InverseCdfSampler::from_weights(&[0.5, 0.5]).unwrap();
            assert_eq!(s.sample_index_from_uniform(0.0), 0);
            assert_eq!(s.sample_index_from_uniform(0.499999), 0);
            assert_eq!(s.sample_index_from_uniform(0.5), 1);
            assert_eq!(s.sample_index_from_uniform(0.999999), 1);
        }
    }

    #[test]
    fn linear_categorical_matches_alias() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let weights = [1.0, 3.0, 2.0, 4.0];
            let alias = categorical(&weights).unwrap();
            let mut rng = SplitMix64::seed_from_u64(SEED);
            let mut acc = [0usize; 4];
            for _ in 0..40000 {
                let a = alias.sample(&mut rng);
                let b = sample_categorical(&weights, &mut rng).unwrap();
                acc[a] += 1;
                acc[b] += 1;
            }
            let total = acc.iter().sum::<usize>() as f64;
            let sum = weights.iter().sum::<f64>();
            for i in 0..4 {
                let expected = weights[i] / sum;
                assert!((acc[i] as f64 / total - expected).abs() < 0.02, "bin {i}");
            }
        }
    }

    #[test]
    fn systematic_resample_counts_bounded() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let weights = [1.0, 1.0, 2.0];
            let r = SystematicResampler::from_weights(&weights).unwrap();
            let mut rng = SplitMix64::seed_from_u64(SEED);
            let n = 1000usize;
            let mut out = vec![0usize; n];
            // Each scheme should keep every index inside the category range.
            for &scheme in &[
                ResampleScheme::Multinomial,
                ResampleScheme::Systematic,
                ResampleScheme::Stratified,
            ] {
                r.sample_indices(&mut rng, scheme, &mut out);
                assert!(out.iter().all(|&i| i < 3));
                // counts within [floor, ceil] of n*p_i
                let mut counts = [0usize; 3];
                for &i in &out {
                    counts[i] += 1;
                }
                let total = weights.iter().sum::<f64>();
                let n_f = n as f64;
                for (i, &c) in counts.iter().enumerate() {
                    let p = weights[i] / total;
                    if scheme == ResampleScheme::Systematic {
                        // Systematic resampling keeps each count within
                        // [floor, ceil] of its expectation.
                        let lo = (n_f * p).floor() as usize;
                        let hi = (n_f * p).ceil() as usize;
                        assert!(
                            c >= lo && c <= hi,
                            "scheme {scheme:?} bin {i}: {c} not in [{lo},{hi}]"
                        );
                    } else {
                        // Multinomial / stratified have variance; only check
                        // the empirical mean is approximately correct.
                        let mean = c as f64 / n_f;
                        assert!(
                            (mean - p).abs() < 0.05,
                            "scheme {scheme:?} bin {i}: mean {mean} vs {p}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn reservoir_uniform_among_stream() {
        let mut rng = SplitMix64::seed_from_u64(SEED);
        let mut slots = [0u32; 10];
        let mut rs = ReservoirSampler::new(&mut slots);
        for x in 0u32..100 {
            rs.offer(x, &mut rng);
        }
        assert_eq!(rs.capacity(), 10);
        assert!(slots.iter().all(|&v| v < 100));
        assert!(slots.iter().any(|&v| v >= 90));

        // Small stream: reservoir holds everything.
        let mut small = [0u32; 5];
        let mut rs2 = ReservoirSampler::new(&mut small);
        for x in 0u32..3 {
            rs2.offer(x, &mut rng);
        }
        let mut collected: Vec<u32> = rs2.sample().to_vec();
        collected.sort_unstable();
        assert_eq!(collected, vec![0, 1, 2]);
    }

    #[test]
    fn reservoir_is_fair_over_trials() {
        let mut rng = SplitMix64::seed_from_u64(SEED);
        let mut hits = [0u32; 100];
        for _ in 0..20000u32 {
            let mut slots = [0u32; 10];
            let mut rs = ReservoirSampler::new(&mut slots);
            for x in 0u32..100 {
                rs.offer(x, &mut rng);
            }
            for &v in rs.sample() {
                hits[v as usize] += 1;
            }
        }
        for &h in &hits {
            // expected ~ (10/100) * 20000 = 2000
            assert!((h as f64 - 2000.0).abs() < 350.0, "hit count {h}");
        }
    }

    #[test]
    fn uniform_index_is_unbiased() {
        let mut rng = SplitMix64::seed_from_u64(SEED);
        let n = 6usize;
        let mut counts = [0u32; 6];
        for _ in 0..60000 {
            let i = uniform_index(&mut rng, n).unwrap();
            counts[i] += 1;
        }
        for &c in &counts {
            assert!((c as f64 - 10000.0).abs() < 600.0, "count {c}");
        }
        assert_eq!(uniform_index(&mut rng, 0), None);
        assert_eq!(uniform_index(&mut rng, 1), Some(0));
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = SplitMix64::seed_from_u64(SEED);
        let mut v = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        shuffle(&mut v, &mut rng);
        let mut sorted = v;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn rejection_triangular_mean() {
        // Target: f(x) = 2x on [0,1] (mean 2/3). Proposal: Uniform(0,1), m=2.
        let target = DensityFn(|x: f64| {
            if (0.0..=1.0).contains(&x) {
                2.0 * x
            } else {
                0.0
            }
        });
        let proposal = Uniform::new(0.0, 1.0).unwrap();
        let rs = RejectionSampler::new(target, proposal, 2.0).unwrap();
        let mut rng = SplitMix64::seed_from_u64(SEED);
        let mut sum = 0.0;
        let trials = 50000u32;
        for _ in 0..trials {
            let x = rs.sample(&mut rng);
            assert!((0.0..=1.0).contains(&x));
            sum += x;
        }
        let mean = sum / trials as f64;
        assert!((mean - 2.0 / 3.0).abs() < 0.02, "mean {mean}");
        // Acceptance probability ~ 1/m.
        let _ = rs.acceptance_probability();
    }

    #[test]
    fn rejection_invalid_bound() {
        let target = DensityFn(|_x: f64| 1.0);
        let proposal = Uniform::new(0.0, 1.0).unwrap();
        assert!(matches!(
            RejectionSampler::new(target, proposal, 0.0),
            Err(SamplerError::InvalidBound)
        ));
    }

    #[test]
    fn bernoulli_mean() {
        let b = Bernoulli::new(0.3).unwrap();
        let mut rng = SplitMix64::seed_from_u64(SEED);
        let trials = 50000u32;
        let mut ones = 0u32;
        for _ in 0..trials {
            if b.sample(&mut rng) {
                ones += 1;
            }
        }
        let p = ones as f64 / trials as f64;
        assert!((p - 0.3).abs() < 0.02, "p {p}");
    }

    #[test]
    fn sampler_slice_fills() {
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let s = categorical(&[1.0, 1.0, 1.0, 1.0]).unwrap();
            let mut rng = SplitMix64::seed_from_u64(SEED);
            let mut buf = [0usize; 16];
            s.sample_slice(&mut rng, &mut buf);
            assert!(buf.iter().all(|&i| i < 4));
        }
    }

    #[test]
    fn deterministic_with_seed() {
        let mut a = SplitMix64::seed_from_u64(SEED);
        let mut b = SplitMix64::seed_from_u64(SEED);
        #[cfg(any(feature = "std", feature = "alloc"))]
        {
            let s = categorical(&[1.0, 2.0, 3.0, 4.0]).unwrap();
            let x1 = s.sample(&mut a);
            let x2 = s.sample(&mut b);
            assert_eq!(x1, x2);
        }
        let _ = (&mut a, &mut b);
    }
}
