//! FFT (wrap [`rustfft`]).
//!
//! [`rustfft`] is a fast, pure-Rust FFT implementation, but its API is planner
//! oriented and in-place: you build a [`rustfft::FftPlanner`], ask it for an
//! `Arc<dyn rustfft::Fft<f64>>` of a given length, and hand that object a
//! mutable slice of [`Complex<f64>`] which it transforms in place. That is the
//! right shape for a kernel, and the wrong shape for casual call sites.
//!
//! This crate is the thin ergonomic layer over it:
//!
//! * [`Fft`] — a reusable engine holding a planner, so repeated transforms of
//!   the same length reuse the cached plan and scratch space.
//! * [`fft`] / [`ifft`] — one-shot free functions for when you just want a
//!   spectrum and do not care about planner reuse.
//! * [`magnitude`] / [`power_spectrum`] — the two spectral summaries that
//!   almost every caller writes by hand otherwise.
//!
//! Nothing here hides `rustfft`: the crate re-exports it (and [`Complex`]) so
//! that callers can drop down to the raw planner at any time without adding a
//! second, possibly version-skewed, `rustfft` dependency.
//!
//! # Normalization
//!
//! Like `rustfft` — and like FFTW, and like NumPy's `fft` before its `1/n`
//! rescaling — the transforms here are **unnormalized**. The forward transform
//! computes
//!
//! ```text
//! X[k] = sum_{t=0}^{n-1} x[t] * exp(-2*pi*i*k*t/n)
//! ```
//!
//! and the inverse computes the same sum with `exp(+2*pi*i*k*t/n)`. Composing
//! them therefore multiplies the signal by `n`; divide by `n` (or use
//! [`Fft::inverse_normalized`] / [`ifft_normalized`]) to recover the original
//! samples.
//!
//! # Examples
//!
//! ```
//! use tpt_math_signal_fft::{fft, ifft_normalized};
//!
//! let signal = [1.0, 2.0, 3.0, 4.0];
//! let spectrum = fft(&signal);
//! assert_eq!(spectrum.len(), 4);
//! // X[0] is the DC term: the sum of the samples.
//! assert!((spectrum[0].re - 10.0).abs() < 1e-12);
//!
//! // Round-trip back to the original samples.
//! let recovered = ifft_normalized(&spectrum);
//! for (got, want) in recovered.iter().zip(signal.iter()) {
//!     assert!((got.re - want).abs() < 1e-12);
//!     assert!(got.im.abs() < 1e-12);
//! }
//! ```
//!
//! [`rustfft`]: rustfft

#![warn(missing_docs)]

pub use rustfft;
pub use rustfft::num_complex::Complex;
pub use tpt_math_numeric;

use rustfft::FftPlanner;
use tpt_math_numeric::Float;

/// A reusable FFT engine wrapping a [`rustfft::FftPlanner<f64>`].
///
/// Planning an FFT is not free: `rustfft` factors the length, selects an
/// algorithm, and precomputes twiddle factors. A [`FftPlanner`] caches that
/// work, so transforming many buffers of the same length through one [`Fft`]
/// value is substantially cheaper than calling the free [`fft`]/[`ifft`]
/// helpers in a loop.
///
/// Every method takes `&mut self` because planning mutates the planner's
/// cache.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_fft::Fft;
///
/// let mut engine = Fft::new();
/// let a = engine.forward(&[1.0, 0.0, 1.0, 0.0]);
/// let b = engine.forward(&[0.0, 1.0, 0.0, 1.0]); // reuses the cached plan
/// assert_eq!(a.len(), 4);
/// assert_eq!(b.len(), 4);
/// ```
pub struct Fft {
    planner: FftPlanner<f64>,
}

impl core::fmt::Debug for Fft {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // `FftPlanner` is an opaque cache and is not `Debug`.
        f.debug_struct("Fft").finish_non_exhaustive()
    }
}

impl Default for Fft {
    fn default() -> Self {
        Self::new()
    }
}

impl Fft {
    /// Create a new engine with an empty plan cache.
    ///
    /// `rustfft` picks the best available instruction set (AVX/SSE/NEON) at
    /// runtime when the plan is built.
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    /// Forward transform of a real-valued signal.
    ///
    /// The transform length `n` is `signal.len()`; the samples are widened to
    /// complex with zero imaginary part. The returned spectrum has `n` bins.
    /// For a real input the spectrum is conjugate-symmetric, so bins
    /// `n/2 + 1 ..` are mirrors of bins `1 ..= n/2` and carry no new
    /// information.
    ///
    /// An empty signal yields an empty spectrum.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_fft::Fft;
    ///
    /// let mut engine = Fft::new();
    /// // A constant signal is pure DC.
    /// let spectrum = engine.forward(&[2.0; 8]);
    /// assert!((spectrum[0].re - 16.0).abs() < 1e-12);
    /// assert!(spectrum[1..].iter().all(|z| z.norm() < 1e-12));
    /// ```
    pub fn forward(&mut self, signal: &[f64]) -> Vec<Complex<f64>> {
        let mut buffer = real_to_complex(signal);
        self.run_forward(&mut buffer);
        buffer
    }

    /// Forward transform of a real-valued signal at an explicit length `n`.
    ///
    /// The signal is zero-padded when `n > signal.len()` and truncated when
    /// `n < signal.len()`. Zero-padding does not add information, but it
    /// interpolates the spectrum onto a finer bin grid and lets you round a
    /// length up to a highly composite one (e.g. a power of two), which is the
    /// fastest case for `rustfft`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_fft::Fft;
    ///
    /// let mut engine = Fft::new();
    /// let spectrum = engine.forward_n(&[1.0, 2.0, 3.0], 8);
    /// assert_eq!(spectrum.len(), 8);
    /// // Still the sum of the (non-zero) samples at DC.
    /// assert!((spectrum[0].re - 6.0).abs() < 1e-12);
    /// ```
    pub fn forward_n(&mut self, signal: &[f64], n: usize) -> Vec<Complex<f64>> {
        let mut buffer = Vec::with_capacity(n);
        buffer.extend(signal.iter().take(n).map(|&re| Complex::new(re, 0.0)));
        buffer.resize(n, Complex::new(0.0, 0.0));
        self.run_forward(&mut buffer);
        buffer
    }

    /// Forward transform of a complex-valued signal.
    ///
    /// This is the general case: unlike [`forward`](Self::forward) the
    /// spectrum of a complex input has no symmetry, so all `n` bins are
    /// meaningful.
    pub fn forward_cplx(&mut self, signal: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut buffer = signal.to_vec();
        self.run_forward(&mut buffer);
        buffer
    }

    /// Inverse transform of a spectrum.
    ///
    /// **The result is not normalized.** `inverse(forward(x))` equals
    /// `n * x`, where `n` is the transform length. Divide by `n` yourself, or
    /// use [`inverse_normalized`](Self::inverse_normalized).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_fft::Fft;
    ///
    /// let mut engine = Fft::new();
    /// let signal = [1.0, 2.0, 3.0, 4.0];
    /// let spectrum = engine.forward(&signal);
    /// let back = engine.inverse(&spectrum);
    /// // Scaled by n = 4.
    /// assert!((back[0].re - 4.0 * signal[0]).abs() < 1e-12);
    /// ```
    pub fn inverse(&mut self, spectrum: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut buffer = spectrum.to_vec();
        self.run_inverse(&mut buffer);
        buffer
    }

    /// Inverse transform of a spectrum; alias of [`inverse`](Self::inverse).
    ///
    /// Provided for symmetry with [`forward_cplx`](Self::forward_cplx), since
    /// an inverse transform always consumes complex input. Like
    /// [`inverse`](Self::inverse) it is **not normalized**.
    pub fn inverse_cplx(&mut self, spectrum: &[Complex<f64>]) -> Vec<Complex<f64>> {
        self.inverse(spectrum)
    }

    /// Inverse transform scaled by `1/n`, so that it undoes
    /// [`forward`](Self::forward) exactly.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_fft::Fft;
    ///
    /// let mut engine = Fft::new();
    /// let signal = [1.0, -2.0, 0.5, 7.0];
    /// let spectrum = engine.forward(&signal);
    /// let back = engine.inverse_normalized(&spectrum);
    /// for (got, want) in back.iter().zip(signal.iter()) {
    ///     assert!((got.re - want).abs() < 1e-12);
    /// }
    /// ```
    pub fn inverse_normalized(&mut self, spectrum: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut buffer = self.inverse(spectrum);
        normalize(&mut buffer);
        buffer
    }

    /// Plan (or reuse a plan) and run a forward transform in place.
    fn run_forward(&mut self, buffer: &mut [Complex<f64>]) {
        if buffer.is_empty() {
            return;
        }
        self.planner.plan_fft_forward(buffer.len()).process(buffer);
    }

    /// Plan (or reuse a plan) and run an inverse transform in place.
    fn run_inverse(&mut self, buffer: &mut [Complex<f64>]) {
        if buffer.is_empty() {
            return;
        }
        self.planner.plan_fft_inverse(buffer.len()).process(buffer);
    }
}

/// Forward FFT of a real-valued signal.
///
/// A convenience wrapper that builds a planner internally. For repeated
/// transforms prefer [`Fft`], which caches plans across calls.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_fft::fft;
///
/// // A unit impulse has a flat spectrum.
/// let spectrum = fft(&[1.0, 0.0, 0.0, 0.0]);
/// assert!(spectrum.iter().all(|z| (z.re - 1.0).abs() < 1e-12));
/// ```
pub fn fft(signal: &[f64]) -> Vec<Complex<f64>> {
    Fft::new().forward(signal)
}

/// Inverse FFT of a spectrum, **not normalized** (scaled by `n`).
///
/// A convenience wrapper that builds a planner internally. For repeated
/// transforms prefer [`Fft`]; to undo [`fft`] exactly, use
/// [`ifft_normalized`].
pub fn ifft(spectrum: &[Complex<f64>]) -> Vec<Complex<f64>> {
    Fft::new().inverse(spectrum)
}

/// Inverse FFT of a spectrum, scaled by `1/n` so that it undoes [`fft`].
///
/// # Examples
///
/// ```
/// use tpt_math_signal_fft::{fft, ifft_normalized};
///
/// let signal = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
/// let back = ifft_normalized(&fft(&signal));
/// for (got, want) in back.iter().zip(signal.iter()) {
///     assert!((got.re - want).abs() < 1e-9);
/// }
/// ```
pub fn ifft_normalized(spectrum: &[Complex<f64>]) -> Vec<Complex<f64>> {
    Fft::new().inverse_normalized(spectrum)
}

/// Per-bin magnitudes `|X[k]|` of a spectrum.
///
/// Computed with `hypot`, so it does not overflow or underflow for extreme
/// component magnitudes the way `sqrt(re*re + im*im)` can.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_fft::{fft, magnitude};
///
/// let mags = magnitude(&fft(&[1.0, 0.0, 0.0, 0.0]));
/// assert!(mags.iter().all(|m| (m - 1.0).abs() < 1e-12));
/// ```
pub fn magnitude(spectrum: &[Complex<f64>]) -> Vec<f64> {
    spectrum.iter().map(|z| Float::hypot(z.re, z.im)).collect()
}

/// Power spectrum `|X[k]|^2` of a real-valued signal.
///
/// This is the raw (unnormalized, one-sided-*un*folded) periodogram: bin `k`
/// holds the squared magnitude of the forward transform, and for a real input
/// the second half of the output mirrors the first. Divide by `n` (or `n^2`,
/// depending on convention) if you need a density estimate.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_fft::power_spectrum;
///
/// let power = power_spectrum(&[1.0, 1.0, 1.0, 1.0]);
/// // All energy sits at DC: |4|^2 = 16.
/// assert!((power[0] - 16.0).abs() < 1e-12);
/// assert!(power[1..].iter().all(|p| *p < 1e-12));
/// ```
pub fn power_spectrum(signal: &[f64]) -> Vec<f64> {
    fft(signal)
        .iter()
        .map(|z| z.re * z.re + z.im * z.im)
        .collect()
}

/// Widen real samples to complex with zero imaginary part.
fn real_to_complex(signal: &[f64]) -> Vec<Complex<f64>> {
    signal.iter().map(|&re| Complex::new(re, 0.0)).collect()
}

/// Scale a buffer by `1/n` in place (no-op when empty).
fn normalize(buffer: &mut [Complex<f64>]) {
    if buffer.is_empty() {
        return;
    }
    let scale = 1.0 / buffer.len() as f64;
    for value in buffer.iter_mut() {
        *value *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;

    /// Absolute tolerance for values that should be exactly recovered.
    const TOL: f64 = 1e-9;

    fn assert_close(got: f64, want: f64, tol: f64) {
        assert!(
            (got - want).abs() <= tol,
            "expected {want}, got {got} (|diff| = {})",
            (got - want).abs()
        );
    }

    /// `cos(2*pi*k*t/n)` sampled at `n` points.
    fn cosine(n: usize, k: usize) -> Vec<f64> {
        (0..n)
            .map(|t| (2.0 * PI * k as f64 * t as f64 / n as f64).cos())
            .collect()
    }

    #[test]
    fn impulse_has_flat_unit_spectrum() {
        let spectrum = fft(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(spectrum.len(), 4);
        for bin in &spectrum {
            assert_close(bin.re, 1.0, TOL);
            assert_close(bin.im, 0.0, TOL);
        }
    }

    #[test]
    fn dc_bin_is_the_sum_of_samples() {
        let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = fft(&signal);
        assert_close(spectrum[0].re, signal.iter().sum::<f64>(), TOL);
        assert_close(spectrum[0].im, 0.0, TOL);
    }

    #[test]
    fn forward_then_inverse_round_trips() {
        let signal = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let mut engine = Fft::new();

        let spectrum = engine.forward(&signal);
        let raw = engine.inverse(&spectrum);
        let normalized = engine.inverse_normalized(&spectrum);

        for (i, &want) in signal.iter().enumerate() {
            // The raw inverse is scaled by n.
            assert_close(raw[i].re, want * signal.len() as f64, TOL);
            assert_close(raw[i].im, 0.0, TOL);
            // The normalized inverse recovers the signal exactly.
            assert_close(normalized[i].re, want, TOL);
            assert_close(normalized[i].im, 0.0, TOL);
        }
    }

    #[test]
    fn complex_round_trip_matches_input() {
        let signal: Vec<Complex<f64>> = (0..16)
            .map(|i| Complex::new(i as f64 * 0.25, (i as f64).sin()))
            .collect();
        let mut engine = Fft::new();

        let spectrum = engine.forward_cplx(&signal);
        let mut back = engine.inverse_cplx(&spectrum);
        normalize(&mut back);

        for (got, want) in back.iter().zip(signal.iter()) {
            assert_close(got.re, want.re, TOL);
            assert_close(got.im, want.im, TOL);
        }
    }

    #[test]
    fn cosine_has_a_single_peak_at_the_expected_bin() {
        let n = 64;
        let k = 5;
        let mags = magnitude(&fft(&cosine(n, k)));

        // A real cosine splits its energy between bin k and its mirror n - k,
        // each carrying magnitude n/2.
        assert_close(mags[k], n as f64 / 2.0, 1e-9);
        assert_close(mags[n - k], n as f64 / 2.0, 1e-9);

        // Every other bin is empty.
        for (bin, mag) in mags.iter().enumerate() {
            if bin == k || bin == n - k {
                continue;
            }
            assert!(*mag < 1e-9, "bin {bin} should be empty, got {mag}");
        }

        // The peak really is the largest bin of the one-sided spectrum
        // (bins 0 ..= n/2; the mirror at n - k ties with it exactly).
        let peak = mags[..=n / 2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(bin, _)| bin)
            .unwrap();
        assert_eq!(peak, k);
    }

    #[test]
    fn power_spectrum_is_squared_magnitude() {
        let signal = cosine(32, 3);
        let power = power_spectrum(&signal);
        let mags = magnitude(&fft(&signal));

        assert_eq!(power.len(), signal.len());
        for (p, m) in power.iter().zip(mags.iter()) {
            assert_close(*p, m * m, 1e-9);
        }
        // Peak power for a length-32 cosine: (32/2)^2 = 256.
        assert_close(power[3], 256.0, 1e-7);
    }

    #[test]
    fn zero_padding_preserves_dc_and_changes_length() {
        let mut engine = Fft::new();
        let signal = [1.0, 2.0, 3.0];

        let padded = engine.forward_n(&signal, 8);
        assert_eq!(padded.len(), 8);
        assert_close(padded[0].re, 6.0, TOL);

        // Truncation drops the tail.
        let truncated = engine.forward_n(&signal, 2);
        assert_eq!(truncated.len(), 2);
        assert_close(truncated[0].re, 3.0, TOL);
    }

    #[test]
    fn non_power_of_two_lengths_work() {
        let signal: Vec<f64> = (0..21).map(|i| (i as f64 * 0.7).sin()).collect();
        let back = ifft_normalized(&fft(&signal));
        for (got, want) in back.iter().zip(signal.iter()) {
            assert_close(got.re, *want, TOL);
        }
    }

    #[test]
    fn empty_input_is_handled() {
        let mut engine = Fft::new();
        assert!(engine.forward(&[]).is_empty());
        assert!(engine.inverse(&[]).is_empty());
        assert!(engine.inverse_normalized(&[]).is_empty());
        assert!(fft(&[]).is_empty());
        assert!(ifft(&[]).is_empty());
        assert!(magnitude(&[]).is_empty());
        assert!(power_spectrum(&[]).is_empty());
    }

    #[test]
    fn magnitude_matches_hypotenuse() {
        let spectrum = [Complex::new(3.0, 4.0), Complex::new(-5.0, 12.0)];
        let mags = magnitude(&spectrum);
        assert_close(mags[0], 5.0, TOL);
        assert_close(mags[1], 13.0, TOL);
    }
}
