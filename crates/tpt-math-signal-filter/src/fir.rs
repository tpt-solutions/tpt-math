//! Finite impulse response (FIR) filtering.

use core::f64::consts::PI;

use tpt_math_numeric::Float;
use tpt_math_signal_fft::{Complex, Fft};

use crate::window::hamming;

/// A finite impulse response filter: a fixed list of tap weights applied by
/// direct convolution.
///
/// # Output convention
///
/// [`filter`](Self::filter) is **centred**: the output has exactly the same
/// length as the input, and the filter's group delay is compensated so that
/// feature `i` of the input lines up with sample `i` of the output. Concretely,
/// for taps `h[0..m]` and group delay `d = (m - 1) / 2`,
///
/// ```text
/// y[i] = sum_{k=0}^{m-1} h[k] * x[i + d - k],   x[j] = 0 outside 0 <= j < n
/// ```
///
/// For the linear-phase (symmetric, odd-length) filters produced by
/// [`lowpass`](Self::lowpass) this makes the filter exactly zero-phase in the
/// interior: a sine wave comes out with the same phase it went in with. With an
/// even number of taps the true group delay is a half-integer, so `filter`
/// compensates the integer part `m/2 - 1` and leaves a residual half-sample
/// delay.
///
/// Outside the signal the input is treated as zero, so roughly the first and
/// last `d` output samples are edge transients. If you want the textbook causal
/// convolution `y[i] = sum h[k] * x[i-k]` instead — the streaming convention,
/// with the delay left in — use [`filter_causal`](Self::filter_causal).
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::FirFilter;
///
/// // A 3-tap smoother, applied to an impulse, reproduces the taps centred on
/// // the impulse's position: no net delay.
/// let fir = FirFilter::new(vec![0.25, 0.5, 0.25]);
/// let mut impulse = vec![0.0; 7];
/// impulse[3] = 1.0;
///
/// let smoothed = fir.filter(&impulse);
/// assert_eq!(smoothed, vec![0.0, 0.0, 0.25, 0.5, 0.25, 0.0, 0.0]);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FirFilter {
    /// Tap weights `h[0..m]`, in impulse-response order.
    coeffs: Vec<f64>,
}

impl FirFilter {
    /// Build a filter from an explicit list of tap weights.
    ///
    /// The taps are the filter's impulse response, in order: `coeffs[0]`
    /// multiplies the newest sample. No normalization is applied — if you want
    /// unit DC gain, divide by the sum yourself (the design constructors below
    /// already do).
    ///
    /// An empty tap list is legal and filters everything to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// let identity = FirFilter::new(vec![1.0]);
    /// assert_eq!(identity.filter(&[1.0, -2.0, 3.0]), vec![1.0, -2.0, 3.0]);
    /// ```
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    /// Design a windowed-sinc low-pass filter with `n` taps.
    ///
    /// `cutoff` is the -6 dB corner expressed in **cycles per sample**, so it
    /// must lie in `(0, 0.5)`; `0.5` is the Nyquist frequency. For a 48 kHz
    /// signal and a 4 kHz corner, pass `4000.0 / 48000.0`.
    ///
    /// The design is the classic one: truncate the ideal brick-wall impulse
    /// response `2*fc*sinc(2*fc*t)` to `n` taps centred on `(n-1)/2`, taper it
    /// with a [`hamming`](crate::hamming) window to tame the Gibbs ringing, then
    /// rescale to unit DC gain. The transition band is roughly `3.3/n` cycles
    /// per sample wide and the stopband sits near -53 dB, so use more taps for a
    /// sharper corner.
    ///
    /// The taps are symmetric, hence the filter is exactly linear phase and
    /// [`filter`](Self::filter) removes that phase entirely.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`, or if `cutoff` is not in the open interval
    /// `(0, 0.5)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// let lp = FirFilter::lowpass(31, 0.1);
    /// assert_eq!(lp.len(), 31);
    /// // Unit DC gain: a constant passes through untouched (away from edges).
    /// assert!((lp.coeffs().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    /// ```
    pub fn lowpass(n: usize, cutoff: f64) -> Self {
        assert!(n > 0, "FirFilter::lowpass: need at least one tap");
        assert!(
            cutoff > 0.0 && cutoff < 0.5,
            "FirFilter::lowpass: cutoff must be in (0, 0.5) cycles/sample, got {cutoff}"
        );

        let center = (n - 1) as f64 / 2.0;
        let mut coeffs: Vec<f64> = hamming(n)
            .into_iter()
            .enumerate()
            .map(|(i, w)| {
                let t = i as f64 - center;
                2.0 * cutoff * sinc(2.0 * cutoff * t) * w
            })
            .collect();

        // Normalize to unit gain at DC.
        let sum: f64 = coeffs.iter().sum();
        if sum != 0.0 {
            for c in &mut coeffs {
                *c /= sum;
            }
        }
        Self { coeffs }
    }

    /// Design a windowed-sinc high-pass filter with `n` taps.
    ///
    /// Built by spectral inversion of [`lowpass`](Self::lowpass): negate the
    /// low-pass taps and add one to the centre tap, which subtracts the
    /// low-frequency content from an all-pass. `cutoff` has the same meaning as
    /// for `lowpass`, and the result has unit gain at Nyquist and zero gain at
    /// DC.
    ///
    /// # Panics
    ///
    /// Panics if `n` is even (spectral inversion needs a single centre tap), or
    /// under the same conditions as [`lowpass`](Self::lowpass).
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// let hp = FirFilter::highpass(31, 0.1);
    /// // Zero DC gain: a constant is annihilated.
    /// assert!(hp.coeffs().iter().sum::<f64>().abs() < 1e-12);
    /// ```
    pub fn highpass(n: usize, cutoff: f64) -> Self {
        assert!(
            n % 2 == 1,
            "FirFilter::highpass: needs an odd number of taps, got {n}"
        );
        let mut filter = Self::lowpass(n, cutoff);
        for c in &mut filter.coeffs {
            *c = -*c;
        }
        filter.coeffs[n / 2] += 1.0;
        filter
    }

    /// The filter's tap weights.
    pub fn coeffs(&self) -> &[f64] {
        &self.coeffs
    }

    /// The number of taps.
    pub fn len(&self) -> usize {
        self.coeffs.len()
    }

    /// Whether the filter has no taps (and therefore outputs only zeros).
    pub fn is_empty(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// The integer group delay `(m - 1) / 2` compensated by
    /// [`filter`](Self::filter).
    ///
    /// For a symmetric filter this is the true delay of the causal convolution,
    /// which is why removing it yields a zero-phase result.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// assert_eq!(FirFilter::new(vec![1.0]).group_delay(), 0);
    /// assert_eq!(FirFilter::lowpass(31, 0.2).group_delay(), 15);
    /// ```
    pub fn group_delay(&self) -> usize {
        self.coeffs.len().saturating_sub(1) / 2
    }

    /// Filter a signal by direct convolution, delay-compensated.
    ///
    /// Returns a new vector of the same length as `signal`; see the
    /// [type documentation](Self) for the exact convention.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// // A boxcar average of three samples.
    /// let fir = FirFilter::new(vec![1.0 / 3.0; 3]);
    /// let out = fir.filter(&[1.0, 1.0, 1.0, 1.0, 1.0]);
    /// // The interior is exactly the constant; the edges taper (zero padding).
    /// assert!((out[2] - 1.0).abs() < 1e-12);
    /// assert!((out[0] - 2.0 / 3.0).abs() < 1e-12);
    /// ```
    pub fn filter(&self, signal: &[f64]) -> Vec<f64> {
        let m = self.coeffs.len();
        let n = signal.len();
        let delay = self.group_delay();
        let mut out = vec![0.0; n];

        for (i, y) in out.iter_mut().enumerate() {
            // Taps k with 0 <= i + delay - k < n contribute; the rest read
            // zeros from outside the signal.
            let first = (i + delay + 1).saturating_sub(n);
            let last = (i + delay + 1).min(m);
            let mut acc = 0.0;
            for k in first..last {
                acc += self.coeffs[k] * signal[i + delay - k];
            }
            *y = acc;
        }
        out
    }

    /// Filter a signal in place, overwriting it with the result.
    ///
    /// Identical to [`filter`](Self::filter) in every respect except that it
    /// writes back into `signal` and allocates only a small delay line of
    /// `len()` samples instead of a full copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// let fir = FirFilter::lowpass(9, 0.2);
    /// let mut signal: Vec<f64> = (0..32).map(|t| (t as f64 * 0.3).sin()).collect();
    /// let expected = fir.filter(&signal);
    ///
    /// fir.filter_in_place(&mut signal);
    /// assert_eq!(signal, expected);
    /// ```
    pub fn filter_in_place(&self, signal: &mut [f64]) {
        let m = self.coeffs.len();
        if m == 0 {
            signal.fill(0.0);
            return;
        }
        let n = signal.len();
        let delay = self.group_delay();

        // Ring buffer of the last `m` *original* samples: taps reaching back
        // before the write cursor would otherwise read already-filtered data.
        // Samples at or after the cursor are still pristine in `signal`.
        let mut history = vec![0.0; m];
        for i in 0..n {
            let first = (i + delay + 1).saturating_sub(n);
            let last = (i + delay + 1).min(m);
            let mut acc = 0.0;
            for k in first..last {
                let t = i + delay - k;
                let x = if t >= i { signal[t] } else { history[t % m] };
                acc += self.coeffs[k] * x;
            }
            history[i % m] = signal[i];
            signal[i] = acc;
        }
    }

    /// Filter a signal with the plain causal convolution, delay included.
    ///
    /// `y[i] = sum_k h[k] * x[i - k]`, with zeros assumed before the start of
    /// the signal. This is the streaming convention: no lookahead, but a
    /// symmetric filter shifts its output by [`group_delay`](Self::group_delay)
    /// samples.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// let fir = FirFilter::new(vec![0.0, 1.0]); // a pure one-sample delay
    /// assert_eq!(fir.filter_causal(&[1.0, 2.0, 3.0]), vec![0.0, 1.0, 2.0]);
    /// ```
    pub fn filter_causal(&self, signal: &[f64]) -> Vec<f64> {
        let m = self.coeffs.len();
        let n = signal.len();
        let mut out = vec![0.0; n];

        for (i, y) in out.iter_mut().enumerate() {
            let last = (i + 1).min(m);
            let mut acc = 0.0;
            for k in 0..last {
                acc += self.coeffs[k] * signal[i - k];
            }
            *y = acc;
        }
        out
    }

    /// The filter's frequency response, sampled at `n` equally spaced points.
    ///
    /// Bin `k` holds `H(exp(-2*pi*i*k/n))`, the response at `k/n` cycles per
    /// sample; bin `0` is DC and bin `n/2` is Nyquist. This is the response of
    /// the causal coefficient sequence (computed as its zero-padded FFT), so it
    /// carries the linear phase term that [`filter`](Self::filter) compensates;
    /// magnitudes are unaffected by that choice.
    ///
    /// # Panics
    ///
    /// Panics if `n` is smaller than the number of taps, which would truncate
    /// the impulse response instead of zero-padding it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::FirFilter;
    ///
    /// let lp = FirFilter::lowpass(31, 0.1);
    /// let h = lp.frequency_response(256);
    ///
    /// assert!((h[0].norm() - 1.0).abs() < 1e-9); // unit gain at DC
    /// assert!(h[128].norm() < 1e-3);             // stopband at Nyquist
    /// ```
    pub fn frequency_response(&self, n: usize) -> Vec<Complex<f64>> {
        assert!(
            n >= self.coeffs.len(),
            "FirFilter::frequency_response: n = {n} would truncate {} taps",
            self.coeffs.len()
        );
        Fft::new().forward_n(&self.coeffs, n)
    }
}

/// Normalized cardinal sine, `sin(pi*x) / (pi*x)`, with `sinc(0) = 1`.
fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        let px = PI * x;
        Float::sin(px) / px
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::TAU;

    /// Root mean square of a slice.
    fn rms(x: &[f64]) -> f64 {
        Float::sqrt(x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64)
    }

    /// `sin(2*pi*f*t)` sampled at `n` points, `f` in cycles per sample.
    fn sine(n: usize, f: f64) -> Vec<f64> {
        (0..n).map(|t| Float::sin(TAU * f * t as f64)).collect()
    }

    #[test]
    fn lowpass_attenuates_high_frequencies_more_than_low() {
        let n = 512;
        let taps = 51;
        let fir = FirFilter::lowpass(taps, 0.1);

        let low = sine(n, 0.02);
        let high = sine(n, 0.35);
        let mixed: Vec<f64> = low.iter().zip(&high).map(|(l, h)| l + h).collect();

        // Measure away from the zero-padded edges.
        let interior = taps..n - taps;
        let low_out = fir.filter(&low)[interior.clone()].to_vec();
        let high_out = fir.filter(&high)[interior.clone()].to_vec();
        let mixed_out = fir.filter(&mixed)[interior.clone()].to_vec();

        let low_rms = rms(&low_out);
        let high_rms = rms(&high_out);

        // The passband tone survives essentially intact (a unit sine has
        // RMS 1/sqrt(2)); the stopband tone is crushed.
        assert!((low_rms - 0.5_f64.sqrt()).abs() < 0.01, "low RMS {low_rms}");
        assert!(
            high_rms < 0.01 * low_rms,
            "high RMS {high_rms} is not far below low RMS {low_rms}"
        );

        // Filtering the sum therefore recovers the low component alone.
        let mixed_rms = rms(&mixed_out);
        assert!(
            (mixed_rms - low_rms).abs() < 0.01,
            "mixed RMS {mixed_rms} vs low RMS {low_rms}"
        );
        let worst = mixed_out
            .iter()
            .zip(&low_out)
            .map(|(m, l)| (m - l).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < 0.01, "worst pointwise error {worst}");
    }

    #[test]
    fn lowpass_is_zero_phase_in_the_interior() {
        let n = 256;
        let taps = 31;
        let fir = FirFilter::lowpass(taps, 0.2);
        let signal = sine(n, 0.05);
        let out = fir.filter(&signal);

        // Passband: same amplitude *and* same phase, so a pointwise comparison
        // holds (this is what the group-delay compensation buys).
        for i in taps..n - taps {
            assert!(
                (out[i] - signal[i]).abs() < 0.02,
                "at {i}: {} vs {}",
                out[i],
                signal[i]
            );
        }
    }

    #[test]
    fn lowpass_passes_a_constant_unchanged() {
        let fir = FirFilter::lowpass(21, 0.1);
        let out = fir.filter(&[3.5; 64]);
        for value in &out[21..64 - 21] {
            assert!((value - 3.5).abs() < 1e-12, "got {value}");
        }
        assert!((fir.coeffs().iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn lowpass_taps_are_symmetric() {
        let fir = FirFilter::lowpass(33, 0.15);
        let h = fir.coeffs();
        for (front, back) in h.iter().zip(h.iter().rev()) {
            assert!((front - back).abs() < 1e-15);
        }
    }

    #[test]
    fn highpass_is_the_mirror_of_lowpass() {
        let n = 512;
        let taps = 51;
        let fir = FirFilter::highpass(taps, 0.1);

        let low = fir.filter(&sine(n, 0.02))[taps..n - taps].to_vec();
        let high = fir.filter(&sine(n, 0.35))[taps..n - taps].to_vec();

        assert!(rms(&high) > 0.7, "passband RMS {}", rms(&high));
        assert!(rms(&low) < 0.01, "stopband RMS {}", rms(&low));
        // Constant input is annihilated.
        let dc = fir.filter(&[1.0; 128]);
        assert!(dc[taps..128 - taps].iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn identity_filter_returns_the_input() {
        let signal = [1.0, -2.0, 3.5, 0.0, 7.25];
        let fir = FirFilter::new(vec![1.0]);
        assert_eq!(fir.filter(&signal), signal.to_vec());
        assert_eq!(fir.filter_causal(&signal), signal.to_vec());
        assert_eq!(fir.group_delay(), 0);
    }

    #[test]
    fn centred_convolution_reproduces_the_taps_on_an_impulse() {
        let fir = FirFilter::new(vec![0.25, 0.5, 0.25]);
        let mut impulse = vec![0.0; 7];
        impulse[3] = 1.0;
        assert_eq!(
            fir.filter(&impulse),
            vec![0.0, 0.0, 0.25, 0.5, 0.25, 0.0, 0.0]
        );

        // The causal convolution puts the same taps one group delay later.
        assert_eq!(
            fir.filter_causal(&impulse),
            vec![0.0, 0.0, 0.0, 0.25, 0.5, 0.25, 0.0]
        );
    }

    #[test]
    fn filter_in_place_matches_filter() {
        let signal: Vec<f64> = (0..97).map(|t| Float::sin(t as f64 * 0.37) * 2.0).collect();
        for taps in [1usize, 2, 3, 8, 17] {
            let fir = FirFilter::new((0..taps).map(|k| (k as f64 + 1.0) * 0.1).collect());
            let expected = fir.filter(&signal);
            let mut in_place = signal.clone();
            fir.filter_in_place(&mut in_place);
            assert_eq!(in_place, expected, "taps = {taps}");
        }
    }

    #[test]
    fn empty_inputs_and_empty_filters_are_handled() {
        let fir = FirFilter::lowpass(9, 0.2);
        assert!(fir.filter(&[]).is_empty());
        assert!(fir.filter_causal(&[]).is_empty());

        let empty = FirFilter::new(Vec::new());
        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.filter(&[1.0, 2.0]), vec![0.0, 0.0]);
        let mut signal = [1.0, 2.0];
        empty.filter_in_place(&mut signal);
        assert_eq!(signal, [0.0, 0.0]);
    }

    #[test]
    fn signals_shorter_than_the_filter_still_work() {
        let fir = FirFilter::new(vec![0.5, 0.5, 0.5, 0.5, 0.5]);
        let out = fir.filter(&[1.0, 1.0]);
        assert_eq!(out.len(), 2);
        assert_eq!(out, vec![1.0, 1.0]);
    }

    #[test]
    fn frequency_response_matches_the_design() {
        let fir = FirFilter::lowpass(31, 0.1);
        let h = fir.frequency_response(512);
        assert_eq!(h.len(), 512);

        assert!((h[0].norm() - 1.0).abs() < 1e-12, "DC gain {}", h[0].norm());
        // 0.02 cycles/sample => bin 10 of 512: still in the passband.
        assert!((h[10].norm() - 1.0).abs() < 0.01);
        // 0.35 cycles/sample => bin 179: deep in the stopband.
        assert!(h[179].norm() < 1e-3, "stopband gain {}", h[179].norm());
        // Real taps give a conjugate-symmetric response.
        assert!((h[10].norm() - h[512 - 10].norm()).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "cutoff must be in (0, 0.5)")]
    fn lowpass_rejects_an_out_of_range_cutoff() {
        let _ = FirFilter::lowpass(9, 0.75);
    }

    #[test]
    #[should_panic(expected = "need at least one tap")]
    fn lowpass_rejects_zero_taps() {
        let _ = FirFilter::lowpass(0, 0.25);
    }

    #[test]
    #[should_panic(expected = "odd number of taps")]
    fn highpass_rejects_an_even_tap_count() {
        let _ = FirFilter::highpass(10, 0.25);
    }

    #[test]
    fn sinc_is_one_at_zero_and_vanishes_at_integers() {
        assert_eq!(sinc(0.0), 1.0);
        for k in 1..8 {
            assert!(sinc(k as f64).abs() < 1e-15);
            assert!(sinc(-(k as f64)).abs() < 1e-15);
        }
        assert!((sinc(0.5) - 2.0 / PI).abs() < 1e-15);
    }
}
