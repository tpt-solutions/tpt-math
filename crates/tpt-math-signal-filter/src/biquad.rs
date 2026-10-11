//! Second-order IIR sections (biquads) and Butterworth low-pass cascades.

use core::f64::consts::PI;

/// A second-order IIR section with transfer function
///
/// ```text
/// H(z) = (b0 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²)
/// ```
///
/// Samples are processed in direct form II transposed, which keeps the internal
/// state small and is numerically well-behaved for stable low-order sections.
#[derive(Clone, Debug, PartialEq)]
pub struct Biquad {
    /// Feed-forward coefficients `b0, b1, b2`.
    pub b: [f64; 3],
    /// Feedback coefficients `a1, a2` (with the leading `a0` normalised to 1).
    pub a: [f64; 2],
    z1: f64,
    z2: f64,
}

impl Biquad {
    /// Build a section from coefficients already normalised so that `a0 = 1`.
    pub fn new(b: [f64; 3], a: [f64; 2]) -> Self {
        Biquad {
            b,
            a,
            z1: 0.0,
            z2: 0.0,
        }
    }

    /// Second-order low-pass section (RBJ audio-EQ cookbook form).
    ///
    /// `cutoff` is the −3 dB-style corner frequency as a fraction of the sample
    /// rate (cycles per sample, in `(0, 0.5)`); `q` is the quality factor. A
    /// Butterworth response uses `q = 1/√2`.
    ///
    /// Returns `None` if `cutoff` is outside `(0, 0.5)` or `q` is not positive.
    pub fn lowpass(cutoff: f64, q: f64) -> Option<Self> {
        let (cos_w, alpha) = rbj_terms(cutoff, q)?;
        let a0 = 1.0 + alpha;
        let b0 = (1.0 - cos_w) / 2.0 / a0;
        Some(Biquad::new(
            [b0, (1.0 - cos_w) / a0, b0],
            [-2.0 * cos_w / a0, (1.0 - alpha) / a0],
        ))
    }

    /// Second-order high-pass section (RBJ audio-EQ cookbook form).
    ///
    /// Same parameters and `None` conditions as [`Biquad::lowpass`].
    pub fn highpass(cutoff: f64, q: f64) -> Option<Self> {
        let (cos_w, alpha) = rbj_terms(cutoff, q)?;
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w) / 2.0 / a0;
        Some(Biquad::new(
            [b0, -(1.0 + cos_w) / a0, b0],
            [-2.0 * cos_w / a0, (1.0 - alpha) / a0],
        ))
    }

    /// Second-order band-pass section with constant 0 dB peak gain (RBJ).
    ///
    /// `center` is the centre frequency in cycles per sample, and `q` sets the
    /// bandwidth: a larger `q` gives a narrower band. The response is exactly 1
    /// at `center` and 0 at DC and Nyquist.
    ///
    /// Returns `None` for an out-of-range `center` or a non-positive `q`.
    pub fn bandpass(center: f64, q: f64) -> Option<Self> {
        let (cos_w, alpha) = rbj_terms(center, q)?;
        let a0 = 1.0 + alpha;
        Some(Biquad::new(
            [alpha / a0, 0.0, -alpha / a0],
            [-2.0 * cos_w / a0, (1.0 - alpha) / a0],
        ))
    }

    /// Second-order notch (band-reject) section (RBJ).
    ///
    /// The response is exactly 0 at `center` and 1 at DC and Nyquist. The
    /// parameters are as for [`Biquad::bandpass`].
    pub fn notch(center: f64, q: f64) -> Option<Self> {
        let (cos_w, alpha) = rbj_terms(center, q)?;
        let a0 = 1.0 + alpha;
        Some(Biquad::new(
            [1.0 / a0, -2.0 * cos_w / a0, 1.0 / a0],
            [-2.0 * cos_w / a0, (1.0 - alpha) / a0],
        ))
    }

    /// Process one sample, updating the internal state.
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b[0] * x + self.z1;
        self.z1 = self.b[1] * x - self.a[0] * y + self.z2;
        self.z2 = self.b[2] * x - self.a[1] * y;
        y
    }

    /// Filter a whole signal, starting from the current state.
    pub fn filter(&mut self, signal: &[f64]) -> Vec<f64> {
        signal.iter().map(|&x| self.process(x)).collect()
    }

    /// Clear the internal state.
    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Magnitude of the frequency response `|H(e^{i2πf})|` at `freq`, given in
    /// cycles per sample.
    pub fn magnitude(&self, freq: f64) -> f64 {
        let w = 2.0 * PI * freq;
        // z⁻¹ = e^{-iw}; evaluate numerator and denominator as complex numbers.
        let (c1, s1) = (w.cos(), -w.sin());
        let (c2, s2) = ((2.0 * w).cos(), -(2.0 * w).sin());
        let num_re = self.b[0] + self.b[1] * c1 + self.b[2] * c2;
        let num_im = self.b[1] * s1 + self.b[2] * s2;
        let den_re = 1.0 + self.a[0] * c1 + self.a[1] * c2;
        let den_im = self.a[0] * s1 + self.a[1] * s2;
        (num_re * num_re + num_im * num_im).sqrt() / (den_re * den_re + den_im * den_im).sqrt()
    }
}

/// Shared RBJ terms `(cos ω₀, α)` for a section at `frequency` (cycles per
/// sample) and quality factor `q`, with `α = sin ω₀ / (2q)`.
///
/// Returns `None` unless `frequency` lies in `(0, 0.5)` and `q` is positive.
/// The comparisons are written so that NaN is rejected.
fn rbj_terms(frequency: f64, q: f64) -> Option<(f64, f64)> {
    let frequency_ok = frequency > 0.0 && frequency < 0.5;
    let q_ok = q > 0.0;
    if !frequency_ok || !q_ok {
        return None;
    }
    let w0 = 2.0 * PI * frequency;
    Some((w0.cos(), w0.sin() / (2.0 * q)))
}

/// Butterworth low-pass filter of even `order` as a cascade of biquads.
///
/// The poles sit on the unit circle of the analogue prototype, so the response
/// is maximally flat in the passband and equal to `1/√2` at `cutoff`.
///
/// Returns `None` if `order` is zero or odd (a first-order section is not
/// provided), or if `cutoff` is outside `(0, 0.5)`.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::butterworth_lowpass;
///
/// let sections = butterworth_lowpass(4, 0.1).unwrap();
/// let gain: f64 = sections.iter().map(|s| s.magnitude(0.0)).product();
/// assert!((gain - 1.0).abs() < 1e-12); // unity DC gain
/// ```
pub fn butterworth_lowpass(order: usize, cutoff: f64) -> Option<Vec<Biquad>> {
    if order == 0 || order % 2 != 0 {
        return None;
    }
    let n = order as f64;
    (1..=order / 2)
        .map(|k| {
            // Pole angle for the k-th conjugate pair; Q = 1 / (2 cos θ_k) with
            // θ_k measured from the imaginary axis.
            let theta = PI * (2 * k - 1) as f64 / (2.0 * n);
            Biquad::lowpass(cutoff, 1.0 / (2.0 * theta.cos()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn highpass_blocks_dc_and_passes_high_frequencies() {
        let hp = Biquad::highpass(0.1, 0.5).unwrap();
        assert!(hp.magnitude(0.0) < 1e-12);
        assert!((hp.magnitude(0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn highpass_is_minus_3db_at_butterworth_cutoff() {
        let hp = Biquad::highpass(0.1, core::f64::consts::FRAC_1_SQRT_2).unwrap();
        assert!((hp.magnitude(0.1) - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9);
    }

    #[test]
    fn bandpass_has_unity_peak_at_center_and_rejects_dc() {
        let bp = Biquad::bandpass(0.1, 5.0).unwrap();
        assert!((bp.magnitude(0.1) - 1.0).abs() < 1e-9);
        assert!(bp.magnitude(0.0) < 1e-12);
        assert!(bp.magnitude(0.5) < 1e-12);
    }

    #[test]
    fn notch_is_zero_at_center_and_unity_at_dc() {
        let notch = Biquad::notch(0.1, 5.0).unwrap();
        assert!(notch.magnitude(0.1) < 1e-9);
        assert!((notch.magnitude(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn new_designs_reject_out_of_range_parameters() {
        assert!(Biquad::highpass(0.0, 1.0).is_none());
        assert!(Biquad::highpass(0.1, 0.0).is_none());
        assert!(Biquad::bandpass(0.6, 1.0).is_none());
        assert!(Biquad::notch(0.1, f64::NAN).is_none());
    }

    #[test]
    fn lowpass_has_unity_dc_gain_and_rejects_nyquist() {
        let s = Biquad::lowpass(0.1, core::f64::consts::FRAC_1_SQRT_2).unwrap();
        assert!((s.magnitude(0.0) - 1.0).abs() < 1e-12);
        assert!(s.magnitude(0.5) < 1e-6);
    }

    #[test]
    fn butterworth_is_minus_3db_at_cutoff() {
        // Each cascade section has the same corner, so the total gain at the
        // cutoff is exactly 1/√2 for any even order.
        for order in [2, 4, 6, 8] {
            let sections = butterworth_lowpass(order, 0.2).unwrap();
            let gain: f64 = sections.iter().map(|s| s.magnitude(0.2)).product();
            assert!(
                (gain - core::f64::consts::FRAC_1_SQRT_2).abs() < 1e-9,
                "order {order}: gain {gain}"
            );
        }
    }

    #[test]
    fn butterworth_rolls_off_faster_with_order() {
        let gain_at = |order: usize| -> f64 {
            let sections = butterworth_lowpass(order, 0.05).unwrap();
            sections.iter().map(|s| s.magnitude(0.2)).product()
        };
        assert!(gain_at(4) < gain_at(2));
        assert!(gain_at(8) < gain_at(4));
    }

    #[test]
    fn filtering_a_constant_settles_to_that_constant() {
        // A unity-DC-gain filter passes a constant once the transient decays.
        let mut sections = butterworth_lowpass(4, 0.1).unwrap();
        let mut y = vec![1.0_f64; 2000];
        for s in sections.iter_mut() {
            y = s.filter(&y);
        }
        assert!((y[1999] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_designs_are_rejected() {
        assert!(butterworth_lowpass(3, 0.1).is_none());
        assert!(butterworth_lowpass(0, 0.1).is_none());
        assert!(butterworth_lowpass(2, 0.6).is_none());
        assert!(Biquad::lowpass(0.1, 0.0).is_none());
    }
}
