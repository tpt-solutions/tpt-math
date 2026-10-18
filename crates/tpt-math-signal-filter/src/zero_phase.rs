//! Zero-phase filtering: run a filter forward and backward so that the phase
//! shift cancels, as `scipy.signal.filtfilt` does.

use crate::Biquad;

/// Zero-phase filtering of `signal` by a cascade of biquad sections.
///
/// The signal is extended at both ends by odd reflection (`2·x₀ − x_k`), the
/// cascade is run forward, the result is reversed and run again, and the
/// extension is cut away. The magnitude response is the square of the
/// cascade's, and the phase is zero. The sections are reset first, so their
/// incoming state is ignored.
///
/// Returns an empty vector for an empty signal.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::{butterworth_lowpass, zero_phase_filter};
///
/// let sections = butterworth_lowpass(2, 0.2).unwrap();
/// let x: Vec<f64> = (0..200).map(|i| (0.02 * i as f64 * 6.283_185).sin()).collect();
/// let y = zero_phase_filter(&sections, &x);
/// assert_eq!(y.len(), x.len());
/// ```
pub fn zero_phase_filter(sections: &[Biquad], signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }
    // Pad by up to 3·(filter order) samples, the usual choice for a biquad cascade.
    let pad = (3 * (2 * sections.len() + 1)).min(n - 1);
    let mut ext: Vec<f64> = Vec::with_capacity(n + 2 * pad);
    for k in (1..=pad).rev() {
        ext.push(2.0 * signal[0] - signal[k]);
    }
    ext.extend_from_slice(signal);
    for k in 1..=pad {
        ext.push(2.0 * signal[n - 1] - signal[n - 1 - k]);
    }

    let mut stages: Vec<Biquad> = sections.to_vec();
    let mut pass = ext;
    for _ in 0..2 {
        for stage in stages.iter_mut() {
            stage.reset();
        }
        for stage in stages.iter_mut() {
            pass = stage.filter(&pass);
        }
        pass.reverse();
    }
    pass[pad..pad + n].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::butterworth_lowpass;

    #[test]
    fn passband_sine_is_kept_without_phase_shift() {
        let sections = butterworth_lowpass(2, 0.2).unwrap();
        let n = 400;
        let w = 0.02 * core::f64::consts::TAU;
        let x: Vec<f64> = (0..n).map(|i| (w * i as f64).sin()).collect();
        let y = zero_phase_filter(&sections, &x);
        // Away from the edges, the output tracks the input (gain 1, phase 0).
        for i in 40..(n - 40) {
            assert!(
                (y[i] - x[i]).abs() < 1e-3,
                "sample {i}: {} vs {}",
                y[i],
                x[i]
            );
        }
    }

    #[test]
    fn stopband_sine_is_removed() {
        let sections = butterworth_lowpass(2, 0.05).unwrap();
        let n = 600;
        let w = 0.4 * core::f64::consts::TAU;
        let x: Vec<f64> = (0..n).map(|i| (w * i as f64).sin()).collect();
        let y = zero_phase_filter(&sections, &x);
        let peak = y[100..n - 100].iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        assert!(peak < 1e-3, "residual {peak}");
    }

    #[test]
    fn empty_and_short_inputs() {
        let sections = butterworth_lowpass(2, 0.2).unwrap();
        assert!(zero_phase_filter(&sections, &[]).is_empty());
        assert_eq!(zero_phase_filter(&sections, &[1.0]).len(), 1);
    }
}
