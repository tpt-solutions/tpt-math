//! Window functions.
//!
//! Every generator here returns a `Vec<f64>` of exactly `n` samples using the
//! **symmetric** convention (`w[i] == w[n - 1 - i]`), which is the right choice
//! for designing linear-phase FIR filters. Spectral analysis code sometimes
//! prefers the *periodic* (DFT-even) variant instead; build it by taking the
//! first `n` samples of an `n + 1` point symmetric window.
//!
//! Degenerate lengths follow the usual convention: length `0` yields an empty
//! window, and length `1` yields `[1.0]` for every shape, since a single sample
//! has no taper to apply.

use core::f64::consts::PI;

use tpt_math_numeric::Float;

/// A named window shape.
///
/// This is the value-level counterpart to the free functions in this module,
/// for code that has to choose a window at run time (a CLI flag, a config file,
/// a UI dropdown).
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::{hamming, Window};
///
/// assert_eq!(Window::Hamming.coefficients(8), hamming(8));
/// assert_eq!(Window::Rectangular.coefficients(3), vec![1.0, 1.0, 1.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Window {
    /// No taper: every sample weighted by `1.0` (a plain boxcar).
    Rectangular,
    /// Hamming window, `0.54 - 0.46*cos(2*pi*i/(n-1))`.
    Hamming,
    /// Hann ("hanning") window, `0.5 - 0.5*cos(2*pi*i/(n-1))`.
    Hanning,
    /// Blackman window, `0.42 - 0.5*cos(x) + 0.08*cos(2x)`.
    Blackman,
    /// Bartlett (triangular) window with zero endpoints.
    Bartlett,
}

impl Window {
    /// Generate the `n` window coefficients for this shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_signal_filter::Window;
    ///
    /// let w = Window::Hanning.coefficients(5);
    /// assert_eq!(w.len(), 5);
    /// assert!(w[0].abs() < 1e-12 && w[4].abs() < 1e-12);
    /// assert!((w[2] - 1.0).abs() < 1e-12);
    /// ```
    pub fn coefficients(self, n: usize) -> Vec<f64> {
        match self {
            Window::Rectangular => rectangular(n),
            Window::Hamming => hamming(n),
            Window::Hanning => hanning(n),
            Window::Blackman => blackman(n),
            Window::Bartlett => bartlett(n),
        }
    }

    /// The window's lowercase name, e.g. `"hamming"`.
    ///
    /// Useful for logging and for round-tripping a configuration value.
    pub fn name(self) -> &'static str {
        match self {
            Window::Rectangular => "rectangular",
            Window::Hamming => "hamming",
            Window::Hanning => "hanning",
            Window::Blackman => "blackman",
            Window::Bartlett => "bartlett",
        }
    }
}

/// Rectangular (boxcar) window: `n` copies of `1.0`.
///
/// The "no window" window. It has the narrowest main lobe of any window and
/// the worst sidelobes (-13 dB), so it leaks badly for anything that is not
/// periodic in the frame.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::rectangular;
///
/// assert_eq!(rectangular(4), vec![1.0; 4]);
/// assert!(rectangular(0).is_empty());
/// ```
pub fn rectangular(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Hamming window of length `n`.
///
/// `w[i] = 0.54 - 0.46*cos(2*pi*i/(n-1))`.
///
/// The endpoints are `0.08` rather than zero — the classic Hamming trade: the
/// discontinuity at the edges buys a much lower first sidelobe (-43 dB) than
/// the Hann window at the cost of a slower asymptotic roll-off. This is the
/// default taper used by [`FirFilter::lowpass`](crate::FirFilter::lowpass).
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::hamming;
///
/// let w = hamming(9);
/// assert!((w[0] - 0.08).abs() < 1e-12);
/// assert!((w[4] - 1.00).abs() < 1e-12);
/// assert_eq!(w[0], w[8]); // symmetric
/// ```
pub fn hamming(n: usize) -> Vec<f64> {
    cosine_sum(n, &[0.54, 0.46])
}

/// Hann window of length `n`, spelled "hanning" as in MATLAB and NumPy.
///
/// `w[i] = 0.5 - 0.5*cos(2*pi*i/(n-1))`.
///
/// A raised cosine that reaches exactly zero at both ends, so a windowed frame
/// joins smoothly onto silence. Sidelobes start at -31 dB but fall off at
/// 18 dB/octave, which beats Hamming far from the main lobe.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::hanning;
///
/// let w = hanning(5);
/// assert!(w[0].abs() < 1e-12);
/// assert!((w[2] - 1.0).abs() < 1e-12);
/// ```
pub fn hanning(n: usize) -> Vec<f64> {
    cosine_sum(n, &[0.5, 0.5])
}

/// Blackman window of length `n`.
///
/// `w[i] = 0.42 - 0.5*cos(x) + 0.08*cos(2x)`, with `x = 2*pi*i/(n-1)`.
///
/// A three-term cosine sum: wider main lobe than Hamming, but sidelobes buried
/// at -58 dB. Use it when stopband leakage matters more than resolution.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::blackman;
///
/// let w = blackman(7);
/// assert!(w[0].abs() < 1e-12);
/// assert!((w[3] - 1.0).abs() < 1e-12);
/// ```
pub fn blackman(n: usize) -> Vec<f64> {
    cosine_sum(n, &[0.42, 0.5, 0.08])
}

/// Bartlett (triangular) window of length `n`, with zero endpoints.
///
/// `w[i] = 1 - |(i - c) / c|` where `c = (n - 1) / 2`.
///
/// The convolution of two rectangular windows, hence a squared-sinc spectrum:
/// no negative sidelobes, but only -26 dB of suppression.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::bartlett;
///
/// assert_eq!(bartlett(5), vec![0.0, 0.5, 1.0, 0.5, 0.0]);
/// ```
pub fn bartlett(n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let center = (n - 1) as f64 / 2.0;
            (0..n)
                .map(|i| 1.0 - Float::abs((i as f64 - center) / center))
                .collect()
        }
    }
}

/// Multiply a signal by a window of the same length.
///
/// The window is generated at `signal.len()` points, so the result always has
/// the same length as the input. This is the tapering step that precedes an
/// FFT: it forces the frame to decay to (nearly) zero at both ends, which
/// suppresses the spectral leakage caused by the implicit periodic extension.
///
/// # Examples
///
/// ```
/// use tpt_math_signal_filter::{apply_window, Window};
///
/// let frame = [1.0; 5];
/// let tapered = apply_window(&frame, Window::Hanning);
/// assert_eq!(tapered.len(), 5);
/// assert!(tapered[0].abs() < 1e-12);      // edges are pulled to zero
/// assert!((tapered[2] - 1.0).abs() < 1e-12); // the centre is untouched
///
/// // The rectangular window is the identity.
/// assert_eq!(apply_window(&frame, Window::Rectangular), frame.to_vec());
/// ```
pub fn apply_window(signal: &[f64], window: Window) -> Vec<f64> {
    let coefficients = window.coefficients(signal.len());
    signal
        .iter()
        .zip(coefficients)
        .map(|(&x, w)| x * w)
        .collect()
}

/// Generalized cosine-sum window: `sum_j (-1)^j * a[j] * cos(j * 2*pi*i/(n-1))`.
///
/// Hamming, Hann, and Blackman are all members of this family; only the
/// coefficient vector differs.
fn cosine_sum(n: usize, a: &[f64]) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let step = 2.0 * PI / (n - 1) as f64;
            (0..n)
                .map(|i| {
                    let x = step * i as f64;
                    a.iter()
                        .enumerate()
                        .map(|(j, &aj)| {
                            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
                            sign * aj * Float::cos(j as f64 * x)
                        })
                        .sum()
                })
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Window; 5] = [
        Window::Rectangular,
        Window::Hamming,
        Window::Hanning,
        Window::Blackman,
        Window::Bartlett,
    ];

    #[test]
    fn hamming_edges_are_near_zero_and_centre_is_one() {
        let n = 65;
        let w = hamming(n);
        assert_eq!(w.len(), n);

        // The Hamming pedestal is 0.08: small compared to the unit centre.
        assert!(w[0] < 0.1, "first sample {} is not near zero", w[0]);
        assert!(w[n - 1] < 0.1, "last sample {} is not near zero", w[n - 1]);
        assert!((w[0] - 0.08).abs() < 1e-12);
        assert!((w[n - 1] - 0.08).abs() < 1e-12);

        // The centre tap is unity.
        assert!((w[n / 2] - 1.0).abs() < 1e-12, "centre = {}", w[n / 2]);
    }

    #[test]
    fn windows_are_symmetric_and_bounded() {
        for window in ALL {
            for n in [2usize, 3, 8, 9, 64, 65] {
                let w = window.coefficients(n);
                assert_eq!(w.len(), n, "{} at n = {n}", window.name());
                for i in 0..n {
                    assert!(
                        (w[i] - w[n - 1 - i]).abs() < 1e-12,
                        "{} is not symmetric at n = {n}",
                        window.name()
                    );
                    assert!(
                        (-1e-12..=1.0 + 1e-12).contains(&w[i]),
                        "{} left [0, 1] at n = {n}: {}",
                        window.name(),
                        w[i]
                    );
                }
            }
        }
    }

    #[test]
    fn tapered_windows_peak_at_the_centre() {
        for window in [Window::Hamming, Window::Hanning, Window::Blackman] {
            let n = 33;
            let w = window.coefficients(n);
            assert!((w[n / 2] - 1.0).abs() < 1e-12, "{}", window.name());
            for (i, &value) in w.iter().enumerate() {
                if i != n / 2 {
                    assert!(value < w[n / 2], "{} peaks off-centre", window.name());
                }
            }
        }
    }

    #[test]
    fn zero_length_windows_are_empty_and_unit_length_windows_are_one() {
        for window in ALL {
            assert!(window.coefficients(0).is_empty(), "{}", window.name());
            assert_eq!(window.coefficients(1), vec![1.0], "{}", window.name());
        }
    }

    #[test]
    fn hann_and_bartlett_vanish_at_the_endpoints() {
        for window in [Window::Hanning, Window::Bartlett] {
            let w = window.coefficients(16);
            assert!(w[0].abs() < 1e-12, "{}", window.name());
            assert!(w[15].abs() < 1e-12, "{}", window.name());
        }
    }

    #[test]
    fn bartlett_is_a_triangle() {
        assert_eq!(bartlett(5), vec![0.0, 0.5, 1.0, 0.5, 0.0]);

        // Even lengths straddle the peak; the slope is still uniform.
        let w = bartlett(4);
        let expected = [0.0, 2.0 / 3.0, 2.0 / 3.0, 0.0];
        for (got, want) in w.iter().zip(expected) {
            assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn apply_window_multiplies_elementwise() {
        let signal = [2.0, -3.0, 4.0, 0.5, 1.0];
        let expected: Vec<f64> = signal
            .iter()
            .zip(hamming(signal.len()))
            .map(|(x, w)| x * w)
            .collect();
        assert_eq!(apply_window(&signal, Window::Hamming), expected);
    }

    #[test]
    fn rectangular_window_is_the_identity() {
        let signal = [1.5, -2.5, 3.5];
        assert_eq!(apply_window(&signal, Window::Rectangular), signal.to_vec());
        assert!(apply_window(&[], Window::Blackman).is_empty());
    }

    #[test]
    fn enum_dispatch_matches_the_free_functions() {
        let n = 12;
        assert_eq!(Window::Rectangular.coefficients(n), rectangular(n));
        assert_eq!(Window::Hamming.coefficients(n), hamming(n));
        assert_eq!(Window::Hanning.coefficients(n), hanning(n));
        assert_eq!(Window::Blackman.coefficients(n), blackman(n));
        assert_eq!(Window::Bartlett.coefficients(n), bartlett(n));
    }
}
