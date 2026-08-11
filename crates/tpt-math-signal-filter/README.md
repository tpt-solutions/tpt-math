# tpt-math-signal-filter

Digital filtering and windowing for `tpt-math`. This crate wraps no upstream
filter library: the FIR design, convolution, and window generators are written
directly against `f64` slices, using `tpt-math-signal-fft` only to evaluate
frequency responses.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
This is a leaf crate of the signal layer, sitting above `tpt-math-signal-fft`
(used for `FirFilter::frequency_response`) and `tpt-math-numeric` (for the
`Float` trait glue). It is re-exported by the `tpt-math-signal` umbrella as
`tpt_math_signal::filter`.

## Features

- No optional features: `default = []`, and the crate always builds the full
  API.
- Requires `std`. Every generator and filter operation returns an owned
  `Vec<f64>`, and the transitive `tpt-math-signal-fft` / `rustfft` dependency is
  `std`-only, so there is no `no_std` configuration.
- Two areas of functionality:
  - **FIR filtering** — `FirFilter`, with explicit taps (`FirFilter::new`) or
    windowed-sinc `lowpass` / `highpass` designs, plus centred (`filter`,
    `filter_in_place`), causal (`filter_causal`), and spectral
    (`frequency_response`) evaluation.
  - **Windowing** — `rectangular`, `hamming`, `hanning`, `blackman`,
    `bartlett`, the run-time-dispatched `Window` enum, and `apply_window`.

## Quick start

```toml
[dependencies]
tpt-math-signal-filter = "0.1"
```

Design a 31-tap low-pass at 0.1 cycles/sample and apply it:

```rust
use tpt_math_signal_filter::FirFilter;

let lp = FirFilter::lowpass(31, 0.1);
assert_eq!(lp.len(), 31);
// Unit DC gain: a constant passes through untouched (away from edges).
assert!((lp.coeffs().iter().sum::<f64>() - 1.0).abs() < 1e-12);

let signal: Vec<f64> = (0..256).map(|t| (t as f64 * 0.05).sin()).collect();
let smoothed = lp.filter(&signal);
assert_eq!(smoothed.len(), signal.len());
```

Taper a frame before an FFT:

```rust
use tpt_math_signal_filter::{apply_window, Window};

let frame = [1.0; 5];
let tapered = apply_window(&frame, Window::Hanning);
assert_eq!(tapered.len(), 5);
assert!(tapered[0].abs() < 1e-12);          // edges are pulled to zero
assert!((tapered[2] - 1.0).abs() < 1e-12);  // the centre is untouched

// The rectangular window is the identity.
assert_eq!(apply_window(&frame, Window::Rectangular), frame.to_vec());
```

## Notes

- **`filter` is centred, not causal.** The output has the same length as the
  input and the filter's integer group delay `(m - 1) / 2` is compensated, so
  for the symmetric (linear-phase) designs produced by `lowpass` / `highpass`
  the result is zero-phase in the interior. Use `filter_causal` for the
  textbook streaming convolution `y[i] = sum_k h[k] * x[i-k]` with the delay
  left in.
- Outside the signal the input is treated as zero, so roughly the first and
  last `group_delay()` samples of `filter` are edge transients.
- With an even number of taps the true group delay is a half-integer; `filter`
  compensates only the integer part and leaves a residual half-sample delay.
- Cutoffs are given in **cycles per sample** in the open interval `(0, 0.5)`,
  where `0.5` is Nyquist. `lowpass` panics outside that range or with zero taps;
  `highpass` additionally requires an odd tap count, because it is built by
  spectral inversion around a single centre tap.
- Windows use the **symmetric** convention (`w[i] == w[n - 1 - i]`), which is
  the right choice for linear-phase FIR design. For the periodic (DFT-even)
  variant preferred by some spectral-analysis code, take the first `n` samples
  of an `n + 1` point window. Length `0` yields an empty window and length `1`
  yields `[1.0]` for every shape.
- `frequency_response(n)` panics if `n` is smaller than the tap count, which
  would truncate rather than zero-pad the impulse response.
- Despite the crate description, only FIR filtering is implemented today; IIR
  designs are not yet part of the public API.

## License

Licensed under either of MIT or Apache-2.0 at your option.
