# tpt-math-signal

Umbrella crate for the `tpt-math` signal-processing layer: it re-exports
`tpt-math-signal-fft` (an ergonomic layer over
[`rustfft`](https://crates.io/crates/rustfft)) and `tpt-math-signal-filter`
(FIR filtering and window functions) behind Cargo features. Depend on this crate
when you want spectral analysis and filtering from a single dependency line.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
This crate sits at the top of the signal layer and contains no algorithms of its
own; it is purely a re-export facade over the two leaf crates below it:

| Feature | Re-exported as | Source crate |
|---|---|---|
| `tpt-math-signal-fft` | `fft` | [`tpt-math-signal-fft`](https://docs.rs/tpt-math-signal-fft) — FFT / spectra via `rustfft` |
| `tpt-math-signal-filter` | `filter` | [`tpt-math-signal-filter`](https://docs.rs/tpt-math-signal-filter) — FIR filtering and window functions |

The two work together: `tpt-math-signal-filter` already depends on
`tpt-math-signal-fft` internally for `FirFilter::frequency_response`, and both
speak plain `&[f64]` / `Vec<f64>` plus `Complex<f64>`, so tapering, filtering,
and transforming compose without conversion.

## Features

- `tpt-math-signal-fft` *(default)* — pulls in `tpt-math-signal-fft` and exposes
  it as the `fft` module: the reusable `Fft` engine, the one-shot `fft` / `ifft`
  / `ifft_normalized` functions, `magnitude`, `power_spectrum`, and the
  `rustfft` / `Complex` re-exports.
- `tpt-math-signal-filter` *(default)* — pulls in `tpt-math-signal-filter` and
  exposes it as the `filter` module: `FirFilter` (explicit taps or windowed-sinc
  `lowpass` / `highpass`), the `Window` enum and its free-function window
  generators, and `apply_window`.
- Default: both features are on. Disable default features to take only the half
  you need — note that selecting only `tpt-math-signal-filter` still pulls in
  `rustfft` transitively, since the filter crate uses it for frequency
  responses.

This crate requires `std`; `rustfft` is `std`-only and both sub-crates return
owned `Vec`s, so there is no `no_std` configuration of this umbrella.

## Quick start

```toml
[dependencies]
tpt-math-signal = "0.1"
```

Window a frame, filter it, and take its spectrum, using both re-exports:

```rust
use tpt_math_signal::fft::{magnitude, Fft};
use tpt_math_signal::filter::{apply_window, FirFilter, Window};

// A low-frequency tone buried under a high-frequency one.
let signal: Vec<f64> = (0..256)
    .map(|t| (t as f64 * 0.1).sin() + (t as f64 * 2.0).sin())
    .collect();

// Keep the low tone: 31-tap windowed-sinc low-pass at 0.1 cycles/sample.
let lp = FirFilter::lowpass(31, 0.1);
let filtered = lp.filter(&signal);
assert_eq!(filtered.len(), signal.len());

// Taper the frame, then transform it with a reusable planner.
let tapered = apply_window(&filtered, Window::Hanning);
let mut engine = Fft::new();
let spectrum = engine.forward(&tapered);
let mags = magnitude(&spectrum);
assert_eq!(mags.len(), 256);
```

Taking only one half:

```toml
[dependencies]
tpt-math-signal = { version = "0.1", default-features = false, features = ["tpt-math-signal-fft"] }
```

## Notes

- The feature names deliberately match the crate names, so
  `--features tpt-math-signal-fft` enables exactly the `tpt-math-signal-fft`
  dependency and nothing else.
- Transforms reached through `fft` are **unnormalized**, as in `rustfft`:
  `ifft(fft(x))` equals `n * x`. Use `fft::ifft_normalized` /
  `Fft::inverse_normalized` to round-trip exactly.
- `FirFilter::filter` is delay-compensated (centred), not causal; see the
  `tpt-math-signal-filter` README for the exact convention.
- Only FIR filtering is implemented in the `filter` module today; IIR designs
  are not yet part of the public API.
- `rustfft`, reached transitively through `fft`, is itself dual-licensed
  `MIT OR Apache-2.0`, matching this crate.

## License

Licensed under either of MIT or Apache-2.0 at your option.
