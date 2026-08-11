# tpt-math-signal-fft

A thin, ergonomic layer over [`rustfft`](https://crates.io/crates/rustfft), the
fast pure-Rust FFT implementation. `rustfft` is planner-oriented and transforms
in place; this crate keeps that engine available while adding slice-in/`Vec`-out
entry points, a reusable planner handle, and the two spectral summaries most
callers write by hand.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
This is a leaf crate of the signal layer: it depends only on
`tpt-math-numeric` (for the `Float` trait glue) and `rustfft`. It is consumed by
`tpt-math-signal-filter` (for frequency responses) and re-exported by the
`tpt-math-signal` umbrella as `tpt_math_signal::fft`.

## Features

- No optional features: `default = []`, and the crate always builds the full
  API.
- Requires `std`. `rustfft` allocates and dispatches on runtime CPU feature
  detection, and every entry point here returns an owned `Vec`, so there is no
  `no_std` configuration.
- Re-exports `rustfft` and `rustfft::num_complex::Complex` (plus
  `tpt_math_numeric`), so callers can drop down to the raw planner without
  adding a second, possibly version-skewed, `rustfft` dependency.

## Quick start

```toml
[dependencies]
tpt-math-signal-fft = "0.1"
```

One-shot transforms with the free functions:

```rust
use tpt_math_signal_fft::{fft, ifft_normalized};

let signal = [1.0, 2.0, 3.0, 4.0];
let spectrum = fft(&signal);
assert_eq!(spectrum.len(), 4);
// X[0] is the DC term: the sum of the samples.
assert!((spectrum[0].re - 10.0).abs() < 1e-12);

// Round-trip back to the original samples.
let recovered = ifft_normalized(&spectrum);
for (got, want) in recovered.iter().zip(signal.iter()) {
    assert!((got.re - want).abs() < 1e-12);
    assert!(got.im.abs() < 1e-12);
}
```

For repeated transforms, hold an `Fft` so the plan and its twiddle factors are
cached across calls:

```rust
use tpt_math_signal_fft::Fft;

let mut engine = Fft::new();
let a = engine.forward(&[1.0, 0.0, 1.0, 0.0]);
let b = engine.forward(&[0.0, 1.0, 0.0, 1.0]); // reuses the cached plan
assert_eq!(a.len(), 4);
assert_eq!(b.len(), 4);
```

The API surface is small:

- `Fft::new` / `Fft::default` — a reusable engine over a `rustfft::FftPlanner<f64>`.
- `Fft::forward`, `Fft::forward_n` (zero-pad or truncate to `n`), `Fft::forward_cplx`.
- `Fft::inverse`, `Fft::inverse_cplx`, `Fft::inverse_normalized`.
- `fft`, `ifft`, `ifft_normalized` — one-shot equivalents that build a planner internally.
- `magnitude` (per-bin `|X[k]|`, computed with `hypot`) and `power_spectrum`
  (`|X[k]|^2` of a real signal).

## Notes

- **Transforms are unnormalized**, exactly as in `rustfft`, FFTW, and NumPy:
  the forward transform is `X[k] = sum_t x[t] * exp(-2*pi*i*k*t/n)` and the
  inverse is the same sum with `exp(+2*pi*i*k*t/n)`, so composing them scales
  the signal by `n`. Use `ifft_normalized` / `Fft::inverse_normalized` (or
  divide by `n` yourself) to recover the original samples.
- Every method on `Fft` takes `&mut self`, because planning mutates the
  planner's cache.
- Real inputs give a conjugate-symmetric spectrum: bins `n/2 + 1 ..` mirror
  bins `1 ..= n/2` and carry no new information.
- Empty inputs are handled: they produce empty outputs rather than panicking.
- All lengths work, but highly composite lengths (powers of two above all) are
  the fastest case for `rustfft`; `Fft::forward_n` exists to round a length up
  to one.
- `rustfft` is itself dual-licensed `MIT OR Apache-2.0`, matching this crate.

## License

Licensed under either of MIT or Apache-2.0 at your option.
