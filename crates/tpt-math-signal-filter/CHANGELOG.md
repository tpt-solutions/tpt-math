# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `FirFilter::filter_par` behind the new `rayon` feature. The output is bitwise identical to `filter`.
- Benchmarks for direct and overlap-add filtering at 9 and 255 taps.
- `FirFilter::filter_overlap_add`: FFT overlap-add filtering for long kernels. It gives the same output as `filter`, at `O(n log m)` instead of `O(n·m)`.
- `butterworth_bandpass`: Butterworth band-pass as a high-pass stage at the low edge cascaded with a low-pass stage at the high edge. This is a cascade, not an LP→BP prototype transform, so band edges are only approximately −3 dB for narrow bands.
- `butterworth_highpass`: Butterworth high-pass cascade of even order, the high-pass counterpart of `butterworth_lowpass`.
- `Biquad::highpass`, `Biquad::bandpass` (constant 0 dB peak gain) and `Biquad::notch` (RBJ cookbook forms). `Biquad::lowpass` now shares their parameter validation.
- `Biquad`: second-order IIR section (RBJ low-pass, direct form II transposed), with `process`, `filter`, `reset`, and `magnitude`.
- `butterworth_lowpass`: even-order Butterworth low-pass as a biquad cascade (−3 dB at the cutoff).
- `zero_phase_filter`: forward-backward filtering over a biquad cascade (`filtfilt`), with odd-extension padding at the ends. It has no phase lag.

### Changed

- `FirFilter::filter_overlap_add` uses the real-input transforms `rfft`/`irfft`, roughly halving the spectrum work.

## [0.1.0]

- Initial workspace release.
