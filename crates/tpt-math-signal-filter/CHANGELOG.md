# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `butterworth_bandpass`: Butterworth band-pass as a high-pass stage at the low edge cascaded with a low-pass stage at the high edge. This is a cascade, not an LP→BP prototype transform, so band edges are only approximately −3 dB for narrow bands.
- `butterworth_highpass`: Butterworth high-pass cascade of even order, the high-pass counterpart of `butterworth_lowpass`.
- `Biquad::highpass`, `Biquad::bandpass` (constant 0 dB peak gain) and `Biquad::notch` (RBJ cookbook forms). `Biquad::lowpass` now shares their parameter validation.
- `Biquad`: second-order IIR section (RBJ low-pass, direct form II transposed), with `process`, `filter`, `reset`, and `magnitude`.
- `butterworth_lowpass`: even-order Butterworth low-pass as a biquad cascade (−3 dB at the cutoff).

## [0.1.0]

- Initial workspace release.
