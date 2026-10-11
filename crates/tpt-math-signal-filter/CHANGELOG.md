# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Biquad::highpass`, `Biquad::bandpass` (constant 0 dB peak gain) and `Biquad::notch` (RBJ cookbook forms). `Biquad::lowpass` now shares their parameter validation.
- `Biquad`: second-order IIR section (RBJ low-pass, direct form II transposed), with `process`, `filter`, `reset`, and `magnitude`.
- `butterworth_lowpass`: even-order Butterworth low-pass as a biquad cascade (−3 dB at the cutoff).

## [0.1.0]

- Initial workspace release.
