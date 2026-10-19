# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `Fft::rfft` and `Fft::irfft`: real-input transforms returning the `n/2 + 1` non-redundant bins. For even `n` they use one complex transform of length `n/2`. Checked against the full transform for even and odd lengths.
- `stft`: short-time Fourier transform. Frames are complex, so `|·|²` gives the spectrogram.
- `welch`: Welch power spectral density, one-sided.
- `fft_convolve`: linear convolution by FFT.

## [0.1.0]

- Initial workspace release.
