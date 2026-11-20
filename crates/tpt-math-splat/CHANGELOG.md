# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to SemVer.

## [Unreleased]

## [0.1.0] - 2026-11-21

### Added

- Initial release: packed `Sym3`/`Sym4`, Cholesky / LDLᵀ, SPD inverse and
  determinant, Jacobi eigensolver for 3×3 symmetric matrices.
- PSD-by-construction 3D/4D covariances (`R S Sᵀ Rᵀ`), 4D time-conditioning,
  EWA projection to a 2D covariance and conic.
- Real spherical harmonics, degrees 0–3 (3DGS convention).
- Morton (2D/3D/4D) and N-dimensional Hilbert keys, quantisation helpers.
