# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-21

### Changed (breaking)

- The `dist` re-export is `tpt-math-prob-dist` 0.2.0, where `CoreRng` implements `rand::TryRng` instead of `rand::RngCore`. Code that used `RngCore` methods on `CoreRng` must import `rand::Rng` instead.

## [0.1.0]

- Initial workspace release.
