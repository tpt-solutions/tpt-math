# Changelog

All notable changes to this crate are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.1] - 2026-08-19

- Added `try_set_initial`/`try_set_transition`/`try_set_emission` on the HMM,
  returning `Result<(), MarkovError>` instead of panicking on an invalid
  state, observation, or probability; the existing panicking setters now
  delegate to these.

## [0.1.0]

- Initial workspace release.
