# crates.io Publish Checklist — current release queue

Status as of 2026-11-21. Versions checked against the crates.io sparse index,
not against the CHANGELOGs.

## 1. Where crates.io stands

The previous release (2026-10-20) put 28 crates and the facade on crates.io, each
with a git tag `tpt-math-<crate>-v<version>` (facade: `tpt-math-v0.1.0`).

Against the index today, 33 of the 34 workspace crates match their local version.
The exception is `tpt-math-splat`, which has never been published.

Crates whose published version is behind the local version: `exact`, `prob-dist`,
`prob`, `tpt-math`. The first three are breaking changes, so they take a minor
bump under the pre-1.0 policy. `tpt-math` moves with them because its public
re-exports change.

## 2. Release queue

Publish in this order. Each crate depends only on crates earlier in the list or
already on the index.

| # | Crate | On crates.io | Local | Why |
|---|---|---|---|---|
| 1 | `tpt-math-splat` | — | 0.1.0 | New crate. Dry run passes. |
| 2 | `tpt-math-exact` | 0.1.1 | **0.2.0** | Breaking: `BigRational` is now defined in this crate instead of re-exported from `num-rational`; `num-bigint` moves 0.4 → 0.5. |
| 3 | `tpt-math-prob-dist` | 0.1.1 | **0.2.0** | Breaking: `CoreRng` implements `rand::TryRng` instead of `rand::RngCore` (rand 0.10). |
| 4 | `tpt-math-prob` | 0.1.0 | **0.2.0** | Re-exports `tpt-math-prob-dist` as `dist`, so the breaking change reaches users only through a new `prob` release. Needs step 3 on the index first. |
| 5 | `tpt-math` (facade) | 0.1.0 | **0.2.0** | Re-exports `exact` (breaking) and `prob`; adds the `splat` feature. Needs steps 1, 2 and 4 on the index first. |

Not in the queue, with no release needed:

- The other 29 crates pass the semver check against their published version
  (section 3). Their local manifests now point at `exact` 0.2.0 or `prob-dist`
  0.2.0 where they depend on them (`symbolic`, `autodiff`). Their published
  versions still resolve the 0.1.x dependencies, so consumers are unaffected.
  Release them only when they change; that release will pick up the new
  dependency.
- Benchmark files that moved to `std::hint::black_box`, and the `fir.rs` loop
  tidy in `signal-filter`, are internal. They have no changelog entry and no
  release.

Decisions taken in this pass (reverse any of them before publishing):

- The facade goes to 0.2.0 rather than 0.1.1 because its `exact` re-exports
  change type. The `splat` feature alone would have been a patch release.
- Changelog entries for the new versions are stamped `2026-11-21`. Change the
  date if publishing slips.

## 3. Verification (this pass)

- `cargo check --workspace --all-features --all-targets`: passes. Two
  `unused dependency` warnings are reported; not investigated.
- `cargo test` for `exact`, `prob-dist`, `splat`, `symbolic`, `autodiff` and
  `tpt-math` with `--all-features`: all pass, including doctests.
- `cargo semver-checks check-release --workspace`: all crates report "no change"
  except the three bumped crates, which report "major change". The tool skips
  the checks for a 0.x minor bump, so the breaking classification rests on the
  changelogs and the code diff. The run stops at `tpt-math-splat`: it has no
  crates.io baseline, so the registry lookup fails. This is the same failure
  that broke the push job, and the `--exclude tpt-math-splat` in
  `.github/workflows/ci.yml` stays until splat is published.
- `cargo publish --dry-run`: passes for `tpt-math-splat`, `tpt-math-exact` and
  `tpt-math-prob-dist`. `tpt-math-prob` and `tpt-math` cannot be dry-run until
  their dependencies are on the index.

## 4. Publish steps

- [ ] Commit the version bumps, changelog and README changes, and the CI change.
- [ ] `cargo publish -p tpt-math-splat`, then confirm it on the index.
- [ ] `cargo publish -p tpt-math-exact`, then confirm.
- [ ] `cargo publish -p tpt-math-prob-dist`, then confirm.
- [ ] `cargo publish -p tpt-math-prob`, then confirm.
- [ ] `cargo publish -p tpt-math`, then confirm.
- [ ] Check docs.rs builds for all five at the new versions.
- [ ] Tag: `tpt-math-splat-v0.1.0`, `tpt-math-exact-v0.2.0`,
      `tpt-math-prob-dist-v0.2.0`, `tpt-math-prob-v0.2.0`, `tpt-math-v0.2.0`.
- [ ] Remove `--exclude tpt-math-splat` from the push step in
      `.github/workflows/ci.yml`.
- [ ] Move each published crate's `## [Unreleased]` heading below the new version
      heading if the new entries were not stamped before publishing.

## 5. Open items

- `release-plz.toml` keeps `publish = false`. The file's comment asks for a manual
  review of the first release, which this one is not. Leave it unless you decide
  otherwise.
- `symbolic` implements its coefficient trait for `tpt_math_exact::BigRational`.
  Its next release should be checked against `exact` 0.2.0 before publishing.
- `todo.md` item 7.5 (release-plz) still reflects the earlier state.

## 6. Corrections to the previous version of this document

- Section 3 said `cargo publish --dry-run` could not run standalone. It can, once
  the crate's dependencies are on the index; `splat`, `exact` and `prob-dist`
  now pass.
- The earlier release table listed 28 crates plus the facade. Every crate in that
  table is on the index at its listed version; that history lives in the git tags.
