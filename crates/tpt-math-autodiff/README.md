# tpt-math-autodiff

Umbrella crate for the `tpt-math` automatic-differentiation family. It pulls in
the forward-mode (dual-number) and reverse-mode (tape) crates behind Cargo
features and re-exports them under short module names, so a downstream crate can
depend on one name instead of two.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
This crate contains no algorithms of its own; it is the top of the autodiff
layer and only re-exports its siblings:

| Feature | Re-exported as | Source crate |
|---|---|---|
| `tpt-math-autodiff-fwd` | `fwd` | [`tpt-math-autodiff-fwd`](https://docs.rs/tpt-math-autodiff-fwd) — forward-mode dual numbers |
| `tpt-math-autodiff-rev` | `rev` | [`tpt-math-autodiff-rev`](https://docs.rs/tpt-math-autodiff-rev) — reverse-mode gradient tape |

## Features

- `tpt-math-autodiff-fwd` *(default)* — pulls in `tpt-math-autodiff-fwd` and
  re-exports it as `fwd`.
- `tpt-math-autodiff-rev` *(default)* — pulls in `tpt-math-autodiff-rev` and
  re-exports it as `rev`.

Both features are enabled by default; each maps one-to-one onto an optional
dependency, so disabling one drops the dependency entirely. `fwd` is `no_std`
(its own `std` feature is default-on but can be turned off); `rev` needs an
allocator and `std`, so enabling the `tpt-math-autodiff-rev` feature makes the
build std-only.

## Quick start

```toml
[dependencies]
tpt-math-autodiff = { version = "0.1", features = ["tpt-math-autodiff-fwd", "tpt-math-autodiff-rev"] }
```

Both modes differentiate the same function through the re-exported modules:

```rust
use tpt_math_autodiff::{fwd::Dual, rev::GradientTape};

// Forward mode: f(x) = x^2, so f'(3) = 6.
let x = Dual::<f64>::variable(3.0, 0);
let y = x * x;
assert_eq!(y.re(), 9.0);
assert_eq!(y.du(0), 6.0);

// Reverse mode: the same derivative, read off a tape.
let tape = GradientTape::new();
let x = tape.var(3.0);
let y = x * x;
assert!((tape.gradient(y, &[x])[0] - 6.0).abs() < 1e-12);
```

Taking only what you need works too — `default-features = false` plus
`features = ["tpt-math-autodiff-fwd"]` gives just `tpt_math_autodiff::fwd`.

## Notes

- The feature names deliberately match the crate names, which is what lets each
  feature gate exactly one optional dependency.
- Pick forward mode for few inputs / many outputs (cost scales with the number
  of input directions) and reverse mode for many inputs / one scalar output
  (one backward pass yields the whole gradient).
- Reverse mode is `f64`-only and allocates; forward mode is generic over the
  scalar type and `no_std`-capable.
- Dual-licensed `MIT OR Apache-2.0`, matching the rest of the workspace. Neither
  re-exported crate wraps a third-party autodiff library, so no upstream
  autodiff licence applies.

## License

Licensed under either of MIT or Apache-2.0 at your option.
