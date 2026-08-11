# tpt-math-autodiff-rev

Reverse-mode (tape / Wengert-list) automatic differentiation. Where forward
mode propagates one derivative direction per evaluation, this crate records the
computation on a `GradientTape` and replays it backwards, producing **all**
partial derivatives of a single scalar output in one pass — the right tool for
gradients of many-input, one-output functions such as loss and objective
functions. It consolidates the prior TPT reverse-mode crates (`tpt-grad`,
`tpt-grad-macro`, `tpt-zero-grad`) into one permissively-licensed, dependency-light
implementation.

## Part of tpt-math

Part of the [`tpt-math`](https://github.com/tpt-solutions/tpt-math) workspace —
the numeric substrate for `tpt-science`, `tpt-engineering`, and `tpt-formal`.
This is the reverse-mode half of the autodiff layer: it sits directly above
`tpt-math-autodiff-fwd` (its only dependency, used to cross-check the two modes
in tests) and beneath the `tpt-math-autodiff` umbrella crate, which re-exports
it as `rev`.

## Features

- `std` *(default)* — enables `std` on `tpt-math-autodiff-fwd`.

The tape needs an allocator (it is a `Vec` of nodes) and `f64` transcendentals,
so this crate is **std-only** for now. The `std` feature exists for consistency
with the sibling autodiff crates; building with `default-features = false` is
not currently supported.

## Quick start

```toml
[dependencies]
tpt-math-autodiff-rev = "0.1"
```

Build an expression on a tape and read off the whole gradient:

```rust
use tpt_math_autodiff_rev::GradientTape;

let tape = GradientTape::new();
let x = tape.var(2.0);
let y = tape.var(3.0);
let z = x * x + y.sin();
let g = tape.gradient(z, &[x, y]);

assert!((g[0] - 4.0).abs() < 1e-9);            // dz/dx = 2x
assert!((g[1] - 3.0_f64.cos()).abs() < 1e-9);  // dz/dy = cos(y)
```

When the tape itself is not interesting, `value_and_gradient` creates it, seeds
one `Variable` per input, and runs the backward pass for you:

```rust
use tpt_math_autodiff_rev::value_and_gradient;

// f(x, y) = x * y + sin(x)
let (value, grad) = value_and_gradient(&[2.0, 3.0], |_tape, v| v[0] * v[1] + v[0].sin());

assert!((value - (6.0 + 2.0_f64.sin())).abs() < 1e-9);
assert!((grad[0] - (3.0 + 2.0_f64.cos())).abs() < 1e-9);
assert!((grad[1] - 2.0).abs() < 1e-9);
```

`GradientTape::backward` returns a `Gradient` covering *every* node, which can
be queried per variable with `Gradient::wrt` or the `Index<Variable>` impl.

## Notes

- Supported primitives: `+`, `-`, `*`, `/` (between variables and with plain
  `f64` on either side), the `*Assign` counterparts, unary `-`, `sin`, `cos`,
  `tan`, `exp`, `ln`, `sqrt`, `powi`, `powf`, `recip`, plus `constant`s that
  never carry a gradient. `GradientTape::push` is public, so further primitives
  can be layered on top of the tape.
- `Variable<'t>` borrows its tape, and the tape uses interior mutability, so
  recording only needs `&self` while variables stay `Copy`. Mixing variables
  from two different tapes panics.
- Scalars are `f64` only; use `tpt-math-autodiff-fwd` for generic scalar types
  or `no_std` targets.
- `#![forbid(unsafe_code)]`.
- Dual-licensed `MIT OR Apache-2.0`, matching the rest of the workspace. The
  implementation is self-contained: apart from `tpt-math-autodiff-fwd` there are
  no dependencies, so no upstream autodiff licence applies.

## License

Licensed under either of MIT or Apache-2.0 at your option.
