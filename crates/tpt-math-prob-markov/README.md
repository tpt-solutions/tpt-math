# tpt-math-prob-markov

Finite Markov chains and discrete hidden Markov models. This crate consolidates
the prior TPT `tpt-zero-markov` work into the `tpt-math` probability stack:
transition matrices with stationary-distribution solving and trajectory
simulation, plus an HMM with Viterbi decoding and a scaled forward pass. Its
only dependency is `tpt-math-prob-core`.

## Part of tpt-math

Part of [tpt-math](https://github.com/tpt-solutions/tpt-math), the numeric
substrate for the TPT science / engineering / formal stack. It is a leaf crate
in the probability layer, built on the `Rng`/`Distribution` traits from
`tpt-math-prob-core` and re-exported by the `tpt-math-prob` umbrella. It is not
an umbrella crate.

## Features

- `default = []` — the crate declares no optional features; the full API is
  always available.
- **`std` required.** This crate is not `no_std`: matrices and traces are plain
  `Vec`s, so no linear-algebra backend is needed but an allocator and `std`
  are.
- `#![forbid(unsafe_code)]`, `#![warn(missing_docs, missing_debug_implementations)]`.

## Quick start

```toml
[dependencies]
tpt-math-prob-markov = "0.1"
```

```rust
use tpt_math_prob_markov::{Hmm, MarkovChain, SplitMix64};

// A two-state chain: rows are outgoing distributions.
let mut chain = MarkovChain::new(2);
chain.set_transition(0, 0, 0.9);
chain.set_transition(0, 1, 0.1);
chain.set_transition(1, 0, 0.5);
chain.set_transition(1, 1, 0.5);

let pi = chain.stationary();          // power iteration
assert!((pi[0] - 5.0 / 6.0).abs() < 1e-9);

let mut rng = SplitMix64::seed_from_u64(42);
let path = chain.run(&mut rng, 0, 1_000);
assert_eq!(path.len(), 1_000);

// Hidden: 0 = healthy, 1 = fever. Observed: 0 = normal, 1 = cold, 2 = dizzy.
let model = Hmm::from_parts(
    vec![0.6, 0.4],
    vec![vec![0.7, 0.3], vec![0.4, 0.6]],
    vec![vec![0.5, 0.4, 0.1], vec![0.1, 0.3, 0.6]],
)
.unwrap();

assert_eq!(model.viterbi(&[0, 1, 2]), vec![0, 0, 1]);
```

What is in the box:

- `MarkovChain` — `new`/`uniform`/`identity`/`from_rows`/`from_rows_normalized`,
  `set_transition`, `normalize`/`normalize_row`, `is_stochastic`, `validate`,
  `lazy`, `step_distribution`, `distribution_after`,
  `stationary`/`stationary_with`/`try_stationary`, `step`, `run`,
  `empirical_distribution`.
- `Hmm` (aliased `HiddenMarkovModel`) — `new`/`uniform`/`from_parts`,
  `set_initial`/`set_transition`/`set_emission`, `normalize`, `validate`,
  `hidden_chain`, `viterbi`/`try_viterbi`/`try_viterbi_with_log_prob`,
  `forward` (scaled), `log_likelihood`, `sample`.
- `sample_categorical` — inverse-CDF draw from relative (unnormalised) weights;
  the primitive behind `MarkovChain::step` and `Hmm::sample`.
- `MarkovError`, `PROBABILITY_TOLERANCE`, `DEFAULT_STATIONARY_TOLERANCE`,
  `DEFAULT_STATIONARY_MAX_ITER`.

## Notes

- Conventions: states are `usize` indices into `0..states`, observation symbols
  are `usize` indices into `0..observations`, and row `i` of a transition matrix
  is the outgoing distribution of state `i`.
- Reading a probability out of range returns `0.0` rather than panicking, so a
  partially built model stays inspectable. Operations that can fail on malformed
  input have a checked `try_*` (or `validate`) form returning `MarkovError`; the
  plain form either falls back to a documented default or panics on a caller
  mistake such as an out-of-range index.
- `sample_categorical` skips negative, infinite and `NaN` weights and returns
  `None` when no positive finite mass remains.
- All sampling is generic over `Rng`, so seeding `SplitMix64` makes a whole
  simulation reproducible. `Distribution`, `Rng`, `Sampler`, `SplitMix64`,
  `Standard` and the `tpt_math_prob_core` module itself are re-exported.

## License

Licensed under either of MIT or Apache-2.0 at your option.
