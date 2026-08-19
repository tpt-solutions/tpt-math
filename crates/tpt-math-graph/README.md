# tpt-math-graph

Graph mathematics for the `tpt-math` substrate.

* **Thin wrap over [`petgraph`](https://crates.io/crates/petgraph)** (dual
  `Apache-2.0 OR MIT`, `no_std` + `alloc`): adjacency structures
  (`Graph`/`StableGraph`/`GraphMap`), directions (`Directed`/`Undirected`),
  `NodeIndex`/`EdgeIndex`, and topological sort (`toposort`).
* **In-house** max-flow (`max_flow`, Edmonds–Karp) built on top of the wrapped
  directed `Graph` with `f64` edge capacities — `petgraph` has no built-in
  max-flow (an upstream feature request has been open since 2021).
* **In-house** shortest-path helpers (`dijkstra`, `astar`) with a `no_std`-safe
  implementation (on `std` they delegate to `petgraph`'s algorithms).

## Usage

```rust
use tpt_math_graph::{Graph, Directed, NodeIndex, max_flow};

let mut g: Graph<&str, f64> = Graph::new();
let s = g.add_node("s");
let t = g.add_node("t");
let a = g.add_node("a");
g.add_edge(s, a, 3.0);
g.add_edge(a, t, 3.0);
let mf = max_flow(&g, s, t).unwrap();
assert_eq!(mf.value, 3.0);
```

## Features

* `std` (default) — enable the `std` support of dependencies.
* `alloc` — signal allocator availability (the algorithms need it).

## License

Dual-licensed under either of `MIT` or `Apache-2.0` at your option.
