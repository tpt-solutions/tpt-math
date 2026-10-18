# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to semantic versioning.

## [Unreleased]

### Added

- `try_dijkstra` and `try_astar`, returning `Result`; `GraphError::InvalidWeight`; the `AStarPath` alias.

### Changed

- **Breaking:** `GraphError` gained a variant.
- `astar` uses a binary heap instead of an O(V²) scan; the `no_std` Dijkstra branch uses the same heap.
- `dijkstra` and `astar` panic on negative or NaN weights (use the `try_*` forms). `max_flow` rejects negative, NaN or infinite capacities.

### Fixed

- `astar` no longer loops forever when every open `f` is infinite or NaN.
- `max_flow` keeps small capacities; the tolerance is now relative to the largest capacity instead of an absolute `1e-12`.

### Added

- `dijkstra_path`: shortest path as the node route as well as its cost.
- `bfs_order` and `dfs_order`: breadth-first and depth-first (preorder) traversal from a start node.
- `has_path`, `is_cyclic`, `connected_components` (weakly connected for directed graphs) and `strongly_connected_components` (Tarjan), thin wraps over `petgraph`.
- `minimum_spanning_tree`: Kruskal spanning forest as `(source, target, weight)` triples. Requires `N: Clone`, because the wrapped `petgraph` iterator does.

## [0.1.0] - 2026-08-19

### Added

- Initial release: thin wrap over `petgraph` (v0.8, dual `Apache-2.0 OR MIT`,
  `no_std` + `alloc`) for `Graph`/`StableGraph`/`GraphMap`, `NodeIndex`/
  `EdgeIndex`, `Directed`/`Undirected`, and `toposort`; in-house `max_flow`
  (Edmonds–Karp) and `dijkstra`/`astar` shortest-path helpers.
