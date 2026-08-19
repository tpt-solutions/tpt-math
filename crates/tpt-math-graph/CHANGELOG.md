# Changelog

All notable changes to this crate will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to semantic versioning.

## [0.1.0] - Unreleased

### Added

- Initial release: thin wrap over `petgraph` (v0.8, dual `Apache-2.0 OR MIT`,
  `no_std` + `alloc`) for `Graph`/`StableGraph`/`GraphMap`, `NodeIndex`/
  `EdgeIndex`, `Directed`/`Undirected`, and `toposort`; in-house `max_flow`
  (Edmonds–Karp) and `dijkstra`/`astar` shortest-path helpers.
