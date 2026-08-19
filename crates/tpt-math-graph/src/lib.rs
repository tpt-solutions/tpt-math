#![no_std]
#![forbid(unsafe_code)]
#![allow(clippy::needless_range_loop)]
//! Graph mathematics for the `tpt-math` substrate.
//!
//! `tpt-math-graph` wraps [`petgraph`] (dual `Apache-2.0 OR MIT`, `no_std` +
//! `alloc`) for the adjacency structures, topological sort and shortest-path
//! algorithms, and builds the **max-flow** solver in-house on top (petgraph
//! has no built-in max-flow — an upstream feature request has been open since
//! 2021).
//!
//! # Wrap-vs-in-house split
//!
//! * **Thin wrap** — [`Graph`], [`StableGraph`], [`GraphMap`], [`NodeIndex`],
//!   [`EdgeIndex`], [`Directed`]/[`Undirected`], and [`toposort`] are re-exported
//!   directly from `petgraph`.
//! * **In-house** — [`max_flow`] (Edmonds–Karp) is implemented here, operating
//!   on `petgraph`'s directed [`Graph`] with `f64` edge capacities.
//!
//! [`petgraph`]: petgraph
//!
//! # Features
//!
//! * `std` (default) — enable the `std` support of dependencies.
//! * `alloc` — signal allocator availability (the algorithms need it).
//!
//! # Examples
//!
//! ```
//! use tpt_math_graph::{Graph, Directed, NodeIndex};
//!
//! let mut g: Graph<&str, f64> = Graph::new();
//! let a = g.add_node("a");
//! let b = g.add_node("b");
//! g.add_edge(a, b, 1.0);
//! assert_eq!(g.node_count(), 2);
//! ```

extern crate alloc;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

use petgraph::graph::IndexType;
use petgraph::visit::EdgeRef;
use petgraph::Direction;

pub use petgraph;

// Re-export the adjacency types so downstream code can build graphs through
// `tpt-math-graph` without naming `petgraph` directly.
pub use petgraph::algo::toposort;
pub use petgraph::graph::{EdgeIndex, Graph, NodeIndex};
pub use petgraph::graphmap::GraphMap;
pub use petgraph::stable_graph::StableGraph;
pub use petgraph::{Directed, Undirected};

/// Errors returned by the graph algorithms in this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The requested operation cannot run because the source equals the sink.
    SourceEqualsSink,
    /// A topological sort was requested on a graph that contains a cycle.
    Cycle,
}

// ===========================================================================
// In-house max-flow (Edmonds–Karp)
// ===========================================================================

/// The result of a [`max_flow`] computation.
#[derive(Debug, Clone, PartialEq)]
pub struct MaxFlow {
    /// Total flow from source to sink.
    pub value: f64,
    /// Flow on each original directed edge, indexed by `EdgeIndex::index()`.
    pub edge_flow: Vec<f64>,
}

#[derive(Clone)]
struct ResEdge {
    to: usize,
    cap: f64,
    rev: usize,
}

/// Compute the maximum flow from `source` to `sink` in a directed
/// [`Graph`] whose edge weights are capacities (`f64`).
///
/// Uses the Edmonds–Karp algorithm (BFS-augmented Ford–Fulkerson) over an
/// in-house residual graph, so it needs no `std` collections.
pub fn max_flow<N, Ix>(
    graph: &Graph<N, f64, Directed, Ix>,
    source: NodeIndex<Ix>,
    sink: NodeIndex<Ix>,
) -> Result<MaxFlow, GraphError>
where
    Ix: IndexType,
{
    if source == sink {
        return Err(GraphError::SourceEqualsSink);
    }
    let n = graph.node_count();
    let m = graph.edge_count();
    let mut res: Vec<Vec<ResEdge>> = (0..n).map(|_| Vec::new()).collect();
    let mut caps: Vec<f64> = Vec::with_capacity(m);
    // For each original edge, record the (node, residual-entry) of its forward
    // residual edge so we can recover the realised flow at the end.
    let mut fwd_of: Vec<(usize, usize)> = Vec::with_capacity(m);

    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        let ui = u.index();
        let vi = v.index();
        let c = graph[e];
        caps.push(c);
        let fwd = res[ui].len();
        let bwd = res[vi].len();
        res[ui].push(ResEdge {
            to: vi,
            cap: c,
            rev: bwd,
        });
        res[vi].push(ResEdge {
            to: ui,
            cap: 0.0,
            rev: fwd,
        });
        fwd_of.push((ui, fwd));
    }

    let s = source.index();
    let t = sink.index();
    let eps = 1e-12;
    let mut total = 0.0_f64;

    loop {
        let mut parent: Vec<usize> = (0..n).map(|_| usize::MAX).collect();
        let mut parent_edge: Vec<usize> = (0..n).map(|_| 0).collect();
        let mut cap_to: Vec<f64> = (0..n).map(|_| 0.0).collect();
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(s);
        parent[s] = s;
        cap_to[s] = f64::INFINITY;

        while let Some(u) = queue.pop_front() {
            let mut ki = 0;
            while ki < res[u].len() {
                let re = &res[u][ki];
                let to = re.to;
                let c = re.cap;
                if c > eps && parent[to] == usize::MAX {
                    parent[to] = u;
                    parent_edge[to] = ki;
                    cap_to[to] = cap_to[u].min(c);
                    if to == t {
                        queue.clear();
                        break;
                    }
                    queue.push_back(to);
                }
                ki += 1;
            }
        }

        if parent[t] == usize::MAX {
            break;
        }
        let flow = cap_to[t];
        let mut v = t;
        while v != s {
            let u = parent[v];
            let k = parent_edge[v];
            res[u][k].cap -= flow;
            let rev = res[u][k].rev;
            res[v][rev].cap += flow;
            v = u;
        }
        total += flow;
    }

    let mut edge_flow = vec![0.0_f64; m];
    for (ei, &(ui, fwd)) in fwd_of.iter().enumerate() {
        edge_flow[ei] = caps[ei] - res[ui][fwd].cap;
    }

    Ok(MaxFlow {
        value: total,
        edge_flow,
    })
}

// ===========================================================================
// Convenience shortest-path wrappers (in-house, on petgraph Graph)
// ===========================================================================

/// Shortest-path distance from `source` to `target` via Dijkstra, using edge
/// weights as non-negative costs. Returns `None` if `target` is unreachable.
pub fn dijkstra<N, Ix>(
    graph: &Graph<N, f64, Directed, Ix>,
    source: NodeIndex<Ix>,
    target: Option<NodeIndex<Ix>>,
) -> BTreeMap<usize, f64>
where
    Ix: IndexType,
{
    // Reuse petgraph's Dijkstra when on `std`; in `no_std` we run our own
    // binary-heap-free Dijkstra (small graphs, simple priority by repeated
    // scan) so the algorithm stays allocator-only.
    #[cfg(feature = "std")]
    {
        let _ = target;
        use petgraph::algo::dijkstra as pg_dijkstra;
        let dist = pg_dijkstra(graph, source, target, |e| *e.weight());
        let mut out = BTreeMap::new();
        for (k, v) in dist {
            out.insert(k.index(), v);
        }
        out
    }
    #[cfg(not(feature = "std"))]
    {
        let n = graph.node_count();
        let mut dist: BTreeMap<usize, f64> = BTreeMap::new();
        dist.insert(source.index(), 0.0);
        // Simple Dijkstra with a linear-scan "priority queue".
        let mut visited: BTreeMap<usize, bool> = BTreeMap::new();
        for _ in 0..n {
            // Pick the unvisited node with the smallest distance.
            let mut u = usize::MAX;
            let mut best = f64::INFINITY;
            for (node, d) in dist.iter() {
                if !visited.get(node).copied().unwrap_or(false) && *d < best {
                    best = *d;
                    u = *node;
                }
            }
            if u == usize::MAX {
                break;
            }
            visited.insert(u, true);
            if let Some(tg) = target {
                if u == tg.index() {
                    break;
                }
            }
            for e in graph.edges_directed(NodeIndex::<Ix>::new(u), Direction::Outgoing) {
                let v = e.target().index();
                let w = *e.weight();
                let nd = best + w;
                let cur = dist.get(&v).copied().unwrap_or(f64::INFINITY);
                if nd < cur {
                    dist.insert(v, nd);
                }
            }
        }
        // Keep only reachable entries.
        let mut out = BTreeMap::new();
        for (k, v) in dist.iter() {
            if v.is_finite() {
                out.insert(*k, *v);
            }
        }
        out
    }
}

/// A* shortest path from `start` to the first node satisfying `is_goal`,
/// using edge weights as costs and `heuristic` as the admissible estimate.
/// Returns `(cost, path)` or `None` if no path exists.
pub fn astar<N, Ix, F, H>(
    graph: &Graph<N, f64, Directed, Ix>,
    start: NodeIndex<Ix>,
    is_goal: F,
    heuristic: H,
) -> Option<(f64, Vec<NodeIndex<Ix>>)>
where
    Ix: IndexType,
    F: Fn(NodeIndex<Ix>) -> bool,
    H: Fn(NodeIndex<Ix>) -> f64,
{
    let s = start.index();
    let mut g_score: BTreeMap<usize, f64> = BTreeMap::new();
    g_score.insert(s, 0.0);
    let mut came_from: BTreeMap<usize, usize> = BTreeMap::new();
    let mut open: BTreeMap<usize, f64> = BTreeMap::new();
    open.insert(s, heuristic(start));

    while !open.is_empty() {
        // Pop the open node with lowest f = g + h.
        let mut u = usize::MAX;
        let mut best_f = f64::INFINITY;
        for (node, f) in open.iter() {
            if *f < best_f {
                best_f = *f;
                u = *node;
            }
        }
        open.remove(&u);

        let gu = *g_score.get(&u).unwrap_or(&f64::INFINITY);
        let node_u = NodeIndex::<Ix>::new(u);
        if is_goal(node_u) {
            // Reconstruct path.
            let mut path = Vec::new();
            let mut cur = u;
            path.push(NodeIndex::<Ix>::new(cur));
            while let Some(&p) = came_from.get(&cur) {
                path.push(NodeIndex::<Ix>::new(p));
                cur = p;
            }
            path.reverse();
            return Some((gu, path));
        }

        for e in graph.edges_directed(node_u, Direction::Outgoing) {
            let v = e.target().index();
            let w = *e.weight();
            let tentative = gu + w;
            let vg = g_score.get(&v).copied().unwrap_or(f64::INFINITY);
            if tentative < vg {
                came_from.insert(v, u);
                g_score.insert(v, tentative);
                let f = tentative + heuristic(NodeIndex::new(v));
                open.insert(v, f);
            }
        }
    }
    None
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn small_dag() -> (Graph<&'static str, f64>, NodeIndex, NodeIndex, NodeIndex) {
        let mut g: Graph<&str, f64> = Graph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        let c = g.add_node("c");
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(a, c, 1.0);
        (g, a, b, c)
    }

    #[test]
    fn toposort_orders_dag() {
        let (g, a, b, c) = {
            let (g, a, b, c) = small_dag();
            // rename to avoid unused
            (g, a, b, c)
        };
        let order = toposort(&g, None).expect("DAG has a topological order");
        let pos = |n: NodeIndex| order.iter().position(|&x| x == n).unwrap();
        assert!(pos(a) < pos(b));
        assert!(pos(b) < pos(c));
        assert!(pos(a) < pos(c));
    }

    #[test]
    fn dijkstra_finds_shortest_distance() {
        let (g, a, _b, c) = small_dag();
        let dist = dijkstra(&g, a, Some(c));
        // Direct a->c costs 1.0; a->b->c also costs 2.0; shortest is 1.0.
        assert!((dist[&c.index()] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn astar_finds_path() {
        let (g, a, _b, c) = small_dag();
        let res = astar(&g, a, |n| n == c, |_| 0.0);
        assert!(res.is_some());
        let (cost, path) = res.unwrap();
        assert!((cost - 1.0).abs() < 1e-12);
        assert_eq!(path[0], a);
        assert_eq!(path[path.len() - 1], c);
    }

    #[test]
    fn max_flow_classic_network() {
        // s -> a (3), s -> b (2), a -> b (1), a -> t (2), b -> t (3).
        // Min cut = 5 (source out-capacity = sink in-capacity), so max flow = 5.
        let mut g: Graph<&str, f64> = Graph::new();
        let s = g.add_node("s");
        let a = g.add_node("a");
        let b = g.add_node("b");
        let t = g.add_node("t");
        g.add_edge(s, a, 3.0);
        g.add_edge(s, b, 2.0);
        g.add_edge(a, b, 1.0);
        g.add_edge(a, t, 2.0);
        g.add_edge(b, t, 3.0);
        let mf = max_flow(&g, s, t).unwrap();
        assert!((mf.value - 5.0).abs() < 1e-9, "got {}", mf.value);
    }

    #[test]
    fn max_flow_zero_when_no_path() {
        let mut g: Graph<&str, f64> = Graph::new();
        let s = g.add_node("s");
        let t = g.add_node("t");
        g.add_node("mid");
        let mf = max_flow(&g, s, t).unwrap();
        assert_eq!(mf.value, 0.0);
    }

    #[test]
    fn max_flow_bottleneck() {
        // s -> a (3), s -> b (2); a -> t (3), b -> t (1); no a<->b edge.
        // Sink in-capacity = 3 + 1 = 4, so max flow is capped at 4.
        let mut g: Graph<&str, f64> = Graph::new();
        let s = g.add_node("s");
        let a = g.add_node("a");
        let b = g.add_node("b");
        let t = g.add_node("t");
        g.add_edge(s, a, 3.0);
        g.add_edge(s, b, 2.0);
        g.add_edge(a, t, 3.0);
        g.add_edge(b, t, 1.0);
        let mf = max_flow(&g, s, t).unwrap();
        assert!((mf.value - 4.0).abs() < 1e-9, "got {}", mf.value);
        // b->t caps at 1, so s->b carries exactly 1; the rest (3) goes s->a -> a->t.
        assert!(
            (mf.edge_flow[0] - 3.0).abs() < 1e-9,
            "s->a flow = {}",
            mf.edge_flow[0]
        );
        assert!(
            (mf.edge_flow[1] - 1.0).abs() < 1e-9,
            "s->b flow = {}",
            mf.edge_flow[1]
        );
        assert!(
            (mf.edge_flow[2] - 3.0).abs() < 1e-9,
            "a->t flow = {}",
            mf.edge_flow[2]
        );
        assert!(
            (mf.edge_flow[3] - 1.0).abs() < 1e-9,
            "b->t flow = {}",
            mf.edge_flow[3]
        );
    }

    #[test]
    fn max_flow_rejects_equal_endpoints() {
        let mut g: Graph<&str, f64> = Graph::new();
        let s = g.add_node("s");
        assert_eq!(max_flow(&g, s, s), Err(GraphError::SourceEqualsSink));
    }
}
