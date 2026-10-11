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

use alloc::collections::{BTreeMap, BinaryHeap, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

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
    /// An edge weight (or, for [`max_flow`], a capacity) is negative, NaN or,
    /// for capacities, infinite. Dijkstra and A* require non-negative weights.
    InvalidWeight,
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
    if graph.edge_weights().any(|c| !c.is_finite() || *c < 0.0) {
        return Err(GraphError::InvalidWeight);
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
    // Residual capacities are considered present while they exceed a tiny
    // tolerance *relative to the largest capacity*, so small-but-real
    // capacities are kept while rounding residue is still ignored.
    let max_cap = caps.iter().fold(0.0_f64, |acc, &c| acc.max(c));
    let eps = max_cap * f64::EPSILON;
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

/// Open-set entry for Dijkstra and A*, ordered so that [`BinaryHeap`] pops the
/// smallest `f` first. Only non-NaN keys are ever inserted.
#[derive(Clone, Copy, PartialEq)]
struct OpenNode {
    f: f64,
    node: usize,
}

impl Eq for OpenNode {}

impl PartialOrd for OpenNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OpenNode {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .f
            .total_cmp(&self.f)
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// Reject negative and NaN edge weights, which Dijkstra and A* cannot handle.
fn check_non_negative<N, Ix>(graph: &Graph<N, f64, Directed, Ix>) -> Result<(), GraphError>
where
    Ix: IndexType,
{
    if graph.edge_weights().any(|w| w.is_nan() || *w < 0.0) {
        Err(GraphError::InvalidWeight)
    } else {
        Ok(())
    }
}

/// Shortest-path distance from `source` to `target` via Dijkstra, using edge
/// weights as non-negative costs. Returns `None` if `target` is unreachable.
///
/// # Panics
///
/// Panics if any edge weight is negative or NaN. Use [`try_dijkstra`] to get
/// an error instead.
pub fn dijkstra<N, Ix>(
    graph: &Graph<N, f64, Directed, Ix>,
    source: NodeIndex<Ix>,
    target: Option<NodeIndex<Ix>>,
) -> BTreeMap<usize, f64>
where
    Ix: IndexType,
{
    try_dijkstra(graph, source, target).unwrap_or_else(|e| panic!("dijkstra failed: {e:?}"))
}

/// Fallible [`dijkstra`].
///
/// # Errors
///
/// Returns [`GraphError::InvalidWeight`] if any edge weight is negative or NaN.
pub fn try_dijkstra<N, Ix>(
    graph: &Graph<N, f64, Directed, Ix>,
    source: NodeIndex<Ix>,
    target: Option<NodeIndex<Ix>>,
) -> Result<BTreeMap<usize, f64>, GraphError>
where
    Ix: IndexType,
{
    check_non_negative(graph)?;
    // Reuse petgraph's Dijkstra when on `std`; in `no_std` run the same
    // algorithm over our own binary heap so it stays allocator-only.
    #[cfg(feature = "std")]
    {
        let _ = target;
        use petgraph::algo::dijkstra as pg_dijkstra;
        let dist = pg_dijkstra(graph, source, target, |e| *e.weight());
        let mut out = BTreeMap::new();
        for (k, v) in dist {
            out.insert(k.index(), v);
        }
        Ok(out)
    }
    #[cfg(not(feature = "std"))]
    {
        let mut dist: BTreeMap<usize, f64> = BTreeMap::new();
        dist.insert(source.index(), 0.0);
        let mut visited: BTreeMap<usize, bool> = BTreeMap::new();
        let mut heap = BinaryHeap::new();
        heap.push(OpenNode {
            f: 0.0,
            node: source.index(),
        });
        while let Some(OpenNode { f: d, node: u }) = heap.pop() {
            if visited.contains_key(&u) {
                continue; // settled earlier with a distance no larger than `d`
            }
            visited.insert(u, true);
            if target.is_some_and(|tg| tg.index() == u) {
                break;
            }
            for e in graph.edges_directed(NodeIndex::<Ix>::new(u), Direction::Outgoing) {
                let v = e.target().index();
                let nd = d + *e.weight();
                if nd < dist.get(&v).copied().unwrap_or(f64::INFINITY) {
                    dist.insert(v, nd);
                    heap.push(OpenNode { f: nd, node: v });
                }
            }
        }
        Ok(dist)
    }
}

/// A* shortest path from `start` to the first node satisfying `is_goal`,
/// using edge weights as costs and `heuristic` as the admissible estimate.
/// Returns `(cost, path)` or `None` if no path exists.
///
/// # Panics
///
/// Panics if any edge weight is negative or NaN. Use [`try_astar`] to get an
/// error instead.
pub fn astar<N, Ix, F, H>(
    graph: &Graph<N, f64, Directed, Ix>,
    start: NodeIndex<Ix>,
    is_goal: F,
    heuristic: H,
) -> Option<AStarPath<Ix>>
where
    Ix: IndexType,
    F: Fn(NodeIndex<Ix>) -> bool,
    H: Fn(NodeIndex<Ix>) -> f64,
{
    try_astar(graph, start, is_goal, heuristic).unwrap_or_else(|e| panic!("astar failed: {e:?}"))
}

/// The cost and node sequence of a shortest path found by A*.
pub type AStarPath<Ix> = (f64, Vec<NodeIndex<Ix>>);

/// Fallible [`astar`].
///
/// # Errors
///
/// Returns [`GraphError::InvalidWeight`] if any edge weight is negative or NaN.
pub fn try_astar<N, Ix, F, H>(
    graph: &Graph<N, f64, Directed, Ix>,
    start: NodeIndex<Ix>,
    is_goal: F,
    heuristic: H,
) -> Result<Option<AStarPath<Ix>>, GraphError>
where
    Ix: IndexType,
    F: Fn(NodeIndex<Ix>) -> bool,
    H: Fn(NodeIndex<Ix>) -> f64,
{
    check_non_negative(graph)?;
    let s = start.index();
    let mut g_score: BTreeMap<usize, f64> = BTreeMap::new();
    g_score.insert(s, 0.0);
    let mut came_from: BTreeMap<usize, usize> = BTreeMap::new();
    let mut open: BinaryHeap<OpenNode> = BinaryHeap::new();
    let h0 = heuristic(start);
    if !h0.is_nan() {
        open.push(OpenNode { f: h0, node: s });
    }

    // Each push follows a strict improvement of some g-score, so the search
    // terminates on a finite graph whatever the heuristic returns.
    while let Some(OpenNode { f, node: u }) = open.pop() {
        let gu = *g_score.get(&u).unwrap_or(&f64::INFINITY);
        let node_u = NodeIndex::<Ix>::new(u);
        // Skip stale entries: a cheaper route to `u` was found after this push.
        if f > gu + heuristic(node_u) {
            continue;
        }
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
            return Ok(Some((gu, path)));
        }

        for e in graph.edges_directed(node_u, Direction::Outgoing) {
            let v = e.target().index();
            let tentative = gu + *e.weight();
            let vg = g_score.get(&v).copied().unwrap_or(f64::INFINITY);
            if tentative < vg {
                came_from.insert(v, u);
                g_score.insert(v, tentative);
                let f = tentative + heuristic(NodeIndex::new(v));
                if !f.is_nan() {
                    open.push(OpenNode { f, node: v });
                }
            }
        }
    }
    Ok(None)
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

    #[test]
    fn max_flow_keeps_small_capacities() {
        // A fixed 1e-12 cutoff used to discard this edge entirely.
        let mut g: Graph<&str, f64> = Graph::new();
        let s = g.add_node("s");
        let t = g.add_node("t");
        g.add_edge(s, t, 1e-14);
        let mf = max_flow(&g, s, t).unwrap();
        assert!((mf.value - 1e-14).abs() < 1e-20, "got {}", mf.value);
    }

    #[test]
    fn max_flow_rejects_invalid_capacity() {
        let mut g: Graph<&str, f64> = Graph::new();
        let s = g.add_node("s");
        let t = g.add_node("t");
        g.add_edge(s, t, -1.0);
        assert_eq!(max_flow(&g, s, t), Err(GraphError::InvalidWeight));
    }

    #[test]
    fn astar_terminates_when_heuristic_is_infinite() {
        // Every open node has f = inf. The old scan never removed anything and
        // looped forever; the search must now finish. The goal is reachable,
        // so it is still found with the true cost.
        let (g, a, _b, c) = small_dag();
        let (cost, _) = astar(&g, a, |n| n == c, |_| f64::INFINITY).unwrap();
        assert!((cost - 1.0).abs() < 1e-12);

        // The original hang: the goal is unreachable and every f is infinite.
        let mut g2: Graph<&str, f64> = Graph::new();
        let s = g2.add_node("s");
        let isolated = g2.add_node("isolated");
        g2.add_edge(s, s, 1.0);
        assert!(astar(&g2, s, |n| n == isolated, |_| f64::INFINITY).is_none());
    }

    #[test]
    fn astar_terminates_when_heuristic_is_nan() {
        let (g, a, _b, c) = small_dag();
        assert!(astar(&g, a, |n| n == c, |_| f64::NAN).is_none());
    }

    #[test]
    fn astar_reopens_nodes_reached_by_a_cheaper_route() {
        // a -> c costs 10 directly, but a -> b -> c costs 2. Whichever route
        // reaches c first, the reported cost must be the cheaper one.
        let mut g: Graph<&str, f64> = Graph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        let c = g.add_node("c");
        g.add_edge(a, c, 10.0);
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        let (cost, path) = astar(&g, a, |n| n == c, |_| 0.0).unwrap();
        assert!((cost - 2.0).abs() < 1e-12);
        assert_eq!(path, vec![a, b, c]);
    }

    #[test]
    fn try_dijkstra_rejects_negative_weight() {
        let mut g: Graph<&str, f64> = Graph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        g.add_edge(a, b, -1.0);
        assert_eq!(try_dijkstra(&g, a, None), Err(GraphError::InvalidWeight));
    }

    #[test]
    fn try_astar_rejects_nan_weight() {
        let mut g: Graph<&str, f64> = Graph::new();
        let a = g.add_node("a");
        let b = g.add_node("b");
        g.add_edge(a, b, f64::NAN);
        assert_eq!(
            try_astar(&g, a, |n| n == b, |_| 0.0),
            Err(GraphError::InvalidWeight)
        );
    }
}
