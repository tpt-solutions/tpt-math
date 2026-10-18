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

/// A path as its total cost and the node sequence from source to target.
pub type ShortestPath<Ix> = (f64, Vec<NodeIndex<Ix>>);

/// Shortest path from `source` to `target`, as its total cost and the node
/// sequence `source, …, target`.
///
/// Returns `Ok(None)` when `target` is unreachable. Uses the in-house Dijkstra
/// loop on every feature set, so the predecessor bookkeeping is the same under
/// `std` and `no_std`. [`dijkstra`] returns distances only.
///
/// # Errors
///
/// Returns [`GraphError::InvalidWeight`] if any edge weight is negative or NaN.
///
/// # Examples
///
/// ```
/// use tpt_math_graph::{Graph, NodeIndex};
///
/// let mut g: Graph<(), f64> = Graph::new();
/// let a = g.add_node(());
/// let b = g.add_node(());
/// let c = g.add_node(());
/// g.add_edge(a, b, 1.0);
/// g.add_edge(b, c, 1.0);
/// g.add_edge(a, c, 5.0);
/// let (cost, path) = tpt_math_graph::dijkstra_path(&g, a, c).unwrap().unwrap();
/// assert_eq!(cost, 2.0);
/// assert_eq!(path, vec![a, b, c]);
/// ```
pub fn dijkstra_path<N, Ix>(
    graph: &Graph<N, f64, Directed, Ix>,
    source: NodeIndex<Ix>,
    target: NodeIndex<Ix>,
) -> Result<Option<ShortestPath<Ix>>, GraphError>
where
    Ix: IndexType,
{
    check_non_negative(graph)?;
    let goal = target.index();
    let mut dist: BTreeMap<usize, f64> = BTreeMap::new();
    let mut pred: BTreeMap<usize, usize> = BTreeMap::new();
    let mut settled: BTreeMap<usize, bool> = BTreeMap::new();
    dist.insert(source.index(), 0.0);
    let mut heap = BinaryHeap::new();
    heap.push(OpenNode {
        f: 0.0,
        node: source.index(),
    });
    while let Some(OpenNode { f: d, node: u }) = heap.pop() {
        if settled.contains_key(&u) {
            continue;
        }
        settled.insert(u, true);
        if u == goal {
            break;
        }
        for e in graph.edges_directed(NodeIndex::<Ix>::new(u), Direction::Outgoing) {
            let v = e.target().index();
            let nd = d + *e.weight();
            if nd < dist.get(&v).copied().unwrap_or(f64::INFINITY) {
                dist.insert(v, nd);
                pred.insert(v, u);
                heap.push(OpenNode { f: nd, node: v });
            }
        }
    }
    if !settled.contains_key(&goal) {
        return Ok(None);
    }
    let cost = dist[&goal];
    let mut path = vec![NodeIndex::<Ix>::new(goal)];
    let mut cur = goal;
    while cur != source.index() {
        cur = pred[&cur];
        path.push(NodeIndex::<Ix>::new(cur));
    }
    path.reverse();
    Ok(Some((cost, path)))
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
// Traversal, spanning trees and components (thin wraps over petgraph)
// ===========================================================================

/// Nodes reachable from `start` in breadth-first order, `start` first.
///
/// Edges are followed in their stored direction for directed graphs.
pub fn bfs_order<N, E, Ty, Ix>(
    graph: &petgraph::Graph<N, E, Ty, Ix>,
    start: NodeIndex<Ix>,
) -> Vec<NodeIndex<Ix>>
where
    Ty: petgraph::EdgeType,
    Ix: IndexType,
{
    let mut order = Vec::new();
    let mut bfs = petgraph::visit::Bfs::new(graph, start);
    while let Some(n) = bfs.next(graph) {
        order.push(n);
    }
    order
}

/// Nodes reachable from `start` in depth-first (preorder) order, `start` first.
///
/// Edges are followed in their stored direction for directed graphs.
pub fn dfs_order<N, E, Ty, Ix>(
    graph: &petgraph::Graph<N, E, Ty, Ix>,
    start: NodeIndex<Ix>,
) -> Vec<NodeIndex<Ix>>
where
    Ty: petgraph::EdgeType,
    Ix: IndexType,
{
    let mut order = Vec::new();
    let mut dfs = petgraph::visit::Dfs::new(graph, start);
    while let Some(n) = dfs.next(graph) {
        order.push(n);
    }
    order
}

/// Whether a directed path from `from` to `to` exists (`from == to` is `true`).
pub fn has_path<N, E, Ix>(
    graph: &Graph<N, E, Directed, Ix>,
    from: NodeIndex<Ix>,
    to: NodeIndex<Ix>,
) -> bool
where
    Ix: IndexType,
{
    petgraph::algo::has_path_connecting(graph, from, to, None)
}

/// Whether the directed graph contains a cycle.
pub fn is_cyclic<N, E, Ix>(graph: &Graph<N, E, Directed, Ix>) -> bool
where
    Ix: IndexType,
{
    petgraph::algo::is_cyclic_directed(graph)
}

/// The number of connected components of an undirected graph.
///
/// Edge directions are ignored. For a directed graph this counts the weakly
/// connected components.
pub fn connected_components<N, E, Ty, Ix>(graph: &petgraph::Graph<N, E, Ty, Ix>) -> usize
where
    Ty: petgraph::EdgeType,
    Ix: IndexType,
{
    petgraph::algo::connected_components(graph)
}

/// The strongly connected components of a directed graph, in reverse
/// topological order of the condensation (Tarjan's algorithm).
pub fn strongly_connected_components<N, E, Ix>(
    graph: &Graph<N, E, Directed, Ix>,
) -> Vec<Vec<NodeIndex<Ix>>>
where
    Ix: IndexType,
{
    petgraph::algo::tarjan_scc(graph)
}

/// A minimum spanning forest, as `(source, target, weight)` triples.
///
/// Edge directions are ignored. Each component of the input contributes a
/// spanning tree, so a disconnected graph yields a forest. Ties are broken by
/// petgraph's ordering, so the result is deterministic for a given graph.
///
/// The total weight can be summed by the caller; the edges are returned in the
/// order Kruskal's algorithm accepts them.
pub fn minimum_spanning_tree<N, E, Ty, Ix>(
    graph: &petgraph::Graph<N, E, Ty, Ix>,
) -> Vec<(NodeIndex<Ix>, NodeIndex<Ix>, E)>
where
    N: Clone,
    E: Clone + PartialOrd,
    Ty: petgraph::EdgeType,
    Ix: IndexType,
{
    use petgraph::data::Element;
    petgraph::algo::min_spanning_tree(graph)
        .filter_map(|element| match element {
            Element::Edge {
                source,
                target,
                weight,
            } => Some((NodeIndex::new(source), NodeIndex::new(target), weight)),
            Element::Node { .. } => None,
        })
        .collect()
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
    fn dijkstra_path_reconstructs_the_shortest_route() {
        let (g, a, _b, c) = small_dag();
        // Direct a->c and a->b->c both cost 1.0 in small_dag, so a tie is allowed.
        let (cost, path) = dijkstra_path(&g, a, c).unwrap().unwrap();
        assert!((cost - 1.0).abs() < 1e-12);
        assert_eq!(path.first(), Some(&a));
        assert_eq!(path.last(), Some(&c));
        // Path edges must exist and sum to the cost.
        let mut total = 0.0;
        for w in path.windows(2) {
            let e = g.find_edge(w[0], w[1]).expect("path uses real edges");
            total += g[e];
        }
        assert!((total - cost).abs() < 1e-12);
        assert_eq!(dijkstra_path(&g, a, a).unwrap(), Some((0.0, vec![a])));
        assert_eq!(dijkstra_path(&g, c, a).unwrap(), None);
    }

    #[test]
    fn dijkstra_path_prefers_the_longer_cheaper_route() {
        let mut g: Graph<(), f64> = Graph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(a, c, 5.0);
        let (cost, path) = dijkstra_path(&g, a, c).unwrap().unwrap();
        assert_eq!(cost, 2.0);
        assert_eq!(path, vec![a, b, c]);
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

#[cfg(test)]
mod traversal_tests {
    use super::*;

    fn chain() -> (Graph<u8, f64>, Vec<NodeIndex>) {
        // 0 -> 1 -> 2, 0 -> 3, 4 isolated.
        let mut g: Graph<u8, f64> = Graph::new();
        let n: Vec<NodeIndex> = (0..5).map(|i| g.add_node(i)).collect();
        g.add_edge(n[0], n[1], 1.0);
        g.add_edge(n[1], n[2], 1.0);
        g.add_edge(n[0], n[3], 1.0);
        (g, n)
    }

    #[test]
    fn bfs_visits_by_level() {
        let (g, n) = chain();
        let order = bfs_order(&g, n[0]);
        assert_eq!(order[0], n[0]);
        assert_eq!(order.len(), 4);
        // Both children of 0 come before the grandchild 2.
        let pos = |x: NodeIndex| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(n[1]) < pos(n[2]));
        assert!(pos(n[3]) < pos(n[2]));
        assert!(!order.contains(&n[4]));
    }

    #[test]
    fn dfs_goes_deep_first() {
        let (g, n) = chain();
        let order = dfs_order(&g, n[0]);
        // Preorder: 0, then the first child's subtree before its sibling.
        assert_eq!(order.len(), 4);
        let pos = |x: NodeIndex| order.iter().position(|&y| y == x).unwrap();
        assert!(pos(n[2]) < pos(n[3]) || pos(n[1]) < pos(n[3]));
        assert_eq!(order[0], n[0]);
    }

    #[test]
    fn reachability_and_cycles() {
        let (mut g, n) = chain();
        assert!(has_path(&g, n[0], n[2]));
        assert!(!has_path(&g, n[2], n[0]));
        assert!(has_path(&g, n[4], n[4]));
        assert!(!is_cyclic(&g));
        g.add_edge(n[2], n[0], 1.0);
        assert!(is_cyclic(&g));
    }

    #[test]
    fn components_ignore_direction() {
        let (g, _) = chain();
        // {0,1,2,3} and {4}.
        assert_eq!(connected_components(&g), 2);
    }

    #[test]
    fn strongly_connected_components_split_a_cycle() {
        let mut g: Graph<(), ()> = Graph::new();
        let a = g.add_node(());
        let b = g.add_node(());
        let c = g.add_node(());
        g.add_edge(a, b, ());
        g.add_edge(b, a, ());
        g.add_edge(b, c, ());
        let sccs = strongly_connected_components(&g);
        assert_eq!(sccs.len(), 2);
        let big = sccs.iter().find(|s| s.len() == 2).unwrap();
        assert!(big.contains(&a) && big.contains(&b));
    }

    #[test]
    fn spanning_tree_of_a_weighted_square() {
        // Square 0-1-2-3-0 with a heavy diagonal 0-2 that must be skipped.
        let mut g: Graph<(), f64> = Graph::new();
        let n: Vec<NodeIndex> = (0..4).map(|_| g.add_node(())).collect();
        g.add_edge(n[0], n[1], 1.0);
        g.add_edge(n[1], n[2], 2.0);
        g.add_edge(n[2], n[3], 3.0);
        g.add_edge(n[3], n[0], 4.0);
        g.add_edge(n[0], n[2], 10.0);
        let mst = minimum_spanning_tree(&g);
        assert_eq!(mst.len(), 3);
        let total: f64 = mst.iter().map(|&(_, _, w)| w).sum();
        assert_eq!(total, 6.0);
    }

    #[test]
    fn spanning_forest_for_disconnected_graph() {
        let (g, _) = {
            let mut g: Graph<(), f64> = Graph::new();
            let n: Vec<NodeIndex> = (0..4).map(|_| g.add_node(())).collect();
            g.add_edge(n[0], n[1], 1.0);
            g.add_edge(n[2], n[3], 2.0);
            (g, n)
        };
        // Two components, so two edges: a forest with n - c = 2 edges.
        assert_eq!(minimum_spanning_tree(&g).len(), 2);
    }
}
