//! Cross-crate example: `tpt-math-graph` shortest paths and topological order.
//!
//! Builds a small weighted DAG, orders it topologically, and finds the shortest
//! distances from the source with Dijkstra's algorithm.

use tpt_math_graph::{dijkstra, toposort, Graph};

fn main() {
    let mut g: Graph<&str, f64> = Graph::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    let d = g.add_node("d");
    g.add_edge(a, b, 4.0);
    g.add_edge(a, c, 1.0);
    g.add_edge(c, b, 2.0);
    g.add_edge(b, d, 1.0);

    let order = toposort(&g, None).expect("the graph is acyclic");
    let names: Vec<&str> = order.iter().map(|&n| g[n]).collect();
    println!("topological order: {names:?}");

    let distances = dijkstra(&g, a, None);
    for (node, dist) in &distances {
        println!(
            "shortest distance from a to {}: {dist}",
            g[tpt_math_graph::NodeIndex::new(*node)]
        );
    }
}
