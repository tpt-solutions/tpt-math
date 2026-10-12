use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_graph::{dijkstra, Graph};

fn bench_graph(c: &mut Criterion) {
    let n = 40;
    let mut g: Graph<(), f64> = Graph::new();
    let nodes: Vec<_> = (0..n * n).map(|_| g.add_node(())).collect();
    for r in 0..n {
        for col in 0..n {
            let i = r * n + col;
            if col + 1 < n {
                g.add_edge(nodes[i], nodes[i + 1], 1.0);
                g.add_edge(nodes[i + 1], nodes[i], 1.0);
            }
            if r + 1 < n {
                g.add_edge(nodes[i], nodes[i + n], 1.0);
                g.add_edge(nodes[i + n], nodes[i], 1.0);
            }
        }
    }
    c.bench_function("dijkstra 40x40 grid", |b| {
        b.iter(|| black_box(dijkstra(&g, nodes[0], None)))
    });
}

criterion_group!(benches, bench_graph);
criterion_main!(benches);
