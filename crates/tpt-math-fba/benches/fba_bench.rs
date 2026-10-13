use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_fba::{fba, Model, SimplexSolver};

/// Layered synthetic network: uptake feeds `layers` stages of `width`
/// metabolites, each connected to the next by sparse cross-links, and a
/// biomass sink at the end.
fn layered_model(layers: usize, width: usize) -> Model {
    let mut m = Model::new();
    let name = |l: usize, w: usize| format!("M{l}_{w}");
    for l in 0..layers {
        for w in 0..width {
            m.add_metabolite(&name(l, w)).unwrap();
        }
    }
    m.add_reaction("EX_src", 0.0, 10.0, &[(name(0, 0).as_str(), 1.0)])
        .unwrap();
    for l in 0..layers - 1 {
        for w in 0..width {
            let a = name(l, w);
            let b = name(l + 1, w);
            let c = name(l + 1, (w + 1) % width);
            let id = format!("R{l}_{w}");
            m.add_reaction(&id, 0.0, 1000.0, &[(a.as_str(), -1.0), (b.as_str(), 1.0)])
                .unwrap();
            let id = format!("X{l}_{w}");
            m.add_reaction(&id, 0.0, 1000.0, &[(a.as_str(), -1.0), (c.as_str(), 1.0)])
                .unwrap();
        }
    }
    let last = name(layers - 1, 0);
    m.add_reaction("BIOMASS", 0.0, 1000.0, &[(last.as_str(), -1.0)])
        .unwrap();
    m.set_objective("BIOMASS", 1.0).unwrap();
    m
}

fn bench_fba(c: &mut Criterion) {
    let model = layered_model(20, 15);
    let solver = SimplexSolver::new();
    c.bench_function("fba layered 20x15", |b| {
        b.iter(|| black_box(fba(black_box(&model), &solver).unwrap()))
    });
}

criterion_group!(benches, bench_fba);
criterion_main!(benches);
