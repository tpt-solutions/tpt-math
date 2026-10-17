use criterion::{black_box, criterion_group, criterion_main, Criterion};
use tpt_math_pde_operators::{Boundary, Centering, Field, Grid, Operators};

fn bench_laplacian(c: &mut Criterion) {
    let n = 256;
    let h = 1.0 / n as f64;
    let ops = Operators::new(Grid::new([n, n], [h, h], [Boundary::Dirichlet0; 2]).unwrap());
    let u = Field::from_fn(Centering::Cell, [n, n], |c| {
        ((c[0] + c[1]) as f64 * h).sin()
    });
    c.bench_function("Laplacian 2D Dirichlet, 256x256", |b| {
        b.iter(|| black_box(ops.laplacian(black_box(&u))))
    });
}

criterion_group!(benches, bench_laplacian);
criterion_main!(benches);
