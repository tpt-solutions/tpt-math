use core::f64::consts::PI;

use tpt_math_pde_operators::{Boundary, Centering, Field, Grid, Operators};

#[test]
fn dirichlet_laplacian_is_exact_for_quadratics_in_the_interior() {
    let n = 16;
    let h = 1.0 / n as f64;
    let ops = Operators::new(Grid::new([n], [h], [Boundary::Dirichlet0]).unwrap());
    // u = x(1 - x) vanishes on both walls and has u'' = -2 everywhere.
    let u = Field::from_fn(Centering::Cell, [n], |c| {
        let x = (c[0] as f64 + 0.5) * h;
        x * (1.0 - x)
    });
    let lap = ops.laplacian(&u);
    for &value in &lap.as_slice()[1..n - 1] {
        assert!((value + 2.0).abs() < 1e-9, "interior Laplacian was {value}");
    }
}

fn periodic_laplacian_error(n: usize) -> f64 {
    let h = 1.0 / n as f64;
    let k = 2.0 * PI;
    let ops = Operators::new(Grid::new([n], [h], [Boundary::Periodic]).unwrap());
    let u = Field::from_fn(Centering::Cell, [n], |c| {
        (k * (c[0] as f64 + 0.5) * h).sin()
    });
    let lap = ops.laplacian(&u);
    lap.as_slice()
        .iter()
        .zip(u.as_slice())
        .map(|(l, x)| (l + k * k * x).abs())
        .fold(0.0, f64::max)
}

#[test]
fn periodic_laplacian_converges_at_second_order() {
    let coarse = periodic_laplacian_error(32);
    let fine = periodic_laplacian_error(64);
    let order = (coarse / fine).log2();
    assert!((order - 2.0).abs() < 0.05, "observed order {order}");
}

#[test]
fn neumann_laplacian_annihilates_constants() {
    let n = 10;
    let ops = Operators::new(Grid::new([n, n], [0.1_f64, 0.2], [Boundary::Neumann0; 2]).unwrap());
    let ones = Field::from_fn(Centering::Cell, [n, n], |_| 1.0_f64);
    assert!(ops
        .laplacian(&ones)
        .as_slice()
        .iter()
        .all(|x| x.abs() < 1e-12));
}

#[test]
fn verify_reports_every_contract_as_passing_for_a_mixed_3d_grid() {
    let grid = Grid::new(
        [4, 3, 5],
        [0.5, 0.25, 1.0],
        [Boundary::Periodic, Boundary::Dirichlet0, Boundary::Neumann0],
    )
    .unwrap();
    let ops = Operators::new(grid);
    let reports = ops.verify();
    assert!(!reports.is_empty());
    for report in &reports {
        assert!(report.passed, "{report:?}");
    }
}
