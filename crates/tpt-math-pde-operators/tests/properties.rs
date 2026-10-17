use proptest::prelude::*;

use tpt_math_pde_operators::{Boundary, Grid, Operators};

fn boundary() -> impl Strategy<Value = Boundary> {
    prop_oneof![
        Just(Boundary::Periodic),
        Just(Boundary::Dirichlet0),
        Just(Boundary::Neumann0),
    ]
}

fn contracts_hold<const D: usize>(shape: [usize; D], bc: [Boundary; D], h: f64) -> bool {
    let grid = Grid::new(shape, [h; D], bc).unwrap();
    Operators::new(grid).verify().iter().all(|r| r.passed)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn contracts_hold_in_1d(n in 1usize..=24, bc in boundary(), h in 0.01f64..2.0) {
        prop_assert!(contracts_hold([n], [bc], h));
    }

    #[test]
    fn contracts_hold_in_2d(
        nx in 1usize..=12,
        ny in 1usize..=12,
        bx in boundary(),
        by in boundary(),
        h in 0.01f64..2.0,
    ) {
        prop_assert!(contracts_hold([nx, ny], [bx, by], h));
    }

    #[test]
    fn contracts_hold_in_3d(
        nx in 1usize..=6,
        ny in 1usize..=6,
        nz in 1usize..=6,
        bx in boundary(),
        by in boundary(),
        bz in boundary(),
        h in 0.01f64..2.0,
    ) {
        prop_assert!(contracts_hold([nx, ny, nz], [bx, by, bz], h));
    }
}
