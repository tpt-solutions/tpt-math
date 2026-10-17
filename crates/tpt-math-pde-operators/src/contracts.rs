//! Structural contracts every discrete operator here must satisfy.
//!
//! Each check evaluates an identity on deterministic pseudo-random probe
//! fields and reports the residual against a tolerance scaled to the size of
//! the terms involved. Passing reports are evidence on the probes, not a proof.

use alloc::vec::Vec;

use tpt_math_numeric::{NumCast, Scalar};

use crate::grid::unflatten;
use crate::{Boundary, Centering, Field, Grid, Operators};

/// Relative tolerance, in units of machine epsilon per cell.
const RTOL: f64 = 1.0e3;

/// Probe seeds. Each seed draws a fresh set of fields.
const SEEDS: [u64; 3] = [1, 2, 3];

/// Outcome of one contract check on one probe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContractReport {
    /// What the contract states.
    pub name: &'static str,
    /// Absolute gap between the two sides of the identity.
    pub residual: f64,
    /// Largest residual the check accepts.
    pub tolerance: f64,
    /// Whether `residual <= tolerance`.
    pub passed: bool,
}

impl<T: Scalar, const D: usize> Operators<T, D> {
    /// Evaluates every structural contract on deterministic probe fields.
    ///
    /// Always runs, whatever the build profile. [`Operators::new`] calls it
    /// automatically when contracts are active.
    pub fn verify(&self) -> Vec<ContractReport> {
        let grid = *self.grid();
        let mut reports = Vec::new();
        for seed in SEEDS {
            let mut rng = Rng::new(seed);
            let u: Field<T, D> = probe(&mut rng, Centering::Cell, grid.shape());
            let v: Field<T, D> = probe(&mut rng, Centering::Cell, grid.shape());
            let flux: Vec<Field<T, D>> = (0..D)
                .map(|axis| probe(&mut rng, Centering::Face(axis), grid.face_shape(axis)))
                .collect();

            reports.push(self.check_adjointness(&u, &flux));
            reports.push(self.check_divergence_theorem(&flux));
            reports.push(self.check_laplacian_self_adjoint(&u, &v));
            reports.push(self.check_laplacian_negative_semidefinite(&u));
            for axis in 0..D {
                if grid.boundary()[axis].is_periodic() {
                    reports.push(self.check_skew_symmetry(&u, &v, axis));
                }
            }
        }
        if grid.boundary().iter().all(|&b| b != Boundary::Dirichlet0) {
            reports.push(self.check_constant_kernel());
        }
        reports
    }

    fn check_adjointness(&self, u: &Field<T, D>, flux: &[Field<T, D>]) -> ContractReport {
        let grid = *self.grid();
        let (mut lhs, mut scale) = (T::zero(), T::zero());
        for (axis, (grad, component)) in self.gradient(u).iter().zip(flux).enumerate() {
            let (sum, abs) = face_dot(&grid, axis, grad, component);
            lhs = lhs + sum;
            scale = scale + abs;
        }
        let (rhs, cell_scale) = cell_dot(&grid, u, &self.divergence(flux));
        report(
            "adjointness: <grad u, F> = -<u, div F>",
            lhs,
            -rhs,
            scale + cell_scale,
            grid.num_cells(),
        )
    }

    fn check_divergence_theorem(&self, flux: &[Field<T, D>]) -> ContractReport {
        let grid = *self.grid();
        let shape = grid.shape();
        let div = self.divergence(flux);
        let ones = Field::from_fn(Centering::Cell, shape, |_| T::one());
        let (lhs, cell_scale) = cell_dot(&grid, &ones, &div);

        // Net outflow through the walls: only non-periodic axes have walls.
        let (mut rhs, mut flux_scale) = (T::zero(), T::zero());
        for (axis, component) in flux.iter().enumerate() {
            if grid.boundary()[axis].is_periodic() {
                continue;
            }
            let cross_section = grid.cell_volume() / grid.spacing()[axis];
            let n = shape[axis];
            for (flat, &value) in component.as_slice().iter().enumerate() {
                let index = unflatten(&component.shape(), flat);
                let sign = if index[axis] == n {
                    T::one()
                } else if index[axis] == 0 {
                    -T::one()
                } else {
                    continue;
                };
                let effective = if grid.boundary()[axis] == Boundary::Neumann0 {
                    T::zero()
                } else {
                    value
                };
                rhs = rhs + sign * cross_section * effective;
                flux_scale = flux_scale + cross_section * effective.abs();
            }
        }
        report(
            "discrete conservation: sum(vol * div F) = net wall flux",
            lhs,
            rhs,
            cell_scale + flux_scale,
            grid.num_cells(),
        )
    }

    fn check_laplacian_self_adjoint(&self, u: &Field<T, D>, v: &Field<T, D>) -> ContractReport {
        let grid = *self.grid();
        let (lhs, lhs_scale) = cell_dot(&grid, u, &self.laplacian(v));
        let (rhs, rhs_scale) = cell_dot(&grid, &self.laplacian(u), v);
        report(
            "Laplacian is self-adjoint: <u, Lv> = <Lu, v>",
            lhs,
            rhs,
            lhs_scale + rhs_scale,
            grid.num_cells(),
        )
    }

    fn check_laplacian_negative_semidefinite(&self, u: &Field<T, D>) -> ContractReport {
        let grid = *self.grid();
        let (quadratic, scale) = cell_dot(&grid, u, &self.laplacian(u));
        report(
            "Laplacian is negative semi-definite: <u, Lu> <= 0",
            quadratic.max(T::zero()),
            T::zero(),
            scale,
            grid.num_cells(),
        )
    }

    fn check_skew_symmetry(&self, u: &Field<T, D>, v: &Field<T, D>, axis: usize) -> ContractReport {
        let grid = *self.grid();
        let du = self.central_difference(u, axis);
        let dv = self.central_difference(v, axis);
        let (lhs, lhs_scale) = cell_dot(&grid, &du, v);
        let (rhs, rhs_scale) = cell_dot(&grid, u, &dv);
        report(
            "central difference is skew-symmetric: <Dx, y> = -<x, Dy>",
            lhs,
            -rhs,
            lhs_scale + rhs_scale,
            grid.num_cells(),
        )
    }

    fn check_constant_kernel(&self) -> ContractReport {
        let grid = *self.grid();
        let ones = Field::from_fn(Centering::Cell, grid.shape(), |_| T::one());
        let peak = self
            .laplacian(&ones)
            .as_slice()
            .iter()
            .fold(T::zero(), |acc, &x| acc.max(x.abs()));
        report(
            "constants are in the Laplacian kernel (no Dirichlet walls)",
            peak,
            T::zero(),
            T::one(),
            grid.num_cells(),
        )
    }
}

/// Builds a report, accepting `|lhs - rhs| <= RTOL * eps * cells * max(scale, 1)`.
fn report<T: Scalar>(name: &'static str, lhs: T, rhs: T, scale: T, cells: usize) -> ContractReport {
    let tolerance =
        lit::<T>(RTOL) * T::epsilon() * lit::<T>(cells.max(1) as f64) * scale.max(T::one());
    let residual = (lhs - rhs).abs();
    ContractReport {
        name,
        residual: to_f64(residual),
        tolerance: to_f64(tolerance),
        passed: residual <= tolerance,
    }
}

/// Cell-weighted inner product: the signed sum and the sum of absolute terms.
fn cell_dot<T: Scalar, const D: usize>(
    grid: &Grid<T, D>,
    a: &Field<T, D>,
    b: &Field<T, D>,
) -> (T, T) {
    let volume = grid.cell_volume();
    weighted_dot(a, b, |_| volume)
}

/// Face-weighted inner product on the faces normal to `axis`.
fn face_dot<T: Scalar, const D: usize>(
    grid: &Grid<T, D>,
    axis: usize,
    a: &Field<T, D>,
    b: &Field<T, D>,
) -> (T, T) {
    weighted_dot(a, b, |face| grid.face_weight(axis, face))
}

fn weighted_dot<T: Scalar, const D: usize>(
    a: &Field<T, D>,
    b: &Field<T, D>,
    weight: impl Fn([usize; D]) -> T,
) -> (T, T) {
    let shape = a.shape();
    let (mut sum, mut abs) = (T::zero(), T::zero());
    for (flat, (&x, &y)) in a.as_slice().iter().zip(b.as_slice()).enumerate() {
        let term = weight(unflatten(&shape, flat)) * x * y;
        sum = sum + term;
        abs = abs + term.abs();
    }
    (sum, abs)
}

fn probe<T: Scalar, const D: usize>(
    rng: &mut Rng,
    centering: Centering,
    shape: [usize; D],
) -> Field<T, D> {
    Field::from_fn(centering, shape, |_| rng.next_unit())
}

/// Deterministic xorshift64 generator for probe fields.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    /// Uniform in `[-1, 1)`.
    fn next_unit<T: Scalar>(&mut self) -> T {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let unit = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        lit(2.0 * unit - 1.0)
    }
}

fn lit<T: Scalar>(x: f64) -> T {
    <T as NumCast>::from(x).unwrap_or_else(T::zero)
}

fn to_f64<T: Scalar>(x: T) -> f64 {
    x.to_f64().unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_accepts_rounding_and_rejects_real_gaps() {
        let ok = report("rounding", 1.0_f64, 1.0 + 1e-14, 1.0, 10);
        assert!(ok.passed);
        let bad = report("gap", 1.0_f64, 1.5, 1.0, 10);
        assert!(!bad.passed);
    }

    #[test]
    fn report_rejects_nan() {
        let nan = report("nan", f64::NAN, 0.0, 1.0, 10);
        assert!(!nan.passed);
    }
}
