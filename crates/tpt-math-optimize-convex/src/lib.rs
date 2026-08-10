//! Convex / quadratic optimization (thin wrapper around [`clarabel`]).
//!
//! This crate exposes a small, ergonomic API over the [`clarabel`] conic
//! interior-point solver for convex quadratic programs (QPs) of the form
//!
//! ```text
//! minimize    ½ xᵀ P x + qᵀ x
//! subject to  A_eq x = b_eq
//!             A_ineq x ≤ b_ineq
//!             l ≤ x ≤ u            (per-variable bounds)
//! ```
//!
//! The problem is converted into [`clarabel`]'s standard conic form:
//!
//! ```text
//! minimize    qᵀ x
//! subject to  A x + s = b
//!             s ∈ K
//! ```
//!
//! where equality constraints use the zero cone `ZeroConeT`, inequality
//! constraints and variable bounds use the nonnegative cone `NonnegativeConeT`.
//! Bounds `l ≤ x ≤ u` are mapped to two nonnegative slack inequalities each
//! (`x - l ≥ 0` and `u - x ≥ 0`). The quadratic cost matrix `P` is symmetrized
//! as `(P + Pᵀ) / 2` and clarabel consumes only its upper triangle (clarabel
//! also performs this extraction internally, but we do it explicitly for
//! deterministic behaviour).
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg::nalgebra::{dvector, dmatrix};
//! use tpt_math_optimize_convex::solve_qp;
//!
//! // minimize x² + y²  subject to  x + y = 1
//! // optimum is (0.5, 0.5)
//! let P = dmatrix![2.0, 0.0; 0.0, 2.0];
//! let q = dvector![0.0, 0.0];
//! let A_eq = dmatrix![1.0, 1.0];
//! let b_eq = dvector![1.0];
//! let x = solve_qp(&P, &q, &A_eq, &b_eq, &[]).unwrap();
//! assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6);
//! ```
//!
//! [`clarabel`]: crate::clarabel

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettings, IPSolver, SolverStatus, SupportedConeT, SupportedConeT::*,
};
use std::fmt;
use tpt_math_linalg::nalgebra::{DMatrix, DVector};

/// Re-export of [`nalgebra`] (via `tpt-math-linalg`) for convenience in
/// constructing dense inputs such as [`DMatrix`] and [`DVector`].
pub use tpt_math_linalg::nalgebra;

/// Re-export of the underlying [`clarabel`] solver crate.
///
/// Useful when you need lower-level control (e.g. custom cone types,
/// settings, or solution inspection).
pub use clarabel;

/// Errors returned by the QP solvers in this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvexError {
    /// A matrix or vector had an incompatible dimension (e.g. `A_eq` column
    /// count does not match the number of variables `n`).
    DimensionMismatch {
        /// Human-readable description of the conflict.
        what: String,
    },
    /// An input contained a non-finite (`NaN` or `infinite`) entry.
    NotFinite {
        /// Which input was non-finite.
        what: String,
    },
    /// The conic solver did not return a (near) optimal solution.
    Solver {
        /// The [`SolverStatus`] reported by clarabel (formatted).
        status: String,
    },
}

impl fmt::Display for ConvexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvexError::DimensionMismatch { what } => write!(f, "dimension mismatch: {what}"),
            ConvexError::NotFinite { what } => write!(f, "non-finite input: {what}"),
            ConvexError::Solver { status } => write!(f, "solver failed: {status}"),
        }
    }
}

impl std::error::Error for ConvexError {}

/// A successful solution to a quadratic program.
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// The optimal primal variable vector `x`.
    pub x: DVector<f64>,
    /// The optimal objective value `½ xᵀ P x + qᵀ x`.
    pub objective: f64,
    /// The solver status. Usually [`SolverStatus::Solved`] (or
    /// [`SolverStatus::AlmostSolved`] at reduced accuracy).
    pub status: SolverStatus,
}

/// Build a sparse CSC matrix from a dense `nalgebra::DMatrix`, dropping exact
/// zeros to keep the representation compact. clarabel consumes the upper
/// triangle of `P` only, but dense `A` is passed whole.
fn dense_to_csc(m: &DMatrix<f64>) -> CscMatrix<f64> {
    let rows = m.nrows();
    let cols = m.ncols();
    let mut colptr = vec![0usize; cols + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for c in 0..cols {
        for r in 0..rows {
            let v = m[(r, c)];
            if v != 0.0 {
                rowval.push(r);
                nzval.push(v);
            }
        }
        colptr[c + 1] = rowval.len();
    }
    CscMatrix::new(rows, cols, colptr, rowval, nzval)
}

/// Symmetrize `P` as `(P + Pᵀ) / 2` and return its upper triangle as a sparse
/// CSC matrix (clarabel's convention for the quadratic cost term).
fn symmetric_upper_csc(p: &DMatrix<f64>) -> CscMatrix<f64> {
    let n = p.nrows();
    let mut up = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let v = 0.5 * (p[(i, j)] + p[(j, i)]);
            up[(i, j)] = v;
        }
    }
    dense_to_csc(&up)
}

/// Validate that every element of a dense matrix is finite.
fn check_finite_matrix(m: &DMatrix<f64>, what: &str) -> Result<(), ConvexError> {
    if !m.iter().all(|v| v.is_finite()) {
        return Err(ConvexError::NotFinite { what: what.into() });
    }
    Ok(())
}

/// Validate that every element of a dense vector is finite.
fn check_finite_vector(v: &DVector<f64>, what: &str) -> Result<(), ConvexError> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(ConvexError::NotFinite { what: what.into() });
    }
    Ok(())
}

/// Internal worker that assembles the conic form and calls clarabel.
#[allow(clippy::too_many_arguments)]
fn solve_qp_internal(
    P: &DMatrix<f64>,
    q: &DVector<f64>,
    A_eq: &DMatrix<f64>,
    b_eq: &DVector<f64>,
    A_ineq: &DMatrix<f64>,
    b_ineq: &DVector<f64>,
    bounds: &[(f64, f64)],
) -> Result<QpSolution, ConvexError> {
    let n = q.len();

    if P.nrows() != n || P.ncols() != n {
        return Err(ConvexError::DimensionMismatch {
            what: format!("P is {}x{}, expected {}x{}", P.nrows(), P.ncols(), n, n),
        });
    }
    if A_eq.ncols() != n {
        return Err(ConvexError::DimensionMismatch {
            what: format!("A_eq has {} columns, expected {n}", A_eq.ncols()),
        });
    }
    if A_eq.nrows() != b_eq.len() {
        return Err(ConvexError::DimensionMismatch {
            what: format!(
                "A_eq has {} rows but b_eq has {} entries",
                A_eq.nrows(),
                b_eq.len()
            ),
        });
    }
    if A_ineq.ncols() != n {
        return Err(ConvexError::DimensionMismatch {
            what: format!("A_ineq has {} columns, expected {n}", A_ineq.ncols()),
        });
    }
    if A_ineq.nrows() != b_ineq.len() {
        return Err(ConvexError::DimensionMismatch {
            what: format!(
                "A_ineq has {} rows but b_ineq has {} entries",
                A_ineq.nrows(),
                b_ineq.len()
            ),
        });
    }
    if bounds.len() != n && !bounds.is_empty() {
        return Err(ConvexError::DimensionMismatch {
            what: format!("bounds has {} entries, expected {n} or empty", bounds.len()),
        });
    }

    check_finite_matrix(P, "P")?;
    check_finite_vector(q, "q")?;
    check_finite_matrix(A_eq, "A_eq")?;
    check_finite_vector(b_eq, "b_eq")?;
    check_finite_matrix(A_ineq, "A_ineq")?;
    check_finite_vector(b_ineq, "b_ineq")?;

    let p_csc = symmetric_upper_csc(P);
    let c: Vec<f64> = q.iter().copied().collect();

    // Number of constraints: equality + inequality + two bounds per variable.
    let n_eq = A_eq.nrows();
    let n_ineq = A_ineq.nrows();
    let n_bound = if bounds.is_empty() { 0 } else { 2 * n };
    let m = n_eq + n_ineq + n_bound;

    let mut A = DMatrix::<f64>::zeros(m, n);
    let mut b = DVector::<f64>::zeros(m);
    let mut cones: Vec<SupportedConeT<f64>> = Vec::new();

    let mut row = 0usize;

    // Equality constraints -> zero cone.
    for i in 0..n_eq {
        for j in 0..n {
            A[(row, j)] = A_eq[(i, j)];
        }
        b[row] = b_eq[i];
        row += 1;
    }
    if n_eq > 0 {
        cones.push(ZeroConeT(n_eq));
    }

    // Inequality constraints A_ineq x ≤ b_ineq -> nonnegative cone on the slack.
    for i in 0..n_ineq {
        for j in 0..n {
            A[(row, j)] = A_ineq[(i, j)];
        }
        b[row] = b_ineq[i];
        row += 1;
    }
    if n_ineq > 0 {
        cones.push(NonnegativeConeT(n_ineq));
    }

    // Bounds l ≤ x ≤ u -> (x - l ≥ 0) and (u - x ≥ 0) -> nonnegative cone.
    if n_bound > 0 {
        for j in 0..n {
            let (l, u) = bounds[j];
            if !l.is_finite() {
                return Err(ConvexError::NotFinite { what: "lower bound".into() });
            }
            if !u.is_finite() {
                return Err(ConvexError::NotFinite { what: "upper bound".into() });
            }
            // x - l ≥ 0  =>  x + s = l with s ≥ 0
            A[(row, j)] = 1.0;
            b[row] = l;
            row += 1;
            // u - x ≥ 0  =>  -x + s = u with s ≥ 0
            A[(row, j)] = -1.0;
            b[row] = u;
            row += 1;
        }
        cones.push(NonnegativeConeT(n_bound));
    }

    let A_csc = dense_to_csc(&A);
    let bb: Vec<f64> = b.iter().copied().collect();

    let settings = DefaultSettings::<f64>::default();
    let mut solver = clarabel::solver::DefaultSolver::new(&p_csc, &c, &A_csc, &bb, &cones, settings)
        .map_err(|e| ConvexError::Solver { status: e.to_string() })?;
    solver.solve();

    let sol = &solver.solution;
    if matches!(sol.status, SolverStatus::Solved | SolverStatus::AlmostSolved) {
        Ok(QpSolution {
            x: DVector::from_vec(sol.x.clone()),
            objective: sol.obj_val,
            status: sol.status,
        })
    } else {
        Err(ConvexError::Solver {
            status: format!("{:?}", sol.status),
        })
    }
}

/// Convenience wrapper for a quadratic program with **equality constraints and
/// per-variable bounds** (the most common convex QP form).
///
/// Solves
///
/// ```text
/// minimize    ½ xᵀ P x + qᵀ x
/// subject to  A_eq x = b_eq
///             l ≤ x ≤ u   for (l, u) = bounds[i]
/// ```
///
/// # Arguments
///
/// * `P` — symmetric `n×n` quadratic-cost matrix (will be symmetrized).
/// * `q` — linear-cost vector of length `n`.
/// * `A_eq` — `p×n` equality-constraint matrix.
/// * `b_eq` — right-hand side of length `p`.
/// * `bounds` — per-variable `(lower, upper)` bounds. Pass `&[]` to impose no
///   bounds. Use `-inf`/`inf` style via finite bounds only; unbounded
///   directions are not supported by this convenience function.
///
/// # Errors
///
/// Returns [`ConvexError`] on dimension mismatch, non-finite input, or if the
/// solver fails to converge to an optimum.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg::nalgebra::{dvector, dmatrix};
/// use tpt_math_optimize_convex::solve_qp;
///
/// // minimize x²  subject to  x ≥ 1   ->  optimum x = 1
/// let P = dmatrix![2.0];
/// let q = dvector![0.0];
/// let A_eq = dmatrix![];
/// let b_eq = dvector![];
/// let x = solve_qp(&P, &q, &A_eq, &b_eq, &[(1.0, f64::INFINITY)]).unwrap();
/// assert!((x[0] - 1.0).abs() < 1e-6);
/// ```
pub fn solve_qp(
    P: &DMatrix<f64>,
    q: &DVector<f64>,
    A_eq: &DMatrix<f64>,
    b_eq: &DVector<f64>,
    bounds: &[(f64, f64)],
) -> Result<DVector<f64>, ConvexError> {
    let A_ineq = DMatrix::<f64>::zeros(0, q.len());
    let b_ineq = DVector::<f64>::zeros(0);
    solve_qp_internal(P, q, A_eq, b_eq, &A_ineq, &b_ineq, bounds).map(|s| s.x)
}

/// A builder for quadratic programs supporting equality constraints, inequality
/// constraints, and per-variable bounds.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg::nalgebra::{dvector, dmatrix};
/// use tpt_math_optimize_convex::{QuadraticProgram, solve_qp};
///
/// // minimize x² + y²  subject to  x + y = 1, x ≥ 0, y ≥ 0
/// let qp = QuadraticProgram::new(dvector![0.0, 0.0])
///     .objective(dmatrix![2.0, 0.0; 0.0, 2.0])
///     .equality(dmatrix![1.0, 1.0], dvector![1.0])
///     .bounds(&[(0.0, f64::INFINITY), (0.0, f64::INFINITY)]);
/// let sol = qp.solve().unwrap();
/// assert!((sol.x[0] - 0.5).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct QuadraticProgram {
    n: usize,
    P: DMatrix<f64>,
    q: DVector<f64>,
    A_eq: DMatrix<f64>,
    b_eq: DVector<f64>,
    A_ineq: DMatrix<f64>,
    b_ineq: DVector<f64>,
    bounds: Vec<(f64, f64)>,
}

impl QuadraticProgram {
    /// Create a new QP with `n` variables and zero quadratic cost.
    ///
    /// The linear cost `q` must have length `n`.
    pub fn new(q: DVector<f64>) -> Self {
        let n = q.len();
        QuadraticProgram {
            n,
            P: DMatrix::zeros(n, n),
            q,
            A_eq: DMatrix::zeros(0, n),
            b_eq: DVector::zeros(0),
            A_ineq: DMatrix::zeros(0, n),
            b_ineq: DVector::zeros(0),
            bounds: Vec::new(),
        }
    }

    /// Set the quadratic cost matrix `P` (`n×n`). It will be symmetrized.
    #[allow(non_snake_case)]
    pub fn objective(mut self, P: DMatrix<f64>) -> Self {
        self.P = P;
        self
    }

    /// Set the linear cost vector `q`.
    pub fn linear_cost(mut self, q: DVector<f64>) -> Self {
        self.q = q;
        self
    }

    /// Set equality constraints `A_eq x = b_eq`.
    pub fn equality(mut self, A_eq: DMatrix<f64>, b_eq: DVector<f64>) -> Self {
        self.A_eq = A_eq;
        self.b_eq = b_eq;
        self
    }

    /// Set linear inequality constraints `A_ineq x ≤ b_ineq`.
    pub fn inequality(mut self, A_ineq: DMatrix<f64>, b_ineq: DVector<f64>) -> Self {
        self.A_ineq = A_ineq;
        self.b_ineq = b_ineq;
        self
    }

    /// Set per-variable bounds `(lower, upper)`; length must equal `n`.
    pub fn bounds(mut self, bounds: &[(f64, f64)]) -> Self {
        self.bounds = bounds.to_vec();
        self
    }

    /// Solve the program and return a [`QpSolution`].
    pub fn solve(self) -> Result<QpSolution, ConvexError> {
        solve_qp_internal(
            &self.P,
            &self.q,
            &self.A_eq,
            &self.b_eq,
            &self.A_ineq,
            &self.b_ineq,
            &self.bounds,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_linalg::nalgebra::{dmatrix, dvector};

    #[test]
    fn test_minimize_x2_eq_1() {
        // minimize x²  s.t. x ≥ 1  ->  x = 1
        let P = dmatrix![2.0];
        let q = dvector![0.0];
        let A_eq = dmatrix![];
        let b_eq = dvector![];
        let x = solve_qp(&P, &q, &A_eq, &b_eq, &[(1.0, f64::INFINITY)]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6, "got x = {}", x[0]);
    }

    #[test]
    fn test_minimize_x2_y2_eq_sum() {
        // minimize x² + y²  s.t. x + y = 1  ->  (0.5, 0.5)
        let P = dmatrix![2.0, 0.0; 0.0, 2.0];
        let q = dvector![0.0, 0.0];
        let A_eq = dmatrix![1.0, 1.0];
        let b_eq = dvector![1.0];
        let x = solve_qp(&P, &q, &A_eq, &b_eq, &[]).unwrap();
        assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6, "got x = {:?}", x);
    }

    #[test]
    fn test_minimize_with_inequality() {
        // minimize x² + y²  s.t. x ≥ 1  ->  x = 1, y = 0
        let P = dmatrix![2.0, 0.0; 0.0, 2.0];
        let q = dvector![0.0, 0.0];
        let A_eq = dmatrix![];
        let b_eq = dvector![];
        let x = solve_qp(&P, &q, &A_eq, &b_eq, &[(1.0, f64::INFINITY), (0.0, f64::INFINITY)]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6 && (x[1] - 0.0).abs() < 1e-6, "got x = {:?}", x);
    }

    #[test]
    fn test_builder_twoside_bounds() {
        // minimize x²  s.t. 2 ≤ x ≤ 3  ->  x = 2
        let qp = QuadraticProgram::new(dvector![0.0])
            .objective(dmatrix![2.0])
            .bounds(&[(2.0, 3.0)]);
        let sol = qp.solve().unwrap();
        assert!((sol.x[0] - 2.0).abs() < 1e-6, "got x = {}", sol.x[0]);
    }

    #[test]
    fn test_inequality_constraints() {
        // minimize x² + y²  s.t. x + y ≤ 1, x ≥ 0, y ≥ 0
        // The unconstrained min at (0,0) already satisfies the constraint, so
        // optimum is (0, 0).
        let P = dmatrix![2.0, 0.0; 0.0, 2.0];
        let q = dvector![0.0, 0.0];
        let A_ineq = dmatrix![1.0, 1.0];
        let b_ineq = dvector![1.0];
        let sol = solve_qp_internal(
            &P,
            &q,
            &DMatrix::zeros(0, 2),
            &DVector::zeros(0),
            &A_ineq,
            &b_ineq,
            &[],
        )
        .unwrap();
        assert!((sol.x[0]).abs() < 1e-6 && (sol.x[1]).abs() < 1e-6, "got x = {:?}", sol.x);
    }

    #[test]
    fn test_dimension_error() {
        let P = dmatrix![2.0, 0.0; 0.0, 2.0];
        let q = dvector![0.0]; // wrong length
        let A_eq = dmatrix![1.0, 1.0];
        let b_eq = dvector![1.0];
        let err = solve_qp(&P, &q, &A_eq, &b_eq, &[]).unwrap_err();
        assert!(matches!(err, ConvexError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_nonfinite_error() {
        let P = dmatrix![2.0, 0.0; 0.0, 2.0];
        let q = dvector![f64::NAN, 0.0];
        let A_eq = dmatrix![1.0, 1.0];
        let b_eq = dvector![1.0];
        let err = solve_qp(&P, &q, &A_eq, &b_eq, &[]).unwrap_err();
        assert!(matches!(err, ConvexError::NotFinite { .. }));
    }
}
