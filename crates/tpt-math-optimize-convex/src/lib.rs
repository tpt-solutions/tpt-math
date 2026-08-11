//! Convex / quadratic optimization (in-house dense interior-point solver).
//!
//! This crate solves convex quadratic programs (QPs) of the form
//!
//! ```text
//! minimize    ½ xᵀ P x + qᵀ x
//! subject to  A_eq x = b_eq
//!             A_ineq x ≤ b_ineq
//!             l ≤ x ≤ u            (per-variable bounds)
//! ```
//!
//! with a self-contained dense primal-dual interior-point method
//! (Mehrotra predictor-corrector). The KKT linear systems are inverted with
//! [`tpt_math_linalg_dense`]'s faer-backed dense solver, so there is no
//! Apache-2.0-only dependency anywhere in the tree (the old `clarabel` backend
//! was removed for that reason — see `spec.txt` / ADR-0007).
//!
//! The problem is converted into conic form
//!
//! ```text
//! minimize    qᵀ x + ½ xᵀ P x
//! subject to  A_eq x        = b_eq
//!             A x + s       = b,   s ≥ 0
//! ```
//!
//! where the equality constraints use the zero cone (no slack) and the
//! inequality constraints plus per-variable bounds use the nonnegative cone
//! (one slack `s_j ≥ 0` each). Bounds `l ≤ x ≤ u` are mapped to two
//! nonnegative slack inequalities each (`x - l ≥ 0` and `u - x ≥ 0`); a
//! non-finite bound is skipped (treated as "unbounded" on that side).
//!
//! # Examples
//!
//! ```
//! use tpt_math_linalg_dense::{DMatrix, DVector};
//! use tpt_math_optimize_convex::solve_qp;
//!
//! // minimize x² + y²  subject to  x + y = 1
//! // optimum is (0.5, 0.5)
//! let p = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
//! let q = DVector::from_vec(vec![0.0, 0.0]);
//! let a_eq = DMatrix::from_row_slice(1, 2, &[1.0, 1.0]);
//! let b_eq = DVector::from_vec(vec![1.0]);
//! let x = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap();
//! assert!((x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6);
//! ```
//!
//! [`tpt_math_linalg_dense`]: crate::tpt_math_linalg_dense

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use tpt_math_linalg_dense::{DMatrix, DVector};

/// Re-export of [`tpt_math_linalg_dense`] (faer-backed dense storage) for
/// constructing dense inputs such as [`DMatrix`] and [`DVector`].
pub use tpt_math_linalg_dense;

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
    /// The interior-point solver did not return a (near) optimal solution —
    /// non-convergence, infeasibility, or unboundedness. The `status` string
    /// carries the reason.
    Solver {
        /// Human-readable solver outcome.
        status: String,
    },
}

impl std::fmt::Display for ConvexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvexError::DimensionMismatch { what } => write!(f, "dimension mismatch: {what}"),
            ConvexError::NotFinite { what } => write!(f, "non-finite input: {what}"),
            ConvexError::Solver { status } => write!(f, "solver failed: {status}"),
        }
    }
}

impl std::error::Error for ConvexError {}

/// Outcome of a single interior-point solve, used by the status field of
/// [`QpSolution`].
#[derive(Debug, Clone, PartialEq)]
pub enum QpStatus {
    /// The optimizer converged to a solution within tolerance.
    Solved,
    /// The optimizer converged to a solution at reduced accuracy.
    AlmostSolved,
    /// The solve terminated without an optimum; the string explains why.
    Failed(String),
}

/// A successful solution to a quadratic program.
#[derive(Debug, Clone)]
pub struct QpSolution {
    /// The optimal primal variable vector `x`.
    pub x: DVector<f64>,
    /// The optimal objective value `½ xᵀ P x + qᵀ x`.
    pub objective: f64,
    /// The solve outcome. Usually [`QpStatus::Solved`].
    pub status: QpStatus,
}

/// Solve a QP with equality constraints and per-variable bounds (the most
/// common convex QP form). See the crate-level docs for the formulation.
///
/// See [`solve_qp_internal`] for the full form including inequality
/// constraints.
///
/// # Errors
///
/// Returns [`ConvexError`] on dimension mismatch, non-finite input, or if the
/// solver fails to converge to an optimum.
pub fn solve_qp(
    p: &DMatrix<f64>,
    q: &DVector<f64>,
    a_eq: &DMatrix<f64>,
    b_eq: &DVector<f64>,
    bounds: &[(f64, f64)],
) -> Result<DVector<f64>, ConvexError> {
    let a_ineq = DMatrix::<f64>::zeros(0, q.len());
    let b_ineq = DVector::<f64>::zeros(0);
    solve_qp_internal(p, q, a_eq, b_eq, &a_ineq, &b_ineq, bounds).map(|s| s.x)
}

/// Full QP solver: equality constraints, inequality constraints, and
/// per-variable bounds.
///
/// # Errors
///
/// Returns [`ConvexError`] on dimension mismatch, non-finite input, or if the
/// solver fails to converge to an optimum.
pub fn solve_qp_internal(
    p: &DMatrix<f64>,
    q: &DVector<f64>,
    a_eq: &DMatrix<f64>,
    b_eq: &DVector<f64>,
    a_ineq: &DMatrix<f64>,
    b_ineq: &DVector<f64>,
    bounds: &[(f64, f64)],
) -> Result<QpSolution, ConvexError> {
    let n = q.len();

    if p.nrows() != n || p.ncols() != n {
        return Err(ConvexError::DimensionMismatch {
            what: format!("P is {}x{}, expected {}x{}", p.nrows(), p.ncols(), n, n),
        });
    }
    if !a_eq.is_empty() && a_eq.ncols() != n {
        return Err(ConvexError::DimensionMismatch {
            what: format!("A_eq has {} columns, expected {n}", a_eq.ncols()),
        });
    }
    if a_eq.nrows() != b_eq.len() {
        return Err(ConvexError::DimensionMismatch {
            what: format!(
                "A_eq has {} rows but b_eq has {} entries",
                a_eq.nrows(),
                b_eq.len()
            ),
        });
    }
    if !a_ineq.is_empty() && a_ineq.ncols() != n {
        return Err(ConvexError::DimensionMismatch {
            what: format!("A_ineq has {} columns, expected {n}", a_ineq.ncols()),
        });
    }
    if a_ineq.nrows() != b_ineq.len() {
        return Err(ConvexError::DimensionMismatch {
            what: format!(
                "A_ineq has {} rows but b_ineq has {} entries",
                a_ineq.nrows(),
                b_ineq.len()
            ),
        });
    }
    if bounds.len() != n && !bounds.is_empty() {
        return Err(ConvexError::DimensionMismatch {
            what: format!("bounds has {} entries, expected {n} or empty", bounds.len()),
        });
    }

    check_finite_matrix(p, "P")?;
    check_finite_vector(q, "q")?;
    check_finite_matrix(a_eq, "A_eq")?;
    check_finite_vector(b_eq, "b_eq")?;
    check_finite_matrix(a_ineq, "A_ineq")?;
    check_finite_vector(b_ineq, "b_ineq")?;

    // Assemble the nonnegative-cone system (inequalities + finite bounds).
    let (a, b) = build_inequality_system(a_ineq, b_ineq, bounds, n);

    // Work in plain `Vec`/`Vec<Vec>` form for the iteration; the KKT solve is
    // delegated to `tpt-math-linalg-dense`'s faer-backed `DMatrix::solve`.
    let p_sym = symmetrize(p);
    let qv = q.iter().copied().collect::<Vec<f64>>();
    let a_eqv = matrix_to_vec(a_eq);
    let b_eqv = b_eq.iter().copied().collect::<Vec<f64>>();

    let (x, status) = interior_point(&p_sym, &qv, &a_eqv, &b_eqv, &a, &b)?;

    let xv = DVector::from_vec(x);
    let objective = qp_objective(&p_sym, &qv, &xv.iter().copied().collect::<Vec<f64>>());
    Ok(QpSolution {
        x: xv,
        objective,
        status,
    })
}

/// A builder for quadratic programs supporting equality constraints, inequality
/// constraints, and per-variable bounds.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_dense::{DMatrix, DVector};
/// use tpt_math_optimize_convex::{QuadraticProgram, solve_qp};
///
/// // minimize x² + y²  subject to  x + y = 1, x ≥ 0, y ≥ 0
/// let qp = QuadraticProgram::new(DVector::from_vec(vec![0.0, 0.0]))
///     .objective(DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]))
///     .equality(DMatrix::from_row_slice(1, 2, &[1.0, 1.0]), DVector::from_vec(vec![1.0]))
///     .bounds(&[(0.0, f64::INFINITY), (0.0, f64::INFINITY)]);
/// let sol = qp.solve().unwrap();
/// assert!((sol.x[0] - 0.5).abs() < 1e-6 && (sol.x[1] - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct QuadraticProgram {
    p: DMatrix<f64>,
    q: DVector<f64>,
    a_eq: DMatrix<f64>,
    b_eq: DVector<f64>,
    a_ineq: DMatrix<f64>,
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
            p: DMatrix::zeros(n, n),
            q,
            a_eq: DMatrix::zeros(0, n),
            b_eq: DVector::zeros(0),
            a_ineq: DMatrix::zeros(0, n),
            b_ineq: DVector::zeros(0),
            bounds: Vec::new(),
        }
    }

    /// Set the quadratic cost matrix `P` (`n×n`). It will be symmetrized.
    pub fn objective(mut self, p: DMatrix<f64>) -> Self {
        self.p = p;
        self
    }

    /// Set the linear cost vector `q`.
    pub fn linear_cost(mut self, q: DVector<f64>) -> Self {
        self.q = q;
        self
    }

    /// Set equality constraints `A_eq x = b_eq`.
    pub fn equality(mut self, a_eq: DMatrix<f64>, b_eq: DVector<f64>) -> Self {
        self.a_eq = a_eq;
        self.b_eq = b_eq;
        self
    }

    /// Set linear inequality constraints `A_ineq x ≤ b_ineq`.
    pub fn inequality(mut self, a_ineq: DMatrix<f64>, b_ineq: DVector<f64>) -> Self {
        self.a_ineq = a_ineq;
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
            &self.p,
            &self.q,
            &self.a_eq,
            &self.b_eq,
            &self.a_ineq,
            &self.b_ineq,
            &self.bounds,
        )
    }
}

// ---------------------------------------------------------------------------
// Internal helpers operating on `Vec` / `Vec<Vec>` form.
// ---------------------------------------------------------------------------

fn matrix_to_vec(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..m.nrows())
        .map(|i| (0..m.ncols()).map(|j| m[(i, j)]).collect())
        .collect()
}

fn symmetrize(m: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let n = m.nrows();
    (0..n)
        .map(|i| {
            (0..m.ncols())
                .map(|j| 0.5 * (m[(i, j)] + m[(j, i)]))
                .collect()
        })
        .collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn qp_objective(p: &[Vec<f64>], q: &[f64], x: &[f64]) -> f64 {
    let n = x.len();
    let mut quad = 0.0;
    for i in 0..n {
        let mut row = 0.0;
        for k in 0..n {
            row += p[i][k] * x[k];
        }
        quad += row * x[i];
    }
    0.5 * quad + dot(q, x)
}

fn check_finite_matrix(m: &DMatrix<f64>, what: &str) -> Result<(), ConvexError> {
    if !m.iter().all(|v| v.is_finite()) {
        return Err(ConvexError::NotFinite { what: what.into() });
    }
    Ok(())
}

fn check_finite_vector(v: &DVector<f64>, what: &str) -> Result<(), ConvexError> {
    if !v.iter().all(|x| x.is_finite()) {
        return Err(ConvexError::NotFinite { what: what.into() });
    }
    Ok(())
}

/// Build the nonnegative-cone system `A x + s = b, s ≥ 0` from the inequality
/// constraints `A_ineq x ≤ b_ineq` and the per-variable bounds. Each inequality
/// becomes one row; each finite bound becomes one row (`x - l ≥ 0` and/or
/// `u - x ≥ 0`).
fn build_inequality_system(
    a_ineq: &DMatrix<f64>,
    b_ineq: &DVector<f64>,
    bounds: &[(f64, f64)],
    n: usize,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();

    for r in 0..a_ineq.nrows() {
        let row: Vec<f64> = (0..n).map(|j| a_ineq[(r, j)]).collect();
        rows.push(row);
        rhs.push(b_ineq[r]);
    }

    for (j, &(l, u)) in bounds.iter().enumerate() {
        if l.is_finite() {
            // x_j ≥ l  =>  -x_j + s = -l
            let mut row = vec![0.0; n];
            row[j] = -1.0;
            rows.push(row);
            rhs.push(-l);
        }
        if u.is_finite() {
            // x_j ≤ u  =>   x_j + s = u
            let mut row = vec![0.0; n];
            row[j] = 1.0;
            rows.push(row);
            rhs.push(u);
        }
    }

    (rows, rhs)
}

/// Maximum step length `α ∈ [0, cap]` keeping `v + α dv ≥ 0`.
fn step_length(v: &[f64], dv: &[f64], cap: f64) -> f64 {
    let mut alpha = cap;
    for i in 0..v.len() {
        if dv[i] < 0.0 {
            alpha = alpha.min(-v[i] / dv[i]);
        }
    }
    alpha.max(0.0)
}

/// Solve the reduced KKT system `K [dx; dy] = rhs` and return the concatenated
/// step. The matrix `K` is `(n + m_eq) × (n + m_eq)`:
///
/// ```text
/// K = [ P + Aᵀ Σ A    A_eqᵀ ]
///     [ A_eq            0    ]
/// ```
///
/// with `Σ = diag(z / s)`. The solve uses `tpt-math-linalg-dense`'s faer-backed
/// dense LU.
///
/// `a` is the nonnegative-cone constraint matrix (`m_c × n`), `a_eq` the
/// equality matrix (`m_eq × n`).
fn solve_kkt(
    p: &[Vec<f64>],
    a_eq: &[Vec<f64>],
    a: &[Vec<f64>],
    sigma: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, ConvexError> {
    let n = p.len();
    let m_eq = a_eq.len();
    let m_c = a.len();
    let dim = n + m_eq;

    let k = DMatrix::<f64>::from_fn(dim, dim, |i, j| {
        if i < n && j < n {
            let mut v = p[i][j];
            for k in 0..m_c {
                v += a[k][i] * sigma[k] * a[k][j];
            }
            v
        } else if i < n && j >= n {
            let e = j - n;
            if e < m_eq {
                a_eq[e][i]
            } else {
                0.0
            }
        } else if i >= n && j < n {
            let e = i - n;
            if e < m_eq {
                a_eq[e][j]
            } else {
                0.0
            }
        } else {
            0.0
        }
    });

    let rh = DVector::from_vec(rhs.to_vec());
    let delta = k
        .solve(&rh)
        .map_err(|_| ConvexError::Solver {
            status: "singular KKT system (infeasible or ill-posed)".into(),
        })?;
    Ok(delta.iter().copied().collect())
}

/// One interior-point iteration step (affine when `w` is zero, corrector
/// otherwise). Returns `(dx, dy, ds, dz)`.
#[allow(clippy::too_many_arguments)]
fn compute_step(
    p: &[Vec<f64>],
    a_eq: &[Vec<f64>],
    a: &[Vec<f64>],
    _b: &[f64],
    x: &[f64],
    s: &[f64],
    z: &[f64],
    sigma: &[f64],
    r_dual: &[f64],
    r_pri_eq: &[f64],
    r_pri2: &[f64],
    w: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), ConvexError> {
    let n = x.len();
    let m_eq = a_eq.len();
    let m_c = a.len();

    let mut rhs = vec![0.0; n + m_eq];
    for i in 0..n {
        let mut v = -r_dual[i];
        for j in 0..m_c {
            v += a[j][i] * z[j];
            v -= a[j][i] * sigma[j] * r_pri2[j];
            v += a[j][i] * (w[j] / s[j]);
        }
        rhs[i] = v;
    }
    for e in 0..m_eq {
        rhs[n + e] = -r_pri_eq[e];
    }

    let delta = solve_kkt(p, a_eq, a, sigma, &rhs)?;
    let dx = delta[0..n].to_vec();
    let dy = delta[n..n + m_eq].to_vec();

    let mut ds = vec![0.0; m_c];
    let mut dz = vec![0.0; m_c];
    for j in 0..m_c {
        let ax = dot(&a[j], &dx);
        ds[j] = -r_pri2[j] - ax;
        dz[j] = -z[j] + w[j] / s[j] - sigma[j] * ds[j];
    }
    Ok((dx, dy, ds, dz))
}

/// The Mehrotra predictor-corrector interior-point loop.
fn interior_point(
    p: &[Vec<f64>],
    q: &[f64],
    a_eq: &[Vec<f64>],
    b_eq: &[f64],
    a: &[Vec<f64>],
    b: &[f64],
) -> Result<(Vec<f64>, QpStatus), ConvexError> {
    let n = q.len();
    let m_eq = a_eq.len();
    let m_c = a.len();

    let tol = 1e-8;
    let max_iter = 200;
    let tau = 0.99;
    let huge = 1e14;

    let mut x = vec![0.0; n];
    let mut y = vec![0.0; m_eq];
    let mut s = vec![0.0; m_c];
    let mut z = vec![0.0; m_c];
    for j in 0..m_c {
        let ax = dot(&a[j], &x);
        let mut sj = b[j] - ax;
        if !sj.is_finite() || sj <= 0.0 {
            sj = 1.0;
        }
        s[j] = sj;
        z[j] = 1.0;
    }

    for _iter in 0..max_iter {
        let r_dual = resid_dual(p, q, a_eq, &y, a, &z, &x);
        let r_pri_eq = resid_pri_eq(a_eq, &x, b_eq);
        let r_pri2 = resid_pri2(a, &x, &s, b);
        let mu = if m_c > 0 {
            dot(&s, &z) / (m_c as f64)
        } else {
            0.0
        };

        let prim = norm(&r_pri_eq) + norm(&r_pri2);
        let dual = norm(&r_dual);
        if prim < tol && dual < tol && (m_c == 0 || mu < tol) {
            return Ok((x, QpStatus::Solved));
        }
        if prim < tol * 10.0 && dual < tol * 10.0 && (m_c == 0 || mu < tol * 10.0) {
            return Ok((x, QpStatus::AlmostSolved));
        }

        let sigma_vec: Vec<f64> = (0..m_c).map(|j| z[j] / s[j]).collect();

        // Predictor (affine) step.
        let w_zero = vec![0.0; m_c];
        let (_dx_aff, _dy_aff, ds_aff, dz_aff) = compute_step(
            p, a_eq, a, b, &x, &s, &z, &sigma_vec, &r_dual, &r_pri_eq, &r_pri2, &w_zero,
        )?;

        let a_aff_pri = step_length(&s, &ds_aff, 1.0);
        let a_aff_dual = step_length(&z, &dz_aff, 1.0);
        let a_aff = a_aff_pri.min(a_aff_dual).min(1.0);
        let mu_aff = if m_c > 0 {
            let sp: Vec<f64> = (0..m_c).map(|j| s[j] + a_aff * ds_aff[j]).collect();
            let zp: Vec<f64> = (0..m_c).map(|j| z[j] + a_aff * dz_aff[j]).collect();
            dot(&sp, &zp) / (m_c as f64)
        } else {
            0.0
        };
        let sigma_cent = if m_c > 0 {
            let ratio = mu_aff / mu.max(1e-300);
            (ratio.max(0.0).min(1.0)).powi(3)
        } else {
            0.0
        };

        // Corrector step.
        let w = (0..m_c)
            .map(|j| sigma_cent * ds_aff[j] * dz_aff[j])
            .collect::<Vec<f64>>();
        let (dx, dy, ds, dz) = compute_step(
            p, a_eq, a, b, &x, &s, &z, &sigma_vec, &r_dual, &r_pri_eq, &r_pri2, &w,
        )?;

        let a_pri = tau * step_length(&s, &ds, 1.0);
        let a_dual = tau * step_length(&z, &dz, 1.0);

        for i in 0..n {
            x[i] += a_pri * dx[i];
        }
        for e in 0..m_eq {
            y[e] += a_dual * dy[e];
        }
        for j in 0..m_c {
            s[j] += a_pri * ds[j];
            z[j] += a_dual * dz[j];
        }

        if x.iter().any(|v| !v.is_finite()) || norm(&x) > huge {
            return Err(ConvexError::Solver {
                status: "primal iterate diverged (problem is unbounded)".into(),
            });
        }
        if s.iter().chain(z.iter()).any(|v| !v.is_finite()) {
            return Err(ConvexError::Solver {
                status: "dual iterate diverged (problem is infeasible)".into(),
            });
        }
    }

    Err(ConvexError::Solver {
        status: "did not converge within the iteration budget".into(),
    })
}

fn resid_dual(
    p: &[Vec<f64>],
    q: &[f64],
    a_eq: &[Vec<f64>],
    y: &[f64],
    a: &[Vec<f64>],
    z: &[f64],
    x: &[f64],
) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut v = q[i];
            for k in 0..n {
                v += p[i][k] * x[k];
            }
            for e in 0..a_eq.len() {
                v += a_eq[e][i] * y[e];
            }
            for j in 0..a.len() {
                v += a[j][i] * z[j];
            }
            v
        })
        .collect()
}

fn resid_pri_eq(a_eq: &[Vec<f64>], x: &[f64], b_eq: &[f64]) -> Vec<f64> {
    a_eq.iter()
        .enumerate()
        .map(|(e, row)| dot(row, x) - b_eq[e])
        .collect()
}

fn resid_pri2(a: &[Vec<f64>], x: &[f64], s: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter()
        .enumerate()
        .map(|(j, row)| dot(row, x) + s[j] - b[j])
        .collect()
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_math_linalg_dense::{DMatrix, DVector};

    fn dv(xs: &[f64]) -> DVector<f64> {
        DVector::from_vec(xs.to_vec())
    }

    fn dm(rows: usize, cols: usize, xs: &[f64]) -> DMatrix<f64> {
        DMatrix::from_row_slice(rows, cols, xs)
    }

    #[test]
    fn test_minimize_x2_eq_1() {
        // minimize x²  s.t. x ≥ 1  ->  x = 1
        let p = dm(1, 1, &[2.0]);
        let q = dv(&[0.0]);
        let a_eq = dm(0, 1, &[]);
        let b_eq = dv(&[]);
        let x = solve_qp(&p, &q, &a_eq, &b_eq, &[(1.0, f64::INFINITY)]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-6, "got x = {}", x[0]);
    }

    #[test]
    fn test_minimize_x2_y2_eq_sum() {
        // minimize x² + y²  s.t. x + y = 1  ->  (0.5, 0.5)
        let p = dm(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let q = dv(&[0.0, 0.0]);
        let a_eq = dm(1, 2, &[1.0, 1.0]);
        let b_eq = dv(&[1.0]);
        let x = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap();
        assert!(
            (x[0] - 0.5).abs() < 1e-6 && (x[1] - 0.5).abs() < 1e-6,
            "got x = {:?}",
            x
        );
    }

    #[test]
    fn test_minimize_with_inequality() {
        // minimize x² + y²  s.t. x ≥ 1  ->  x = 1, y = 0
        let p = dm(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let q = dv(&[0.0, 0.0]);
        let a_eq = dm(0, 2, &[]);
        let b_eq = dv(&[]);
        let x = solve_qp(
            &p,
            &q,
            &a_eq,
            &b_eq,
            &[(1.0, f64::INFINITY), (0.0, f64::INFINITY)],
        )
        .unwrap();
        assert!(
            (x[0] - 1.0).abs() < 1e-4 && (x[1] - 0.0).abs() < 1e-4,
            "got x = {:?}",
            x
        );
    }

    #[test]
    fn test_builder_twoside_bounds() {
        // minimize x²  s.t. 2 ≤ x ≤ 3  ->  x = 2
        let qp = QuadraticProgram::new(dv(&[0.0]))
            .objective(dm(1, 1, &[2.0]))
            .bounds(&[(2.0, 3.0)]);
        let sol = qp.solve().unwrap();
        assert!((sol.x[0] - 2.0).abs() < 1e-6, "got x = {}", sol.x[0]);
    }

    #[test]
    fn test_inequality_constraints() {
        // minimize x² + y²  s.t. x + y ≤ 1, x ≥ 0, y ≥ 0
        // unconstrained min (0,0) already feasible -> (0, 0)
        let p = dm(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let q = dv(&[0.0, 0.0]);
        let a_ineq = dm(1, 2, &[1.0, 1.0]);
        let b_ineq = dv(&[1.0]);
        let sol = solve_qp_internal(
            &p,
            &q,
            &dm(0, 2, &[]),
            &dv(&[]),
            &a_ineq,
            &b_ineq,
            &[],
        )
        .unwrap();
        assert!(
            sol.x[0].abs() < 1e-6 && sol.x[1].abs() < 1e-6,
            "got x = {:?}",
            sol.x
        );
    }

    #[test]
    fn test_dimension_error() {
        let p = dm(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let q = dv(&[0.0]); // wrong length
        let a_eq = dm(1, 2, &[1.0, 1.0]);
        let b_eq = dv(&[1.0]);
        let err = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap_err();
        assert!(matches!(err, ConvexError::DimensionMismatch { .. }));
    }

    #[test]
    fn test_nonfinite_error() {
        let p = dm(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let q = dv(&[f64::NAN, 0.0]);
        let a_eq = dm(1, 2, &[1.0, 1.0]);
        let b_eq = dv(&[1.0]);
        let err = solve_qp(&p, &q, &a_eq, &b_eq, &[]).unwrap_err();
        assert!(matches!(err, ConvexError::NotFinite { .. }));
    }

    #[test]
    fn test_infeasible_qp() {
        // x ≥ 2 and x ≤ 1: empty feasible set.
        let p = dm(1, 1, &[2.0]);
        let q = dv(&[0.0]);
        let a_eq = dm(0, 1, &[]);
        let b_eq = dv(&[]);
        let x = solve_qp(&p, &q, &a_eq, &b_eq, &[(2.0, 1.0)]);
        assert!(x.is_err(), "expected infeasible error, got {:?}", x);
    }

    #[test]
    fn test_unbounded_qp() {
        // minimize -x  s.t. x ≥ 1: objective unbounded below as x → +∞.
        let p = dm(1, 1, &[0.0]);
        let q = dv(&[-1.0]);
        let a_eq = dm(0, 1, &[]);
        let b_eq = dv(&[]);
        let x = solve_qp(&p, &q, &a_eq, &b_eq, &[(1.0, f64::INFINITY)]);
        assert!(x.is_err(), "expected unbounded error, got {:?}", x);
    }

    #[test]
    fn test_random_qp_kkt_conditions() {
        // Random PSD P, random q, box bounds. Verify KKT conditions at the
        // returned optimum (a cross-check that the solve is correct).
        let n = 5;
        let mut rng_state: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            // xorshift64*
            rng_state ^= rng_state >> 12;
            rng_state ^= rng_state << 25;
            rng_state ^= rng_state >> 27;
            ((rng_state.wrapping_mul(0x2545_f491_4f6c_dd1d)) >> 11) as f64 / (1u64 << 53) as f64
        };

        // Build PSD P = Mᵀ M + I.
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for k in 0..n {
                m[i][k] = rng() * 2.0 - 1.0;
            }
        }
        let mut p = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut v = if i == j { 1.0 } else { 0.0 };
                for k in 0..n {
                    v += m[k][i] * m[k][j];
                }
                p[i][j] = v;
            }
        }
        let q: Vec<f64> = (0..n).map(|_| rng() * 2.0 - 1.0).collect();
        let p_mat = dm(n, n, &p.iter().flatten().copied().collect::<Vec<_>>());
        let q_vec = dv(&q);

        let bounds: Vec<(f64, f64)> = (0..n).map(|_| (-5.0, 5.0)).collect();
        let sol = solve_qp(&p_mat, &q_vec, &dm(0, n, &[]), &dv(&[]), &bounds).unwrap();
        let x = sol.iter().copied().collect::<Vec<_>>();

        // Stationarity: P x + q + Aᵀ z = 0 on the interior of active bounds.
        // For each variable, if strictly inside the box, the gradient must be 0.
        for i in 0..n {
            let mut grad = q[i];
            for k in 0..n {
                grad += p[i][k] * x[k];
            }
            let inside = x[i] > -5.0 + 1e-6 && x[i] < 5.0 - 1e-6;
            if inside {
                assert!(
                    grad.abs() < 1e-4,
                    "variable {i} interior but gradient = {grad} (x = {})",
                    x[i]
                );
            } else {
                // At a bound: sign of gradient must push back into the feasible
                // region (complementarity).
                let at_lower = (x[i] + 5.0_f64).abs() < 1e-4;
                let at_upper = (5.0_f64 - x[i]).abs() < 1e-4;
                if at_lower {
                    assert!(grad > -1e-4, "at lower bound but gradient = {grad}");
                }
                if at_upper {
                    assert!(grad < 1e-4, "at upper bound but gradient = {grad}");
                }
            }
        }
        // Feasibility.
        for &xi in &x {
            assert!(xi >= -5.0 - 1e-6 && xi <= 5.0 + 1e-6, "infeasible x = {xi}");
        }
    }
}
