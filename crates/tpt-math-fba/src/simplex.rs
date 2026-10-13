//! Bounded-variable revised simplex on a sparse LU-factored basis.
//!
//! Phase 1 starts from an all-artificial basis (one artificial per row, sign
//! chosen so it is feasible at the current non-basic point) and minimises the
//! sum of artificials. Phase 2 then minimises the real objective with the
//! artificials frozen at zero. Pricing is Dantzig's rule, falling back to
//! Bland's rule after a run of degenerate pivots to break cycling.
//!
//! The basis is refactorised on every iteration (`B` for FTRAN and `Bᵀ` for
//! BTRAN). That keeps the code simple and numerically clean at the cost of
//! speed; a Forrest–Tomlin update is the planned optimisation.

use tpt_math_linalg_dense::DVector;
use tpt_math_linalg_sparse::{CooMatrix, SparseLu};

use crate::error::FbaError;
use crate::lp::{LpSolution, LpSolver, LpStatus, SparseLp};
use crate::presolve::{scale, Scaled};

const PIV_TOL: f64 = 1e-9;
const BLAND_AFTER: usize = 50;
const TIE_TOL: f64 = 1e-12;

/// Tolerances and limits for [`SimplexSolver`].
#[derive(Clone, Debug, PartialEq)]
pub struct SimplexOptions {
    /// Maximum number of pivots and bound flips across both phases.
    pub max_iter: usize,
    /// Primal feasibility tolerance (relative to the size of `b`).
    pub feas_tol: f64,
    /// Dual feasibility (reduced-cost) tolerance.
    pub opt_tol: f64,
    /// Number of geometric-mean scaling passes; `0` disables scaling.
    pub scale_passes: usize,
}

impl Default for SimplexOptions {
    fn default() -> Self {
        SimplexOptions {
            max_iter: 100_000,
            feas_tol: 1e-9,
            opt_tol: 1e-9,
            scale_passes: 4,
        }
    }
}

/// Bounded-variable revised simplex solver.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SimplexSolver {
    /// Solver options.
    pub options: SimplexOptions,
}

impl SimplexSolver {
    /// Solver with [`SimplexOptions::default`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Solver with the given options.
    pub fn with_options(options: SimplexOptions) -> Self {
        SimplexSolver { options }
    }
}

impl LpSolver for SimplexSolver {
    fn solve(&self, lp: &SparseLp) -> Result<LpSolution, FbaError> {
        lp.validate()?;
        let Scaled {
            lp: scaled,
            row,
            col,
        } = scale(lp, self.options.scale_passes);
        let mut sol = solve_scaled(&self.options, &scaled)?;
        for (x, s) in sol.x.iter_mut().zip(&col) {
            *x *= s;
        }
        for (d, s) in sol.reduced_costs.iter_mut().zip(&col) {
            *d /= s;
        }
        for (y, r) in sol.duals.iter_mut().zip(&row) {
            *y *= r;
        }
        sol.objective = lp.c.iter().zip(&sol.x).map(|(c, x)| c * x).sum();
        Ok(sol)
    }
}

/// Working state shared by both phases.
struct State<'a> {
    opts: &'a SimplexOptions,
    m: usize,
    /// Column-wise constraint matrix, including the artificial columns.
    cols: Vec<Vec<(usize, f64)>>,
    b: Vec<f64>,
    lo: Vec<f64>,
    hi: Vec<f64>,
    /// Current value of every variable (structural, then artificial).
    x: Vec<f64>,
    /// `basis[k]` is the variable basic in row `k`.
    basis: Vec<usize>,
    is_basic: Vec<bool>,
    iterations: usize,
    degenerate: usize,
}

fn solve_scaled(opts: &SimplexOptions, lp: &SparseLp) -> Result<LpSolution, FbaError> {
    let m = lp.a.nrows();
    let n = lp.a.ncols();
    let total = n + m;

    let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    for (r, c, &v) in lp.a.iter() {
        cols[c].push((r, v));
    }
    let mut lo = lp.lb.clone();
    lo.resize(total, 0.0);
    let mut hi = lp.ub.clone();
    hi.resize(total, f64::INFINITY);

    // Structurals start at a finite bound (or zero for free variables).
    let mut x = vec![0.0; total];
    for j in 0..n {
        x[j] = if lo[j].is_finite() {
            lo[j]
        } else if hi[j].is_finite() {
            hi[j]
        } else {
            0.0
        };
    }
    // Artificial for row i absorbs the residual `b - A x_N` with a sign that
    // makes it non-negative, so the initial basis is feasible for phase 1.
    let mut resid = lp.b.clone();
    for j in 0..n {
        for &(i, v) in &cols[j] {
            resid[i] -= v * x[j];
        }
    }
    let mut basis = Vec::with_capacity(m);
    for i in 0..m {
        let s = if resid[i] >= 0.0 { 1.0 } else { -1.0 };
        cols.push(vec![(i, s)]);
        x[n + i] = resid[i].abs();
        basis.push(n + i);
    }
    let mut is_basic = vec![false; total];
    for &j in &basis {
        is_basic[j] = true;
    }

    let mut st = State {
        opts,
        m,
        cols,
        b: lp.b.clone(),
        lo,
        hi,
        x,
        basis,
        is_basic,
        iterations: 0,
        degenerate: 0,
    };

    // Phase 1: minimise the sum of artificials.
    let mut cost1 = vec![0.0; total];
    cost1[n..].iter_mut().for_each(|c| *c = 1.0);
    let (status1, _) = st.iterate(&cost1)?;
    if status1 == LpStatus::IterationLimit {
        return Ok(finish(&st, n, status1, &lp.c, Vec::new()));
    }
    let scale_b = 1.0 + lp.b.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let infeasibility: f64 = st.x[n..].iter().sum();
    if infeasibility > opts.feas_tol * scale_b {
        return Ok(finish(&st, n, LpStatus::Infeasible, &lp.c, Vec::new()));
    }

    // Phase 2: freeze artificials at zero and optimise the real objective.
    for j in n..total {
        st.hi[j] = 0.0;
    }
    let mut cost2 = vec![0.0; total];
    cost2[..n].copy_from_slice(&lp.c);
    let (status2, y) = st.iterate(&cost2)?;
    Ok(finish(&st, n, status2, &lp.c, y))
}

/// Build the structural part of the solution. Duals and reduced costs are
/// only meaningful at an optimum, so they are zeroed otherwise.
fn finish(st: &State<'_>, n: usize, status: LpStatus, c: &[f64], y: Vec<f64>) -> LpSolution {
    let optimal = status == LpStatus::Optimal;
    let (duals, reduced_costs) = if optimal {
        let d = (0..n)
            .map(|j| c[j] - st.cols[j].iter().map(|&(i, v)| y[i] * v).sum::<f64>())
            .collect();
        (y, d)
    } else {
        (vec![0.0; st.m], vec![0.0; n])
    };
    LpSolution {
        status,
        objective: c.iter().zip(&st.x).map(|(c, x)| c * x).sum(),
        x: st.x[..n].to_vec(),
        duals,
        reduced_costs,
        iterations: st.iterations,
    }
}

fn factor_basis(
    cols: &[Vec<(usize, f64)>],
    basis: &[usize],
    m: usize,
) -> Result<(SparseLu, SparseLu), FbaError> {
    let (mut r, mut c, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for (k, &j) in basis.iter().enumerate() {
        for &(i, val) in &cols[j] {
            r.push(i);
            c.push(k);
            v.push(val);
        }
    }
    let bt = CooMatrix::from_triplets(m, m, c.clone(), r.clone(), v.clone());
    let b = CooMatrix::from_triplets(m, m, r, c, v);
    let lu = SparseLu::factor(&b.to_csr()).map_err(|_| FbaError::SingularBasis)?;
    let lut = SparseLu::factor(&bt.to_csr()).map_err(|_| FbaError::SingularBasis)?;
    Ok((lu, lut))
}

fn solve_with(lu: &SparseLu, rhs: Vec<f64>) -> Result<Vec<f64>, FbaError> {
    lu.solve(&DVector::from_vec(rhs))
        .map(|v| v.as_slice().to_vec())
        .map_err(|_| FbaError::SingularBasis)
}

impl State<'_> {
    /// Run simplex iterations for the given cost vector until no improving
    /// column exists. Returns the status and the final duals `y`.
    fn iterate(&mut self, cost: &[f64]) -> Result<(LpStatus, Vec<f64>), FbaError> {
        let total = cost.len();
        loop {
            if self.iterations >= self.opts.max_iter {
                return Ok((LpStatus::IterationLimit, Vec::new()));
            }
            let (lu, lut) = factor_basis(&self.cols, &self.basis, self.m)?;

            // Basic values from the non-basic point: B x_B = b - N x_N.
            let mut rhs = self.b.clone();
            for j in 0..total {
                if self.is_basic[j] || self.x[j] == 0.0 {
                    continue;
                }
                for &(i, v) in &self.cols[j] {
                    rhs[i] -= v * self.x[j];
                }
            }
            let xb = solve_with(&lu, rhs)?;
            for (k, &j) in self.basis.iter().enumerate() {
                self.x[j] = xb[k];
            }

            // Duals: Bᵀ y = c_B.
            let cb: Vec<f64> = self.basis.iter().map(|&j| cost[j]).collect();
            let y = solve_with(&lut, cb)?;

            let bland = self.degenerate >= BLAND_AFTER;
            let mut enter: Option<(usize, f64)> = None;
            let mut best_merit = 0.0;
            for (j, &cj) in cost.iter().enumerate() {
                if self.is_basic[j] || self.lo[j] == self.hi[j] {
                    continue;
                }
                let d = cj - self.cols[j].iter().map(|&(i, v)| y[i] * v).sum::<f64>();
                let (dir, merit) = if d < -self.opts.opt_tol && self.x[j] < self.hi[j] {
                    (1.0, -d)
                } else if d > self.opts.opt_tol && self.x[j] > self.lo[j] {
                    (-1.0, d)
                } else {
                    continue;
                };
                if bland {
                    enter = Some((j, dir));
                    break;
                }
                if merit > best_merit {
                    best_merit = merit;
                    enter = Some((j, dir));
                }
            }
            let Some((q, dir)) = enter else {
                return Ok((LpStatus::Optimal, y));
            };

            // Direction of the basic variables: alpha = B⁻¹ a_q.
            let mut aq = vec![0.0; self.m];
            for &(i, v) in &self.cols[q] {
                aq[i] = v;
            }
            let alpha = solve_with(&lu, aq)?;

            // Ratio test. The entering variable's own range is a bound flip.
            let mut t_best = f64::INFINITY;
            if self.lo[q].is_finite() && self.hi[q].is_finite() {
                t_best = self.hi[q] - self.lo[q];
            }
            let mut leave: Option<(usize, f64)> = None;
            let mut best_delta = 0.0f64;
            for (k, &ak) in alpha.iter().enumerate() {
                let delta = -dir * ak;
                if delta.abs() <= PIV_TOL {
                    continue;
                }
                let jb = self.basis[k];
                let (ratio, bound) = if delta < 0.0 {
                    if !self.lo[jb].is_finite() {
                        continue;
                    }
                    (((self.x[jb] - self.lo[jb]) / -delta).max(0.0), self.lo[jb])
                } else {
                    if !self.hi[jb].is_finite() {
                        continue;
                    }
                    (((self.hi[jb] - self.x[jb]) / delta).max(0.0), self.hi[jb])
                };
                let take = if ratio < t_best - TIE_TOL {
                    true
                } else if (ratio - t_best).abs() <= TIE_TOL {
                    match leave {
                        Some((kl, _)) if bland => jb < self.basis[kl],
                        Some(_) => delta.abs() > best_delta,
                        None => false,
                    }
                } else {
                    false
                };
                if take {
                    t_best = ratio;
                    leave = Some((k, bound));
                    best_delta = delta.abs();
                }
            }
            if !t_best.is_finite() {
                return Ok((LpStatus::Unbounded, y));
            }

            self.degenerate = if t_best <= self.opts.feas_tol {
                self.degenerate + 1
            } else {
                0
            };
            match leave {
                None => {
                    self.x[q] = if dir > 0.0 { self.hi[q] } else { self.lo[q] };
                }
                Some((k, bound)) => {
                    let jb = self.basis[k];
                    self.x[jb] = bound;
                    self.is_basic[jb] = false;
                    self.basis[k] = q;
                    self.is_basic[q] = true;
                    self.x[q] += dir * t_best;
                }
            }
            self.iterations += 1;
        }
    }
}
