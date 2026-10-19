//! Linear programming by the two-phase simplex method.
//!
//! Solves
//!
//! ```text
//! maximize    cᵀ x
//! subject to  A x ≤ b,   x ≥ 0
//! ```
//!
//! for a dense `A`. Rows with a negative right-hand side are handled by a
//! phase-one search for a feasible basis. Bland's rule picks the entering and
//! leaving variables, so the method cannot cycle on degenerate problems.
//! Minimisation is `maximize` applied to `-c`.

/// Outcome of a linear program.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LpStatus {
    /// An optimal vertex was found.
    Optimal,
    /// No `x ≥ 0` satisfies `A x ≤ b`.
    Infeasible,
    /// The objective grows without bound over the feasible region.
    Unbounded,
}

/// Result of [`maximize`].
#[derive(Clone, Debug, PartialEq)]
pub struct LpSolution {
    /// Whether the problem is optimal, infeasible or unbounded.
    pub status: LpStatus,
    /// The optimal point. Empty unless `status` is [`LpStatus::Optimal`].
    pub x: Vec<f64>,
    /// `cᵀ x` at the optimum, or `NaN` if there is none.
    pub objective: f64,
}

/// Errors from [`maximize`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LpError {
    /// `A` does not have one row per entry of `b`, or a row does not have one
    /// entry per variable in `c`.
    DimensionMismatch,
}

impl core::fmt::Display for LpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LpError::DimensionMismatch => write!(f, "constraint matrix dimensions do not match"),
        }
    }
}

impl std::error::Error for LpError {}

/// Maximise `cᵀ x` subject to `A x ≤ b` and `x ≥ 0`.
///
/// `a` holds one row per constraint. `b` may have negative entries.
///
/// # Errors
///
/// Returns [`LpError::DimensionMismatch`] if `a` and `b` disagree in length,
/// or if a row of `a` does not have `c.len()` entries.
///
/// # Examples
///
/// ```
/// use tpt_math_optimize_general::linear_program::{maximize, LpStatus};
///
/// // maximize 3x + 5y  s.t.  x ≤ 4, 2y ≤ 12, 3x + 2y ≤ 18  →  (2, 6), 36.
/// let c = [3.0, 5.0];
/// let a = [vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]];
/// let sol = maximize(&c, &a, &[4.0, 12.0, 18.0]).unwrap();
/// assert_eq!(sol.status, LpStatus::Optimal);
/// assert!((sol.objective - 36.0).abs() < 1e-9);
/// ```
pub fn maximize(c: &[f64], a: &[Vec<f64>], b: &[f64]) -> Result<LpSolution, LpError> {
    let n = c.len();
    let m = a.len();
    if b.len() != m || a.iter().any(|row| row.len() != n) {
        return Err(LpError::DimensionMismatch);
    }

    // Columns are the n variables, then one slack per row, then one artificial
    // per row whose right-hand side is negative. The last column is the rhs.
    let n_art = b.iter().filter(|&&v| v < 0.0).count();
    let ncols = n + m + n_art;
    let rhs = ncols;
    let mut rows = Vec::with_capacity(m);
    let mut basis = Vec::with_capacity(m);
    let mut next_art = n + m;
    for i in 0..m {
        // A row with b < 0 is negated, so its slack enters with coefficient -1
        // and an artificial variable supplies the initial basic column.
        let sign = if b[i] < 0.0 { -1.0 } else { 1.0 };
        let mut row = vec![0.0; ncols + 1];
        for j in 0..n {
            row[j] = sign * a[i][j];
        }
        row[n + i] = sign;
        row[rhs] = sign * b[i];
        if sign < 0.0 {
            row[next_art] = 1.0;
            basis.push(next_art);
            next_art += 1;
        } else {
            basis.push(n + i);
        }
        rows.push(row);
    }
    let mut t = Tableau {
        rows,
        reduced: Vec::new(),
        basis,
    };

    // Phase one: drive the artificial variables to zero.
    if n_art > 0 {
        let mut cost1 = vec![0.0; ncols];
        for cost in &mut cost1[n + m..] {
            *cost = -1.0;
        }
        t.price(&cost1);
        // The phase-one objective is bounded above by zero, so this cannot
        // report unboundedness.
        t.optimise(ncols);
        // The objective is -reduced[rhs]; a strictly negative optimum means the
        // artificials cannot all vanish.
        if t.reduced[rhs] > EPS {
            return Ok(no_solution(LpStatus::Infeasible));
        }
        // Pivot any artificial still basic (at value zero) onto a real column.
        // A row with no such column is redundant and its artificial stays put.
        for i in 0..t.rows.len() {
            if t.basis[i] >= n + m {
                if let Some(j) = (0..n + m).find(|&j| t.rows[i][j].abs() > EPS) {
                    t.pivot(i, j);
                }
            }
        }
    }

    // Phase two: optimise the real objective. Artificial columns may not enter.
    let mut cost2 = vec![0.0; ncols];
    cost2[..n].copy_from_slice(c);
    t.price(&cost2);
    if !t.optimise(n + m) {
        return Ok(no_solution(LpStatus::Unbounded));
    }

    let mut x = vec![0.0; n];
    for (i, &col) in t.basis.iter().enumerate() {
        if col < n {
            x[col] = t.rows[i][rhs];
        }
    }
    let objective = c.iter().zip(&x).map(|(ci, xi)| ci * xi).sum();
    Ok(LpSolution {
        status: LpStatus::Optimal,
        x,
        objective,
    })
}

/// Tolerance for pivot selection and feasibility checks.
const EPS: f64 = 1e-9;

fn no_solution(status: LpStatus) -> LpSolution {
    LpSolution {
        status,
        x: Vec::new(),
        objective: f64::NAN,
    }
}

/// Dense simplex tableau in canonical form for the current basis.
struct Tableau {
    /// One row per constraint; the last entry of each row is the rhs.
    rows: Vec<Vec<f64>>,
    /// Reduced costs `c_j − c_Bᵀ B⁻¹ A_j`, with the negated objective value in
    /// the last slot. A positive entry improves the objective if it enters.
    reduced: Vec<f64>,
    /// The column that is basic in each row.
    basis: Vec<usize>,
}

impl Tableau {
    fn rhs_col(&self) -> usize {
        self.reduced.len() - 1
    }

    /// Recomputes the reduced costs for `cost` (one entry per column, no rhs)
    /// under the current basis.
    fn price(&mut self, cost: &[f64]) {
        self.reduced = cost.iter().copied().chain(core::iter::once(0.0)).collect();
        for (row, &b) in self.rows.iter().zip(&self.basis) {
            let cb = cost[b];
            if cb != 0.0 {
                for (r, x) in self.reduced.iter_mut().zip(row) {
                    *r -= cb * x;
                }
            }
        }
    }

    /// Makes column `col` basic in row `r` by Gaussian elimination.
    fn pivot(&mut self, r: usize, col: usize) {
        let p = self.rows[r][col];
        for x in self.rows[r].iter_mut() {
            *x /= p;
        }
        let pivot_row = self.rows[r].clone();
        for (i, row) in self.rows.iter_mut().enumerate() {
            let f = row[col];
            if i != r && f != 0.0 {
                for (x, y) in row.iter_mut().zip(&pivot_row) {
                    *x -= f * y;
                }
            }
        }
        let f = self.reduced[col];
        if f != 0.0 {
            for (x, y) in self.reduced.iter_mut().zip(&pivot_row) {
                *x -= f * y;
            }
        }
        self.basis[r] = col;
    }

    /// Runs simplex iterations. Only columns `< eligible` may enter the basis.
    /// Returns `false` if the objective is unbounded.
    fn optimise(&mut self, eligible: usize) -> bool {
        let rhs = self.rhs_col();
        loop {
            // Bland's rule: the lowest-index improving column enters.
            let Some(e) = (0..eligible).find(|&j| self.reduced[j] > EPS) else {
                return true;
            };
            // Minimum-ratio test; ties go to the lowest-index basic column.
            let mut leave: Option<usize> = None;
            for i in 0..self.rows.len() {
                let a = self.rows[i][e];
                if a <= EPS {
                    continue;
                }
                let better = match leave {
                    None => true,
                    Some(l) => {
                        let ratio = self.rows[i][rhs] / a;
                        let best = self.rows[l][rhs] / self.rows[l][e];
                        ratio < best - EPS || (ratio <= best + EPS && self.basis[i] < self.basis[l])
                    }
                };
                if better {
                    leave = Some(i);
                }
            }
            let Some(r) = leave else {
                return false;
            };
            self.pivot(r, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classic_two_variable_problem() {
        let c = [3.0, 5.0];
        let a = [vec![1.0, 0.0], vec![0.0, 2.0], vec![3.0, 2.0]];
        let sol = maximize(&c, &a, &[4.0, 12.0, 18.0]).unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-9);
        assert!((sol.x[1] - 6.0).abs() < 1e-9);
        assert!((sol.objective - 36.0).abs() < 1e-9);
    }

    #[test]
    fn negative_rhs_is_handled_by_phase_one() {
        // maximize -x  s.t.  -x ≤ -2 (so x ≥ 2) and x ≤ 5  →  x = 2, value -2.
        let sol = maximize(&[-1.0], &[vec![-1.0], vec![1.0]], &[-2.0, 5.0]).unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((sol.x[0] - 2.0).abs() < 1e-9);
        assert!((sol.objective + 2.0).abs() < 1e-9);
    }

    #[test]
    fn detects_infeasible() {
        // x ≥ 1 and x ≤ 0.5.
        let sol = maximize(&[1.0], &[vec![-1.0], vec![1.0]], &[-1.0, 0.5]).unwrap();
        assert_eq!(sol.status, LpStatus::Infeasible);
        assert!(sol.x.is_empty());
    }

    #[test]
    fn detects_unbounded() {
        // maximize x with no upper bound on x.
        let sol = maximize(&[1.0], &[vec![-1.0]], &[0.0]).unwrap();
        assert_eq!(sol.status, LpStatus::Unbounded);
    }

    #[test]
    fn degenerate_vertex_terminates() {
        // Three constraints meet at the optimum (1, 1): a degenerate vertex.
        let c = [1.0, 1.0];
        let a = [vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let sol = maximize(&c, &a, &[1.0, 1.0, 2.0]).unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((sol.objective - 2.0).abs() < 1e-9);
    }

    #[test]
    fn minimisation_by_negated_objective() {
        // minimize x + y  s.t.  x + y ≥ 3 (as -x - y ≤ -3), x ≤ 10, y ≤ 10.
        let a = [vec![-1.0, -1.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let sol = maximize(&[-1.0, -1.0], &a, &[-3.0, 10.0, 10.0]).unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!((-sol.objective - 3.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_dimension_mismatch() {
        assert_eq!(
            maximize(&[1.0, 2.0], &[vec![1.0]], &[1.0]),
            Err(LpError::DimensionMismatch)
        );
        assert_eq!(
            maximize(&[1.0], &[vec![1.0], vec![1.0]], &[1.0]),
            Err(LpError::DimensionMismatch)
        );
    }
}
