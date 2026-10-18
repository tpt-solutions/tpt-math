//! Fill-reducing orderings for sparse symmetric factorisations.
//!
//! The sparsity pattern of a Cholesky factor depends on the order in which the
//! unknowns are numbered. A poor numbering can fill the factor in almost
//! completely, while a banded numbering keeps it close to the original
//! pattern. [`reverse_cuthill_mckee`] computes a bandwidth-reducing numbering,
//! and [`ReorderedCholesky`] factors a matrix in that numbering and hides the
//! permutation from the caller.
//!
//! RCM is a bandwidth heuristic, not the minimum-degree ordering that AMD
//! provides. On irregular meshes AMD usually gives less fill, and it is not
//! implemented here.

use alloc::{
    collections::{BTreeSet, VecDeque},
    vec,
    vec::Vec,
};

use tpt_math_linalg_dense::DVector;

use crate::{CooMatrix, CsrMatrix, SparseCholesky, SparseError, SparseLu};

/// The adjacency lists of the symmetrised pattern of `a`, without self-loops.
fn symmetric_adjacency(a: &CsrMatrix<f64>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, j, _) in a.iter() {
        if i != j {
            adj[i].push(j);
            adj[j].push(i);
        }
    }
    for list in &mut adj {
        list.sort_unstable();
        list.dedup();
    }
    adj
}

/// Breadth-first levels from `start`, returned as `(last_level_nodes, depth)`.
fn bfs_levels(adj: &[Vec<usize>], start: usize, seen: &mut [bool]) -> (Vec<usize>, usize) {
    let mut frontier = vec![start];
    let mut visited = vec![start];
    seen[start] = true;
    let mut depth = 0;
    let mut last = frontier.clone();
    while !frontier.is_empty() {
        last = frontier.clone();
        let mut next = Vec::new();
        for &u in &frontier {
            for &v in &adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    next.push(v);
                    visited.push(v);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        depth += 1;
        frontier = next;
    }
    // Undo the marks so the caller can reuse `seen` for the real traversal.
    for v in visited {
        seen[v] = false;
    }
    (last, depth)
}

/// A pseudo-peripheral node of the component containing `start`: a node at
/// (near-)maximal eccentricity, found by repeated breadth-first searches from
/// the lowest-degree node of the last level.
fn pseudo_peripheral(adj: &[Vec<usize>], start: usize) -> usize {
    let mut seen = vec![false; adj.len()];
    let mut node = start;
    let (mut last, mut depth) = bfs_levels(adj, node, &mut seen);
    loop {
        let candidate = *last.iter().min_by_key(|&&v| adj[v].len()).unwrap_or(&node);
        let (next_last, next_depth) = bfs_levels(adj, candidate, &mut seen);
        if next_depth > depth {
            node = candidate;
            last = next_last;
            depth = next_depth;
        } else {
            return candidate;
        }
    }
}

/// A reverse Cuthill–McKee ordering of the symmetrised pattern of `a`.
///
/// Returns `perm` with `perm[new] = old`: the unknown that moves to position
/// `new` in the reordered system. Components are numbered one after another,
/// each starting from a pseudo-peripheral node. The result is the reverse of
/// the Cuthill–McKee order, which tends to keep factor fill small.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{reverse_cuthill_mckee, CooMatrix};
///
/// // A path 0-2-1-3 has bandwidth 2 in its natural order and 1 after RCM.
/// let mut coo = CooMatrix::<f64>::new(4, 4);
/// for &(i, j) in &[(0, 2), (2, 1), (1, 3)] {
///     coo.push(i, j, 1.0);
///     coo.push(j, i, 1.0);
/// }
/// let perm = reverse_cuthill_mckee(&coo.to_csr()).unwrap();
/// assert_eq!(perm.len(), 4);
/// ```
pub fn reverse_cuthill_mckee(a: &CsrMatrix<f64>) -> Result<Vec<usize>, SparseError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SparseError::DimensionMismatch);
    }
    let adj = symmetric_adjacency(a);
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    // Start each component from its lowest-degree node, refined to a
    // pseudo-peripheral one. Scanning in index order keeps the result
    // deterministic.
    for seed in 0..n {
        if visited[seed] {
            continue;
        }
        let start = pseudo_peripheral(&adj, seed);
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(u) = queue.pop_front() {
            order.push(u);
            // Visit unvisited neighbours in order of increasing degree.
            let mut neighbours: Vec<usize> =
                adj[u].iter().copied().filter(|&v| !visited[v]).collect();
            neighbours.sort_by_key(|&v| adj[v].len());
            for v in neighbours {
                visited[v] = true;
                queue.push_back(v);
            }
        }
    }
    order.reverse();
    Ok(order)
}

/// A minimum-degree ordering of the symmetrised pattern of `a`.
///
/// Repeatedly eliminates the unnumbered node of smallest current degree in the
/// elimination graph, joining its remaining neighbours into a clique. This is
/// the minimum-degree idea that AMD approximates. Degrees here are exact, so
/// the cost grows with the square of the fill. That is fine for matrices of a
/// few thousand unknowns, but not for very large problems. Ties go to the lowest
/// index, so the result is deterministic.
///
/// Returns `perm` with `perm[new] = old`, the elimination order.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square.
///
/// # Examples
///
/// ```
/// use tpt_math_linalg_sparse::{minimum_degree_ordering, CooMatrix};
///
/// // An arrow matrix: the hub (node 0) should be eliminated last.
/// let n = 5;
/// let mut coo = CooMatrix::<f64>::new(n, n);
/// for i in 0..n {
///     coo.push(i, i, 2.0);
///     coo.push(0, i, 1.0);
///     coo.push(i, 0, 1.0);
/// }
/// let perm = minimum_degree_ordering(&coo.to_csr()).unwrap();
/// assert_eq!(*perm.last().unwrap(), 0);
/// ```
pub fn minimum_degree_ordering(a: &CsrMatrix<f64>) -> Result<Vec<usize>, SparseError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(SparseError::DimensionMismatch);
    }
    let mut adj: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for (i, j, _) in a.iter() {
        if i != j {
            adj[i].insert(j);
            adj[j].insert(i);
        }
    }
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let next = (0..n)
            .filter(|&v| !eliminated[v])
            .min_by_key(|&v| (adj[v].len(), v))
            .ok_or(SparseError::DimensionMismatch)?;
        eliminated[next] = true;
        order.push(next);
        let neighbours: Vec<usize> = adj[next].iter().copied().collect();
        for &u in &neighbours {
            adj[u].remove(&next);
        }
        // Eliminating `next` makes its neighbours a clique.
        for (idx, &u) in neighbours.iter().enumerate() {
            for &v in &neighbours[idx + 1..] {
                adj[u].insert(v);
                adj[v].insert(u);
            }
        }
        adj[next].clear();
    }
    Ok(order)
}

/// The symmetric permutation `P A Pᵀ` of `a`, where `perm[new] = old`.
///
/// Entry `(i, j)` of `a` moves to `(new(i), new(j))`, with `new` the inverse of
/// `perm`.
///
/// # Errors
///
/// Returns [`SparseError::DimensionMismatch`] if `a` is not square or `perm`
/// is not a permutation of `0..n`.
pub fn permute_symmetric(
    a: &CsrMatrix<f64>,
    perm: &[usize],
) -> Result<CsrMatrix<f64>, SparseError> {
    let n = a.nrows();
    if a.ncols() != n || perm.len() != n {
        return Err(SparseError::DimensionMismatch);
    }
    let mut new_of = vec![usize::MAX; n];
    for (new, &old) in perm.iter().enumerate() {
        if old >= n || new_of[old] != usize::MAX {
            return Err(SparseError::DimensionMismatch);
        }
        new_of[old] = new;
    }
    let mut coo = CooMatrix::new(n, n);
    for (i, j, &v) in a.iter() {
        coo.push(new_of[i], new_of[j], v);
    }
    Ok(coo.to_csr())
}

/// A sparse Cholesky factorisation computed in a reverse Cuthill–McKee order.
///
/// Reordering is applied and undone internally, so [`solve`](Self::solve) takes
/// and returns vectors in the original numbering.
#[derive(Clone, Debug)]
pub struct ReorderedCholesky {
    /// `perm[new] = old`.
    perm: Vec<usize>,
    factor: SparseCholesky,
}

impl ReorderedCholesky {
    /// Factor the SPD matrix `a` after an RCM reordering.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`SparseCholesky::factor`] for the permuted matrix,
    /// and [`SparseError::DimensionMismatch`] for a non-square `a`.
    pub fn factor(a: &CsrMatrix<f64>) -> Result<Self, SparseError> {
        let perm = reverse_cuthill_mckee(a)?;
        let permuted = permute_symmetric(a, &perm)?;
        let factor = SparseCholesky::factor(&permuted)?;
        Ok(ReorderedCholesky { perm, factor })
    }

    /// The permutation used, with `perm[new] = old`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Nonzeros in the Cholesky factor. Compare with an unordered factor to see
    /// the fill reduction.
    pub fn nnz(&self) -> usize {
        self.factor.nnz()
    }

    /// Solve `A x = b` in the original numbering.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `b` has the wrong length.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, SparseError> {
        let n = self.perm.len();
        if b.len() != n {
            return Err(SparseError::DimensionMismatch);
        }
        let permuted_rhs = DVector::from_fn(n, |new| b[self.perm[new]]);
        let y = self.factor.solve(&permuted_rhs)?;
        // `inverse[old]` is the position of `old` in the permuted numbering.
        let mut inverse = vec![0; n];
        for (new, &old) in self.perm.iter().enumerate() {
            inverse[old] = new;
        }
        Ok(DVector::from_fn(n, |old| y[inverse[old]]))
    }
}

/// The fill-reducing ordering used to number the unknowns before a factorisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillOrdering {
    /// Reverse Cuthill–McKee: keeps the bandwidth small.
    ReverseCuthillMckee,
    /// Minimum degree: the greedy elimination ordering (see
    /// [`minimum_degree_ordering`]). Usually gives less fill than RCM.
    MinimumDegree,
}

impl FillOrdering {
    /// The permutation `perm[new] = old` for `a` under this ordering.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `a` is not square.
    pub fn permutation(self, a: &CsrMatrix<f64>) -> Result<Vec<usize>, SparseError> {
        match self {
            FillOrdering::ReverseCuthillMckee => reverse_cuthill_mckee(a),
            FillOrdering::MinimumDegree => minimum_degree_ordering(a),
        }
    }
}

impl ReorderedCholesky {
    /// Factor the SPD matrix `a` after a reordering chosen by `ordering`.
    ///
    /// # Errors
    ///
    /// As [`ReorderedCholesky::factor`].
    pub fn factor_with(a: &CsrMatrix<f64>, ordering: FillOrdering) -> Result<Self, SparseError> {
        let perm = ordering.permutation(a)?;
        let permuted = permute_symmetric(a, &perm)?;
        let factor = SparseCholesky::factor(&permuted)?;
        Ok(ReorderedCholesky { perm, factor })
    }
}

/// A sparse LU factorisation computed after a symmetric fill-reducing reordering.
///
/// The matrix is permuted as `P A Pᵀ` before partial-pivoting LU, so the same
/// numbering is applied to rows and columns. Pivoting can still move rows, but
/// the ordering keeps the structure close to the symmetric pattern.
#[derive(Debug)]
pub struct ReorderedLu {
    /// `perm[new] = old`.
    perm: Vec<usize>,
    lu: SparseLu,
}

impl ReorderedLu {
    /// Factor `a` after a reordering chosen by `ordering`.
    ///
    /// # Errors
    ///
    /// Returns the errors of [`SparseLu::factor`] for the permuted matrix, and
    /// [`SparseError::DimensionMismatch`] for a non-square `a`.
    pub fn factor(a: &CsrMatrix<f64>, ordering: FillOrdering) -> Result<Self, SparseError> {
        let perm = ordering.permutation(a)?;
        let permuted = permute_symmetric(a, &perm)?;
        let lu = SparseLu::factor(&permuted)?;
        Ok(ReorderedLu { perm, lu })
    }

    /// The permutation used, with `perm[new] = old`.
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Solve `A x = b` in the original numbering.
    ///
    /// # Errors
    ///
    /// Returns [`SparseError::DimensionMismatch`] if `b` has the wrong length.
    pub fn solve(&self, b: &DVector<f64>) -> Result<DVector<f64>, SparseError> {
        let n = self.perm.len();
        if b.len() != n {
            return Err(SparseError::DimensionMismatch);
        }
        let permuted_rhs = DVector::from_fn(n, |new| b[self.perm[new]]);
        let y = self.lu.solve(&permuted_rhs)?;
        let mut inverse = vec![0; n];
        for (new, &old) in self.perm.iter().enumerate() {
            inverse[old] = new;
        }
        Ok(DVector::from_fn(n, |old| y[inverse[old]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Symmetric tridiagonal (path) matrix with diagonal 4 and off-diagonal −1,
    /// with its unknowns scrambled by `shuffle`.
    fn scrambled_path(shuffle: &[usize]) -> CsrMatrix<f64> {
        let n = shuffle.len();
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(shuffle[i], shuffle[i], 4.0);
            if i + 1 < n {
                coo.push(shuffle[i], shuffle[i + 1], -1.0);
                coo.push(shuffle[i + 1], shuffle[i], -1.0);
            }
        }
        coo.to_csr()
    }

    fn is_permutation(p: &[usize]) -> bool {
        let mut seen = vec![false; p.len()];
        p.iter()
            .all(|&v| v < p.len() && !core::mem::replace(&mut seen[v], true))
    }

    #[test]
    fn rcm_returns_a_permutation() {
        let a = scrambled_path(&[3, 0, 5, 1, 4, 2]);
        let perm = reverse_cuthill_mckee(&a).unwrap();
        assert!(is_permutation(&perm));
    }

    #[test]
    fn rcm_restores_a_banded_ordering_of_a_scrambled_path() {
        // A path has bandwidth 1 in its natural order. RCM should find an order
        // with bandwidth 1 again, however the unknowns were scrambled.
        let a = scrambled_path(&[3, 0, 5, 1, 4, 2, 7, 6]);
        let perm = reverse_cuthill_mckee(&a).unwrap();
        let permuted = permute_symmetric(&a, &perm).unwrap();
        let bandwidth = permuted
            .iter()
            .map(|(i, j, _)| i.abs_diff(j))
            .max()
            .unwrap_or(0);
        assert_eq!(bandwidth, 1);
    }

    #[test]
    fn rcm_handles_disconnected_components() {
        // Two independent 2 × 2 blocks: {0, 2} and {1, 3}.
        let mut coo = CooMatrix::new(4, 4);
        for &(i, j) in &[
            (0, 0),
            (2, 2),
            (1, 1),
            (3, 3),
            (0, 2),
            (2, 0),
            (1, 3),
            (3, 1),
        ] {
            coo.push(i, j, 1.0);
        }
        let perm = reverse_cuthill_mckee(&coo.to_csr()).unwrap();
        assert!(is_permutation(&perm));
        assert_eq!(perm.len(), 4);
    }

    #[test]
    fn permute_symmetric_round_trips() {
        let a = scrambled_path(&[2, 0, 1]);
        let perm = vec![1, 2, 0];
        let back = {
            let mut inv = vec![0; 3];
            for (new, &old) in perm.iter().enumerate() {
                inv[old] = new;
            }
            permute_symmetric(&permute_symmetric(&a, &perm).unwrap(), &inv).unwrap()
        };
        let original: Vec<_> = a.iter().map(|(i, j, v)| (i, j, *v)).collect();
        let round: Vec<_> = back.iter().map(|(i, j, v)| (i, j, *v)).collect();
        assert_eq!(original, round);
    }

    #[test]
    fn permute_rejects_a_non_permutation() {
        let a = scrambled_path(&[0, 1, 2]);
        assert!(permute_symmetric(&a, &[0, 0, 1]).is_err());
        assert!(permute_symmetric(&a, &[0, 1]).is_err());
    }

    #[test]
    fn reordered_cholesky_solves_in_the_original_numbering() {
        let a = scrambled_path(&[3, 0, 5, 1, 4, 2, 7, 6]);
        let b = DVector::from_fn(8, |i| (i as f64 + 1.0).sin());
        let x = ReorderedCholesky::factor(&a).unwrap().solve(&b).unwrap();
        let ax = a.matvec(&x).unwrap();
        for i in 0..8 {
            assert!((ax[i] - b[i]).abs() < 1e-10, "row {i}");
        }
    }

    #[test]
    fn reordering_reduces_fill_for_a_scrambled_band() {
        let shuffle = [5, 2, 7, 0, 4, 1, 6, 3, 9, 8];
        let a = scrambled_path(&shuffle);
        let natural = SparseCholesky::factor(&a).unwrap().nnz();
        let reordered = ReorderedCholesky::factor(&a).unwrap().nnz();
        assert!(
            reordered <= natural,
            "reordered {reordered}, natural {natural}"
        );
    }

    /// The 5-point Laplacian on a `k × k` grid (SPD), numbered row by row.
    fn grid_laplacian(k: usize) -> CsrMatrix<f64> {
        let n = k * k;
        let mut coo = CooMatrix::new(n, n);
        for r in 0..k {
            for c in 0..k {
                let i = r * k + c;
                coo.push(i, i, 4.0);
                if c + 1 < k {
                    coo.push(i, i + 1, -1.0);
                    coo.push(i + 1, i, -1.0);
                }
                if r + 1 < k {
                    coo.push(i, i + k, -1.0);
                    coo.push(i + k, i, -1.0);
                }
            }
        }
        coo.to_csr()
    }

    #[test]
    fn minimum_degree_reduces_fill_on_a_grid() {
        let a = grid_laplacian(8);
        let natural = SparseCholesky::factor(&a).unwrap().nnz();
        let md = ReorderedCholesky::factor_with(&a, FillOrdering::MinimumDegree)
            .unwrap()
            .nnz();
        assert!(md < natural, "minimum degree {md}, natural {natural}");
        assert!(is_permutation(&minimum_degree_ordering(&a).unwrap()));
    }

    #[test]
    fn both_orderings_solve_in_the_original_numbering() {
        let a = grid_laplacian(6);
        let b = DVector::from_fn(36, |i| (i as f64 * 0.37).sin());
        for ordering in [
            FillOrdering::ReverseCuthillMckee,
            FillOrdering::MinimumDegree,
        ] {
            let x = ReorderedCholesky::factor_with(&a, ordering)
                .unwrap()
                .solve(&b)
                .unwrap();
            let ax = a.matvec(&x).unwrap();
            for i in 0..36 {
                assert!((ax[i] - b[i]).abs() < 1e-10, "{ordering:?} row {i}");
            }
        }
    }

    #[test]
    fn reordered_lu_solves_a_nonsymmetric_system() {
        // Nonsymmetric, diagonally dominant band matrix.
        let n = 9;
        let mut coo = CooMatrix::new(n, n);
        for i in 0..n {
            coo.push(i, i, 5.0);
            if i + 1 < n {
                coo.push(i, i + 1, -1.0);
            }
            if i + 2 < n {
                coo.push(i + 2, i, -2.0);
            }
        }
        let a = coo.to_csr();
        let b = DVector::from_fn(n, |i| 1.0 + i as f64);
        for ordering in [
            FillOrdering::ReverseCuthillMckee,
            FillOrdering::MinimumDegree,
        ] {
            let x = ReorderedLu::factor(&a, ordering)
                .unwrap()
                .solve(&b)
                .unwrap();
            let ax = a.matvec(&x).unwrap();
            for i in 0..n {
                assert!((ax[i] - b[i]).abs() < 1e-10, "{ordering:?} row {i}");
            }
        }
    }
}
