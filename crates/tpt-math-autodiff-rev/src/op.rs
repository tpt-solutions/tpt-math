//! The primitive operations recorded on a [`GradientTape`] and their local
//! derivative rules.
//!
//! Every node of the Wengert list stores exactly one [`Op`]. An `Op` holds the
//! *indices* of the nodes it consumes (never the values themselves), so the
//! tape stays a compact, `Copy`-able list of instructions. The local
//! derivative rule lives in [`Op::diff`], which returns the row of the local
//! Jacobian belonging to the node: one `(operand, ∂out/∂operand)` pair per
//! operand.
//!
//! [`GradientTape`]: crate::GradientTape

/// The maximum number of operands any [`Op`] can reference.
///
/// All primitives in this crate are nullary, unary or binary, so a local
/// Jacobian row never needs more than two entries and [`Partials`] can be a
/// fixed-size, allocation-free value.
pub const MAX_OPERANDS: usize = 2;

/// A single primitive operation in the tape's Wengert list.
///
/// The `usize` payloads are node indices into the tape. Because nodes are only
/// ever appended, every operand index is strictly smaller than the index of
/// the node holding the `Op`; that invariant is what makes a plain reverse
/// scan of the tape a valid topological order for the backward pass.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff_rev::{Op, Partials};
///
/// // Node 2 = node0 * node1, with node0 = 3.0 and node1 = 4.0.
/// let values = [3.0, 4.0, 12.0];
/// let partials = Op::Mul(0, 1).diff(values[2], &values);
/// assert_eq!(partials.as_slice(), &[(0, 4.0), (1, 3.0)]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    /// A literal constant. Carries no gradient.
    Const,
    /// An independent input variable; a leaf that gradients accumulate into.
    Input,
    /// `-a`
    Neg(usize),
    /// `a + b`
    Add(usize, usize),
    /// `a - b`
    Sub(usize, usize),
    /// `a * b`
    Mul(usize, usize),
    /// `a / b`
    Div(usize, usize),
    /// `sin(a)`
    Sin(usize),
    /// `cos(a)`
    Cos(usize),
    /// `tan(a)`
    Tan(usize),
    /// `exp(a)`
    Exp(usize),
    /// `ln(a)`
    Ln(usize),
    /// `sqrt(a)`
    Sqrt(usize),
    /// `a.powi(n)` for an integer exponent `n`.
    Powi(usize, i32),
    /// `a.powf(p)` for a real exponent `p`.
    Powf(usize, f64),
}

impl Op {
    /// The local Jacobian row of this operation.
    ///
    /// Returns one `(operand index, ∂output/∂operand)` pair per operand, which
    /// the backward pass multiplies by the node's adjoint and accumulates into
    /// the operands. Leaves ([`Op::Const`] and [`Op::Input`]) return an empty
    /// row because they have no operands.
    ///
    /// * `value` is the primal value of *this* node. Rules such as `exp` and
    ///   `sqrt` reuse it instead of recomputing a transcendental function.
    /// * `values` is the primal value of every node on the tape, indexed by
    ///   node index.
    ///
    /// # Panics
    ///
    /// Panics if an operand index is out of bounds for `values`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_autodiff_rev::Op;
    ///
    /// let values = [0.5, 0.5_f64.exp()];
    /// let partials = Op::Exp(0).diff(values[1], &values);
    /// assert_eq!(partials.as_slice(), &[(0, 0.5_f64.exp())]);
    /// ```
    #[must_use]
    pub fn diff(&self, value: f64, values: &[f64]) -> Partials {
        match *self {
            Op::Const | Op::Input => Partials::none(),
            Op::Neg(a) => Partials::unary(a, -1.0),
            Op::Add(a, b) => Partials::binary(a, 1.0, b, 1.0),
            Op::Sub(a, b) => Partials::binary(a, 1.0, b, -1.0),
            Op::Mul(a, b) => Partials::binary(a, values[b], b, values[a]),
            Op::Div(a, b) => {
                let den = values[b];
                Partials::binary(a, 1.0 / den, b, -values[a] / (den * den))
            }
            Op::Sin(a) => Partials::unary(a, values[a].cos()),
            Op::Cos(a) => Partials::unary(a, -values[a].sin()),
            // d/dx tan(x) = sec^2(x) = 1 + tan^2(x), and `value` is tan(x).
            Op::Tan(a) => Partials::unary(a, 1.0 + value * value),
            // `value` is already exp(a).
            Op::Exp(a) => Partials::unary(a, value),
            Op::Ln(a) => Partials::unary(a, 1.0 / values[a]),
            // `value` is already sqrt(a).
            Op::Sqrt(a) => Partials::unary(a, 0.5 / value),
            // `saturating_sub` only matters for the degenerate `i32::MIN`
            // exponent, where the derivative under/overflows regardless.
            Op::Powi(a, n) => {
                Partials::unary(a, f64::from(n) * values[a].powi(n.saturating_sub(1)))
            }
            Op::Powf(a, p) => Partials::unary(a, p * values[a].powf(p - 1.0)),
        }
    }

    /// Whether this operation is a leaf, i.e. it has no operands.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, Op::Const | Op::Input)
    }

    /// Whether this operation is a constant, which by definition carries no
    /// gradient.
    #[must_use]
    pub fn is_const(&self) -> bool {
        matches!(self, Op::Const)
    }
}

/// A local Jacobian row: up to [`MAX_OPERANDS`] `(operand index, derivative)`
/// pairs.
///
/// This is a tiny inline vector so that [`Op::diff`] never allocates.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff_rev::Partials;
///
/// let p = Partials::binary(0, 1.0, 3, -1.0);
/// assert_eq!(p.len(), 2);
/// assert_eq!(p.as_slice(), &[(0, 1.0), (3, -1.0)]);
/// assert!(Partials::none().is_empty());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Partials {
    entries: [(usize, f64); MAX_OPERANDS],
    len: usize,
}

impl Partials {
    /// An empty row, for leaves that have no operands.
    #[must_use]
    pub const fn none() -> Self {
        Partials { entries: [(0, 0.0); MAX_OPERANDS], len: 0 }
    }

    /// A row with a single entry, for unary operations.
    #[must_use]
    pub const fn unary(operand: usize, derivative: f64) -> Self {
        Partials { entries: [(operand, derivative), (0, 0.0)], len: 1 }
    }

    /// A row with two entries, for binary operations.
    ///
    /// The two operands may be the same node (as in `x * x`); the backward
    /// pass simply accumulates both contributions.
    #[must_use]
    pub const fn binary(lhs: usize, d_lhs: f64, rhs: usize, d_rhs: f64) -> Self {
        Partials { entries: [(lhs, d_lhs), (rhs, d_rhs)], len: 2 }
    }

    /// The populated `(operand index, derivative)` pairs.
    #[must_use]
    pub fn as_slice(&self) -> &[(usize, f64)] {
        &self.entries[..self.len]
    }

    /// The number of operands contributing to this row.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the row is empty, i.e. the node is a leaf.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate over the `(operand index, derivative)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (usize, f64)> + '_ {
        self.as_slice().iter().copied()
    }
}

impl Default for Partials {
    fn default() -> Self {
        Partials::none()
    }
}

impl<'a> IntoIterator for &'a Partials {
    type Item = (usize, f64);
    type IntoIter = core::iter::Copied<core::slice::Iter<'a, (usize, f64)>>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL
    }

    #[test]
    fn leaves_have_no_partials() {
        assert!(Op::Const.diff(1.0, &[1.0]).is_empty());
        assert!(Op::Input.diff(1.0, &[1.0]).is_empty());
        assert!(Op::Input.is_leaf());
        assert!(Op::Const.is_const());
        assert!(!Op::Input.is_const());
    }

    #[test]
    fn arithmetic_rules() {
        let values = [3.0, 4.0];
        assert_eq!(Op::Add(0, 1).diff(7.0, &values).as_slice(), &[(0, 1.0), (1, 1.0)]);
        assert_eq!(Op::Sub(0, 1).diff(-1.0, &values).as_slice(), &[(0, 1.0), (1, -1.0)]);
        assert_eq!(Op::Mul(0, 1).diff(12.0, &values).as_slice(), &[(0, 4.0), (1, 3.0)]);
        assert_eq!(Op::Neg(0).diff(-3.0, &values).as_slice(), &[(0, -1.0)]);

        let div = Op::Div(0, 1).diff(0.75, &values);
        assert!(close(div.as_slice()[0].1, 1.0 / 4.0));
        assert!(close(div.as_slice()[1].1, -3.0 / 16.0));
    }

    #[test]
    fn transcendental_rules() {
        let x = 0.7_f64;
        let values = [x, 0.0];
        assert!(close(Op::Sin(0).diff(x.sin(), &values).as_slice()[0].1, x.cos()));
        assert!(close(Op::Cos(0).diff(x.cos(), &values).as_slice()[0].1, -x.sin()));
        assert!(close(
            Op::Tan(0).diff(x.tan(), &values).as_slice()[0].1,
            1.0 / (x.cos() * x.cos())
        ));
        assert!(close(Op::Exp(0).diff(x.exp(), &values).as_slice()[0].1, x.exp()));
        assert!(close(Op::Ln(0).diff(x.ln(), &values).as_slice()[0].1, 1.0 / x));
        assert!(close(
            Op::Sqrt(0).diff(x.sqrt(), &values).as_slice()[0].1,
            0.5 / x.sqrt()
        ));
    }

    #[test]
    fn power_rules() {
        let x = 2.0_f64;
        let values = [x];
        assert!(close(Op::Powi(0, 3).diff(x.powi(3), &values).as_slice()[0].1, 12.0));
        assert!(close(Op::Powi(0, 0).diff(1.0, &values).as_slice()[0].1, 0.0));
        assert!(close(
            Op::Powf(0, 0.5).diff(x.sqrt(), &values).as_slice()[0].1,
            0.5 / x.sqrt()
        ));
    }

    #[test]
    fn partials_iteration() {
        let p = Partials::binary(1, 2.0, 5, 3.0);
        let collected: Vec<_> = p.iter().collect();
        assert_eq!(collected, vec![(1, 2.0), (5, 3.0)]);
        let by_ref: Vec<_> = (&p).into_iter().collect();
        assert_eq!(by_ref, collected);
        assert_eq!(Partials::default(), Partials::none());
    }
}
