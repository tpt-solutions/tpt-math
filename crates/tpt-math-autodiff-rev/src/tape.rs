//! The gradient tape (Wengert list) and the result of a backward pass.

use core::ptr;
use std::cell::RefCell;
use std::ops::Index;

use crate::op::Op;
use crate::var::Variable;

/// The recorded program: one [`Op`] and one primal value per node.
///
/// Stored struct-of-arrays so the backward pass can hand [`Op::diff`] a
/// contiguous `&[f64]` of primal values without copying.
#[derive(Clone, Debug, Default)]
struct Wengert {
    ops: Vec<Op>,
    values: Vec<f64>,
}

/// A tape that records every operation performed on its [`Variable`]s so that
/// all partial derivatives can be recovered with a single backward pass.
///
/// The tape uses interior mutability, so recording only needs `&self` and
/// [`Variable`]s can be freely copied around while the expression is built.
/// Nodes are append-only, which guarantees that operand indices are always
/// smaller than the index of the node using them — a reverse scan of the tape
/// is therefore a valid topological order.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff_rev::GradientTape;
///
/// let tape = GradientTape::new();
/// let x = tape.var(2.0);
/// let y = tape.var(3.0);
/// let z = x * x + y.sin();
/// let g = tape.gradient(z, &[x, y]);
///
/// assert!((z.value() - (4.0 + 3.0_f64.sin())).abs() < 1e-12);
/// assert!((g[0] - 4.0).abs() < 1e-12);          // dz/dx = 2x
/// assert!((g[1] - 3.0_f64.cos()).abs() < 1e-12); // dz/dy = cos(y)
/// ```
#[derive(Debug, Default)]
pub struct GradientTape {
    inner: RefCell<Wengert>,
}

impl GradientTape {
    /// Create an empty tape.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty tape with room for `capacity` nodes.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        GradientTape {
            inner: RefCell::new(Wengert {
                ops: Vec::with_capacity(capacity),
                values: Vec::with_capacity(capacity),
            }),
        }
    }

    /// The number of nodes recorded so far.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.borrow().ops.len()
    }

    /// Whether nothing has been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Discard every recorded node.
    ///
    /// This takes `&mut self`, so the borrow checker guarantees that no
    /// [`Variable`] referring to the discarded nodes can still be alive.
    pub fn clear(&mut self) {
        let inner = self.inner.get_mut();
        inner.ops.clear();
        inner.values.clear();
    }

    /// Record an independent input variable holding `value`.
    ///
    /// Gradients accumulate into input variables; they are the leaves you pass
    /// to [`GradientTape::gradient`].
    #[doc(alias = "variable")]
    pub fn var(&self, value: f64) -> Variable<'_> {
        self.push(Op::Input, value)
    }

    /// Alias for [`GradientTape::var`], mirroring the naming used by the
    /// forward-mode crate.
    pub fn variable(&self, value: f64) -> Variable<'_> {
        self.var(value)
    }

    /// Record independent input variables for each of `values`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_autodiff_rev::GradientTape;
    ///
    /// let tape = GradientTape::new();
    /// let v = tape.vars(&[1.0, 2.0, 3.0]);
    /// let sum = v[0] + v[1] + v[2];
    /// assert_eq!(tape.gradient(sum, &v), vec![1.0, 1.0, 1.0]);
    /// ```
    pub fn vars(&self, values: &[f64]) -> Vec<Variable<'_>> {
        values.iter().map(|&v| self.var(v)).collect()
    }

    /// Record a constant holding `value`.
    ///
    /// Constants participate in the forward computation but never receive a
    /// gradient: differentiating with respect to one always yields `0.0`.
    pub fn constant(&self, value: f64) -> Variable<'_> {
        self.push(Op::Const, value)
    }

    /// Append a raw node to the tape and return a handle to it.
    ///
    /// This is the primitive every operator and math method funnels through.
    /// It is public so that additional primitives can be layered on top of the
    /// tape; `value` must be the primal result of `op` and every operand index
    /// inside `op` must already exist on this tape.
    pub fn push(&self, op: Op, value: f64) -> Variable<'_> {
        let mut inner = self.inner.borrow_mut();
        let index = inner.ops.len();
        inner.ops.push(op);
        inner.values.push(value);
        drop(inner);
        Variable::new(self, index, value)
    }

    /// The [`Op`] recorded at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[must_use]
    pub fn op(&self, index: usize) -> Op {
        self.inner.borrow().ops[index]
    }

    /// The primal value recorded at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds.
    #[must_use]
    pub fn value(&self, index: usize) -> f64 {
        self.inner.borrow().values[index]
    }

    /// Whether `var` was recorded on this tape.
    #[must_use]
    pub fn owns(&self, var: Variable<'_>) -> bool {
        ptr::eq(self, var.tape())
    }

    /// Run the backward pass from `output`, returning the adjoint of **every**
    /// node on the tape.
    ///
    /// The output node is seeded with an adjoint of `1.0`; the tape is then
    /// walked in reverse index order (a topological order of the expression
    /// DAG), and each node distributes `adjoint * ∂self/∂operand` into its
    /// operands using [`Op::diff`]. Nodes that the output does not depend on
    /// keep an adjoint of `0.0`, and constants never accumulate one.
    ///
    /// # Panics
    ///
    /// Panics if `output` was recorded on a different tape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_autodiff_rev::GradientTape;
    ///
    /// let tape = GradientTape::new();
    /// let x = tape.var(4.0);
    /// let y = x.sqrt();
    /// let grad = tape.backward(y);
    /// assert!((grad.wrt(x) - 0.25).abs() < 1e-12); // d/dx sqrt(x) = 1/(2 sqrt(x))
    /// ```
    pub fn backward(&self, output: Variable<'_>) -> Gradient {
        assert!(
            self.owns(output),
            "the output variable was recorded on a different GradientTape"
        );

        let inner = self.inner.borrow();
        let mut adjoints = vec![0.0; inner.ops.len()];
        adjoints[output.index()] = 1.0;

        for index in (0..=output.index()).rev() {
            let adjoint = adjoints[index];
            if adjoint == 0.0 {
                // Not an ancestor of the output (or an exactly cancelling
                // contribution): nothing to propagate.
                continue;
            }
            let partials = inner.ops[index].diff(inner.values[index], &inner.values);
            for (operand, local) in partials.iter() {
                // Constants are not differentiable inputs, so gradient never
                // flows into them.
                if inner.ops[operand].is_const() {
                    continue;
                }
                adjoints[operand] += adjoint * local;
            }
        }

        Gradient { adjoints }
    }

    /// Differentiate `output` with respect to each variable in `wrt`.
    ///
    /// Equivalent to `tape.backward(output)` followed by one
    /// [`Gradient::wrt`] lookup per entry, returning `[∂output/∂wrt[0], …]`.
    ///
    /// # Panics
    ///
    /// Panics if `output` or any entry of `wrt` was recorded on a different
    /// tape.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_autodiff_rev::GradientTape;
    ///
    /// let tape = GradientTape::new();
    /// let x = tape.var(3.0);
    /// let g = tape.gradient(x * x, &[x]);
    /// assert!((g[0] - 6.0).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn gradient(&self, output: Variable<'_>, wrt: &[Variable<'_>]) -> Vec<f64> {
        let grad = self.backward(output);
        wrt.iter()
            .map(|&var| {
                assert!(
                    self.owns(var),
                    "a variable in `wrt` was recorded on a different GradientTape"
                );
                grad.wrt(var)
            })
            .collect()
    }
}

/// The adjoints produced by a backward pass: `∂output/∂node` for every node on
/// the tape.
///
/// Look up individual variables with [`Gradient::wrt`] or the [`Index`]
/// implementation.
///
/// # Examples
///
/// ```
/// use tpt_math_autodiff_rev::GradientTape;
///
/// let tape = GradientTape::new();
/// let x = tape.var(1.5);
/// let y = tape.var(-2.0);
/// let grad = tape.backward(x * y);
/// assert_eq!(grad[x], -2.0);
/// assert_eq!(grad.wrt(y), 1.5);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Gradient {
    adjoints: Vec<f64>,
}

impl Gradient {
    /// The partial derivative of the backward pass' output with respect to
    /// `var`.
    ///
    /// Returns `0.0` for variables the output does not depend on, for
    /// constants, and for nodes recorded after the output.
    #[must_use]
    pub fn wrt(&self, var: Variable<'_>) -> f64 {
        self.adjoints.get(var.index()).copied().unwrap_or(0.0)
    }

    /// The adjoints of all nodes, indexed by node index.
    #[must_use]
    pub fn as_slice(&self) -> &[f64] {
        &self.adjoints
    }

    /// Consume the gradient, yielding the raw adjoint vector.
    #[must_use]
    pub fn into_vec(self) -> Vec<f64> {
        self.adjoints
    }

    /// The number of nodes covered by this gradient.
    #[must_use]
    pub fn len(&self) -> usize {
        self.adjoints.len()
    }

    /// Whether the tape was empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.adjoints.is_empty()
    }
}

impl Index<Variable<'_>> for Gradient {
    type Output = f64;

    fn index(&self, var: Variable<'_>) -> &f64 {
        &self.adjoints[var.index()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-9;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < TOL,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn tape_records_nodes() {
        let mut tape = GradientTape::new();
        assert!(tape.is_empty());
        {
            let x = tape.var(2.0);
            let _ = x * x;
            assert_eq!(tape.len(), 2);
            assert_eq!(tape.op(0), Op::Input);
            assert_eq!(tape.op(1), Op::Mul(0, 0));
            assert_close(tape.value(1), 4.0);
        }
        tape.clear();
        assert!(tape.is_empty());
    }

    #[test]
    fn gradient_of_unrelated_variable_is_zero() {
        let tape = GradientTape::new();
        let x = tape.var(2.0);
        let y = tape.var(5.0);
        let z = x * x;
        let g = tape.gradient(z, &[x, y]);
        assert_close(g[0], 4.0);
        assert_close(g[1], 0.0);
    }

    #[test]
    fn constants_carry_no_gradient() {
        let tape = GradientTape::new();
        let x = tape.var(2.0);
        let c = tape.constant(5.0);
        let z = x * c + c;
        let g = tape.gradient(z, &[x, c]);
        assert_close(z.value(), 15.0);
        assert_close(g[0], 5.0);
        assert_close(g[1], 0.0);
    }

    #[test]
    fn gradient_indexing_and_accessors() {
        let tape = GradientTape::new();
        let x = tape.var(1.5);
        let y = tape.var(-2.0);
        let grad = tape.backward(x * y);
        assert_close(grad[x], -2.0);
        assert_close(grad.wrt(y), 1.5);
        assert_eq!(grad.len(), tape.len());
        assert!(!grad.is_empty());
        assert_eq!(grad.as_slice().len(), tape.len());
        assert_eq!(grad.clone().into_vec(), grad.as_slice().to_vec());
    }

    #[test]
    fn push_is_a_usable_extension_point() {
        let tape = GradientTape::new();
        let x = tape.var(0.25);
        // Hand-rolled `2 * x` built straight from tape primitives.
        let two = tape.constant(2.0);
        let doubled = tape.push(Op::Mul(two.index(), x.index()), 2.0 * x.value());
        assert_close(doubled.value(), 0.5);
        assert_close(tape.gradient(doubled, &[x])[0], 2.0);
    }

    #[test]
    #[should_panic(expected = "different GradientTape")]
    fn backward_rejects_foreign_variables() {
        let a = GradientTape::new();
        let b = GradientTape::new();
        let x = b.var(1.0);
        let _ = a.backward(x);
    }

    #[test]
    #[should_panic(expected = "different GradientTape")]
    fn gradient_rejects_foreign_wrt_variables() {
        let a = GradientTape::new();
        let b = GradientTape::new();
        let x = a.var(1.0);
        let foreign = b.var(1.0);
        let _ = a.gradient(x * x, &[foreign]);
    }
}
