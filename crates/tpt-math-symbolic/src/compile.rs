//! Compiling an [`Expr`] to a closure.
//!
//! [`eval`](crate::Symbolic::eval) walks the tree and looks every variable up in a `HashMap` on
//! each call. For an expression evaluated many times (a root finder, a quadrature
//! rule, a plotting loop) that is wasted work. [`compile`] builds the closure
//! tree once, resolves variables to argument positions, and checks names up
//! front, so the per-call cost is just the arithmetic.

use std::fmt;

use crate::{func_pointer, to_f64, Coefficient, Expr};

/// A compiled node: takes the argument slice and yields the value, or `None`
/// where [`eval`](crate::Symbolic::eval) would (division by zero).
type Node<C> = Box<dyn Fn(&[C]) -> Option<C>>;

/// Why an expression could not be compiled.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CompileError {
    /// The expression uses a variable that is not in the argument list.
    UnknownVariable(String),
    /// The expression applies a function name this crate does not know.
    UnknownFunction(String),
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::UnknownVariable(v) => {
                write!(f, "variable `{v}` is not among the compiled arguments")
            }
            CompileError::UnknownFunction(name) => write!(f, "unknown function `{name}`"),
        }
    }
}

impl std::error::Error for CompileError {}

/// An expression compiled to a closure over an argument slice.
///
/// Arguments are matched to the variable list given to [`compile`], by
/// position. [`Compiled::call`] returns the same value as [`eval`](crate::Symbolic::eval) with
/// those bindings, or `None` where `eval` would return `None`.
pub struct Compiled<C> {
    arity: usize,
    run: Node<C>,
}

impl<C> Compiled<C> {
    /// The number of arguments [`call`](Self::call) expects.
    pub fn arity(&self) -> usize {
        self.arity
    }

    /// Evaluate at `args`, in the order of the variable list passed to
    /// [`compile`].
    ///
    /// Returns `None` if `args.len()` differs from [`arity`](Self::arity), or
    /// where [`eval`](crate::Symbolic::eval) would return `None` (division by zero).
    pub fn call(&self, args: &[C]) -> Option<C> {
        if args.len() != self.arity {
            return None;
        }
        (self.run)(args)
    }
}

impl<C> fmt::Debug for Compiled<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Compiled")
            .field("arity", &self.arity)
            .finish_non_exhaustive()
    }
}

/// Compile `expr` to a closure whose arguments are the variables `vars`, in
/// that order.
///
/// Variable names and function names are resolved here, so a bad name is an
/// error at compile time rather than a `None` at every call.
///
/// # Errors
///
/// * [`CompileError::UnknownVariable`] if `expr` mentions a variable not in `vars`.
/// * [`CompileError::UnknownFunction`] if `expr` applies an unknown function.
///
/// # Examples
///
/// ```
/// use tpt_math_symbolic::{compile, Expr};
///
/// // f(x, y) = x^2 + sin(y)
/// let e: Expr<f64> = Expr::var("x").pow(2) + Expr::func("sin", Expr::var("y"));
/// let f = compile(&e, &["x", "y"]).unwrap();
/// assert!((f.call(&[3.0, 0.0]).unwrap() - 9.0).abs() < 1e-12);
/// ```
pub fn compile<C>(expr: &Expr<C>, vars: &[&str]) -> Result<Compiled<C>, CompileError>
where
    C: Coefficient + 'static,
{
    Ok(Compiled {
        arity: vars.len(),
        run: build(expr, vars)?,
    })
}

fn build<C>(expr: &Expr<C>, vars: &[&str]) -> Result<Node<C>, CompileError>
where
    C: Coefficient + 'static,
{
    Ok(match expr {
        Expr::Const(c) => {
            let c = c.clone();
            Box::new(move |_| Some(c.clone()))
        }
        Expr::Var(name) => {
            let slot = vars
                .iter()
                .position(|v| *v == name)
                .ok_or_else(|| CompileError::UnknownVariable(name.clone()))?;
            Box::new(move |args| args.get(slot).cloned())
        }
        Expr::Add(a, b) => binary(a, b, vars, |x, y| Some(x + y))?,
        Expr::Sub(a, b) => binary(a, b, vars, |x, y| Some(x - y))?,
        Expr::Mul(a, b) => binary(a, b, vars, |x, y| Some(x * y))?,
        Expr::Div(a, b) => binary(a, b, vars, |x, d| {
            if d.is_zero_value() {
                None
            } else {
                Some(x / d)
            }
        })?,
        Expr::Neg(a) => {
            let fa = build(a, vars)?;
            Box::new(move |args| Some(-fa(args)?))
        }
        Expr::Pow(a, n) => {
            let fa = build(a, vars)?;
            let n = *n as i32;
            Box::new(move |args| Some(fa(args)?.powi(n)))
        }
        Expr::Func(name, a) => {
            let f =
                func_pointer(name).ok_or_else(|| CompileError::UnknownFunction(name.clone()))?;
            let fa = build(a, vars)?;
            Box::new(move |args| Some(C::from(f(to_f64(fa(args)?)))))
        }
    })
}

fn binary<C, F>(a: &Expr<C>, b: &Expr<C>, vars: &[&str], op: F) -> Result<Node<C>, CompileError>
where
    C: Coefficient + 'static,
    F: Fn(C, C) -> Option<C> + 'static,
{
    let fa = build(a, vars)?;
    let fb = build(b, vars)?;
    Ok(Box::new(move |args| op(fa(args)?, fb(args)?)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Symbolic;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    #[test]
    fn matches_eval_on_a_grid() {
        // (x - y) / (1 + x^2) + exp(-y) * sqrt(2) - tanh(x * y)
        let x = Expr::<f64>::var("x");
        let y = Expr::<f64>::var("y");
        let e = (x.clone() - y.clone()) / (Expr::constant(1.0) + x.clone().pow(2))
            + Expr::func("exp", -y.clone()) * Expr::func("sqrt", Expr::constant(2.0))
            - Expr::func("tanh", x.clone() * y.clone());
        let f = compile(&e, &["x", "y"]).unwrap();
        for i in -5..=5 {
            for j in -5..=5 {
                let (xv, yv) = (i as f64 * 0.37, j as f64 * 0.23);
                let want = e.eval(&env(&[("x", xv), ("y", yv)])).unwrap();
                let got = f.call(&[xv, yv]).unwrap();
                assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0));
            }
        }
    }

    #[test]
    fn argument_order_follows_the_variable_list() {
        let e = Expr::<f64>::var("a") - Expr::var("b") * Expr::constant(10.0);
        let ab = compile(&e, &["a", "b"]).unwrap();
        let ba = compile(&e, &["b", "a"]).unwrap();
        assert_eq!(ab.call(&[3.0, 1.0]), Some(-7.0));
        assert_eq!(ba.call(&[1.0, 3.0]), Some(-7.0));
    }

    #[test]
    fn unknown_variable_is_a_compile_error() {
        let e = Expr::<f64>::var("x") + Expr::var("z");
        assert_eq!(
            compile(&e, &["x"]).unwrap_err(),
            CompileError::UnknownVariable("z".into())
        );
    }

    #[test]
    fn unknown_function_is_a_compile_error() {
        let e = Expr::<f64>::func("gamma", Expr::var("x"));
        assert_eq!(
            compile(&e, &["x"]).unwrap_err(),
            CompileError::UnknownFunction("gamma".into())
        );
    }

    #[test]
    fn division_by_zero_is_none_like_eval() {
        let e = Expr::<f64>::constant(1.0) / Expr::var("x");
        let f = compile(&e, &["x"]).unwrap();
        assert_eq!(f.call(&[0.0]), None);
        assert_eq!(f.call(&[4.0]), Some(0.25));
    }

    #[test]
    fn wrong_arity_is_none() {
        let f = compile(&Expr::<f64>::var("x"), &["x"]).unwrap();
        assert_eq!(f.arity(), 1);
        assert_eq!(f.call(&[]), None);
        assert_eq!(f.call(&[1.0, 2.0]), None);
    }

    #[test]
    fn method_form_compiles_too() {
        let f = Expr::<f64>::var("t").pow(3).compile(&["t"]).unwrap();
        assert_eq!(f.call(&[2.0]), Some(8.0));
    }

    #[test]
    fn constant_expression_ignores_arguments() {
        let f = Expr::<f64>::constant(7.5).compile(&[]).unwrap();
        assert_eq!(f.call(&[]), Some(7.5));
    }
}
