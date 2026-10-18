//! Text in and text out: a parser for infix expressions, and LaTeX output.
//!
//! Grammar, from loosest to tightest binding:
//!
//! ```text
//! expr   := term (('+' | '-') term)*
//! term   := unary (('*' | '/') unary)*
//! unary  := '-' unary | power
//! power  := atom ('^' integer)?
//! atom   := number | 'pi' | 'e' | name '(' expr ')' | name | '(' expr ')'
//! ```
//!
//! Exponents must be non-negative integer literals, because [`Expr::Pow`]
//! takes a `usize`. Function names are the ones [`Expr::eval`] understands.

use core::fmt;

use crate::{Coefficient, Expr};

/// The functions the parser accepts, the same set `eval` and `derivative` know.
const KNOWN_FUNCTIONS: [&str; 12] = [
    "sin", "cos", "tan", "exp", "ln", "sqrt", "asin", "acos", "atan", "sinh", "cosh", "tanh",
];

/// Why an expression string failed to parse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// The input ended while more was expected.
    UnexpectedEnd,
    /// A character that cannot start or continue a token at byte offset `pos`.
    UnexpectedChar {
        /// The offending character.
        ch: char,
        /// Byte offset into the input.
        pos: usize,
    },
    /// A number that is not a finite decimal literal.
    BadNumber(String),
    /// A function name outside the supported set.
    UnknownFunction(String),
    /// An exponent that is not a non-negative integer literal.
    BadExponent(String),
    /// Input left over after a complete expression.
    TrailingInput {
        /// Byte offset where the leftover input starts.
        pos: usize,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedEnd => write!(f, "unexpected end of input"),
            ParseError::UnexpectedChar { ch, pos } => {
                write!(f, "unexpected character {ch:?} at byte {pos}")
            }
            ParseError::BadNumber(s) => write!(f, "bad number {s:?}"),
            ParseError::UnknownFunction(name) => write!(f, "unknown function {name:?}"),
            ParseError::BadExponent(s) => {
                write!(f, "exponent must be a non-negative integer, got {s:?}")
            }
            ParseError::TrailingInput { pos } => write!(f, "trailing input at byte {pos}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse an infix expression such as `"x^2 + sin(x)"`.
///
/// # Errors
///
/// Returns a [`ParseError`] describing the first problem found.
///
/// # Examples
///
/// ```
/// use tpt_math_symbolic::{parse_expr, Expr64, Symbolic};
///
/// let e: Expr64 = parse_expr("x^2 + sin(x)").unwrap();
/// let want = Expr64::var("x").pow(2) + Expr64::func("sin", Expr64::var("x"));
/// assert_eq!(e, want);
/// ```
pub fn parse_expr<C: Coefficient>(src: &str) -> Result<Expr<C>, ParseError> {
    let tokens = tokenize(src)?;
    let mut parser = Parser { tokens, at: 0 };
    let expr = parser.expr()?;
    match parser.peek() {
        None => Ok(expr),
        Some((_, pos)) => Err(ParseError::TrailingInput { pos: *pos }),
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Num(String),
    Ident(String),
    Op(char),
    LParen,
    RParen,
}

/// Split the input into tokens, each paired with its byte offset.
fn tokenize(src: &str) -> Result<Vec<(Token, usize)>, ParseError> {
    let mut out = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c.is_whitespace() {
            i += 1;
        } else if c.is_ascii_digit() || c == '.' {
            let start = i;
            while i < bytes.len() && ((bytes[i] as char).is_ascii_digit() || bytes[i] == b'.') {
                i += 1;
            }
            out.push((Token::Num(src[start..i].to_string()), start));
        } else if c.is_ascii_alphabetic() || c == '_' {
            let start = i;
            while i < bytes.len()
                && ((bytes[i] as char).is_ascii_alphanumeric() || bytes[i] == b'_')
            {
                i += 1;
            }
            out.push((Token::Ident(src[start..i].to_string()), start));
        } else {
            let tok = match c {
                '+' | '-' | '*' | '/' | '^' => Token::Op(c),
                '(' => Token::LParen,
                ')' => Token::RParen,
                _ => return Err(ParseError::UnexpectedChar { ch: c, pos: i }),
            };
            out.push((tok, i));
            i += 1;
        }
    }
    Ok(out)
}

struct Parser {
    tokens: Vec<(Token, usize)>,
    at: usize,
}

impl Parser {
    fn peek(&self) -> Option<&(Token, usize)> {
        self.tokens.get(self.at)
    }

    fn next(&mut self) -> Option<(Token, usize)> {
        let t = self.tokens.get(self.at).cloned();
        self.at += 1;
        t
    }

    /// Position of the token just consumed, for error messages.
    fn char_at(&self, pos: usize) -> char {
        src_char_at(&self.tokens, pos)
    }

    fn expr<C: Coefficient>(&mut self) -> Result<Expr<C>, ParseError> {
        let mut lhs = self.term()?;
        while let Some((Token::Op(op @ ('+' | '-')), _)) = self.peek().cloned() {
            self.at += 1;
            let rhs = self.term()?;
            lhs = if op == '+' { lhs + rhs } else { lhs - rhs };
        }
        Ok(lhs)
    }

    fn term<C: Coefficient>(&mut self) -> Result<Expr<C>, ParseError> {
        let mut lhs = self.unary()?;
        while let Some((Token::Op(op @ ('*' | '/')), _)) = self.peek().cloned() {
            self.at += 1;
            let rhs = self.unary()?;
            lhs = if op == '*' { lhs * rhs } else { lhs / rhs };
        }
        Ok(lhs)
    }

    fn unary<C: Coefficient>(&mut self) -> Result<Expr<C>, ParseError> {
        if let Some((Token::Op('-'), _)) = self.peek() {
            self.at += 1;
            return Ok(-self.unary()?);
        }
        self.power()
    }

    fn power<C: Coefficient>(&mut self) -> Result<Expr<C>, ParseError> {
        let base = self.atom()?;
        if let Some((Token::Op('^'), _)) = self.peek() {
            self.at += 1;
            let (tok, _) = self.next().ok_or(ParseError::UnexpectedEnd)?;
            let text = match tok {
                Token::Num(s) => s,
                other => return Err(ParseError::BadExponent(token_text(&other))),
            };
            let n: usize = text
                .parse()
                .map_err(|_| ParseError::BadExponent(text.clone()))?;
            return Ok(base.pow(n));
        }
        Ok(base)
    }

    fn atom<C: Coefficient>(&mut self) -> Result<Expr<C>, ParseError> {
        let (tok, pos) = self.next().ok_or(ParseError::UnexpectedEnd)?;
        match tok {
            Token::Num(s) => {
                let v: f64 = s.parse().map_err(|_| ParseError::BadNumber(s.clone()))?;
                if !v.is_finite() {
                    return Err(ParseError::BadNumber(s));
                }
                Ok(Expr::constant(v))
            }
            Token::LParen => {
                let inner = self.expr()?;
                self.close_paren()?;
                Ok(inner)
            }
            Token::Ident(name) => {
                if let Some((Token::LParen, _)) = self.peek() {
                    self.at += 1;
                    if !KNOWN_FUNCTIONS.contains(&name.as_str()) {
                        return Err(ParseError::UnknownFunction(name));
                    }
                    let inner = self.expr()?;
                    self.close_paren()?;
                    Ok(Expr::func(name, inner))
                } else {
                    Ok(match name.as_str() {
                        "pi" => Expr::constant(core::f64::consts::PI),
                        "e" => Expr::constant(core::f64::consts::E),
                        _ => Expr::var(name),
                    })
                }
            }
            Token::Op(c) => Err(ParseError::UnexpectedChar { ch: c, pos }),
            Token::RParen => Err(ParseError::UnexpectedChar { ch: ')', pos }),
        }
    }

    fn close_paren(&mut self) -> Result<(), ParseError> {
        match self.next() {
            Some((Token::RParen, _)) => Ok(()),
            Some((_, p)) => Err(ParseError::UnexpectedChar {
                ch: self.char_at(p),
                pos: p,
            }),
            None => Err(ParseError::UnexpectedEnd),
        }
    }
}

/// Text of a token, for error messages.
fn token_text(tok: &Token) -> String {
    match tok {
        Token::Num(s) | Token::Ident(s) => s.clone(),
        Token::Op(c) => c.to_string(),
        Token::LParen => "(".to_string(),
        Token::RParen => ")".to_string(),
    }
}

/// The first character of the token starting at byte offset `pos`.
fn src_char_at(tokens: &[(Token, usize)], pos: usize) -> char {
    tokens
        .iter()
        .find(|(_, p)| *p == pos)
        .map(|(t, _)| token_text(t).chars().next().unwrap_or('?'))
        .unwrap_or('?')
}

impl<C: Coefficient> Expr<C> {
    /// LaTeX source for the expression, with parentheses only where precedence
    /// needs them. Division becomes `\frac`, and products use `\cdot`.
    ///
    /// # Examples
    ///
    /// ```
    /// use tpt_math_symbolic::{parse_expr, Expr64};
    ///
    /// let e: Expr64 = parse_expr("(x + 1) / y^2").unwrap();
    /// assert_eq!(e.to_latex(), r"\frac{x + 1}{y^{2}}");
    /// ```
    pub fn to_latex(&self) -> String {
        latex(self)
    }
}

/// Binding strength used to decide where parentheses are needed.
fn precedence<C: Coefficient>(e: &Expr<C>) -> u8 {
    match e {
        Expr::Add(..) | Expr::Sub(..) => 1,
        Expr::Neg(_) => 2,
        Expr::Const(c) if c.to_string().starts_with('-') => 2,
        Expr::Mul(..) => 3,
        Expr::Pow(..) => 4,
        _ => 5,
    }
}

fn wrap<C: Coefficient>(e: &Expr<C>, min: u8) -> String {
    if precedence(e) < min {
        format!("({})", latex(e))
    } else {
        latex(e)
    }
}

fn latex<C: Coefficient>(e: &Expr<C>) -> String {
    match e {
        Expr::Const(c) => c.to_string(),
        Expr::Var(name) => name.clone(),
        Expr::Add(a, b) => format!("{} + {}", wrap(a, 1), wrap(b, 1)),
        Expr::Sub(a, b) => format!("{} - {}", wrap(a, 1), wrap(b, 2)),
        Expr::Mul(a, b) => format!("{} \\cdot {}", wrap(a, 3), wrap(b, 3)),
        Expr::Div(a, b) => format!("\\frac{{{}}}{{{}}}", latex(a), latex(b)),
        Expr::Neg(a) => format!("-{}", wrap(a, 3)),
        Expr::Pow(a, n) => format!("{}^{{{}}}", wrap(a, 5), n),
        Expr::Func(name, inner) => match name.as_str() {
            "sqrt" => format!("\\sqrt{{{}}}", latex(inner)),
            "asin" => format!("\\arcsin\\left({}\\right)", latex(inner)),
            "acos" => format!("\\arccos\\left({}\\right)", latex(inner)),
            "atan" => format!("\\arctan\\left({}\\right)", latex(inner)),
            "ln" | "exp" | "sin" | "cos" | "tan" | "sinh" | "cosh" | "tanh" => {
                format!("\\{name}\\left({}\\right)", latex(inner))
            }
            other => format!("\\operatorname{{{other}}}\\left({}\\right)", latex(inner)),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Expr64, Symbolic};
    use std::collections::HashMap;

    fn env(x: f64) -> HashMap<String, f64> {
        HashMap::from([("x".to_string(), x)])
    }

    #[test]
    fn parses_precedence_and_associativity() {
        // 1 - 2 - 3 is left-associative: (1 - 2) - 3 = -4.
        let e: Expr64 = parse_expr("1 - 2 - 3").unwrap();
        assert_eq!(e.eval(&HashMap::new()), Some(-4.0));
        // Unary minus binds looser than `^`: -(2^2) = -4.
        let e: Expr64 = parse_expr("-2^2").unwrap();
        assert_eq!(e.eval(&HashMap::new()), Some(-4.0));
        let e: Expr64 = parse_expr("2 * (3 + 4) / 7").unwrap();
        assert_eq!(e.eval(&HashMap::new()), Some(2.0));
    }

    #[test]
    fn parsed_expression_evaluates_and_differentiates() {
        let e: Expr64 = parse_expr("x^2 + sin(x)").unwrap();
        let x = 0.8;
        assert!((e.eval(&env(x)).unwrap() - (x * x + x.sin())).abs() < 1e-12);
        let d = e.derivative("x").simplify();
        assert!((d.eval(&env(x)).unwrap() - (2.0 * x + x.cos())).abs() < 1e-9);
    }

    #[test]
    fn constants_pi_and_e() {
        let e: Expr64 = parse_expr("pi + e").unwrap();
        let want = core::f64::consts::PI + core::f64::consts::E;
        assert!((e.eval(&HashMap::new()).unwrap() - want).abs() < 1e-12);
    }

    #[test]
    fn rejects_bad_input_with_positions() {
        assert_eq!(
            parse_expr::<f64>("1 + $").unwrap_err(),
            ParseError::UnexpectedChar { ch: '$', pos: 4 }
        );
        assert_eq!(
            parse_expr::<f64>("1 +").unwrap_err(),
            ParseError::UnexpectedEnd
        );
        assert_eq!(
            parse_expr::<f64>("foo(x)").unwrap_err(),
            ParseError::UnknownFunction("foo".into())
        );
        assert_eq!(
            parse_expr::<f64>("x^y").unwrap_err(),
            ParseError::BadExponent("y".into())
        );
        assert_eq!(
            parse_expr::<f64>("x^-1").unwrap_err(),
            ParseError::BadExponent("-".into())
        );
        assert!(matches!(
            parse_expr::<f64>("(x + 1"),
            Err(ParseError::UnexpectedEnd)
        ));
        assert!(matches!(
            parse_expr::<f64>("x y"),
            Err(ParseError::TrailingInput { pos: 2 })
        ));
    }

    #[test]
    fn latex_parenthesises_only_where_needed() {
        let cases = [
            ("x^2 + 1", r"x^{2} + 1"),
            ("(x + 1)^2", r"(x + 1)^{2}"),
            ("a - (b - c)", r"a - (b - c)"),
            ("a - b - c", r"a - b - c"),
            ("2 * x * y", r"2 \cdot x \cdot y"),
            ("sqrt(x)", r"\sqrt{x}"),
            ("sin(x^2)", r"\sin\left(x^{2}\right)"),
            ("(x^2)^3", r"(x^{2})^{3}"),
            ("-(x + y)", r"-(x + y)"),
        ];
        for (src, want) in cases {
            let e: Expr64 = parse_expr(src).unwrap();
            assert_eq!(e.to_latex(), want, "input {src}");
        }
    }
}
