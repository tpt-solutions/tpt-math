//! Exact rational numbers over [`BigInt`].
//!
//! A [`BigRational`] is always stored in lowest terms with a positive
//! denominator, so structural equality and hashing agree with numeric equality.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cmp::Ordering;
use core::fmt;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use core::str::FromStr;

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{FromPrimitive, One, Signed, ToPrimitive, Zero};

/// An exact rational number `numer / denom`.
///
/// Invariants: `denom > 0` and `gcd(numer, denom) == 1`. Zero is stored as `0 / 1`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BigRational {
    numer: BigInt,
    denom: BigInt,
}

impl BigRational {
    /// Build `numer / denom` in lowest terms.
    ///
    /// # Panics
    ///
    /// Panics if `denom` is zero.
    pub fn new(numer: BigInt, denom: BigInt) -> Self {
        assert!(!denom.is_zero(), "rational denominator must not be zero");
        let (numer, denom) = if denom.is_negative() {
            (-numer, -denom)
        } else {
            (numer, denom)
        };
        let g = numer.gcd(&denom);
        if g.is_one() {
            BigRational { numer, denom }
        } else {
            BigRational {
                numer: numer / &g,
                denom: denom / g,
            }
        }
    }

    /// Build from a value already in lowest terms with a positive denominator.
    fn reduced(numer: BigInt, denom: BigInt) -> Self {
        debug_assert!(denom.is_positive());
        debug_assert!(numer.gcd(&denom).is_one() || numer.is_zero());
        BigRational { numer, denom }
    }

    /// The integer `n / 1`.
    pub fn from_integer(n: BigInt) -> Self {
        BigRational {
            numer: n,
            denom: BigInt::one(),
        }
    }

    /// The exact value of a finite `f64`, or `None` for `NaN` and infinities.
    pub fn from_float(x: f64) -> Option<Self> {
        if !x.is_finite() {
            return None;
        }
        let bits = x.to_bits();
        let negative = bits >> 63 == 1;
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let frac = bits & ((1u64 << 52) - 1);
        // Subnormals share the exponent of the smallest normal binade.
        let (mant, e) = if exp == 0 {
            (frac, -1074)
        } else {
            (frac | (1u64 << 52), exp - 1075)
        };
        let mut numer = BigInt::from(mant);
        let mut denom = BigInt::one();
        if e >= 0 {
            numer <<= e as usize;
        } else {
            denom <<= (-e) as usize;
        }
        if negative {
            numer = -numer;
        }
        Some(Self::new(numer, denom))
    }

    /// The numerator, carrying the sign.
    pub fn numer(&self) -> &BigInt {
        &self.numer
    }

    /// The denominator, always positive.
    pub fn denom(&self) -> &BigInt {
        &self.denom
    }

    /// True if the value is an integer.
    pub fn is_integer(&self) -> bool {
        self.denom.is_one()
    }

    /// The reciprocal `1 / self`.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero.
    pub fn recip(&self) -> Self {
        assert!(!self.numer.is_zero(), "reciprocal of zero");
        Self::new(self.denom.clone(), self.numer.clone())
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        if self.numer.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }

    /// `self` raised to an integer power. Negative powers invert the base.
    ///
    /// # Panics
    ///
    /// Panics if `self` is zero and `exp` is negative.
    pub fn pow(&self, exp: i32) -> Self {
        let k = exp.unsigned_abs();
        let p = Self::reduced(self.numer.pow(k), self.denom.pow(k));
        if exp >= 0 {
            p
        } else {
            p.recip()
        }
    }

    /// The nearest `f64`, ties to even. Returns `±inf` on overflow and signed
    /// zero on underflow.
    fn nearest_f64(&self) -> f64 {
        if self.numer.is_zero() {
            return 0.0;
        }
        let negative = self.numer.is_negative();
        let n = self.numer.abs();
        let d = &self.denom;

        // Choose k so that round(|x| * 2^k) has 53 bits, i.e. is in [2^52, 2^53).
        // The bit-length difference estimates the binary exponent to within one.
        let mut k = 52 - (n.bits() as i64 - d.bits() as i64);
        for _ in 0..4 {
            let bits = round_scaled(&n, d, k).bits();
            match bits {
                53 => break,
                b if b > 53 => k -= 1,
                _ => k += 1,
            }
        }
        // Below 2^-1022 the precision drops; the quantum is then fixed at 2^-1074.
        if k > 1074 {
            k = 1074;
        }
        let m = round_scaled(&n, d, k)
            .to_u64()
            .expect("rounded mantissa fits in 54 bits");

        // m * 2^-k. Scale in steps that keep every intermediate in normal range.
        let mut x = m as f64;
        let mut e = -k;
        while e > 0 {
            let s = e.min(1000);
            x *= pow2(s);
            e -= s;
        }
        while e < 0 {
            let s = (-e).min(1000);
            x *= pow2(-s);
            e += s;
        }
        if negative {
            -x
        } else {
            x
        }
    }
}

impl BigRational {
    /// The largest integer not greater than `self`.
    pub fn floor(&self) -> Self {
        Self::from_integer(self.numer.div_floor(&self.denom))
    }

    /// The smallest integer not less than `self`.
    pub fn ceil(&self) -> Self {
        -(-self).floor()
    }

    /// The integer part, rounding toward zero.
    pub fn trunc(&self) -> Self {
        Self::from_integer(&self.numer / &self.denom)
    }

    /// The fractional part, `self - self.trunc()`, which has the sign of `self`.
    pub fn fract(&self) -> Self {
        self - &self.trunc()
    }

    /// The nearest integer, with halves rounded away from zero.
    pub fn round(&self) -> Self {
        // floor(|x| + 1/2) = floor((2|n| + d) / 2d) for d > 0.
        let a = self.numer.abs();
        let t = (&a + &a + &self.denom) / (&self.denom + &self.denom);
        let magnitude = Self::from_integer(t);
        if self.numer.is_negative() {
            -magnitude
        } else {
            magnitude
        }
    }

    /// The largest `f64` not greater than `self`. Returns `-inf` below the
    /// finite range and `f64::MAX` above it.
    pub fn to_f64_down(&self) -> f64 {
        let f = self.nearest_f64();
        if f == f64::INFINITY {
            return f64::MAX;
        }
        match BigRational::from_float(f) {
            Some(exact) if exact > *self => next_f64(f, false),
            _ => f,
        }
    }

    /// The smallest `f64` not less than `self`. Returns `inf` above the finite
    /// range and `-f64::MAX` below it.
    pub fn to_f64_up(&self) -> f64 {
        let f = self.nearest_f64();
        if f == f64::NEG_INFINITY {
            return -f64::MAX;
        }
        match BigRational::from_float(f) {
            Some(exact) if exact < *self => next_f64(f, true),
            _ => f,
        }
    }

    /// Rational bounds `lo <= sqrt(self) <= hi` with `hi - lo = 1 / (denom * 2^bits)`.
    /// Returns `None` for negative values.
    pub fn sqrt_bounds(&self, bits: u32) -> Option<(Self, Self)> {
        if self.numer.is_negative() {
            return None;
        }
        // sqrt(n/d) * d * 2^bits = sqrt(n * d * 4^bits), so the integer square
        // root of that gives the bounds directly.
        let scaled = (&self.numer * &self.denom) << ((2 * bits) as usize);
        let r = scaled.sqrt();
        let scale = &self.denom << bits as usize;
        let lo = Self::new(r.clone(), scale.clone());
        let hi = Self::new(r + BigInt::one(), scale);
        Some((lo, hi))
    }

    /// The continued-fraction expansion `[a0; a1, a2, ...]`.
    pub fn continued_fraction(&self) -> Vec<BigInt> {
        let mut terms = Vec::new();
        let (mut p, mut q) = (self.numer.clone(), self.denom.clone());
        loop {
            let a = p.div_floor(&q);
            let r = &p - &a * &q;
            terms.push(a);
            if r.is_zero() {
                break;
            }
            p = q;
            q = r;
        }
        terms
    }

    /// The rational with continued-fraction expansion `terms`, or `None` if
    /// `terms` is empty or an intermediate quotient would divide by zero.
    pub fn from_continued_fraction(terms: &[BigInt]) -> Option<Self> {
        let (last, rest) = terms.split_last()?;
        let mut x = Self::from_integer(last.clone());
        for a in rest.iter().rev() {
            if x.is_zero() {
                return None;
            }
            x = Self::from_integer(a.clone()) + x.recip();
        }
        Some(x)
    }

    /// The closest rational to `self` whose denominator is at most `max_den`.
    ///
    /// # Panics
    ///
    /// Panics if `max_den` is not positive.
    pub fn limit_denominator(&self, max_den: &BigInt) -> Self {
        assert!(max_den.is_positive(), "limit denominator must be positive");
        if self.denom <= *max_den {
            return self.clone();
        }
        if self.numer.is_negative() {
            let positive = self.abs().limit_denominator(max_den);
            return -positive;
        }
        // Walk the continued-fraction convergents of self while the
        // denominator stays within bounds, then pick the better of the last
        // convergent and the semiconvergent that follows it.
        let (mut n, mut d) = (self.numer.clone(), self.denom.clone());
        let (mut p0, mut q0) = (BigInt::zero(), BigInt::one());
        let (mut p1, mut q1) = (BigInt::one(), BigInt::zero());
        loop {
            let a = &n / &d;
            let q2 = &q0 + &a * &q1;
            if q2 > *max_den {
                break;
            }
            let p2 = &p0 + &a * &p1;
            p0 = p1;
            q0 = q1;
            p1 = p2;
            q1 = q2;
            let r = &n - &a * &d;
            n = d;
            d = r;
        }
        let k = (max_den - &q0) / &q1;
        let bound1 = Self::new(&p0 + &k * &p1, &q0 + &k * &q1);
        let bound2 = Self::new(p1, q1);
        let err1 = (&bound1 - self).abs();
        let err2 = (&bound2 - self).abs();
        if err2 <= err1 {
            bound2
        } else {
            bound1
        }
    }

    /// Decimal text with exactly `digits` fractional digits. The value is
    /// rounded to that precision, halves away from zero.
    pub fn to_decimal_string(&self, digits: u32) -> String {
        let scale = BigInt::from(10u32).pow(digits);
        let scaled = (self * &Self::from_integer(scale.clone())).round();
        let n = scaled.numer();
        let sign = if n.is_negative() { "-" } else { "" };
        let magnitude = n.abs();
        let int_part = &magnitude / &scale;
        if digits == 0 {
            return format!("{sign}{int_part}");
        }
        let frac = (&magnitude % &scale).to_string();
        let padding = "0".repeat(digits as usize - frac.len());
        format!("{sign}{int_part}.{padding}{frac}")
    }
}

/// The neighbouring `f64` above (`up`) or below `x`. `x` must be finite.
fn next_f64(x: f64, up: bool) -> f64 {
    if x == 0.0 {
        let smallest = f64::from_bits(1);
        return if up { smallest } else { -smallest };
    }
    // Moving away from zero increases the magnitude bits.
    let away = (x > 0.0) == up;
    let bits = x.to_bits();
    f64::from_bits(if away { bits + 1 } else { bits - 1 })
}

/// Error returned when a string is not a valid [`BigRational`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseRationalError {
    /// The text is not a fraction `p/q` or a decimal number.
    Invalid,
    /// The denominator of a fraction is zero.
    ZeroDenominator,
}

impl fmt::Display for ParseRationalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseRationalError::Invalid => write!(f, "invalid rational literal"),
            ParseRationalError::ZeroDenominator => write!(f, "rational denominator is zero"),
        }
    }
}

impl core::error::Error for ParseRationalError {}

fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// Splits an optional leading `+` or `-` off `s`.
fn split_sign(s: &str) -> (bool, &str) {
    match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    }
}

fn parse_digits(s: &str) -> Option<BigInt> {
    if is_digits(s) {
        BigInt::parse_bytes(s.as_bytes(), 10)
    } else {
        None
    }
}

impl FromStr for BigRational {
    type Err = ParseRationalError;

    /// Parses `p/q` (integers, `q` unsigned) or a decimal such as `-1.25e3`.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use ParseRationalError::{Invalid, ZeroDenominator};

        if let Some((p, q)) = s.split_once('/') {
            let (negative, body) = split_sign(p);
            let numer = parse_digits(body).ok_or(Invalid)?;
            let denom = parse_digits(q).ok_or(Invalid)?;
            if denom.is_zero() {
                return Err(ZeroDenominator);
            }
            let numer = if negative { -numer } else { numer };
            return Ok(Self::new(numer, denom));
        }

        let (mantissa, exponent) = match s.find(['e', 'E']) {
            Some(i) => (&s[..i], s[i + 1..].parse::<i64>().map_err(|_| Invalid)?),
            None => (s, 0),
        };
        let (negative, body) = split_sign(mantissa);
        let (int_digits, frac_digits) = body.split_once('.').unwrap_or((body, ""));
        if int_digits.is_empty() && frac_digits.is_empty() {
            return Err(Invalid);
        }
        let all_digits = |t: &str| t.is_empty() || is_digits(t);
        if !all_digits(int_digits) || !all_digits(frac_digits) {
            return Err(Invalid);
        }
        let digits = format!("{int_digits}{frac_digits}");
        let digits = if digits.is_empty() { "0" } else { &digits };
        let mut value = parse_digits(digits).ok_or(Invalid)?;

        // value * 10^(exponent - frac_len)
        let shift = exponent
            .checked_sub(frac_digits.len() as i64)
            .ok_or(Invalid)?;
        let power = u32::try_from(shift.unsigned_abs()).map_err(|_| Invalid)?;
        let scale = BigInt::from(10u32).pow(power);
        if negative {
            value = -value;
        }
        if shift >= 0 {
            Ok(Self::from_integer(value * scale))
        } else {
            Ok(Self::new(value, scale))
        }
    }
}

/// `2^s` for `-1000 <= s <= 1000`, which are always normal.
fn pow2(s: i64) -> f64 {
    f64::from_bits(((s + 1023) as u64) << 52)
}

/// `round(n / d * 2^k)` with ties to even, for `n, d >= 0` and `d > 0`.
fn round_scaled(n: &BigInt, d: &BigInt, k: i64) -> BigInt {
    let (num, den) = if k >= 0 {
        (n << (k as usize), d.clone())
    } else {
        (n.clone(), d << ((-k) as usize))
    };
    let (q, r) = num.div_rem(&den);
    let twice = &r + &r;
    if twice > den || (twice == den && q.is_odd()) {
        q + BigInt::one()
    } else {
        q
    }
}

// Arithmetic. Each operation returns a value already in lowest terms.

fn add_ref(a: &BigRational, b: &BigRational) -> BigRational {
    if a.denom.is_one() && b.denom.is_one() {
        return BigRational::from_integer(&a.numer + &b.numer);
    }
    // Knuth: with g = gcd(b, d), only the common factor g needs to be removed
    // from the cross terms, keeping intermediates smaller than a*d + c*b.
    let g = a.denom.gcd(&b.denom);
    if g.is_one() {
        // Coprime denominators: the sum is already reduced.
        let numer = &a.numer * &b.denom + &b.numer * &a.denom;
        return BigRational::reduced(numer, &a.denom * &b.denom);
    }
    let bp = &a.denom / &g;
    let dp = &b.denom / &g;
    let t = &a.numer * &dp + &b.numer * &bp;
    let g2 = t.gcd(&g);
    let numer = t / &g2;
    let denom = &a.denom * &dp / g2;
    BigRational::reduced(numer, denom)
}

fn mul_ref(a: &BigRational, b: &BigRational) -> BigRational {
    if a.numer.is_zero() || b.numer.is_zero() {
        return BigRational::from_integer(BigInt::zero());
    }
    if a.denom.is_one() && b.denom.is_one() {
        return BigRational::from_integer(&a.numer * &b.numer);
    }
    // Cross-reduce before multiplying so the product is reduced without a
    // large post-multiplication gcd.
    let g1 = a.numer.gcd(&b.denom);
    let g2 = b.numer.gcd(&a.denom);
    let numer = (&a.numer / &g1) * (&b.numer / &g2);
    let denom = (&a.denom / &g2) * (&b.denom / &g1);
    BigRational::reduced(numer, denom)
}

fn div_ref(a: &BigRational, b: &BigRational) -> BigRational {
    assert!(!b.numer.is_zero(), "rational division by zero");
    mul_ref(a, &b.recip())
}

macro_rules! impl_binop {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $imp:ident) => {
        impl $trait<BigRational> for BigRational {
            type Output = BigRational;
            fn $method(self, rhs: BigRational) -> BigRational {
                $imp(&self, &rhs)
            }
        }
        impl $trait<&BigRational> for BigRational {
            type Output = BigRational;
            fn $method(self, rhs: &BigRational) -> BigRational {
                $imp(&self, rhs)
            }
        }
        impl $trait<BigRational> for &BigRational {
            type Output = BigRational;
            fn $method(self, rhs: BigRational) -> BigRational {
                $imp(self, &rhs)
            }
        }
        impl $trait<&BigRational> for &BigRational {
            type Output = BigRational;
            fn $method(self, rhs: &BigRational) -> BigRational {
                $imp(self, rhs)
            }
        }
        impl $assign_trait<BigRational> for BigRational {
            fn $assign_method(&mut self, rhs: BigRational) {
                *self = $imp(self, &rhs);
            }
        }
        impl $assign_trait<&BigRational> for BigRational {
            fn $assign_method(&mut self, rhs: &BigRational) {
                *self = $imp(self, rhs);
            }
        }
    };
}

impl_binop!(Add, add, AddAssign, add_assign, add_ref);
impl_binop!(Mul, mul, MulAssign, mul_assign, mul_ref);
impl_binop!(Div, div, DivAssign, div_assign, div_ref);

impl Sub<BigRational> for BigRational {
    type Output = BigRational;
    fn sub(self, rhs: BigRational) -> BigRational {
        add_ref(&self, &-rhs)
    }
}

impl Sub<&BigRational> for BigRational {
    type Output = BigRational;
    fn sub(self, rhs: &BigRational) -> BigRational {
        add_ref(&self, &-rhs.clone())
    }
}

impl Sub<BigRational> for &BigRational {
    type Output = BigRational;
    fn sub(self, rhs: BigRational) -> BigRational {
        add_ref(self, &-rhs)
    }
}

impl Sub<&BigRational> for &BigRational {
    type Output = BigRational;
    fn sub(self, rhs: &BigRational) -> BigRational {
        add_ref(self, &-rhs.clone())
    }
}

impl SubAssign<BigRational> for BigRational {
    fn sub_assign(&mut self, rhs: BigRational) {
        *self = add_ref(self, &-rhs);
    }
}

impl SubAssign<&BigRational> for BigRational {
    fn sub_assign(&mut self, rhs: &BigRational) {
        *self = add_ref(self, &-rhs.clone());
    }
}

impl Neg for BigRational {
    type Output = BigRational;
    fn neg(self) -> BigRational {
        BigRational {
            numer: -self.numer,
            denom: self.denom,
        }
    }
}

impl Neg for &BigRational {
    type Output = BigRational;
    fn neg(self) -> BigRational {
        -self.clone()
    }
}

impl PartialOrd for BigRational {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BigRational {
    fn cmp(&self, other: &Self) -> Ordering {
        // Denominators are positive, so cross-multiplication preserves order.
        if self.denom == other.denom {
            return self.numer.cmp(&other.numer);
        }
        (&self.numer * &other.denom).cmp(&(&other.numer * &self.denom))
    }
}

impl fmt::Display for BigRational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom.is_one() {
            write!(f, "{}", self.numer)
        } else {
            write!(f, "{}/{}", self.numer, self.denom)
        }
    }
}

impl From<BigInt> for BigRational {
    fn from(n: BigInt) -> Self {
        BigRational::from_integer(n)
    }
}

impl Zero for BigRational {
    fn zero() -> Self {
        BigRational::from_integer(BigInt::zero())
    }
    fn is_zero(&self) -> bool {
        self.numer.is_zero()
    }
}

impl One for BigRational {
    fn one() -> Self {
        BigRational::from_integer(BigInt::one())
    }
}

impl ToPrimitive for BigRational {
    fn to_i64(&self) -> Option<i64> {
        if self.is_integer() {
            self.numer.to_i64()
        } else {
            None
        }
    }
    fn to_u64(&self) -> Option<u64> {
        if self.is_integer() {
            self.numer.to_u64()
        } else {
            None
        }
    }
    fn to_f64(&self) -> Option<f64> {
        Some(self.nearest_f64())
    }
}

impl FromPrimitive for BigRational {
    fn from_i64(n: i64) -> Option<Self> {
        Some(BigRational::from_integer(BigInt::from(n)))
    }
    fn from_u64(n: u64) -> Option<Self> {
        Some(BigRational::from_integer(BigInt::from(n)))
    }
    fn from_f64(n: f64) -> Option<Self> {
        BigRational::from_float(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(n: i64, d: i64) -> BigRational {
        BigRational::new(BigInt::from(n), BigInt::from(d))
    }

    fn z(n: i64) -> BigRational {
        BigRational::from_integer(BigInt::from(n))
    }

    #[test]
    fn normalizes_to_lowest_terms_with_positive_denominator() {
        assert_eq!(q(2, 4), q(1, 2));
        assert_eq!(q(3, -6), q(-1, 2));
        assert_eq!(q(-4, -8), q(1, 2));
        assert_eq!(q(0, 5), z(0));
        assert_eq!(q(0, -5).denom(), &BigInt::one());
    }

    #[test]
    #[should_panic(expected = "denominator must not be zero")]
    fn zero_denominator_panics() {
        let _ = q(1, 0);
    }

    #[test]
    fn arithmetic_matches_hand_results() {
        assert_eq!(&q(1, 3) + &q(1, 6), q(1, 2));
        assert_eq!(&q(1, 2) - &q(1, 3), q(1, 6));
        assert_eq!(&q(2, 3) * &q(3, 4), q(1, 2));
        assert_eq!(&q(1, 2) / &q(1, 4), z(2));
        assert_eq!(-q(3, 7), q(-3, 7));
        assert_eq!(q(3, 4).pow(2), q(9, 16));
        assert_eq!(q(2, 3).pow(-2), q(9, 4));
        assert_eq!(q(2, 3).pow(0), z(1));
    }

    #[test]
    fn assign_ops_match_binary_ops() {
        let mut a = q(1, 3);
        a += q(1, 6);
        assert_eq!(a, q(1, 2));
        a -= &q(1, 4);
        assert_eq!(a, q(1, 4));
        a *= q(4, 1);
        assert_eq!(a, z(1));
        a /= &q(2, 1);
        assert_eq!(a, q(1, 2));
    }

    #[test]
    fn ordering_is_numeric() {
        assert!(q(-1, 2) < q(1, 3));
        assert!(q(1, 3) < z(1));
        assert!(q(2, 4) == q(1, 2));
        assert!(q(7, 3) > q(9, 4));
        assert!(q(-7, 3) < q(-9, 4));
    }

    #[test]
    fn display_forms() {
        use alloc::string::ToString;
        assert_eq!(q(1, 2).to_string(), "1/2");
        assert_eq!(q(-1, 2).to_string(), "-1/2");
        assert_eq!(q(6, 3).to_string(), "2");
        assert_eq!(z(0).to_string(), "0");
    }

    #[test]
    fn from_float_is_exact() {
        assert_eq!(BigRational::from_float(0.75), Some(q(3, 4)));
        assert_eq!(BigRational::from_float(-2.0), Some(z(-2)));
        assert_eq!(BigRational::from_float(-0.0), Some(z(0)));
        assert_eq!(BigRational::from_float(f64::NAN), None);
        assert_eq!(BigRational::from_float(f64::INFINITY), None);
        assert_eq!(BigRational::from_float(f64::NEG_INFINITY), None);
        // 0.1 is not 1/10 in binary; the exact value is m / 2^55.
        let tenth = BigRational::from_float(0.1).unwrap();
        assert_ne!(tenth, q(1, 10));
        assert_eq!(tenth.nearest_f64(), 0.1);
    }

    #[test]
    fn to_f64_round_trips_floats() {
        let xs = [
            0.1,
            -2.5,
            1e300,
            -1e-300,
            f64::MAX,
            f64::MIN_POSITIVE,
            5e-324,
            -5e-324,
            1.0 / 3.0,
        ];
        for &x in &xs {
            assert_eq!(
                BigRational::from_float(x).unwrap().nearest_f64(),
                x,
                "{x:e}"
            );
        }
    }

    #[test]
    fn to_f64_is_correctly_rounded_for_small_fractions() {
        // Integers below 2^53 are exact in f64, so IEEE division is correctly rounded.
        for n in -40i64..40 {
            for d in 1i64..40 {
                assert_eq!(q(n, d).nearest_f64(), n as f64 / d as f64, "{n}/{d}");
            }
        }
    }

    #[test]
    fn to_f64_overflow_and_underflow() {
        let huge = BigRational::from_integer(BigInt::from(10).pow(400));
        assert_eq!(huge.nearest_f64(), f64::INFINITY);
        let tiny = BigRational::new(BigInt::one(), BigInt::from(10).pow(400));
        assert_eq!(tiny.nearest_f64(), 0.0);
    }

    #[test]
    fn to_f64_ties_round_to_even() {
        // 2^53 + 1 lies exactly between 2^53 and 2^53 + 2; ties go to the even mantissa.
        let x = BigRational::from_integer((BigInt::one() << 53usize) + BigInt::one());
        assert_eq!(x.nearest_f64(), 9007199254740992.0);
    }

    #[test]
    fn rounding_family_on_signed_halves() {
        assert_eq!(q(-7, 2).floor(), z(-4));
        assert_eq!(q(-7, 2).ceil(), z(-3));
        assert_eq!(q(-7, 2).trunc(), z(-3));
        assert_eq!(q(-7, 2).round(), z(-4));
        assert_eq!(q(7, 2).round(), z(4));
        assert_eq!(q(-1, 2).round(), z(-1));
        assert_eq!(q(1, 2).round(), z(1));
        assert_eq!(q(-7, 2).fract(), q(-1, 2));
        assert_eq!(z(5).floor(), z(5));
        assert_eq!(q(1, 3).ceil(), z(1));
    }

    #[test]
    fn continued_fraction_round_trip() {
        let x = q(415, 93);
        let terms: Vec<BigInt> = [4, 2, 6, 7].iter().map(|&a| BigInt::from(a)).collect();
        assert_eq!(x.continued_fraction(), terms);
        assert_eq!(BigRational::from_continued_fraction(&terms), Some(x));
        assert_eq!(BigRational::from_continued_fraction(&[]), None);
        // [0; 0] would divide by zero.
        let bad = [BigInt::from(0), BigInt::from(0)];
        assert_eq!(BigRational::from_continued_fraction(&bad), None);
    }

    #[test]
    fn limit_denominator_finds_closest_fraction() {
        let max = BigInt::from(10);
        assert_eq!(q(3333, 10000).limit_denominator(&max), q(1, 3));
        assert_eq!(q(-3333, 10000).limit_denominator(&max), q(-1, 3));
        assert_eq!(q(2, 5).limit_denominator(&max), q(2, 5));
    }

    #[test]
    fn sqrt_bounds_bracket_the_root() {
        let (lo, hi) = q(2, 1).sqrt_bounds(20).unwrap();
        assert!(&lo * &lo <= q(2, 1));
        assert!(&hi * &hi >= q(2, 1));
        assert_eq!(
            &hi - &lo,
            BigRational::new(BigInt::one(), BigInt::one() << 20usize)
        );

        let (lo, hi) = q(9, 4).sqrt_bounds(10).unwrap();
        assert!(lo <= q(3, 2) && q(3, 2) <= hi);
        assert_eq!(q(-1, 4).sqrt_bounds(10), None);
    }

    #[test]
    fn directed_rounding_brackets_the_value() {
        let x = q(1, 10);
        let down = BigRational::from_float(x.to_f64_down()).unwrap();
        let up = BigRational::from_float(x.to_f64_up()).unwrap();
        assert!(down <= x && x <= up);
        assert!(down < up, "0.1 is not a binary float, so the bounds differ");
        // Exact floats round to themselves in both directions.
        assert_eq!(z(3).to_f64_down(), 3.0);
        assert_eq!(z(3).to_f64_up(), 3.0);
        // Beyond the finite range.
        let huge = BigRational::from_integer(BigInt::from(10).pow(400));
        assert_eq!(huge.to_f64_down(), f64::MAX);
        assert_eq!(huge.to_f64_up(), f64::INFINITY);
    }

    #[test]
    fn decimal_string_rounds_half_away_from_zero() {
        assert_eq!(q(1, 3).to_decimal_string(4), "0.3333");
        assert_eq!(q(-1, 8).to_decimal_string(2), "-0.13");
        assert_eq!(q(5, 2).to_decimal_string(0), "3");
        assert_eq!(q(-1, 1000).to_decimal_string(2), "0.00");
        assert_eq!(z(2).to_decimal_string(0), "2");
        assert_eq!(q(1, 20).to_decimal_string(3), "0.050");
    }

    #[test]
    fn parses_fractions_and_decimals() {
        use ParseRationalError::{Invalid, ZeroDenominator};
        assert_eq!("1/2".parse::<BigRational>(), Ok(q(1, 2)));
        assert_eq!("-6/4".parse::<BigRational>(), Ok(q(-3, 2)));
        assert_eq!("0.125".parse::<BigRational>(), Ok(q(1, 8)));
        assert_eq!("-1.5e2".parse::<BigRational>(), Ok(z(-150)));
        assert_eq!("1e-3".parse::<BigRational>(), Ok(q(1, 1000)));
        assert_eq!(".5".parse::<BigRational>(), Ok(q(1, 2)));
        assert_eq!("5.".parse::<BigRational>(), Ok(z(5)));
        assert_eq!("+7".parse::<BigRational>(), Ok(z(7)));
        assert_eq!("3/-4".parse::<BigRational>(), Err(Invalid));
        assert_eq!("1/0".parse::<BigRational>(), Err(ZeroDenominator));
        assert_eq!("".parse::<BigRational>(), Err(Invalid));
        assert_eq!(".".parse::<BigRational>(), Err(Invalid));
        assert_eq!("1.2.3".parse::<BigRational>(), Err(Invalid));
        assert_eq!("e5".parse::<BigRational>(), Err(Invalid));
        assert_eq!("abc".parse::<BigRational>(), Err(Invalid));
    }
}
