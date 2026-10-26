//! Property tests for exact rational arithmetic and interval enclosures.

use num_traits::ToPrimitive;
use proptest::prelude::*;
use tpt_math_exact::{BigInt, BigRational, Interval};

fn rational(n: i64, d: i64) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

/// Rationals with small numerators and non-zero denominators.
fn rational_strategy() -> impl Strategy<Value = BigRational> {
    (-1000i64..1000, 1i64..1000).prop_map(|(n, d)| rational(n, d))
}

proptest! {
    #[test]
    fn addition_is_commutative_and_associative(
        a in rational_strategy(),
        b in rational_strategy(),
        c in rational_strategy(),
    ) {
        prop_assert_eq!(&a + &b, &b + &a);
        prop_assert_eq!((&a + &b) + &c, &a + (&b + &c));
    }

    #[test]
    fn multiplication_distributes_over_addition(
        a in rational_strategy(),
        b in rational_strategy(),
        c in rational_strategy(),
    ) {
        prop_assert_eq!(&a * (&b + &c), &a * &b + &a * &c);
    }

    #[test]
    fn subtraction_and_division_undo_their_operations(
        a in rational_strategy(),
        b in rational_strategy(),
    ) {
        prop_assert_eq!((&a + &b) - &b, a.clone());
        if b != BigRational::from_integer(BigInt::from(0)) {
            prop_assert_eq!((&a * &b) / &b, a);
        }
    }

    #[test]
    fn interval_sum_encloses_pointwise_sums(
        p in rational_strategy(),
        q in rational_strategy(),
        r in rational_strategy(),
        s in rational_strategy(),
    ) {
        let (lo1, hi1) = if p <= q { (p, q) } else { (q, p) };
        let (lo2, hi2) = if r <= s { (r, s) } else { (s, r) };
        let i1 = Interval::new(lo1.clone(), hi1.clone());
        let i2 = Interval::new(lo2.clone(), hi2.clone());
        let sum = &i1 + &i2;
        // Every pointwise sum of enclosed values lies in the sum interval.
        for x in [&lo1, &hi1] {
            for y in [&lo2, &hi2] {
                prop_assert!(sum.contains(&(x + y)));
            }
        }
    }
}

/// Reference result computed without any shortcut: cross-multiply, then reduce.
fn naive(n: i128, d: i128) -> BigRational {
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

proptest! {
    #[test]
    fn fast_paths_agree_with_naive_arithmetic(
        n1 in -1000i64..1000, d1 in 1i64..1000,
        n2 in -1000i64..1000, d2 in 1i64..1000,
    ) {
        let (n1, d1, n2, d2) = (n1 as i128, d1 as i128, n2 as i128, d2 as i128);
        let a = naive(n1, d1);
        let b = naive(n2, d2);
        prop_assert_eq!(&a + &b, naive(n1 * d2 + n2 * d1, d1 * d2));
        prop_assert_eq!(&a - &b, naive(n1 * d2 - n2 * d1, d1 * d2));
        prop_assert_eq!(&a * &b, naive(n1 * n2, d1 * d2));
        if n2 != 0 {
            prop_assert_eq!(&a / &b, naive(n1 * d2, d1 * n2));
        }
        prop_assert_eq!(a.cmp(&b), (n1 * d2).cmp(&(n2 * d1)));
    }

    #[test]
    fn to_f64_is_within_half_ulp_of_the_exact_value(n in -100_000i64..100_000, d in 1i64..100_000) {
        // Integers this small are exact in f64, so IEEE division is correctly rounded.
        prop_assert_eq!(rational(n, d).to_f64(), Some(n as f64 / d as f64));
    }
}

proptest! {
    #[test]
    fn display_then_parse_round_trips(a in rational_strategy()) {
        let text = a.to_string();
        prop_assert_eq!(text.parse::<BigRational>(), Ok(a));
    }

    #[test]
    fn floor_and_ceil_bracket_the_value(a in rational_strategy()) {
        let one = BigRational::from_integer(BigInt::from(1));
        prop_assert!(a.floor() <= a);
        prop_assert!(a <= a.ceil());
        prop_assert!(a.ceil() - a.floor() <= one);
    }

    #[test]
    fn directed_rounding_brackets_the_value(a in rational_strategy()) {
        let down = BigRational::from_float(a.to_f64_down()).unwrap();
        let up = BigRational::from_float(a.to_f64_up()).unwrap();
        prop_assert!(down <= a);
        prop_assert!(a <= up);
    }

    #[test]
    fn sqrt_bounds_contain_the_root(a in rational_strategy()) {
        if let Some((lo, hi)) = a.sqrt_bounds(24) {
            prop_assert!(&lo * &lo <= a);
            prop_assert!(&hi * &hi >= a);
        }
    }
}
