//! Property tests for exact rational arithmetic and interval enclosures.

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
