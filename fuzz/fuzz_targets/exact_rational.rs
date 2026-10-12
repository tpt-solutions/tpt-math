//! Fuzz exact rational arithmetic and interval enclosure. Ring identities must hold
//! exactly, and an interval sum must contain the sum of its endpoints.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tpt_math_exact::{BigInt, BigRational, Interval};

fn rational(bytes: &[u8]) -> BigRational {
    let n = i64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let d = i64::from_le_bytes(bytes[8..16].try_into().unwrap());
    // A zero denominator is not a rational; map it to 1 so the input stays useful.
    let d = if d == 0 { 1 } else { d };
    BigRational::new(BigInt::from(n), BigInt::from(d))
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 48 {
        return;
    }
    let a = rational(&data[0..16]);
    let b = rational(&data[16..32]);
    let c = rational(&data[32..48]);

    assert_eq!(&a + &b, &b + &a);
    assert_eq!((&a + &b) + &c, &a + (&b + &c));
    assert_eq!(&a * (&b + &c), &a * &b + &a * &c);
    assert_eq!((&a + &b) - &b, a.clone());
    if b != BigRational::from_integer(BigInt::from(0)) {
        assert_eq!((&a * &b) / &b, a.clone());
    }

    let (lo, hi) = if a <= b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) };
    let (lo2, hi2) = if c <= a { (c.clone(), a.clone()) } else { (a.clone(), c.clone()) };
    let i1 = Interval::new(lo.clone(), hi.clone());
    let i2 = Interval::new(lo2.clone(), hi2.clone());
    let sum = &i1 + &i2;
    assert!(sum.contains(&(&lo + &lo2)));
    assert!(sum.contains(&(&hi + &hi2)));
});
