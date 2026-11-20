//! Space-filling-curve keys: Morton (Z-order) and Hilbert.
//!
//! Pure integer arithmetic. Morton places bit `b` of axis `a` at key bit
//! `D·b + a` (axis 0 least significant). Hilbert keys follow Skilling's
//! transpose algorithm, valid in any dimension; consecutive keys are always
//! unit-step neighbours on the grid.

use tpt_math_numeric::Scalar;

// ---------------------------------------------------------------- Morton 2D

fn spread2(x: u32) -> u64 {
    let mut x = u64::from(x);
    x = (x | (x << 16)) & 0x0000_ffff_0000_ffff;
    x = (x | (x << 8)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x << 2)) & 0x3333_3333_3333_3333;
    x = (x | (x << 1)) & 0x5555_5555_5555_5555;
    x
}

fn compact2(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | (x >> 1)) & 0x3333_3333_3333_3333;
    x = (x | (x >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
    x = (x | (x >> 4)) & 0x00ff_00ff_00ff_00ff;
    x = (x | (x >> 8)) & 0x0000_ffff_0000_ffff;
    x = (x | (x >> 16)) & 0x0000_0000_ffff_ffff;
    x as u32
}

/// 2D Morton key (32 bits per axis).
pub fn morton2_encode(x: u32, y: u32) -> u64 {
    spread2(x) | (spread2(y) << 1)
}

/// Inverse of [`morton2_encode`].
pub fn morton2_decode(key: u64) -> [u32; 2] {
    [compact2(key), compact2(key >> 1)]
}

// ---------------------------------------------------------------- Morton 3D

fn spread3(x: u32) -> u64 {
    let mut x = u64::from(x) & 0x1f_ffff;
    x = (x | (x << 32)) & 0x001f_0000_0000_ffff;
    x = (x | (x << 16)) & 0x001f_0000_ff00_00ff;
    x = (x | (x << 8)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x << 4)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x << 2)) & 0x1249_2492_4924_9249;
    x
}

fn compact3(mut x: u64) -> u32 {
    x &= 0x1249_2492_4924_9249;
    x = (x | (x >> 2)) & 0x10c3_0c30_c30c_30c3;
    x = (x | (x >> 4)) & 0x100f_00f0_0f00_f00f;
    x = (x | (x >> 8)) & 0x001f_0000_ff00_00ff;
    x = (x | (x >> 16)) & 0x001f_0000_0000_ffff;
    x = (x | (x >> 32)) & 0x0000_0000_001f_ffff;
    x as u32
}

/// Maximum coordinate for [`morton3_encode`] (21 bits per axis).
pub const MORTON3_MAX: u32 = (1 << 21) - 1;

/// 3D Morton key. Coordinates are masked to 21 bits.
pub fn morton3_encode(c: [u32; 3]) -> u64 {
    spread3(c[0]) | (spread3(c[1]) << 1) | (spread3(c[2]) << 2)
}

/// Inverse of [`morton3_encode`].
pub fn morton3_decode(key: u64) -> [u32; 3] {
    [compact3(key), compact3(key >> 1), compact3(key >> 2)]
}

// ---------------------------------------------------------------- Morton 4D

fn spread4(x: u32) -> u64 {
    let mut x = u64::from(x) & 0xffff;
    x = (x | (x << 24)) & 0x0000_00ff_0000_00ff;
    x = (x | (x << 12)) & 0x000f_000f_000f_000f;
    x = (x | (x << 6)) & 0x0303_0303_0303_0303;
    x = (x | (x << 3)) & 0x1111_1111_1111_1111;
    x
}

fn compact4(mut x: u64) -> u32 {
    x &= 0x1111_1111_1111_1111;
    x = (x | (x >> 3)) & 0x0303_0303_0303_0303;
    x = (x | (x >> 6)) & 0x000f_000f_000f_000f;
    x = (x | (x >> 12)) & 0x0000_00ff_0000_00ff;
    x = (x | (x >> 24)) & 0x0000_0000_0000_ffff;
    x as u32
}

/// Maximum coordinate for [`morton4_encode`] (16 bits per axis).
pub const MORTON4_MAX: u32 = (1 << 16) - 1;

/// 4D Morton key, e.g. `(x, y, z, t)`. Coordinates are masked to 16 bits.
pub fn morton4_encode(c: [u32; 4]) -> u64 {
    spread4(c[0]) | (spread4(c[1]) << 1) | (spread4(c[2]) << 2) | (spread4(c[3]) << 3)
}

/// Inverse of [`morton4_encode`].
pub fn morton4_decode(key: u64) -> [u32; 4] {
    [
        compact4(key),
        compact4(key >> 1),
        compact4(key >> 2),
        compact4(key >> 3),
    ]
}

// ------------------------------------------------------------------ Hilbert

fn hilbert_dims_ok<const N: usize>(bits: u32) -> bool {
    N >= 1 && (1..=32).contains(&bits) && (bits as usize) * N <= 64
}

/// Hilbert key of an `N`-dimensional grid point with `bits` bits per axis.
///
/// Coordinates are masked to `bits` bits. Returns `None` if `bits == 0`,
/// `bits > 32`, or `bits · N > 64`.
pub fn hilbert_encode<const N: usize>(coords: [u32; N], bits: u32) -> Option<u64> {
    if !hilbert_dims_ok::<N>(bits) {
        return None;
    }
    let mask = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let mut x = coords.map(|c| c & mask);
    let m = 1u32 << (bits - 1);

    // Inverse undo excess work.
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..N {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    // Gray encode.
    for i in 1..N {
        x[i] ^= x[i - 1];
    }
    let mut t = 0u32;
    let mut q = m;
    while q > 1 {
        if x[N - 1] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for xi in &mut x {
        *xi ^= t;
    }

    // Interleave the transposed form, axis 0 most significant per level.
    let mut key = 0u64;
    for b in (0..bits).rev() {
        for xi in &x {
            key = (key << 1) | u64::from((xi >> b) & 1);
        }
    }
    Some(key)
}

/// Inverse of [`hilbert_encode`]. Same validity rules; additionally returns
/// `None` if `key` has bits above `bits · N`.
pub fn hilbert_decode<const N: usize>(key: u64, bits: u32) -> Option<[u32; N]> {
    if !hilbert_dims_ok::<N>(bits) {
        return None;
    }
    let total = bits as usize * N;
    if total < 64 && key >> total != 0 {
        return None;
    }
    let mut x = [0u32; N];
    let mut pos = total;
    for b in (0..bits).rev() {
        for xi in &mut x {
            pos -= 1;
            *xi |= (((key >> pos) & 1) as u32) << b;
        }
    }

    // Gray decode by H ^ (H / 2).
    let t = x[N - 1] >> 1;
    for i in (1..N).rev() {
        x[i] ^= x[i - 1];
    }
    x[0] ^= t;
    // Undo excess work.
    let top = if bits == 32 {
        None
    } else {
        Some(2u32 << (bits - 1))
    };
    let mut q = 2u32;
    loop {
        if Some(q) == top {
            break;
        }
        let p = q - 1;
        for i in (0..N).rev() {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        match q.checked_mul(2) {
            Some(n) => q = n,
            None => break,
        }
    }
    Some(x)
}

// ------------------------------------------------------------- Quantisation

/// Maps `v ∈ [lo, hi]` to an integer in `[0, 2^bits − 1]` (clamping; NaN and a
/// degenerate range map to 0). `bits` is clamped to `1..=32`.
pub fn quantize<T: Scalar>(v: T, lo: T, hi: T, bits: u32) -> u32 {
    let bits = bits.clamp(1, 32);
    let extent = hi - lo;
    if !(extent > T::zero()) || !extent.is_finite() || v.is_nan() {
        return 0;
    }
    let max = if bits == 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    };
    let u = ((v - lo) / extent).max(T::zero()).min(T::one());
    let max_t = <T as tpt_math_numeric::NumCast>::from(max).unwrap_or_else(T::one);
    let scaled = (u * max_t).round();
    <u32 as tpt_math_numeric::NumCast>::from(scaled)
        .unwrap_or(max)
        .min(max)
}

/// Quantises a point against an axis-aligned box; see [`quantize`].
pub fn quantize_point<T: Scalar, const N: usize>(
    p: [T; N],
    min: [T; N],
    max: [T; N],
    bits: u32,
) -> [u32; N] {
    let mut out = [0u32; N];
    for i in 0..N {
        out[i] = quantize(p[i], min[i], max[i], bits);
    }
    out
}

/// Hilbert key of a float point within `[min, max]`. `None` for invalid
/// `bits` (see [`hilbert_encode`]).
pub fn hilbert_key<T: Scalar, const N: usize>(
    p: [T; N],
    min: [T; N],
    max: [T; N],
    bits: u32,
) -> Option<u64> {
    hilbert_encode(quantize_point(p, min, max, bits), bits)
}

/// Sorts points in place along the Hilbert curve of the box `[min, max]`.
/// No allocation; keys are recomputed per comparison. Returns `false`
/// (leaving the slice untouched) for invalid `bits`.
pub fn sort_by_hilbert<T: Scalar, const N: usize>(
    points: &mut [[T; N]],
    min: [T; N],
    max: [T; N],
    bits: u32,
) -> bool {
    if !hilbert_dims_ok::<N>(bits) {
        return false;
    }
    points.sort_unstable_by_key(|p| hilbert_key(*p, min, max, bits).unwrap_or(0));
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_interleave<const N: usize>(c: [u32; N], bits: u32) -> u64 {
        let mut key = 0u64;
        for b in 0..bits {
            for a in 0..N {
                key |= u64::from((c[a] >> b) & 1) << (b as usize * N + a);
            }
        }
        key
    }

    #[test]
    fn morton_matches_naive_interleave() {
        let samples = [0u32, 1, 2, 3, 0x55, 0xaaaa, 0xffff, 0x1f_ffff, 123_456, 77];
        for &a in &samples {
            for &b in &samples {
                assert_eq!(morton2_encode(a, b), naive_interleave([a, b], 32));
                let c = [a & 0x1f_ffff, b & 0x1f_ffff, (a ^ b) & 0x1f_ffff];
                assert_eq!(morton3_encode(c), naive_interleave(c, 21));
                assert_eq!(morton3_decode(morton3_encode(c)), c);
                let d = [a & 0xffff, b & 0xffff, (a ^ b) & 0xffff, (a + b) & 0xffff];
                assert_eq!(morton4_encode(d), naive_interleave(d, 16));
                assert_eq!(morton4_decode(morton4_encode(d)), d);
                assert_eq!(morton2_decode(morton2_encode(a, b)), [a, b]);
            }
        }
        assert_eq!(morton2_encode(u32::MAX, u32::MAX), u64::MAX);
    }

    #[test]
    fn morton3_exhaustive_small() {
        for x in 0..8u32 {
            for y in 0..8 {
                for z in 0..8 {
                    let k = morton3_encode([x, y, z]);
                    assert_eq!(k, naive_interleave([x, y, z], 3));
                    assert_eq!(morton3_decode(k), [x, y, z]);
                }
            }
        }
    }

    /// Exhaustively checks bijection and unit-step adjacency on a full grid.
    fn check_hilbert<const N: usize>(bits: u32) {
        let side = 1u32 << bits;
        let total = (side as u64).pow(N as u32);
        let mut prev: Option<[u32; N]> = None;
        for key in 0..total {
            let p = hilbert_decode::<N>(key, bits).unwrap();
            assert_eq!(hilbert_encode(p, bits), Some(key), "roundtrip at {key}");
            assert!(p.iter().all(|&c| c < side));
            if let Some(q) = prev {
                let dist: u32 = (0..N).map(|i| p[i].abs_diff(q[i])).sum();
                assert_eq!(dist, 1, "keys {} and {key} not adjacent", key - 1);
            }
            prev = Some(p);
        }
    }

    #[test]
    fn hilbert_is_a_continuous_bijection() {
        check_hilbert::<2>(5);
        check_hilbert::<3>(4);
        check_hilbert::<4>(3);
        check_hilbert::<1>(6);
    }

    #[test]
    fn hilbert_wide_roundtrip() {
        for &(a, b, c) in &[
            (0u32, 0u32, 0u32),
            (1, 2, 3),
            (0x1f_ffff, 0, 0x10_0000),
            (12345, 678_910, 1_111_111),
        ] {
            let k = hilbert_encode([a, b, c], 21).unwrap();
            assert_eq!(hilbert_decode::<3>(k, 21), Some([a, b, c]));
        }
        let k = hilbert_encode([u32::MAX, 0], 32).unwrap();
        assert_eq!(hilbert_decode::<2>(k, 32), Some([u32::MAX, 0]));
    }

    #[test]
    fn hilbert_invalid_arguments() {
        assert!(hilbert_encode([0u32; 3], 0).is_none());
        assert!(hilbert_encode([0u32; 3], 22).is_none());
        assert!(hilbert_encode([0u32; 2], 33).is_none());
        assert!(hilbert_decode::<2>(1 << 10, 5).is_none());
    }

    #[test]
    fn quantize_behaviour() {
        assert_eq!(quantize(0.5_f64, 0.0, 1.0, 8), 128);
        assert_eq!(quantize(-3.0_f64, 0.0, 1.0, 8), 0);
        assert_eq!(quantize(9.0_f64, 0.0, 1.0, 8), 255);
        assert_eq!(quantize(f64::NAN, 0.0, 1.0, 8), 0);
        assert_eq!(quantize(1.0_f64, 2.0, 2.0, 8), 0);
        assert_eq!(quantize(1.0_f64, 0.0, 1.0, 32), u32::MAX);
        assert_eq!(quantize(1.0_f32, 0.0, 1.0, 32), u32::MAX);
    }

    #[test]
    fn sort_orders_by_key() {
        let mut pts = [[0.9_f64, 0.9], [0.1, 0.1], [0.9, 0.1], [0.1, 0.9]];
        let (lo, hi) = ([0.0, 0.0], [1.0, 1.0]);
        assert!(sort_by_hilbert(&mut pts, lo, hi, 8));
        let keys: [u64; 4] = pts.map(|p| hilbert_key(p, lo, hi, 8).unwrap());
        assert!(keys.windows(2).all(|w| w[0] <= w[1]));
        assert!(!sort_by_hilbert(&mut pts, lo, hi, 0));
    }
}
