#![no_std]
#![forbid(unsafe_code)]
// Small fixed-size matrix kernels are clearest with explicit indexing.
// `!(x > 0)` is used deliberately so that NaN is rejected along with
// non-positive values.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]
//! `no_std`, allocator-free math primitives for 3D and 4D Gaussian splatting.
//!
//! * [`sym`] — packed symmetric matrices ([`Sym3`], [`Sym4`]), Cholesky / LDLᵀ
//!   factorisations, SPD inverse and determinant, and a Jacobi eigensolver for
//!   3×3 symmetric matrices.
//! * [`gaussian`] — building covariances that are positive semi-definite *by
//!   construction* (`Σ = R S Sᵀ Rᵀ`) in 3D and 4D, conditioning a 4D Gaussian on
//!   time to obtain a 3D Gaussian, and EWA projection to a 2D screen-space
//!   covariance.
//! * [`sh`] — real spherical harmonics (degrees 0–3) for view-dependent colour,
//!   using the 3D Gaussian Splatting sign convention.
//! * [`sfc`] — Morton (Z-order) and Hilbert space-filling-curve keys in 2/3/4
//!   (Morton) and any (Hilbert) dimension, plus quantisation helpers.
//!
//! # Correctness stance
//!
//! Floating-point results are *not* claimed to be bit-exact; what is claimed,
//! and tested, is structure:
//!
//! * covariances built by [`gaussian`] are symmetric and positive
//!   semi-definite by construction, never "repaired" afterwards;
//! * the checked factorisations return `None` instead of a garbage factor for a
//!   matrix that is not numerically positive definite;
//! * the space-filling-curve code is pure integer arithmetic, and its bijection
//!   and Hilbert-adjacency properties are verified exhaustively on small grids;
//! * the 4D conditioning is cross-checked against an independent precision
//!   matrix formulation.
//!
//! # Conventions
//!
//! * Quaternions are `[w, x, y, z]` and are normalised internally; a
//!   zero/non-finite quaternion yields `None`.
//! * 4D vectors are `[x, y, z, t]`. A 4D rotation is `L(q_l) · R(q_r)` for two
//!   unit quaternions, with the four components interpreted as quaternion
//!   components in that order (the 4DGS parameterisation).
//! * Matrices are row-major `[[T; N]; N]`, column vectors.
//!
//! # Example
//!
//! ```
//! use tpt_math_splat::gaussian::{cov4_from_rot_scale, condition_on_time};
//!
//! let id = [1.0_f64, 0.0, 0.0, 0.0];
//! let cov = cov4_from_rot_scale(id, id, [1.0, 1.0, 1.0, 0.5]).unwrap();
//! let g = condition_on_time([0.0; 4], &cov, 0.0).unwrap();
//! assert!((g.weight - 1.0).abs() < 1e-12);
//! ```

pub mod gaussian;
pub mod sfc;
pub mod sh;
pub mod sym;

pub use gaussian::{ConditionedGaussian, Sym2};
pub use sym::{Eig3, Sym3, Sym4};
