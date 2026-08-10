//! FIR/IIR filters, windowing.

mod fir;
mod window;

pub use fir::FirFilter;
pub use window::{apply_window, bartlett, blackman, hamming, hanning, rectangular, Window};
