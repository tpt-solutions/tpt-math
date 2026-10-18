//! Fuzz `CooMatrix::from_triplets` and the conversion to CSR. Indices are reduced
//! into range, since out-of-range indices are a documented panic. Duplicates must be
//! summed, and the matrix-vector product must agree with a dense reference.
#![no_main]

use libfuzzer_sys::fuzz_target;
use tpt_math_linalg_sparse::CooMatrix;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let nrows = 1 + data[0] as usize % 6;
    let ncols = 1 + data[1] as usize % 6;
    let body = &data[2..];
    let mut rows = Vec::new();
    let mut cols = Vec::new();
    let mut vals = Vec::new();
    for chunk in body.chunks_exact(3) {
        rows.push(chunk[0] as usize % nrows);
        cols.push(chunk[1] as usize % ncols);
        vals.push(f64::from(chunk[2] as i8));
    }
    let mut dense = vec![vec![0.0_f64; ncols]; nrows];
    for k in 0..rows.len() {
        dense[rows[k]][cols[k]] += vals[k];
    }
    let coo = CooMatrix::from_triplets(nrows, ncols, rows, cols, vals);
    let csr = coo.to_csr();
    let x: Vec<f64> = (0..ncols).map(|j| j as f64 + 1.0).collect();
    let y = csr
        .matvec(&tpt_math_linalg_dense::DVector::from_vec(x.clone()))
        .expect("dimensions match by construction");
    for i in 0..nrows {
        let want: f64 = (0..ncols).map(|j| dense[i][j] * x[j]).sum();
        assert!((y[i] - want).abs() < 1e-9, "row {i}: {} vs {want}", y[i]);
    }
});
