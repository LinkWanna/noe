use crate::backend::ActivationParams;

pub fn mat_vec_mul_i8(
    matrix: &[i8],        // shape: [out_features, in_features]
    vector: &[i8],        // shape: [in_features]
    bias: Option<&[i16]>, // shape: [out_features]
    output: &mut [i8],    // shape: [out_features]
    out_shift: isize,
    activation: ActivationParams,
) {
    let in_features = vector.len();
    let out_features = output.len();
    assert!(matrix.len() == in_features * out_features);
    assert!(bias.map_or(true, |b| b.len() == out_features));

    let rounding = rounding!(out_shift);
    let min = activation.min as isize;
    let max = activation.max as isize;
    let mut bias_fixed = bias
        .map(|b| b.iter().cloned())
        .into_iter()
        .flatten()
        .chain(core::iter::repeat(0));

    let mut matrix_chunks = matrix.chunks_exact(in_features * 2);
    let mut output_chunks = output.chunks_exact_mut(2);
    let mut out_idx = 0;

    // --- 1. 处理双行 (Main Loop) ---
    for (m_rows, o_vals) in matrix_chunks.by_ref().zip(output_chunks.by_ref()) {
        let (row0, row1) = m_rows.split_at(in_features);

        let mut acc0 = bias_fixed.next().unwrap() as isize;
        let mut acc1 = bias_fixed.next().unwrap() as isize;

        for i in 0..in_features {
            let v = vector[i] as isize;
            acc0 += (row0[i] as isize) * v;
            acc1 += (row1[i] as isize) * v;
        }

        o_vals[0] = ((acc0 + rounding) >> out_shift).clamp(min, max) as i8;
        o_vals[1] = ((acc1 + rounding) >> out_shift).clamp(min, max) as i8;
        out_idx += 2;
    }

    // --- 2. 处理剩余行 (Tail) ---
    if out_idx < output.len() {
        let m_row = &matrix[out_idx * in_features..(out_idx + 1) * in_features];
        let mut acc = bias_fixed.next().unwrap() as isize;

        for i in 0..in_features {
            acc += (m_row[i] as isize) * (vector[i] as isize);
        }

        output[out_idx] = ((acc + rounding) >> out_shift).clamp(min, max) as i8;
    }
}
