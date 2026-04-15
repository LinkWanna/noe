use crate::{backend::ActivationParams, basic::mat_vec_mul::mat_vec_mul_i8};

pub fn linear_i8(
    input: &[i8],         // shape: [in]
    weight: &[i8],        // shape: [out, in]
    bias: Option<&[i16]>, // shape: [out]
    output: &mut [i8],    // shape: [out]
    out_shift: isize,
    activation: ActivationParams,
) {
    mat_vec_mul_i8(weight, input, bias, output, out_shift, activation)
}
