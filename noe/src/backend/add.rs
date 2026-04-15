use crate::backend::ActivationParams;

/// out = (A + B << B_shift) >> out_shift
#[allow(non_snake_case)]
pub fn add_i8(
    A: &[i8],
    B: &[i8],
    output: &mut [i8],
    B_shift: usize,
    out_shift: usize,
    activation: ActivationParams,
) {
    for output_val in output.iter_mut().zip(A.iter().zip(B.iter())) {
        let (A_val, B_val) = output_val.1;
        let sum =
            (*A_val as isize + ((*B_val as isize) << B_shift) + rounding!(out_shift)) >> out_shift;
        *output_val.0 = sum.clamp(activation.min, activation.max) as i8;
    }
}
