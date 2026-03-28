use crate::backend::ActivationParams;

/// out = (A + B << B_shift) >> out_shift
#[allow(non_snake_case)]
pub unsafe fn add_i8(
    A: *const i8,
    B: *const i8,
    output: *mut i8,
    B_shift: usize,
    out_shift: usize,
    size: usize,
    activation: ActivationParams,
) {
    for i in 0..size {
        let A_val = unsafe { *A.add(i) } as isize;
        let B_val = (unsafe { *B.add(i) } as isize) << B_shift;
        let sum = (A_val + B_val + rounding!(out_shift)) >> out_shift;

        let clamped_sum = sum.clamp(activation.min, activation.max) as i8;
        unsafe { *output.add(i) = clamped_sum }
    }
}
