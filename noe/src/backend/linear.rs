use crate::backend::ActivationParams;

/// safety: This function is unsafe because it dereferences raw pointers.
/// The caller must ensure that the pointers are valid and point to memory
/// regions that are large enough to hold the required data.
pub unsafe fn linear_i8(
    input: *const i8,         // shape: [in]
    weight: *const i8,        // shape: [out, in]
    bias: Option<*const i16>, // shape: [out]
    output: *mut i8,          // shape: [out]
    in_features: usize,       // number of input features
    out_features: usize,      // number of output features
    out_shift: isize,
    activation: ActivationParams,
) {
    for o in 0..out_features {
        let mut acc = if let Some(bias_ptr) = bias {
            unsafe { *bias_ptr.add(o) as isize }
        } else {
            0
        };
        for i in 0..in_features {
            let w = unsafe { *weight.add(o * in_features + i) } as isize;
            let inp = unsafe { *input.add(i) } as isize;
            acc += w * inp;
        }

        // 四舍五入右移
        acc = (acc + rounding!(out_shift)) >> out_shift;
        let out_val = acc.clamp(activation.min, activation.max) as i8;
        unsafe { *output.add(o) = out_val }
    }
}
