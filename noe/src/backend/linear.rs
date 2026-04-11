use crate::{backend::ActivationParams, basic::mat_vec_mul::mat_vec_mul_i8};

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
    unsafe {
        mat_vec_mul_i8(
            weight,
            input,
            bias,
            output,
            in_features,
            out_features,
            out_shift,
            activation,
        )
    }
}
