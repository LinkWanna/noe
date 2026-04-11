use crate::backend::ActivationParams;

pub unsafe fn mat_vec_mul_i8(
    matrix: *const i8,        // shape: [out, in]
    vector: *const i8,        // shape: [in]
    bias: Option<*const i16>, // shape: [out]
    output: *mut i8,          // shape: [out]
    in_features: usize,       // number of input features
    out_features: usize,      // number of output features
    out_shift: isize,
    activation: ActivationParams,
) {
    let mut o = 0;
    while o <= out_features.saturating_sub(2) {
        let (mut acc0, mut acc1) = if let Some(b) = bias {
            unsafe { (*b.add(o) as isize, *b.add(o + 1) as isize) }
        } else {
            (0, 0)
        };

        unsafe {
            let w_ptr0 = matrix.add(o * in_features);
            let w_ptr1 = matrix.add((o + 1) * in_features);
            for i in 0..in_features {
                let inp = *vector.add(i) as isize;

                let w0 = *w_ptr0.add(i) as isize;
                let w1 = *w_ptr1.add(i) as isize;

                acc0 += w0 * inp;
                acc1 += w1 * inp;
            }
        }

        acc0 = (acc0 + rounding!(out_shift)) >> out_shift;
        acc1 = (acc1 + rounding!(out_shift)) >> out_shift;

        unsafe {
            *output.add(o) = acc0.clamp(activation.min, activation.max) as i8;
            *output.add(o + 1) = acc1.clamp(activation.min, activation.max) as i8;
        }

        o += 2;
    }

    // deal with the last output if out_features is odd
    if o < out_features {
        let mut acc = if let Some(b) = bias {
            unsafe { *b.add(o) as isize }
        } else {
            0
        };

        unsafe {
            let w_ptr = matrix.add(o * in_features);
            for i in 0..in_features {
                acc += (*w_ptr.add(i) as isize) * (*vector.add(i) as isize);
            }
        }

        acc = (acc + rounding!(out_shift)) >> out_shift;
        unsafe { *output.add(o) = acc.clamp(activation.min, activation.max) as i8 }
    }
}
