use crate::backend::ActivationParams;

pub unsafe fn conv1d_chw_i8(
    input: *const i8,             // shape: [in_c, in_len]
    weight: *const i8,            // shape: [out_c, in_c/groups, kernel_size]
    bias: Option<*const i8>,      // shape: [out_c]
    output: *mut i8,              // shape: [out_c, out_len]
    input_shape: (usize, usize),  // (in_c, in_len)
    output_shape: (usize, usize), // (out_c, out_len)
    kernel_size: usize,           // kernel_size
    stride: usize,                // stride
    padding: usize,               // pad
    dilation: usize,              // dilation
    groups: usize,
    out_shift: isize,
    activation: ActivationParams,
) {
    let (in_c, in_len) = (input_shape.0, input_shape.1);
    let (out_c, out_len) = (output_shape.0, output_shape.1);

    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;

    for oc in 0..out_c {
        let group_id = oc / out_c_per_group;
        let ic_start = group_id * in_c_per_group;
        let ic_end = ic_start + in_c_per_group;

        for oh in 0..out_len {
            let base_ih = (oh * stride) as isize - padding as isize;
            let mut acc = if let Some(bias_ptr) = bias {
                unsafe { *bias_ptr.add(oc) as isize }
            } else {
                0
            };

            for ks in 0..kernel_size {
                let dilated_kh = ks * dilation;
                let ih = base_ih + dilated_kh as isize;

                if ih < 0 || ih >= in_len as isize {
                    continue;
                }

                let ih = ih as usize;

                for (local_ic, ic) in (ic_start..ic_end).enumerate() {
                    let input_idx = ic * in_len + ih;
                    let weight_idx =
                        oc * in_c_per_group * kernel_size + local_ic * kernel_size + ks;

                    let inp = unsafe { *input.add(input_idx) } as isize;
                    let w = unsafe { *weight.add(weight_idx) } as isize;
                    acc += inp * w;
                }
            }

            acc = (acc + rounding!(out_shift)) >> out_shift;
            let out_val = acc.clamp(activation.min, activation.max) as i8;
            let output_idx = oc * out_len + oh;
            unsafe { *output.add(output_idx) = out_val }
        }
    }
}
