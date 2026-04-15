use crate::{backend::ActivationParams, basic::mat_vec_mul::mat_vec_mul_i8};

pub unsafe fn conv2d_chw_i8(
    input: *const i8,                    // shape: [in_c, in_h, in_w]
    weight: *const i8,                   // shape: [out_c, in_c/groups, kernel_h, kernel_w]
    bias: Option<*const i16>,            // shape: [out_c]
    output: *mut i8,                     // shape: [out_c, out_h, out_w]
    input_shape: (usize, usize, usize),  // (in_c, in_h, in_w)
    output_shape: (usize, usize, usize), // (out_c, out_h, out_w)
    kernel_size: (usize, usize),         // (kernel_h, kernel_w)
    stride: (usize, usize),              // (stride_h, stride_w)
    padding: (usize, usize),             // (pad_h, pad_w)
    dilation: (usize, usize),            // (dilation_h, dilation_w)
    groups: usize,
    out_shift: isize,
    activation: ActivationParams,
) {
    let (in_c, in_h, in_w) = input_shape;
    let (out_c, out_h, out_w) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;
    let (dilation_h, dilation_w) = dilation;

    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;

    for oc in 0..out_c {
        let group_id = oc / out_c_per_group;
        let ic_start = group_id * in_c_per_group;
        let ic_end = ic_start + in_c_per_group;

        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut acc = if let Some(bias_ptr) = bias {
                    unsafe { *bias_ptr.add(oc) as isize }
                } else {
                    0
                };

                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        // Apply dilation: effective offset in input
                        let dilated_kh = kh * dilation_h;
                        let dilated_kw = kw * dilation_w;

                        let ih = (oh * stride_h) as isize - pad_h as isize + dilated_kh as isize;
                        let iw = (ow * stride_w) as isize - pad_w as isize + dilated_kw as isize;

                        if ih < 0 || ih >= in_h as isize || iw < 0 || iw >= in_w as isize {
                            continue;
                        }

                        let ih = ih as usize;
                        let iw = iw as usize;

                        // Only iterate over input channels in the same group
                        for (local_ic, ic) in (ic_start..ic_end).enumerate() {
                            let input_idx = ic * in_h * in_w + ih * in_w + iw;

                            // Weight layout: [out_c, in_c/groups, k_h, k_w]
                            let weight_idx = oc * (in_c_per_group * kernel_h * kernel_w)
                                + local_ic * (kernel_h * kernel_w)
                                + kh * kernel_w
                                + kw;

                            let inp = unsafe { *input.add(input_idx) } as isize;
                            let w = unsafe { *weight.add(weight_idx) } as isize;
                            acc += inp * w;
                        }
                    }
                }

                acc = (acc + rounding!(out_shift)) >> out_shift;
                let out_val = acc.clamp(activation.min, activation.max) as i8;
                let output_idx = (oc * out_h * out_w) + (oh * out_w) + ow;
                unsafe { *output.add(output_idx) = out_val };
            }
        }
    }
}

pub unsafe fn conv2d_hwc_i8(
    input: *const i8,                    // shape: [in_h, in_w, in_c]
    weight: *const i8,                   // shape: [out_c, kernel_h, kernel_w, in_c/groups]
    bias: Option<*const i16>,            // shape: [out_c]
    output: *mut i8,                     // shape: [out_h, out_w, out_c]
    tmp: *mut i8,                        // shape: [kernel_h * kernel_w * in_c/groups]
    input_shape: (usize, usize, usize),  // (in_h, in_w, in_c)
    output_shape: (usize, usize, usize), // (out_h, out_w, out_c)
    kernel_size: (usize, usize),         // (kernel_h, kernel_w)
    stride: (usize, usize),              // (stride_h, stride_w)
    padding: (usize, usize),             // (pad_h, pad_w)
    dilation: (usize, usize),            // (dilation_h, dilation_w)
    groups: usize,
    out_shift: isize,
    activation: ActivationParams,
) {
    let (in_h, in_w, in_c) = input_shape;
    let (out_h, out_w, out_c) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;
    let (dilation_h, dilation_w) = dilation;

    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;

    let patch_len = kernel_h * kernel_w * in_c_per_group;
    unsafe {
        for g in 0..groups {
            let group_input = input.add(g * in_c_per_group);
            let group_weight = weight.add(g * out_c_per_group * patch_len);
            let group_bias = bias.map(|b| b.add(g * out_c_per_group));
            let group_output = output.add(g * out_c_per_group);

            for oh in 0..out_h {
                let base_h = oh * stride_h;
                for ow in 0..out_w {
                    let base_w = ow * stride_w;

                    // step 1: fill tmp with the current patch (with padding handling)
                    let mut tmp_p = tmp;
                    core::ptr::write_bytes(tmp_p, 0, patch_len); // clear tmp buffer
                    for kh in 0..kernel_h {
                        let ih = base_h as isize - pad_h as isize + (kh * dilation_h) as isize;

                        for kw in 0..kernel_w {
                            let iw = base_w as isize - pad_w as isize + (kw * dilation_w) as isize;

                            if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                let base_offset = (ih as usize * in_w + iw as usize) * in_c;
                                core::ptr::copy_nonoverlapping(
                                    group_input.add(base_offset),
                                    tmp_p,
                                    in_c_per_group,
                                );
                            }

                            tmp_p = tmp_p.add(in_c_per_group);
                        }
                    }

                    // step 2: compute output for this position using vec_mat_mul
                    let out_base = group_output.add((oh * out_w * out_c) + (ow * out_c));

                    // vector: tmp (1 x patch_len)
                    // matrix: group_weight (out_c_per_group x patch_len)
                    let weight_view =
                        core::slice::from_raw_parts(group_weight, out_c_per_group * patch_len);
                    let vector_view = core::slice::from_raw_parts(tmp as *const i8, patch_len);
                    let bias_view =
                        group_bias.map(|b| core::slice::from_raw_parts(b, out_c_per_group));
                    let output_view = core::slice::from_raw_parts_mut(out_base, out_c_per_group);

                    mat_vec_mul_i8(
                        weight_view,
                        vector_view,
                        bias_view,
                        output_view,
                        out_shift,
                        activation,
                    );
                }
            }
        }
    }
}
