use crate::{backend::ActivationParams, basic::mat_vec_mul::mat_vec_mul_i8};

pub fn conv2d_chw_i8(
    input: &[i8],                        // shape: [in_c, in_h, in_w]
    weight: &[i8],                       // shape: [out_c, in_c/groups, kernel_h, kernel_w]
    bias: Option<&[i16]>,                // shape: [out_c]
    output: &mut [i8],                   // shape: [out_c, out_h, out_w]
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
    let (ph, pw) = padding;
    let (sh, sw) = stride;
    let (dh, dw) = dilation;

    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;

    let rounding = rounding!(out_shift);
    let min = activation.min as isize;
    let max = activation.max as isize;

    for (oc, out_channel) in output.chunks_mut(out_h * out_w).enumerate() {
        let group_id = oc / out_c_per_group;
        let ic_start = group_id * in_c_per_group;
        let ic_end = ic_start + in_c_per_group;

        let b_val = bias.map(|b| b[oc] as isize).unwrap_or(0);

        for (oh, out_row) in out_channel.chunks_mut(out_w).enumerate() {
            let base_h = oh * sh;
            for (ow, out_val) in out_row.iter_mut().enumerate() {
                let base_w = ow * sw;
                let mut acc = b_val;

                for kh in 0..kernel_h {
                    let dilated_kh = kh * dh;
                    let ih = base_h as isize - ph as isize + dilated_kh as isize;

                    for kw in 0..kernel_w {
                        let dilated_kw = kw * dw;
                        let iw = base_w as isize - pw as isize + dilated_kw as isize;

                        if iw < 0 || iw >= in_w as isize || ih < 0 || ih >= in_h as isize {
                            continue;
                        }

                        let ih = ih as usize;
                        let iw = iw as usize;

                        for (local_ic, ic) in (ic_start..ic_end).enumerate() {
                            let input_idx = ic * in_h * in_w + ih * in_w + iw;

                            let weight_idx = oc * (in_c_per_group * kernel_h * kernel_w)
                                + local_ic * (kernel_h * kernel_w)
                                + kh * kernel_w
                                + kw;

                            let inp = input[input_idx] as isize;
                            let w = weight[weight_idx] as isize;
                            acc += inp * w;
                        }
                    }
                }

                *out_val = ((acc + rounding) >> out_shift).clamp(min, max) as i8;
            }
        }
    }
}

pub fn conv2d_hwc_i8(
    input: &[i8],                        // shape: [in_h, in_w, in_c]
    weight: &[i8],                       // shape: [out_c, kernel_h, kernel_w, in_c/groups]
    bias: Option<&[i16]>,                // shape: [out_c]
    output: &mut [i8],                   // shape: [out_h, out_w, out_c]
    tmp: &mut [i8],                      // shape: [kernel_h * kernel_w * in_c/groups]
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
    let (ph, pw) = padding;
    let (sh, sw) = stride;
    let (dh, dw) = dilation;

    let in_c_per_group = in_c / groups;
    let out_c_per_group = out_c / groups;
    let patch_len = kernel_h * kernel_w * in_c_per_group;

    for g in 0..groups {
        // 计算当前组的输入、权重和偏置范围
        let in_group_range = (g * in_c_per_group)..((g + 1) * in_c_per_group);
        let weight_group_start = g * out_c_per_group * patch_len;
        let weight_group_end = weight_group_start + out_c_per_group * patch_len;
        let group_weight = &weight[weight_group_start..weight_group_end];

        let group_bias = bias.map(|b| {
            let start = g * out_c_per_group;
            &b[start..start + out_c_per_group]
        });

        for oh in 0..out_h {
            let base_h = oh * sh;
            for ow in 0..out_w {
                let base_w = ow * sw;

                // Step 1: 填充 tmp 缓冲区（处理 Padding）
                tmp.fill(0);

                for kh in 0..kernel_h {
                    let dilated_kh = kh * dh;
                    let ih = base_h as isize - ph as isize + dilated_kh as isize;

                    for kw in 0..kernel_w {
                        let dilated_kw = kw * dw;
                        let iw = base_w as isize - pw as isize + dilated_kw as isize;
                        if ih < 0 || ih >= in_h as isize || iw < 0 || iw >= in_w as isize {
                            continue;
                        }

                        // 计算 tmp 中当前内核位置的偏移
                        let tmp_offset = (kh * kernel_w + kw) * in_c_per_group;
                        let tmp_slice = &mut tmp[tmp_offset..tmp_offset + in_c_per_group];

                        // 计算 input 中对应位置和通道组的偏移
                        let input_base = (ih as usize * in_w + iw as usize) * in_c;
                        let input_slice = &input
                            [input_base + in_group_range.start..input_base + in_group_range.end];

                        tmp_slice.copy_from_slice(input_slice);
                    }
                }

                // Step 2: 计算输出
                // 计算输出位置：HWC 布局下，通道是最后一位
                let out_offset = (oh * out_w * out_c) + (ow * out_c) + (g * out_c_per_group);
                let out_slice = &mut output[out_offset..out_offset + out_c_per_group];

                // 调用矩阵相乘逻辑
                mat_vec_mul_i8(
                    group_weight,
                    tmp,
                    group_bias,
                    out_slice,
                    out_shift,
                    activation,
                );
            }
        }
    }
}
