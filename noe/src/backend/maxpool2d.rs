pub unsafe fn maxpool2d_chw_i8(
    input: *const i8,             // shape: [ch, in_h, in_w]
    output: *mut i8,              // shape: [ch, out_h, out_w]
    input_shape: (usize, usize),  // (in_h, in_w)
    output_shape: (usize, usize), // (out_h, out_w)
    channel: usize,               // channels
    kernel_size: (usize, usize),  // (kernel_h, kernel_w)
    stride: (usize, usize),       // (stride_h, stride_w)
    padding: (usize, usize),      // (pad_h, pad_w)
    out_shift: isize,
) {
    let ch = channel;
    let (in_h, in_w) = input_shape;
    let (out_h, out_w) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;

    for oc in 0..ch {
        let off_c = oc as usize * out_h * out_w;
        for oh in 0..out_h {
            for ow in 0..out_w {
                let mut max_val: i8 = i8::MIN;
                let kh_start = (oh * stride_h) as isize - pad_h as isize;
                let kw_start = (ow * stride_w) as isize - pad_w as isize;
                let kh_end = kh_start + kernel_h as isize;
                let kw_end = kw_start + kernel_w as isize;

                for kh in kh_start..kh_end {
                    for kw in kw_start..kw_end {
                        if kh >= 0 && kw >= 0 && (kh as usize) < in_h && (kw as usize) < in_w {
                            let ih = kh as usize;
                            let iw = kw as usize;
                            let input_idx = (oc * in_h * in_w) + (ih * in_w) + iw;
                            let val = unsafe { *input.add(input_idx) };

                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }

                let output_idx = off_c + (oh * out_w) + ow;
                let out_val = if out_shift > 0 {
                    (max_val as isize) >> out_shift
                } else if out_shift < 0 {
                    (max_val as isize) << (-out_shift)
                } else {
                    max_val as isize
                };
                let clamped = out_val.clamp(-127, 127) as i8;
                unsafe { *output.add(output_idx) = clamped }
            }
        }
    }
}

pub unsafe fn maxpool2d_hwc_i8(
    input: *const i8,             // shape: [in_h, in_w, ch]
    output: *mut i8,              // shape: [out_h, out_w, ch]
    input_shape: (usize, usize),  // (in_h, in_w)
    output_shape: (usize, usize), // (out_h, out_w)
    channel: usize,               // channels
    kernel_size: (usize, usize),  // (kernel_h, kernel_w)
    stride: (usize, usize),       // (stride_h, stride_w)
    padding: (usize, usize),      // (pad_h, pad_w)
    out_shift: isize,
) {
    let ch = channel;
    let (in_h, in_w) = input_shape;
    let (out_h, out_w) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;

    for oh in 0..out_h {
        for ow in 0..out_w {
            let kh_start = (oh * stride_h) as isize - pad_h as isize;
            let kw_start = (ow * stride_w) as isize - pad_w as isize;
            let kh_end = kh_start + kernel_h as isize;
            let kw_end = kw_start + kernel_w as isize;

            for c in 0..ch {
                let mut max_val: i8 = i8::MIN;

                for kh in kh_start..kh_end {
                    for kw in kw_start..kw_end {
                        if kh >= 0 && kw >= 0 && (kh as usize) < in_h && (kw as usize) < in_w {
                            let ih = kh as usize;
                            let iw = kw as usize;
                            let input_idx = ((ih * in_w + iw) * ch) + c;
                            let val = unsafe { *input.add(input_idx) };

                            if val > max_val {
                                max_val = val;
                            }
                        }
                    }
                }

                let output_idx = ((oh * out_w + ow) * ch) + c;
                let out_val = if out_shift > 0 {
                    (max_val as isize) >> out_shift
                } else if out_shift < 0 {
                    (max_val as isize) << (-out_shift)
                } else {
                    max_val as isize
                };
                let clamped = out_val.clamp(-127, 127) as i8;
                unsafe { *output.add(output_idx) = clamped }
            }
        }
    }
}
