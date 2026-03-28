pub unsafe fn maxpool2d_chw_i8(
    input: *const i8,             // shape: [ch, in_h, in_w]
    output: *mut i8,              // shape: [ch, out_h, out_w]
    input_shape: (usize, usize),  // (in_h, in_w)
    output_shape: (usize, usize), // (out_h, out_w)
    channel: usize,               // channels
    kernel_size: (usize, usize),  // (kernel_h, kernel_w)
    stride: (usize, usize),       // (stride_h, stride_w)
    padding: (usize, usize),      // (pad_h, pad_w)
    out_shift: usize,
) {
    let ch = channel;
    let (in_h, in_w) = (input_shape.0, input_shape.1);
    let (out_h, out_w) = (output_shape.0, output_shape.1);
    let (kernel_h, kernel_w) = (kernel_size.0, kernel_size.1);
    let (pad_h, pad_w) = (padding.0, padding.1);
    let (stride_h, stride_w) = (stride.0, stride.1);

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
                let out_val = (max_val as isize) << out_shift;
                let clamped = out_val.clamp(-127, 127) as i8;
                unsafe { *output.add(output_idx) = clamped }
            }
        }
    }
}
