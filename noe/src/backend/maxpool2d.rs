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
    let (in_h, in_w) = input_shape;
    let (out_h, out_w) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;

    let in_plane_size = in_h * in_w;
    let out_plane_size = out_h * out_w;

    for oc in 0..channel {
        // current channel's input and output
        let in_ptr_c = unsafe { input.add(oc * in_plane_size) };
        let out_ptr_c = unsafe { output.add(oc * out_plane_size) };

        for oh in 0..out_h {
            let h_start = (oh * stride_h) as isize - pad_h as isize;
            let h_end = (h_start + kernel_h as isize).min(in_h as isize);
            let h_start = h_start.max(0) as usize;
            let h_end = h_end.max(0) as usize;

            for ow in 0..out_w {
                let w_start = (ow * stride_w) as isize - pad_w as isize;
                let w_end = (w_start + kernel_w as isize).min(in_w as isize);
                let w_start = w_start.max(0) as usize;
                let w_end = w_end.max(0) as usize;

                let mut max_val = i8::MIN;
                for ih in h_start..h_end {
                    let in_ptr_row = unsafe { in_ptr_c.add(ih * in_w) };
                    for iw in w_start..w_end {
                        let val = unsafe { *in_ptr_row.add(iw) };

                        if val > max_val {
                            max_val = val;
                        }
                    }
                }

                // 处理输出逻辑
                let mut out_val = max_val as isize;
                if out_shift > 0 {
                    out_val >>= out_shift;
                } else if out_shift < 0 {
                    out_val <<= -out_shift;
                }

                unsafe { *out_ptr_c.add(oh * out_w + ow) = out_val.clamp(-127, 127) as i8 }
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
    let (in_h, in_w) = input_shape;
    let (out_h, out_w) = output_shape;
    let (kernel_h, kernel_w) = kernel_size;
    let (pad_h, pad_w) = padding;
    let (stride_h, stride_w) = stride;

    let mut out = output;
    for oh in 0..out_h {
        let h_start = (oh * stride_h) as isize - pad_h as isize;
        let h_end = (h_start + kernel_h as isize).min(in_h as isize);
        let h_start = h_start.max(0) as usize;
        let h_end = h_end.max(0) as usize;

        for ow in 0..out_w {
            let w_start = (ow * stride_w) as isize - pad_w as isize;
            let w_end = (w_start + kernel_w as isize).min(in_w as isize);
            let w_start = w_start.max(0) as usize;
            let w_end = w_end.max(0) as usize;

            // step 1: initialize output with the first element in the Kernel window
            let first = unsafe { input.add((h_start * in_w + w_start) * channel) };
            unsafe { core::ptr::copy_nonoverlapping(first, out, channel) }

            for kh in h_start..h_end {
                for kw in w_start..w_end {
                    if kh == h_start && kw == w_start {
                        continue;
                    } else {
                        // step 2: compare and update the output with the current element in the Kernel window
                        let in_ptr = unsafe { input.add((kh * in_w + kw) * channel) };
                        for c in 0..channel {
                            let val = unsafe { *in_ptr.add(c) };
                            let out_val = unsafe { *out.add(c) };
                            if val > out_val {
                                unsafe { *out.add(c) = val }
                            }
                        }
                    }
                }
            }

            // step 3: apply output shift and clamp the output value to the range of i8
            for c in 0..channel {
                let mut out_val = unsafe { *out.add(c) as isize };
                if out_shift > 0 {
                    out_val >>= out_shift
                } else if out_shift < 0 {
                    out_val <<= -out_shift
                };
                unsafe { *out.add(c) = out_val.clamp(-127, 127) as i8 }
            }
            out = unsafe { out.add(channel) };
        }
    }
}
