pub unsafe fn maxpool1d_chw_i8(
    input: *const i8,    // shape: [ch, inp]
    output: *mut i8,     // shape: [ch, out]
    input_shape: usize,  // in
    output_shape: usize, // out
    channel: usize,      // channels
    kernel_size: usize,  // kernel
    stride: usize,       // stride
    padding: usize,      // pad
    out_shift: isize,
) {
    let ch = channel;
    let inp = input_shape;
    let out = output_shape;

    for c in 0..ch {
        for o in 0..out {
            let mut max_val: i8 = i8::MIN;
            let ks_start = (o * stride) as isize - padding as isize;
            let ks_end = ks_start + kernel_size as isize;

            for ks in ks_start..ks_end {
                if ks >= 0 && (ks as usize) < inp {
                    let iw = ks as usize;
                    let input_idx = (c * inp) + iw;
                    let val = unsafe { *input.add(input_idx) };
                    if val > max_val {
                        max_val = val;
                    }
                }
            }

            let output_idx = (c * out) + o;
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
