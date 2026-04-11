use crate::backend::ActivationParams;

pub unsafe fn batchnorm2d_chw_i8(
    data: *mut i8,                // shape: [channel, height, width]
    mul: *const i8,               // shape: [channel]
    add: *const i16,              // shape: [channel]
    shape: (usize, usize, usize), // (channel, height, width)
    out_shift: isize,
    activation: ActivationParams,
) {
    let (channel, height, width) = shape;
    let size = height * width;

    for c in 0..channel {
        let m = unsafe { *mul.add(c) } as isize;
        let a = unsafe { *add.add(c) } as isize;

        let ch_base = unsafe { data.add(c * size) };

        for i in 0..size {
            let x = unsafe { *ch_base.add(i) } as isize;
            let y = ((x * m + a + rounding!(out_shift)) >> out_shift)
                .clamp(activation.min, activation.max);

            unsafe { *ch_base.add(i) = y as i8 }
        }
    }
}

pub unsafe fn batchnorm2d_hwc_i8(
    data: *mut i8,                // shape: [height, width, channel]
    mul: *const i8,               // shape: [channel]
    add: *const i16,              // shape: [channel]
    shape: (usize, usize, usize), // (height, width, channel)
    out_shift: isize,
    activation: ActivationParams,
) {
    let (height, width, channel) = shape;
    let size = height * width;

    for i in 0..size {
        let data_base = unsafe { data.add(i * channel) };

        for c in 0..channel {
            let m = unsafe { *mul.add(c) } as isize;
            let a = unsafe { *add.add(c) } as isize;
            let x = unsafe { *data_base.add(c) } as isize;
            let y = ((x * m + a + rounding!(out_shift)) >> out_shift)
                .clamp(activation.min, activation.max);

            unsafe { *data_base.add(c) = y as i8 }
        }
    }
}
