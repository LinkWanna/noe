use crate::backend::ActivationParams;

pub unsafe fn batchnorm2d_chw_i8(
    data: *mut i8,
    mul: *const i8,
    add: *const i16,
    shape: (usize, usize, usize),
    out_shift: usize,
    activation: ActivationParams,
) {
    let (channel, height, width) = shape;
    let size = height * width;

    for c in 0..channel {
        let m = unsafe { *mul.add(c) } as isize;
        let a = unsafe { *add.add(c) } as isize;
        let ch_ptr = unsafe { data.add(c * size) };

        for i in 0..size {
            let x = unsafe { *ch_ptr.add(i) } as isize;
            let y = ((x * m + a + rounding!(out_shift)) >> out_shift)
                .clamp(activation.min, activation.max);

            unsafe { *ch_ptr.add(i) = y as i8 };
        }
    }
}
