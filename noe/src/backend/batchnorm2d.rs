use crate::backend::ActivationParams;

pub fn batchnorm2d_chw_i8(
    data: &mut [i8],       // shape: [channel, height, width]
    mul: &[i8],            // shape: [channel]
    add: &[i16],           // shape: [channel]
    shape: (usize, usize), // (height, width)
    out_shift: isize,
    activation: ActivationParams,
) {
    let rounding = rounding!(out_shift);
    let min = activation.min as isize;
    let max = activation.max as isize;

    let (height, width) = shape;
    let size = height * width;

    let data_chunks = data.chunks_mut(size);

    for (ch_base, (&m, &a)) in data_chunks.zip(mul.iter().zip(add.iter())) {
        let m = m as isize;
        let a = a as isize;

        let mut chunck = ch_base.chunks_exact_mut(2);
        for base in chunck.by_ref() {
            for i in 0..2 {
                let x = base[i] as isize;
                let y = ((x * m + a + rounding) >> out_shift).clamp(min, max);
                base[i] = y as i8;
            }
        }

        for base in chunck.into_remainder() {
            let inp = *base as isize;
            let y = ((inp * m + a + rounding) >> out_shift).clamp(min, max);

            *base = y as i8;
        }
    }
}

pub fn batchnorm2d_hwc_i8(
    data: &mut [i8],   // shape: [height, width, channel]
    mul: &[i8],        // shape: [channel]
    add: &[i16],       // shape: [channel]
    _: (usize, usize), // (height, width)
    out_shift: isize,
    activation: ActivationParams,
) {
    let rounding = rounding!(out_shift);
    let min = activation.min as isize;
    let max = activation.max as isize;

    let channel = mul.len();
    let mut data_chunks = data.chunks_exact_mut(channel * 4);

    for data_base in data_chunks.by_ref() {
        let (x0, rest) = data_base.split_at_mut(channel);
        let (x1, rest) = rest.split_at_mut(channel);
        let (x2, x3) = rest.split_at_mut(channel);

        for (i, (&m, &a)) in mul.iter().zip(add.iter()).enumerate() {
            let m = m as isize;
            let a = a as isize;

            for x in [&mut x0[i], &mut x1[i], &mut x2[i], &mut x3[i]] {
                let y = ((*x as isize * m + a + rounding) >> out_shift).clamp(min, max);
                *x = y as i8;
            }
        }
    }

    for data_base in data_chunks.into_remainder().chunks_mut(channel) {
        for (i, (&m, &a)) in mul.iter().zip(add.iter()).enumerate() {
            let m = m as isize;
            let a = a as isize;

            for x in data_base.iter_mut().skip(i).step_by(channel) {
                let y = ((*x as isize * m + a + rounding) >> out_shift).clamp(min, max);
                *x = y as i8;
            }
        }
    }
}
