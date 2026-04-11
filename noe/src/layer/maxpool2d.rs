use crate::{
    DataLayout,
    backend::maxpool2d::{maxpool2d_chw_i8, maxpool2d_hwc_i8},
    layer::Module,
};

#[derive(Debug)]
pub struct MaxPool2d {
    input_shape: (usize, usize),
    output_shape: (usize, usize),
    channel: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    _dilation: (usize, usize),
    out_shift: isize,
    input: *const i8,
    output: *mut i8,
}

impl MaxPool2d {
    pub const fn new(
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        channel: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
        dilation: (usize, usize),
        out_shift: isize,
        input: *const i8,
        output: *mut i8,
        _: DataLayout,
    ) -> Self {
        let (h_in, w_in) = input_shape;
        let (h_out, w_out) = output_shape;
        let (k_h, k_w) = kernel_size;
        let (p_t, p_l, p_b, p_r) = padding;
        let (s_h, s_w) = stride;

        assert!(
            p_t == p_b && p_l == p_r,
            "Padding should be symmetric (pad_h_top == pad_h_bottom and pad_w_left == pad_w_right)"
        );

        assert!(
            h_out == ((h_in + 2 * p_t - k_h) / s_h) + 1,
            "Output height does not match the expected value based on input height, kernel size, stride, and padding"
        );

        assert!(
            w_out == ((w_in + 2 * p_l - k_w) / s_w) + 1,
            "Output width does not match the expected value based on input width, kernel size, stride, and padding"
        );

        MaxPool2d {
            input_shape,
            output_shape,
            channel,
            kernel_size,
            stride,
            padding: (p_t, p_l),
            _dilation: dilation,
            out_shift,
            input,
            output,
        }
    }
}

impl Module for MaxPool2d {
    fn forward_chw(&self) {
        unsafe {
            maxpool2d_chw_i8(
                self.input,
                self.output,
                self.input_shape,
                self.output_shape,
                self.channel,
                self.kernel_size,
                self.stride,
                self.padding,
                self.out_shift,
            )
        }
    }

    fn forward_hwc(&self) {
        unsafe {
            maxpool2d_hwc_i8(
                self.input,
                self.output,
                self.input_shape,
                self.output_shape,
                self.channel,
                self.kernel_size,
                self.stride,
                self.padding,
                self.out_shift,
            )
        }
    }
}
