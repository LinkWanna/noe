use crate::{backend::maxpool1d::maxpool1d_chw_i8, layer::Module};

#[derive(Debug)]
pub struct MaxPool1d {
    input_shape: usize,
    output_shape: usize,
    channel: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    _dilation: usize,
    out_shift: isize,
    input: *const i8,
    output: *mut i8,
}

impl MaxPool1d {
    pub const fn new(
        input_shape: usize,
        output_shape: usize,
        channel: usize,
        kernel_size: usize,
        stride: usize,
        padding: (usize, usize),
        dilation: usize,
        out_shift: isize,
        input: *const i8,
        output: *mut i8,
    ) -> Self {
        // sanity checks
        assert!(
            padding.0 == padding.1,
            "Padding should be symmetric (pad_left == pad_right)"
        );
        assert!(
            output_shape == ((input_shape + 2 * padding.0 - kernel_size) / stride) + 1,
            "Output shape does not match the expected value based on input shape, kernel size, stride, and padding"
        );

        MaxPool1d {
            input_shape,
            output_shape,
            channel,
            kernel_size,
            stride,
            padding: padding.0,
            _dilation: dilation,
            out_shift,
            input,
            output,
        }
    }
}

impl Module for MaxPool1d {
    fn forward_chw(&self) {
        unsafe {
            maxpool1d_chw_i8(
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
        todo!("Forward HWC is not implemented yet. Please use forward_chw for now.")
    }
}
