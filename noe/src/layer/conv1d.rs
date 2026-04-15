use crate::{
    backend::{ActivationParams, conv1d::conv1d_chw_i8},
    layer::Module,
};

#[derive(Debug)]
pub struct Conv1d {
    weight: &'static [u8],
    bias: Option<&'static [u8]>,
    input_shape: (usize, usize),
    output_shape: (usize, usize),
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    out_shift: isize,
    input: *const i8,
    output: *mut i8,
    activation: ActivationParams,
}

impl Conv1d {
    pub const fn new(
        weight: &'static [u8],
        bias: Option<&'static [u8]>,
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        kernel_size: usize,
        stride: usize,
        padding: (usize, usize),
        dilation: usize,
        groups: usize,
        out_shift: isize,
        input: *const i8,
        output: *mut i8,
        activation_min: isize,
        activation_max: isize,
    ) -> Self {
        // sanity checks
        assert!(
            weight.len() == (output_shape.0 * input_shape.0 * kernel_size) / groups,
            "Weight length does not match expected size based on output channels, input channels, kernel size, and groups"
        );

        assert!(
            padding.0 == padding.1,
            "Padding should be symmetric (pad_left == pad_right)"
        );

        let activation = ActivationParams {
            min: activation_min,
            max: activation_max,
        };

        Self {
            weight,
            bias,
            input_shape,
            output_shape,
            kernel_size,
            stride,
            padding: padding.0,
            dilation,
            groups,
            out_shift,
            input,
            output,
            activation,
        }
    }
}

impl Module for Conv1d {
    fn forward_chw(&mut self) {
        unsafe {
            conv1d_chw_i8(
                self.input,
                self.weight.as_ptr().cast(),
                self.bias.map(|b| b.as_ptr().cast()),
                self.output,
                self.input_shape,
                self.output_shape,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                self.out_shift,
                self.activation,
            );
        }
    }

    fn forward_hwc(&mut self) {
        todo!("Forward HWC is not implemented yet. Please use forward_chw for now.")
    }
}
