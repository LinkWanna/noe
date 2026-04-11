use crate::{
    DataLayout,
    backend::{
        ActivationParams,
        conv2d::{conv2d_chw_i8, conv2d_hwc_i8},
    },
    layer::Module,
};

#[derive(Debug)]
pub struct Conv2d {
    weight: &'static [u8],
    bias: Option<&'static [u8]>,
    input_shape: (usize, usize, usize),
    output_shape: (usize, usize, usize),
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
    out_shift: isize,
    input: *const i8,
    output: *mut i8,
    activation: ActivationParams,
}

impl Conv2d {
    pub const fn new(
        weight: &'static [u8],
        bias: Option<&'static [u8]>,
        input_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
        dilation: (usize, usize),
        groups: usize,
        out_shift: isize,
        input: *const i8,
        output: *mut i8,
        activation_min: isize,
        activation_max: isize,
        layout: DataLayout,
    ) -> Self {
        // sanity checks
        let (pt, pl, pb, pr) = (padding.0, padding.1, padding.2, padding.3);
        let (kh, kw) = kernel_size;
        let ((ic, ih, iw), (oc, oh, ow)) = match layout {
            DataLayout::CHW => (input_shape, output_shape),
            DataLayout::HWC => {
                let (ih, iw, ic) = input_shape;
                let (oh, ow, oc) = output_shape;
                ((ic, ih, iw), (oc, oh, ow))
            }
        };

        assert!(
            weight.len() == (oc * ic * kh * kw) / groups,
            "Weight length does not match expected size based on output channels, input channels, kernel size, and groups"
        );
        assert!(
            (ih + pt + pb - kh) / stride.0 + 1 == oh,
            "Output height does not match expected size based on input height, padding, kernel height, and stride"
        );
        assert!(
            (iw + pl + pr - kw) / stride.1 + 1 == ow,
            "Output width does not match expected size based on input width, padding, kernel width, and stride"
        );
        assert!(
            pt == pb && pl == pr,
            "Padding should be symmetric (pad_h_top == pad_h_bottom and pad_w_left == pad_w_right)"
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
            padding: (padding.0, padding.1),
            dilation,
            groups,
            out_shift,
            input,
            output,
            activation,
        }
    }
}

impl Module for Conv2d {
    fn forward_chw(&self) {
        unsafe {
            conv2d_chw_i8(
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

    fn forward_hwc(&self) {
        unsafe {
            conv2d_hwc_i8(
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
}
