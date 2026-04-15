use crate::{
    backend::{ActivationParams, linear::linear_i8},
    layer::Module,
};

#[derive(Debug)]
pub struct Linear {
    weight: &'static [i8],
    bias: Option<&'static [i16]>,
    out_shift: isize,
    input: &'static [i8],
    output: &'static mut [i8],
    activation: ActivationParams,
}

impl Linear {
    pub const fn new(
        weight: &'static [u8],
        bias: Option<&'static [u8]>,
        in_features: usize,
        out_features: usize,
        out_shift: isize,
        input: *const i8,
        output: *mut i8,
        activation_min: isize,
        activation_max: isize,
    ) -> Self {
        // sanity checks
        assert!(
            weight.len() == in_features * out_features,
            "Weight size does not match the expected size."
        );

        if let Some(bias) = bias {
            assert!(
                bias.len() == out_features * 2, // bias is i16, so 2 bytes per element
                "Bias size does not match the expected size."
            );
        }

        let activation = ActivationParams {
            min: activation_min,
            max: activation_max,
        };

        let weight = unsafe { core::slice::from_raw_parts(weight.as_ptr().cast(), weight.len()) };
        let bias = match bias {
            Some(b) => Some(unsafe { core::slice::from_raw_parts(b.as_ptr().cast(), b.len() / 2) }),
            None => None,
        };
        let input = unsafe { core::slice::from_raw_parts(input, in_features) };
        let output = unsafe { core::slice::from_raw_parts_mut(output, out_features) };

        Linear {
            weight,
            bias,
            out_shift,
            input,
            output,
            activation,
        }
    }
}

impl Module for Linear {
    fn forward_chw(&mut self) {
        linear_i8(
            self.input,
            self.weight,
            self.bias,
            self.output,
            self.out_shift,
            self.activation,
        );
    }

    fn forward_hwc(&mut self) {
        self.forward_chw();
    }
}
