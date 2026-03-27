use crate::{
    backend::{ActivationParams, linear::linear_i8},
    layer::Module,
};

#[derive(Debug)]
pub struct Linear {
    weight: &'static [u8],
    bias: Option<&'static [u8]>,
    in_features: usize,
    out_features: usize,
    out_shift: isize,
    input: *const i8,
    output: *mut i8,
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

        Linear {
            weight,
            bias,
            in_features,
            out_features,
            out_shift,
            input,
            output,
            activation,
        }
    }
}

impl Module for Linear {
    fn forward_chw(&self) {
        unsafe {
            linear_i8(
                self.input,
                self.weight.as_ptr().cast(),
                self.bias.map(|b| b.as_ptr().cast()),
                self.output,
                self.in_features,
                self.out_features,
                self.out_shift,
                self.activation,
            );
        }
    }

    fn forward_hwc(&self) {
        self.forward_chw();
    }
}
