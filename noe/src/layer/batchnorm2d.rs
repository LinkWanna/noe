use crate::{
    backend::{ActivationParams, batchnorm2d::batchnorm2d_chw_i8},
    layer::Module,
};

#[derive(Debug)]
pub struct BatchNorm2d {
    shape: (usize, usize, usize),
    mul: &'static [u8],
    add: &'static [u8],
    out_shift: isize,
    data: *mut i8,
    activation: ActivationParams,
}

impl BatchNorm2d {
    pub const fn new(
        shape: (usize, usize, usize),
        mul: &'static [u8],
        add: &'static [u8],
        out_shift: isize,
        data: *mut i8,
        activation_min: isize,
        activation_max: isize,
    ) -> Self {
        // sanity check
        assert!(
            mul.len() == shape.0 && add.len() == shape.0 * 2,
            "Mul and Add arrays must have the same length as the number of channels"
        );

        let activation = ActivationParams {
            min: activation_min,
            max: activation_max,
        };

        Self {
            shape,
            mul,
            add,
            out_shift,
            data,
            activation,
        }
    }
}

impl Module for BatchNorm2d {
    fn forward_chw(&self) {
        unsafe {
            batchnorm2d_chw_i8(
                self.data,
                self.mul.as_ptr().cast(),
                self.add.as_ptr().cast(),
                self.shape,
                self.out_shift as usize,
                self.activation,
            );
        }
    }

    fn forward_hwc(&self) {
        todo!("Forward HWC is not implemented yet. Please use forward_chw for now.")
    }
}
