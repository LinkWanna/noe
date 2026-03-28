#![allow(non_snake_case)]

use crate::{
    backend::{ActivationParams, add::add_i8},
    layer::Module,
};

#[derive(Debug)]
pub struct Add {
    B_shift: usize,
    out_shift: usize,
    A: *const i8,
    B: *const i8,
    output: *mut i8,
    activation: ActivationParams,
    size: usize,
}

impl Add {
    pub const fn new(
        A_shape: (usize, usize, usize),
        B_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
        B_shift: usize,
        out_shift: usize,
        A: *const i8,
        B: *const i8,
        output: *mut i8,
        activation_min: isize,
        activation_max: isize,
    ) -> Self {
        // sanity checks
        // We have not implemented broadcasting yet, so A_shape, B_shape and output_shape must be the same
        assert!(
            A_shape.0 == output_shape.0
                && A_shape.1 == output_shape.1
                && A_shape.2 == output_shape.2
                && B_shape.0 == output_shape.0
                && B_shape.1 == output_shape.1
                && B_shape.2 == output_shape.2,
            "A_shape and B_shape must match output_shape"
        );

        let activation = ActivationParams {
            min: activation_min,
            max: activation_max,
        };
        let size = output_shape.0 * output_shape.1 * output_shape.2;

        Self {
            B_shift,
            out_shift,
            A,
            B,
            output,
            activation,
            size,
        }
    }
}

impl Module for Add {
    fn forward_chw(&self) {
        unsafe {
            add_i8(
                self.A,
                self.B,
                self.output,
                self.B_shift,
                self.out_shift,
                self.size,
                self.activation,
            );
        }
    }

    fn forward_hwc(&self) {
        todo!("Forward HWC is not implemented yet. Please use forward_chw for now.")
    }
}
