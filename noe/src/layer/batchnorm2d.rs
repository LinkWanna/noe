use crate::{
    DataLayout,
    backend::{
        ActivationParams,
        batchnorm2d::{batchnorm2d_chw_i8, batchnorm2d_hwc_i8},
    },
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
        layout: DataLayout,
    ) -> Self {
        // sanity check
        let (ch, _, _) = match layout {
            DataLayout::CHW => shape,
            DataLayout::HWC => {
                let (h, w, ch) = shape;
                (ch, h, w)
            }
        };

        assert!(
            mul.len() == ch && add.len() == ch * 2,
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
    fn forward_chw(&mut self) {
        unsafe {
            batchnorm2d_chw_i8(
                self.data,
                self.mul.as_ptr().cast(),
                self.add.as_ptr().cast(),
                self.shape,
                self.out_shift,
                self.activation,
            );
        }
    }

    fn forward_hwc(&mut self) {
        unsafe {
            batchnorm2d_hwc_i8(
                self.data,
                self.mul.as_ptr().cast(),
                self.add.as_ptr().cast(),
                self.shape,
                self.out_shift,
                self.activation,
            );
        }
    }
}
