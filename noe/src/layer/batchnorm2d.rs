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
    shape: (usize, usize),
    mul: &'static [i8],
    add: &'static [i16],
    out_shift: isize,
    data: &'static mut [i8],
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
        let (ch, h, w) = match layout {
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

        let mul = unsafe { core::slice::from_raw_parts(mul.as_ptr() as *const i8, ch) };
        let add = unsafe { core::slice::from_raw_parts(add.as_ptr() as *const i16, ch) };
        let data = unsafe { core::slice::from_raw_parts_mut(data, ch * h * w) };

        Self {
            shape: (h, w),
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
        batchnorm2d_chw_i8(
            self.data,
            self.mul,
            self.add,
            self.shape,
            self.out_shift,
            self.activation,
        );
    }

    fn forward_hwc(&mut self) {
        batchnorm2d_hwc_i8(
            self.data,
            self.mul,
            self.add,
            self.shape,
            self.out_shift,
            self.activation,
        );
    }
}
