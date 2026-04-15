use defmt::info;
use noe::layer::{Linear, Module};

use crate::{make_test_i8_generic, measure};

const IN_SMALL: usize = 32;
const OUT_SMALL: usize = 16;
const IN_LARGE: usize = 64;
const OUT_LARGE: usize = 32;

static mut LINEAR_INPUT: [i8; IN_LARGE] = [0i8; IN_LARGE];
static mut LINEAR_WEIGHT: [i8; IN_LARGE * OUT_LARGE] = [0i8; IN_LARGE * OUT_LARGE];
static mut LINEAR_BIAS: [i16; OUT_LARGE] = [0i16; OUT_LARGE];
static mut LINEAR_OUTPUT: [i8; OUT_LARGE] = [0i8; OUT_LARGE];

pub fn run() {
    info!("--- Linear Operator (32->16, 64->32) ---");

    unsafe {
        make_test_i8_generic(LINEAR_INPUT.as_mut_ptr(), IN_LARGE, 10);
        make_test_i8_generic(LINEAR_WEIGHT.as_mut_ptr(), IN_LARGE * OUT_LARGE, 5);
        make_test_i8_generic(LINEAR_BIAS.as_mut_ptr() as *mut i8, OUT_LARGE * 2, 3);

        let weight_small =
            core::slice::from_raw_parts(LINEAR_WEIGHT.as_ptr() as *const u8, IN_SMALL * OUT_SMALL);
        let bias_small = core::slice::from_raw_parts(
            LINEAR_BIAS.as_ptr() as *const u8,
            OUT_SMALL * core::mem::size_of::<i16>(),
        );

        let mut module_small = Linear::new(
            weight_small,
            Some(bias_small),
            IN_SMALL,
            OUT_SMALL,
            1,
            LINEAR_INPUT.as_ptr(),
            LINEAR_OUTPUT.as_mut_ptr(),
            -127,
            127,
        );

        measure("linear_32x16", 500, || module_small.forward_chw());

        let weight_large =
            core::slice::from_raw_parts(LINEAR_WEIGHT.as_ptr() as *const u8, IN_LARGE * OUT_LARGE);
        let bias_large = core::slice::from_raw_parts(
            LINEAR_BIAS.as_ptr() as *const u8,
            OUT_LARGE * core::mem::size_of::<i16>(),
        );

        let mut module_large = Linear::new(
            weight_large,
            Some(bias_large),
            IN_LARGE,
            OUT_LARGE,
            1,
            LINEAR_INPUT.as_ptr(),
            LINEAR_OUTPUT.as_mut_ptr(),
            -127,
            127,
        );

        measure("linear_64x32", 300, || module_large.forward_chw());
    }
}
