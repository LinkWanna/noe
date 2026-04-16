use defmt::info;
use noe::{
    DataLayout,
    layer::{Conv2d, Module},
};

use crate::{make_test_i8_generic, measure};

const IN_C: usize = 4;
const IN_H: usize = 16;
const IN_W: usize = 16;
const OUT_C: usize = 8;
const K_H: usize = 3;
const K_W: usize = 3;
const STRIDE: usize = 1;
const PAD: usize = 1;

const OUT_H: usize = (IN_H + PAD * 2 - K_H) / STRIDE + 1;
const OUT_W: usize = (IN_W + PAD * 2 - K_W) / STRIDE + 1;

const INPUT_LEN: usize = IN_C * IN_H * IN_W;
const OUTPUT_LEN: usize = OUT_C * OUT_H * OUT_W;
const WEIGHT_LEN: usize = OUT_C * IN_C * K_H * K_W;
const BIAS_LEN: usize = OUT_C;
const TMP_LEN: usize = K_H * K_W * IN_C;

static mut CONV_INPUT: [i8; INPUT_LEN] = [0i8; INPUT_LEN];
static mut CONV_OUTPUT: [i8; OUTPUT_LEN] = [0i8; OUTPUT_LEN];
static mut CONV_WEIGHT: [i8; WEIGHT_LEN] = [0i8; WEIGHT_LEN];
static mut CONV_BIAS: [i16; BIAS_LEN] = [0i16; BIAS_LEN];
static mut CONV_TMP: [i8; TMP_LEN] = [0i8; TMP_LEN];

pub fn run() {
    info!("--- Conv2d Operator (4x16x16 -> 8x16x16, k3, CHW/HWC) ---");

    unsafe {
        make_test_i8_generic(CONV_INPUT.as_mut_ptr(), INPUT_LEN, 5);
        make_test_i8_generic(CONV_OUTPUT.as_mut_ptr(), OUTPUT_LEN, 0);
        make_test_i8_generic(CONV_WEIGHT.as_mut_ptr(), WEIGHT_LEN, 3);
        make_test_i8_generic(CONV_BIAS.as_mut_ptr() as *mut i8, BIAS_LEN * 2, 1);
        make_test_i8_generic(CONV_TMP.as_mut_ptr(), TMP_LEN, 0);

        let weight_bytes =
            core::slice::from_raw_parts(CONV_WEIGHT.as_ptr() as *const u8, WEIGHT_LEN);
        let bias_bytes = core::slice::from_raw_parts(CONV_BIAS.as_ptr() as *const u8, BIAS_LEN * 2);

        let mut conv_chw = Conv2d::new(
            weight_bytes,
            Some(bias_bytes),
            (IN_C, IN_H, IN_W),
            (OUT_C, OUT_H, OUT_W),
            (K_H, K_W),
            (STRIDE, STRIDE),
            (PAD, PAD, PAD, PAD),
            (1, 1),
            1,
            1,
            CONV_INPUT.as_ptr(),
            CONV_OUTPUT.as_mut_ptr(),
            CONV_TMP.as_mut_ptr(),
            -127,
            127,
            DataLayout::CHW,
        );

        measure("conv2d_chw_4x16x16_o8_k3", 80, || conv_chw.forward_chw());

        let mut conv_hwc = Conv2d::new(
            weight_bytes,
            Some(bias_bytes),
            (IN_H, IN_W, IN_C),
            (OUT_H, OUT_W, OUT_C),
            (K_H, K_W),
            (STRIDE, STRIDE),
            (PAD, PAD, PAD, PAD),
            (1, 1),
            1,
            1,
            CONV_INPUT.as_ptr(),
            CONV_OUTPUT.as_mut_ptr(),
            CONV_TMP.as_mut_ptr(),
            -127,
            127,
            DataLayout::HWC,
        );

        measure("conv2d_hwc_4x16x16_o8_k3", 80, || conv_hwc.forward_hwc());
    }
}
