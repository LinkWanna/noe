use defmt::info;
use noe::{
    DataLayout,
    layer::{BatchNorm2d, Module},
};

use crate::{make_test_i8_generic, measure};

const SMALL_CHANNELS: usize = 8;
const SMALL_HEIGHT: usize = 16;
const SMALL_WIDTH: usize = 16;
const LARGE_CHANNELS: usize = 16;
const LARGE_HEIGHT: usize = 32;
const LARGE_WIDTH: usize = 32;
const LARGE_DATA_LEN: usize = LARGE_CHANNELS * LARGE_HEIGHT * LARGE_WIDTH;

static mut BN_DATA: [i8; LARGE_DATA_LEN] = [0i8; LARGE_DATA_LEN];
static mut BN_MUL: [u8; LARGE_CHANNELS] = [0u8; LARGE_CHANNELS];
static mut BN_ADD: [u8; LARGE_CHANNELS * 2] = [0u8; LARGE_CHANNELS * 2];

pub fn run() {
    info!("--- BatchNorm2d Operator (8x16x16, 16x32x32; CHW/HWC) ---");

    unsafe {
        make_test_i8_generic(BN_MUL.as_mut_ptr() as *mut i8, LARGE_CHANNELS, 3);
        make_test_i8_generic(BN_ADD.as_mut_ptr() as *mut i8, LARGE_CHANNELS * 2, 7);

        let run_case = |name: &str, shape: (usize, usize, usize), layout: DataLayout, channels: usize, iterations: u32| {
            make_test_i8_generic(BN_DATA.as_mut_ptr(), shape.0 * shape.1 * shape.2, 11);

            let mut module = BatchNorm2d::new(
                shape,
                &BN_MUL[..channels],
                &BN_ADD[..channels * 2],
                1,
                BN_DATA.as_mut_ptr(),
                -127,
                127,
                layout,
            );

            measure(name, iterations, || match layout {
                DataLayout::CHW => module.forward_chw(),
                DataLayout::HWC => module.forward_hwc(),
            });
        };

        run_case(
            "batchnorm2d_chw_8x16x16",
            (SMALL_CHANNELS, SMALL_HEIGHT, SMALL_WIDTH),
            DataLayout::CHW,
            SMALL_CHANNELS,
            300,
        );

        run_case(
            "batchnorm2d_hwc_8x16x16",
            (SMALL_HEIGHT, SMALL_WIDTH, SMALL_CHANNELS),
            DataLayout::HWC,
            SMALL_CHANNELS,
            300,
        );

        run_case(
            "batchnorm2d_chw_16x32x32",
            (LARGE_CHANNELS, LARGE_HEIGHT, LARGE_WIDTH),
            DataLayout::CHW,
            LARGE_CHANNELS,
            80,
        );

        run_case(
            "batchnorm2d_hwc_16x32x32",
            (LARGE_HEIGHT, LARGE_WIDTH, LARGE_CHANNELS),
            DataLayout::HWC,
            LARGE_CHANNELS,
            80,
        );
    }
}
