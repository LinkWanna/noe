use defmt::info;
use noe::layer::{Add, Module};

use crate::{make_test_i8_generic, measure};

// 512 samples fits comfortably in STM32F103C8 RAM.
static mut ADD_A: [i8; 512] = [0i8; 512];
static mut ADD_B: [i8; 512] = [0i8; 512];
static mut ADD_OUT: [i8; 512] = [0i8; 512];

pub fn run() {
    info!("--- Add Operator ---");

    unsafe {
        make_test_i8_generic(ADD_A.as_mut_ptr(), 256, 5);
        make_test_i8_generic(ADD_B.as_mut_ptr(), 256, 7);

        let mut add_256 = Add::new(
            (1, 1, 256),
            (1, 1, 256),
            (1, 1, 256),
            1,
            1,
            ADD_A.as_mut_ptr() as *const _,
            ADD_B.as_mut_ptr() as *const _,
            ADD_OUT.as_mut_ptr(),
            -127,
            127,
        );

        measure("add_256", 1000, || add_256.forward_chw());

        make_test_i8_generic(ADD_A.as_mut_ptr(), 512, 3);
        make_test_i8_generic(ADD_B.as_mut_ptr(), 512, 9);

        let mut add_512 = Add::new(
            (1, 1, 512),
            (1, 1, 512),
            (1, 1, 512),
            1,
            1,
            ADD_A.as_mut_ptr() as *const _,
            ADD_B.as_mut_ptr() as *const _,
            ADD_OUT.as_mut_ptr(),
            -127,
            127,
        );

        measure("add_512", 500, || add_512.forward_chw());
    }
}
