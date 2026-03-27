#![allow(unused)]

use noe::layer::*;

const MEMORY_SIZE: usize = 1040;
static mut MEMORY: [u8; MEMORY_SIZE] = [0; MEMORY_SIZE];

const fn memory_ptr(off: usize) -> *mut i8 {
    assert!(off < MEMORY_SIZE);
    unsafe {
        let base = &raw mut MEMORY as *mut i8;
        base.add(off)
    }
}


const LINEAR_1: Linear = Linear::new(
    include_bytes!("/home/linkwanna/AI/noe/examples/mlp/mlp_params/features_features_1_weight_quant_export_handler_Constant_1_output_0.bin"),
    Some(include_bytes!("/home/linkwanna/AI/noe/examples/mlp/mlp_params/features_features_1_bias_quant_export_handler_Constant_output_0.bin")),
    784,
    256,
    10,
    memory_ptr(256),
    memory_ptr(0),
    0,
    127,
);

const LINEAR_2: Linear = Linear::new(
    include_bytes!("/home/linkwanna/AI/noe/examples/mlp/mlp_params/features_features_3_weight_quant_export_handler_Constant_output_0.bin"),
    Some(include_bytes!("/home/linkwanna/AI/noe/examples/mlp/mlp_params/features_features_3_bias_quant_export_handler_Constant_output_0.bin")),
    256,
    256,
    7,
    memory_ptr(0),
    memory_ptr(256),
    0,
    127,
);

const LINEAR_3: Linear = Linear::new(
    include_bytes!("/home/linkwanna/AI/noe/examples/mlp/mlp_params/features_features_5_weight_quant_export_handler_Constant_output_0.bin"),
    Some(include_bytes!("/home/linkwanna/AI/noe/examples/mlp/mlp_params/features_features_5_bias_quant_export_handler_Constant_output_0.bin")),
    256,
    10,
    8,
    memory_ptr(256),
    memory_ptr(0),
    -127,
    127,
);


pub fn model_run(input: &[i8], output: &mut [i8]) {
    use core::ptr::copy_nonoverlapping;

    unsafe { copy_nonoverlapping(input.as_ptr(), memory_ptr(256), 784) };

    LINEAR_1.forward_chw();
    LINEAR_2.forward_chw();
    LINEAR_3.forward_chw();
    unsafe { copy_nonoverlapping(memory_ptr(0), output.as_mut_ptr(), 10) }

}
