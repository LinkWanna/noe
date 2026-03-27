use std::fs;

use crate::{
    types::{Format, Layer, Model},
    utils::activation_range,
};

mod types;
mod utils;

/// generate the declaration code for the model
fn gen_declare() -> String {
    let mut body = String::new();
    body.push_str("#![allow(unused)]\n\n");
    body.push_str("use noe::layer::*;\n\n");

    body
}

/// generate the memory pool code for the model
fn gen_memory_pool(size: usize) -> String {
    let mut body = String::new();
    body.push_str(&format!("const MEMORY_SIZE: usize = {};\n", size));
    body.push_str("static mut MEMORY: [u8; MEMORY_SIZE] = [0; MEMORY_SIZE];\n\n");

    body.push_str("const fn memory_ptr(off: usize) -> *mut i8 {\n");
    body.push_str("    assert!(off < MEMORY_SIZE);\n");
    body.push_str("    unsafe {\n");
    body.push_str("        let base = &raw mut MEMORY as *mut i8;\n");
    body.push_str("        base.add(off)\n");
    body.push_str("    }\n");
    body.push_str("}\n\n");

    body
}

/// generate the layer declarations for the model
fn gen_layers(layers: &Vec<Layer>, folder: &str, _: Format) -> String {
    use crate::types::Layer::*;

    let mut body = String::new();

    layers.iter().enumerate().for_each(|(idx, layer)| {
        match layer {
            Linear {
                in_features,
                out_features,
                weight,
                bias,
                out_shift,
                activation,
                input_off,
                output_off,
            } => {
                let (min, max) = activation_range(&activation);

                body.push_str(&format!(
                    "const LINEAR_{idx}: Linear = Linear::new(\n    \
                     include_bytes!(\"{folder}/{weight}\"),\n    \
                     {},\n    \
                     {in_features},\n    \
                     {out_features},\n    \
                     {out_shift},\n    \
                     memory_ptr({input_off}),\n    \
                     memory_ptr({output_off}),\n    \
                     {min},\n    \
                     {max},\n);\n",
                    if let Some(bias_path) = bias {
                        format!("Some(include_bytes!(\"{folder}/{bias_path}\"))")
                    } else {
                        "None".to_string()
                    }
                ));
            }
            Input { .. } | Output { .. } => {}
        }

        body.push('\n');
    });

    body
}

/// generate the model run function for the model
fn gen_model_run(layers: &Vec<Layer>, format: Format) -> String {
    use crate::types::Layer::*;
    let mut body = String::new();
    body.push_str("pub fn model_run(input: &[i8], output: &mut [i8]) {\n");
    body.push_str("    use core::ptr::copy_nonoverlapping;\n\n");

    layers.iter().enumerate().for_each(|(idx, layer)| {
        match layer {
            Input { size, off } => body.push_str(&format!(
                "    unsafe {{ copy_nonoverlapping(input.as_ptr(), memory_ptr({off}), {size}) }};\n"
            )),
            Linear { .. } => body.push_str(&format!("    LINEAR_{idx}.forward_{format}();")),
            Output { size, off } => body.push_str(&format!(
                "    unsafe {{ copy_nonoverlapping(memory_ptr({}), output.as_mut_ptr(), {}) }}\n",
                off, size
            )),
        }

        body.push('\n');
    });

    body.push_str("}\n");

    body
}

pub fn process_model(folder: &str, target: &str) {
    // 将 folder 和 target 转换为绝对路径
    let project_root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let abs_folder = format!("{project_root}/{folder}");
    let abs_target = format!("{project_root}/{target}");
    let mut body = String::new();

    // 1. read JSON file from folder/model.json
    let path = format!("{folder}/model.json");
    let json = fs::read_to_string(path).expect("Failed to read JSON file");

    // 2. deserialize JSON to Model struct
    let config: Model = serde_json::from_str(&json).expect("Failed to parse JSON");
    let format = config.format; // 布局格式

    // 3. generate declaration code
    body.push_str(&gen_declare());

    // 4. generate memory pool code
    body.push_str(&gen_memory_pool(config.memory));

    // 5. generate layer declarations
    body.push_str(&gen_layers(&config.layers, &abs_folder, format));

    // 6. generate model run function
    body.push_str(&gen_model_run(&config.layers, format));

    // 7. write the generated code to target file
    fs::write(abs_target, body).expect("Failed to write declaration file");
}
