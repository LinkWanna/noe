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
fn gen_layers(layers: &Vec<Layer>, folder: &str, format: Format) -> String {
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
            Conv1D {
                input_shape,
                output_shape,
                weight,
                bias,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                activation,
                out_shift,
                input_off,
                output_off,
            } => {
                let (min, max) = activation_range(activation);

                body.push_str(&format!(
                    "const CONV1D_{idx}: Conv1d = Conv1d::new(\n    \
                     include_bytes!(\"{folder}/{weight}\"),\n    \
                     {},\n    \
                     {input_shape:?},\n    \
                     {output_shape:?},\n    \
                     {kernel_size},\n    \
                     {stride},\n    \
                     {padding:?},\n    \
                     {dilation},\n    \
                     {groups},\n    \
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
            Conv2D {
                input_shape,
                output_shape,
                weight,
                bias,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                out_shift,
                activation,
                input_off,
                output_off,
            } => {
                let (min, max) = activation_range(activation);

                body.push_str(&format!(
                    "const CONV2D_{idx}: Conv2d = Conv2d::new(\n    \
                     include_bytes!(\"{folder}/{weight}\"),\n    \
                     {},\n    \
                     {input_shape:?},\n    \
                     {output_shape:?},\n    \
                     {kernel_size:?},\n    \
                     {stride:?},\n    \
                     {padding:?},\n    \
                     {dilation:?},\n    \
                     {groups},\n    \
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
            MaxPool1D {
                input_shape,
                output_shape,
                kernel_size,
                stride,
                padding,
                dilation,
                out_shift,
                input_off,
                output_off,
            } => {
                let (channel, input_shape, output_shape) = if format == Format::CHW {
                    (input_shape.0, input_shape.1, output_shape.1)
                } else {
                    (input_shape.1, input_shape.0, output_shape.0)
                };
                body.push_str(&format!(
                    "const MAXPOOL1D_{idx}: MaxPool1d = MaxPool1d::new(\n    \
                     {input_shape:?},\n    \
                     {output_shape:?},\n    \
                     {channel},\n    \
                     {kernel_size},\n    \
                     {stride},\n    \
                     {padding:?},\n    \
                     {dilation},\n    \
                     {out_shift},\n    \
                     memory_ptr({input_off}),\n    \
                     memory_ptr({output_off}),\n);\n",
                ));
            }
            MaxPool2D {
                input_shape,
                output_shape,
                kernel_size,
                stride,
                padding,
                dilation,
                out_shift,
                input_off,
                output_off,
            } => {
                let (channel, input_shape, output_shape) = if format == Format::CHW {
                    (
                        input_shape.0,
                        (input_shape.1, input_shape.2),
                        (output_shape.1, output_shape.2),
                    )
                } else {
                    (
                        input_shape.2,
                        (input_shape.0, input_shape.1),
                        (output_shape.0, output_shape.1),
                    )
                };

                body.push_str(&format!(
                    "const MAXPOOL2D_{idx}: MaxPool2d = MaxPool2d::new(\n    \
                     {input_shape:?},\n    \
                     {output_shape:?},\n    \
                     {channel},\n    \
                     {kernel_size:?},\n    \
                     {stride:?},\n    \
                     {padding:?},\n    \
                     {dilation:?},\n    \
                     {out_shift:?},\n    \
                     memory_ptr({input_off}),\n    \
                     memory_ptr({output_off}),\n);\n",
                ));
            }
            BatchNorm2d {
                shape,
                mul,
                add,
                out_shift,
                activation,
                off,
            } => {
                let (min, max) = activation_range(activation);

                body.push_str(&format!(
                    "const BATCHNORM2D_{idx}: BatchNorm2d = BatchNorm2d::new(\n    \
                     {shape:?},\n    \
                     include_bytes!(\"{folder}/{mul}\"),\n    \
                     include_bytes!(\"{folder}/{add}\"),\n    \
                     {out_shift},\n    \
                     memory_ptr({off}),\n    \
                     {min},\n    \
                     {max},\n);\n",
                ));
            }
            Add {
                A_shape,
                B_shape,
                output_shape,
                B_shift,
                out_shift,
                activation,
                A_off,
                B_off,
                output_off,
            } => {
                let (min, max) = activation_range(activation);

                body.push_str(&format!(
                    "const ADD_{idx}: Add = Add::new(\n    \
                     {A_shape:?},\n    \
                     {B_shape:?},\n    \
                     {output_shape:?},\n    \
                     {B_shift},\n    \
                     {out_shift},\n    \
                     memory_ptr({A_off}),\n    \
                     memory_ptr({B_off}),\n    \
                     memory_ptr({output_off}),\n    \
                     {min},\n    \
                     {max},\n);\n",
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
            Conv1D { .. } => body.push_str(&format!("    CONV1D_{idx}.forward_{format}();")),
            Conv2D { .. } => body.push_str(&format!("    CONV2D_{idx}.forward_{format}();")),
            MaxPool1D { .. } => body.push_str(&format!("    MAXPOOL1D_{idx}.forward_{format}();")),
            MaxPool2D { .. } => body.push_str(&format!("    MAXPOOL2D_{idx}.forward_{format}();")),
            BatchNorm2d { .. } => {
                body.push_str(&format!("    BATCHNORM2D_{idx}.forward_{format}();"))
            }
            Add { .. } => body.push_str(&format!("    ADD_{idx}.forward_{format}();")),
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
    // get the absolute path of the folder and target file
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
