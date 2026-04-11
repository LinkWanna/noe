#![allow(non_snake_case)]
use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct Model {
    pub memory: usize,
    pub layout: DataLayout,
    pub layers: Vec<Layer>,
}

#[derive(Deserialize, Serialize, Debug)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Layer {
    Input {
        size: usize,
        off: usize,
    },
    Output {
        size: usize,
        off: usize,
    },
    Linear {
        in_features: usize,
        out_features: usize,
        weight: String,
        bias: Option<String>,
        out_shift: isize,
        activation: Option<String>,
        input_off: usize,
        output_off: usize,
    },
    Conv1D {
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        weight: String,
        bias: Option<String>,
        kernel_size: usize,
        stride: usize,
        padding: (usize, usize),
        dilation: usize,
        groups: usize,
        out_shift: usize,
        activation: Option<String>,
        input_off: usize,
        output_off: usize,
    },
    Conv2D {
        input_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
        weight: String,
        bias: Option<String>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
        dilation: (usize, usize),
        groups: usize,
        out_shift: usize,
        activation: Option<String>,
        input_off: usize,
        output_off: usize,
    },
    MaxPool1D {
        input_shape: (usize, usize),
        output_shape: (usize, usize),
        kernel_size: usize,
        stride: usize,
        padding: (usize, usize),
        dilation: usize,
        out_shift: isize,
        input_off: usize,
        output_off: usize,
    },
    MaxPool2D {
        input_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize, usize, usize),
        dilation: (usize, usize),
        out_shift: isize,
        input_off: usize,
        output_off: usize,
    },
    BatchNorm2d {
        shape: (usize, usize, usize),
        mul: String,
        add: String,
        out_shift: isize,
        activation: Option<String>,
        off: usize,
    },
    Add {
        A_shape: (usize, usize, usize),
        B_shape: (usize, usize, usize),
        output_shape: (usize, usize, usize),
        B_shift: usize,
        out_shift: usize,
        activation: Option<String>,
        A_off: usize,
        B_off: usize,
        output_off: usize,
    },
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum DataLayout {
    CHW,
    HWC,
}

impl fmt::Display for DataLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataLayout::CHW => f.write_str("chw"),
            DataLayout::HWC => f.write_str("hwc"),
        }
    }
}
