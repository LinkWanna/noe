use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize, Debug)]
pub struct Model {
    pub memory: usize,
    pub format: Format,
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
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Clone, Copy)]
pub enum Format {
    CHW,
    HWC,
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Format::CHW => f.write_str("chw"),
            Format::HWC => f.write_str("hwc"),
        }
    }
}
