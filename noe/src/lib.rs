#![no_std]

macro_rules! rounding {
    ($out_shift:expr) => {
        if cfg!(not(feature = "truncate")) {
            (1 << $out_shift) >> 1
        } else {
            0
        }
    };
}

mod backend;
mod basic;
pub mod layer;

pub enum DataLayout {
    CHW,
    HWC,
}
