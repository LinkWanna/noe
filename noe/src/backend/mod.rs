macro_rules! rounding {
    ($out_shift:expr) => {
        if cfg!(not(feature = "truncate")) {
            (1 << $out_shift) >> 1
        } else {
            0
        }
    };
}

pub(crate) mod add;
pub(crate) mod batchnorm2d;
pub(crate) mod conv1d;
pub(crate) mod conv2d;
pub(crate) mod linear;
pub(crate) mod maxpool2d;

// For the ReLU and ReLU6 activation functions
#[derive(Debug, Clone, Copy)]
pub struct ActivationParams {
    pub min: isize,
    pub max: isize,
}
