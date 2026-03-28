pub mod batchnorm2d;
pub mod conv2d;
pub mod linear;
pub mod maxpool2d;

// Actually, `batchnorm` should not exist as a separate layer, because it is usually fused with the preceding convolution layer.
// However, for simplicity, we will keep it as a separate layer for now.
pub use batchnorm2d::*;
pub use conv2d::*;
pub use linear::*;
pub use maxpool2d::*;

pub trait Module {
    fn forward_chw(&self);
    fn forward_hwc(&self);
}
