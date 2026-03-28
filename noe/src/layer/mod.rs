pub mod conv2d;
pub mod linear;
pub mod maxpool2d;

pub use conv2d::*;
pub use linear::*;
pub use maxpool2d::*;

pub trait Module {
    fn forward_chw(&self);
    fn forward_hwc(&self);
}
