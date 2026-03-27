pub mod conv2d;
pub mod linear;

pub use conv2d::*;
pub use linear::*;

pub trait Module {
    fn forward_chw(&self);
    fn forward_hwc(&self);
}
