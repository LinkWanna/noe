pub mod linear;
pub use linear::*;

pub trait Module {
    fn forward_chw(&self);
    fn forward_hwc(&self);
}
