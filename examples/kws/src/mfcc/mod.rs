pub mod mel_spectrogram;
pub mod melscale;
pub mod melscale_fbanks;
pub mod mfcc;
pub mod spectrogram;
pub mod stft;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}
