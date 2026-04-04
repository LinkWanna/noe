use super::{melscale::melscale, spectrogram::spectrogram, stft::hann_window};

#[derive(Clone, Copy, Debug)]
pub struct MelSpectrogramConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub win_length: usize,
    pub hop_length: usize,
    pub f_min: f64,
    pub f_max: Option<f64>,
    pub pad: usize,
    pub n_mels: usize,
    pub power: f64,
    pub normalized: bool,
    pub center: bool,
    pub pad_mode: &'static str,
    pub norm: Option<&'static str>,
    pub mel_scale: &'static str,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            win_length: 400,
            hop_length: 160,
            f_min: 0.0,
            f_max: None,
            pad: 0,
            n_mels: 40,
            power: 2.0,
            normalized: false,
            center: true,
            pad_mode: "reflect",
            norm: None,
            mel_scale: "htk",
        }
    }
}

pub fn mel_spectrogram_with_config(input: &[f64], config: &MelSpectrogramConfig) -> Vec<Vec<f64>> {
    assert!(
        config.win_length <= config.n_fft,
        "win_length must be <= n_fft"
    );

    let spectrogram = spectrogram(
        input,
        config.pad,
        Some(&hann_window(config.win_length, true)),
        config.n_fft,
        config.hop_length,
        config.win_length,
        config.power,
        config.normalized,
        config.center,
        config.pad_mode,
        true,
    );

    melscale(
        &spectrogram,
        config.n_mels,
        config.sample_rate,
        config.f_min,
        config.f_max,
        config.n_fft / 2 + 1,
        config.norm,
        config.mel_scale,
    )
}
