use std::f64::consts::PI;

use super::mel_spectrogram::{MelSpectrogramConfig, mel_spectrogram_with_config};

#[derive(Clone, Copy, Debug)]
pub struct MfccConfig {
    pub sample_rate: usize,
    pub n_mfcc: usize,
    pub dct_type: usize,
    pub norm: &'static str,
    pub log_mels: bool,
    pub mel_config: MelSpectrogramConfig,
}

impl Default for MfccConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mfcc: 40,
            dct_type: 2,
            norm: "ortho",
            log_mels: false,
            mel_config: MelSpectrogramConfig {
                n_fft: 512,
                win_length: 512,
                hop_length: 256,
                n_mels: 40,
                center: false,
                ..MelSpectrogramConfig::default()
            },
        }
    }
}

pub fn mfcc_with_config(input: &[f64], config: &MfccConfig) -> Vec<Vec<f64>> {
    assert_eq!(config.dct_type, 2, "only dct_type=2 is supported");
    assert_eq!(config.norm, "ortho", "only norm='ortho' is supported");

    let mut mel_config = config.mel_config;
    mel_config.sample_rate = config.sample_rate;

    let mel_spec = mel_spectrogram_with_config(input, &mel_config);

    let mel_features = if config.log_mels {
        mel_spec
            .iter()
            .map(|row| row.iter().map(|value| (value + 1e-6).ln()).collect())
            .collect()
    } else {
        amplitude_to_db(&mel_spec, 10.0, 1e-10, 80.0)
    };

    let dct = create_dct(config.n_mfcc, mel_config.n_mels);
    apply_dct(&mel_features, &dct)
}

fn amplitude_to_db(input: &[Vec<f64>], multiplier: f64, amin: f64, top_db: f64) -> Vec<Vec<f64>> {
    let rows = input.len();
    let cols = input.first().map_or(0, |row| row.len());
    let mut output = vec![vec![0.0; cols]; rows];

    let mut max_value = f64::NEG_INFINITY;
    for row_index in 0..rows {
        for col_index in 0..cols {
            let value = input[row_index][col_index].max(amin);
            let db_value = multiplier * value.log10();
            output[row_index][col_index] = db_value;
            max_value = max_value.max(db_value);
        }
    }

    let cutoff = max_value - top_db;
    for row in &mut output {
        for value in row {
            if *value < cutoff {
                *value = cutoff;
            }
        }
    }

    output
}

fn create_dct(n_mfcc: usize, n_mels: usize) -> Vec<Vec<f64>> {
    let mut dct = vec![vec![0.0; n_mfcc]; n_mels];

    for mel_index in 0..n_mels {
        for mfcc_index in 0..n_mfcc {
            let angle = PI / (n_mels as f64) * ((mel_index as f64) + 0.5) * (mfcc_index as f64);
            dct[mel_index][mfcc_index] = angle.cos();
        }
    }

    // torchaudio.functional.create_dct(..., norm="ortho")
    let scale = (2.0 / (n_mels as f64)).sqrt();
    for mel_index in 0..n_mels {
        dct[mel_index][0] *= 1.0 / 2.0_f64.sqrt();
        for mfcc_index in 0..n_mfcc {
            dct[mel_index][mfcc_index] *= scale;
        }
    }

    dct
}

fn apply_dct(mel_features: &[Vec<f64>], dct: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n_mels = mel_features.len();
    let frame_count = mel_features.first().map_or(0, |row| row.len());
    let n_mfcc = dct.first().map_or(0, |row| row.len());

    assert_eq!(
        dct.len(),
        n_mels,
        "dct mel dimension must match mel_features rows"
    );

    let mut output = vec![vec![0.0; frame_count]; n_mfcc];
    for mfcc_index in 0..n_mfcc {
        for frame_index in 0..frame_count {
            let mut value = 0.0;
            for mel_index in 0..n_mels {
                value += mel_features[mel_index][frame_index] * dct[mel_index][mfcc_index];
            }
            output[mfcc_index][frame_index] = value;
        }
    }

    output
}
