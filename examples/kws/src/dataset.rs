use std::{
    fs,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
};

use crate::{
    decoder::decode_wav_mono_f64,
    mfcc::mfcc::{MfccConfig, mfcc_with_config},
};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const TARGET_NUM_SAMPLES: usize = 16_000;
const TARGET_TIME_FRAMES: usize = 63;
const NUM_MFCC: usize = 12;
const INPUT_SCALE: f64 = 32.0;

const GLOBAL_MFCC_MEAN: [f64; NUM_MFCC] = [
    -164.69, 37.94, -0.2, 5.39, -5.43, -0.63, -4.05, -0.92, -3.31, -0.44, -4.02, -0.75,
];

const GLOBAL_MFCC_STD: [f64; NUM_MFCC] = [
    106.45, 41.18, 23.14, 17.16, 15.91, 13.01, 11.38, 9.81, 8.6, 7.92, 7.46, 7.01,
];

pub const SPEECH_COMMANDS_CLASS: [&str; 35] = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", "follow", "forward", "four",
    "go", "happy", "house", "learn", "left", "marvin", "nine", "no", "off", "on", "one", "right",
    "seven", "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
];

#[derive(Debug)]
pub struct SpeechCommandsData {
    pub images: Vec<u8>, // Flattened data [N * 1 * 63 * 12] in CHW order
    pub labels: Vec<u8>,
}

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn resolve_dataset_root(root: &str) -> io::Result<PathBuf> {
    let input = Path::new(root);
    let candidates = [
        input.to_path_buf(),
        input.join("speech_commands_v0.02"),
        input.join("SpeechCommands").join("speech_commands_v0.02"),
    ];

    for candidate in candidates {
        if candidate.join("testing_list.txt").is_file() {
            return Ok(candidate);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "speech_commands_v0.02 root not found under {}",
            input.display()
        ),
    ))
}

fn label_to_index(label: &str) -> Option<u8> {
    SPEECH_COMMANDS_CLASS
        .iter()
        .position(|&name| name == label)
        .map(|index| index as u8)
}

fn resample_linear(input: &[f64], src_rate: u32, dst_rate: u32) -> Vec<f64> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }

    let ratio = dst_rate as f64 / src_rate as f64;
    let out_len = ((input.len() as f64) * ratio).round().max(1.0) as usize;
    let step = src_rate as f64 / dst_rate as f64;
    let mut output = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let pos = (i as f64) * step;
        let left = pos.floor() as usize;
        let right = left.saturating_add(1).min(input.len() - 1);
        let frac = pos - (left as f64);
        let value = input[left] * (1.0 - frac) + input[right] * frac;
        output.push(value);
    }

    output
}

fn pad_or_trim_waveform(mut waveform: Vec<f64>) -> Vec<f64> {
    if waveform.len() > TARGET_NUM_SAMPLES {
        waveform.truncate(TARGET_NUM_SAMPLES);
        return waveform;
    }

    if waveform.len() < TARGET_NUM_SAMPLES {
        waveform.resize(TARGET_NUM_SAMPLES, 0.0);
    }

    waveform
}

fn read_subset_list(root: &Path, list_file: &str) -> io::Result<Vec<PathBuf>> {
    let file = fs::File::open(root.join(list_file))?;
    let reader = BufReader::new(file);
    let mut files = Vec::new();

    for line in reader.lines() {
        let rel = line?;
        let rel = rel.trim();
        if rel.is_empty() {
            continue;
        }
        files.push(root.join(rel));
    }

    Ok(files)
}

fn waveform_to_quant_mfcc(waveform: &[f64]) -> io::Result<Vec<u8>> {
    let config = MfccConfig {
        sample_rate: TARGET_SAMPLE_RATE as usize,
        n_mfcc: NUM_MFCC,
        dct_type: 2,
        norm: "ortho",
        log_mels: false,
        mel_config: crate::mfcc::mel_spectrogram::MelSpectrogramConfig {
            n_fft: 512,
            win_length: 512,
            hop_length: 256,
            n_mels: 40,
            center: true,
            ..crate::mfcc::mel_spectrogram::MelSpectrogramConfig::default()
        },
    };

    let mfcc = mfcc_with_config(waveform, &config); // [n_mfcc, time]
    if mfcc.len() != NUM_MFCC {
        return Err(invalid_data(format!(
            "Unexpected MFCC row count: expected {NUM_MFCC}, got {}",
            mfcc.len()
        )));
    }

    for row in &mfcc {
        if row.is_empty() {
            return Err(invalid_data("MFCC output has empty time axis"));
        }
    }

    let mut output = Vec::with_capacity(TARGET_TIME_FRAMES * NUM_MFCC);

    for t in 0..TARGET_TIME_FRAMES {
        for m in 0..NUM_MFCC {
            let raw = if t < mfcc[m].len() { mfcc[m][t] } else { 0.0 };
            let normalized = (raw - GLOBAL_MFCC_MEAN[m]) / GLOBAL_MFCC_STD[m];
            let q = (normalized * INPUT_SCALE).round().clamp(-127.0, 127.0) as i8 as u8;
            output.push(q);
        }
    }

    Ok(output)
}

fn load_sample(path: &Path) -> io::Result<Vec<u8>> {
    let (sample_rate, mut waveform) = decode_wav_mono_f64(path)?;
    if sample_rate != TARGET_SAMPLE_RATE {
        waveform = resample_linear(&waveform, sample_rate, TARGET_SAMPLE_RATE);
    }

    let waveform = pad_or_trim_waveform(waveform);
    waveform_to_quant_mfcc(&waveform)
}

fn extract_label_from_path(path: &Path) -> Option<u8> {
    let label = path.parent()?.file_name()?.to_str()?;
    label_to_index(label)
}

fn load_from_paths(paths: Vec<PathBuf>, num: u32) -> io::Result<SpeechCommandsData> {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for path in paths {
        let Some(label) = extract_label_from_path(&path) else {
            continue;
        };

        let sample = load_sample(&path)?;
        images.extend_from_slice(&sample);
        labels.push(label);

        if num > 0 && labels.len() >= num as usize {
            break;
        }
    }

    if labels.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No valid speech command samples found",
        ));
    }

    Ok(SpeechCommandsData { images, labels })
}

pub fn load_speech_commands(root: &str, num: u32) -> io::Result<SpeechCommandsData> {
    let dataset_root = resolve_dataset_root(root)?;
    let files = read_subset_list(&dataset_root, "testing_list.txt")?;
    load_from_paths(files, num)
}
