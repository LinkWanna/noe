use std::f64::consts::PI;

use super::Complex64;

pub fn hann_window(length: usize, periodic: bool) -> Vec<f64> {
    match length {
        0 => Vec::new(),
        1 => vec![1.0],
        _ => {
            let denominator = if periodic {
                length as f64
            } else {
                (length - 1) as f64
            };
            (0..length)
                .map(|index| {
                    let angle = 2.0 * PI * (index as f64) / denominator;
                    0.5 - 0.5 * angle.cos()
                })
                .collect()
        }
    }
}

/// 获取反射填充的索引，适用于中心化的 STFT 边界处理
pub fn reflect_index(mut index: isize, length: usize) -> usize {
    if length <= 1 {
        return 0;
    }

    // 将索引限制在 [-length, 2*length-2] 范围内，以实现反射效果
    let length = length as isize;
    loop {
        if index < 0 {
            index = -index;
        } else if index >= length {
            index = 2 * length - index - 2;
        } else {
            return index as usize;
        }
    }
}

pub fn reflect_pad(input: &[f64], pad: usize) -> Vec<f64> {
    if pad == 0 {
        return input.to_vec();
    }

    let mut padded = Vec::with_capacity(input.len() + pad * 2);
    for offset in (1..=pad).rev() {
        let reflected = reflect_index(-(offset as isize), input.len());
        padded.push(input[reflected]);
    }
    padded.extend_from_slice(input);
    for offset in 0..pad {
        let reflected = reflect_index((input.len() + offset) as isize, input.len());
        padded.push(input[reflected]);
    }
    padded
}

pub fn pad_window(window: &[f64], target_length: usize) -> Vec<f64> {
    if window.len() == target_length {
        return window.to_vec();
    }

    assert!(
        window.len() <= target_length,
        "window length must not exceed n_fft"
    );

    let mut padded = vec![0.0; target_length];
    let left_pad = (target_length - window.len()) / 2;
    padded[left_pad..left_pad + window.len()].copy_from_slice(window);
    padded
}

pub fn stft(
    input: &[f64],
    n_fft: usize,
    hop_length: usize,
    window: Option<&[f64]>,
    center: bool,
    normalized: bool,
    onesided: bool,
    _align_to_window: bool,
) -> Vec<Vec<Complex64>> {
    // 填充输入信号以适应窗口中心化
    let pad = if center { n_fft / 2 } else { 0 };
    let padded_input = if center {
        reflect_pad(input, pad)
    } else {
        input.to_vec()
    };

    // 准备窗口函数，必要时进行零填充以匹配 n_fft
    let effective_window = match window {
        Some(window) => pad_window(window, n_fft),
        None => vec![1.0; n_fft],
    };

    let frame_count = if padded_input.len() < n_fft {
        0
    } else {
        1 + (padded_input.len() - n_fft) / hop_length
    };

    let frequency_count = if onesided { n_fft / 2 + 1 } else { n_fft };
    let normalization_factor = if normalized {
        1.0 / (n_fft as f64).sqrt()
    } else {
        1.0
    };

    let mut output = vec![vec![Complex64::default(); frame_count]; frequency_count];

    for frame_index in 0..frame_count {
        let start = frame_index * hop_length;
        let frame = &padded_input[start..start + n_fft];

        for frequency_index in 0..frequency_count {
            let mut real = 0.0;
            let mut imaginary = 0.0;

            for sample_index in 0..n_fft {
                let value = frame[sample_index] * effective_window[sample_index];
                let angle =
                    2.0 * PI * (frequency_index as f64) * (sample_index as f64) / (n_fft as f64);
                real += value * angle.cos();
                imaginary -= value * angle.sin();
            }

            output[frequency_index][frame_index] = Complex64::new(
                real * normalization_factor,
                imaginary * normalization_factor,
            );
        }
    }

    output
}
