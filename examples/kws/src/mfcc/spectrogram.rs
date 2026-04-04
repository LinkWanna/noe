use super::{
    Complex64,
    stft::{pad_window, stft},
};

pub fn spectrogram(
    input: &[f64],
    pad: usize,
    window: Option<&[f64]>,
    n_fft: usize,
    hop_length: usize,
    win_length: usize,
    power: f64,
    normalized: bool,
    center: bool,
    pad_mode: &str,
    onesided: bool,
) -> Vec<Vec<f64>> {
    assert!(
        win_length <= n_fft,
        "win_length must be less than or equal to n_fft"
    );
    assert!(power >= 0.0, "power must be non-negative");
    assert!(
        pad_mode == "reflect",
        "only reflect pad_mode is currently supported"
    );

    let mut padded_input = if pad == 0 {
        input.to_vec()
    } else {
        let mut values = Vec::with_capacity(input.len() + pad * 2);
        values.resize(pad, 0.0);
        values.extend_from_slice(input);
        values.resize(values.len() + pad, 0.0);
        values
    };

    if padded_input.is_empty() {
        padded_input.push(0.0);
    }

    // 生成窗口函数，并根据需要进行填充
    let win = window
        .map(|w| pad_window(&w[..win_length.min(w.len())], n_fft))
        .unwrap_or_else(|| vec![1.0; n_fft]);

    // 计算 STFT
    let stft_output: Vec<Vec<Complex64>> = stft(
        &padded_input,
        n_fft,
        hop_length,
        Some(&win),
        center,
        normalized,
        onesided,
        false,
    );

    // 计算幅度谱或功率谱
    let mut output =
        vec![vec![0.0; stft_output.first().map_or(0, |row| row.len())]; stft_output.len()];

    // 计算每个频率和帧的幅度或功率值
    for frequency_index in 0..stft_output.len() {
        for frame_index in 0..stft_output[frequency_index].len() {
            let value = stft_output[frequency_index][frame_index];
            // 幅度平方 = 实部平方 + 虚部平方
            let magnitude_squared = value.re * value.re + value.im * value.im;
            output[frequency_index][frame_index] = if power == 0.0 {
                1.0
            } else {
                // 功率谱 = 幅度平方的 power/2 次方
                magnitude_squared.powf(power / 2.0)
            };
        }
    }

    output
}
