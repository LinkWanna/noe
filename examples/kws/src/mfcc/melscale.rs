use super::melscale_fbanks::melscale_fbanks;

pub fn melscale(
    input: &[Vec<f64>],
    n_mels: usize,
    sample_rate: usize,
    f_min: f64,
    f_max: Option<f64>,
    n_stft: usize,
    norm: Option<&str>,
    mel_scale: &str,
) -> Vec<Vec<f64>> {
    assert!(n_stft > 0, "n_stft must be positive");
    assert!(!input.is_empty(), "input must not be empty");
    assert_eq!(input.len(), n_stft, "input first dim must match n_stft");

    let frame_count = input[0].len();
    for row in input {
        assert_eq!(
            row.len(),
            frame_count,
            "all input rows must have same frame count"
        );
    }

    let f_max = f_max.unwrap_or((sample_rate as f64) / 2.0);
    let fbanks = melscale_fbanks(n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_scale);

    // 对信号使用 Mel 滤波器组
    let mut output = vec![vec![0.0; frame_count]; n_mels];
    for mel_index in 0..n_mels {
        for frame_index in 0..frame_count {
            let mut value = 0.0;
            for freq_index in 0..n_stft {
                value += fbanks[freq_index][mel_index] * input[freq_index][frame_index];
            }
            output[mel_index][frame_index] = value;
        }
    }

    output
}
