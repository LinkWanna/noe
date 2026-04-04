pub fn hz_to_mel(frequency_hz: f64, mel_scale: &str) -> f64 {
    match mel_scale {
        "htk" => 2595.0 * (1.0 + frequency_hz / 700.0).log10(),
        "slaney" => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4_f64).ln() / 27.0;

            if frequency_hz < min_log_hz {
                frequency_hz / f_sp
            } else {
                min_log_mel + (frequency_hz / min_log_hz).ln() / logstep
            }
        }
        _ => panic!("unsupported mel scale: {mel_scale}"),
    }
}

pub fn mel_to_hz(mel: f64, mel_scale: &str) -> f64 {
    match mel_scale {
        "htk" => 700.0 * (10_f64.powf(mel / 2595.0) - 1.0),
        "slaney" => {
            let f_sp = 200.0 / 3.0;
            let min_log_hz = 1000.0;
            let min_log_mel = min_log_hz / f_sp;
            let logstep = (6.4_f64).ln() / 27.0;

            if mel < min_log_mel {
                mel * f_sp
            } else {
                min_log_hz * (logstep * (mel - min_log_mel)).exp()
            }
        }
        _ => panic!("unsupported mel scale: {mel_scale}"),
    }
}

fn linspace(start: f64, end: f64, steps: usize) -> Vec<f64> {
    if steps == 0 {
        return Vec::new();
    }
    if steps == 1 {
        return vec![start];
    }

    let step = (end - start) / ((steps - 1) as f64);
    (0..steps)
        .map(|index| start + (index as f64) * step)
        .collect()
}

pub fn melscale_fbanks(
    n_freqs: usize,
    f_min: f64,
    f_max: f64,
    n_mels: usize,
    sample_rate: usize,
    norm: Option<&str>,
    mel_scale: &str,
) -> Vec<Vec<f64>> {
    assert!(n_freqs > 0, "n_freqs must be positive");
    assert!(n_mels > 0, "n_mels must be positive");
    assert!(f_min >= 0.0, "f_min must be non-negative");
    assert!(f_max > f_min, "f_max must be greater than f_min");
    assert!(
        f_max <= (sample_rate as f64) / 2.0 + 1e-12,
        "f_max must not exceed Nyquist frequency"
    );
    assert!(
        norm.is_none() || norm == Some("slaney"),
        "norm must be None or 'slaney'"
    );

    let all_freqs = linspace(0.0, (sample_rate as f64) / 2.0, n_freqs);

    let mel_min = hz_to_mel(f_min, mel_scale);
    let mel_max = hz_to_mel(f_max, mel_scale);
    let mel_points = linspace(mel_min, mel_max, n_mels + 2);
    let hz_points: Vec<f64> = mel_points
        .iter()
        .map(|mel| mel_to_hz(*mel, mel_scale))
        .collect();

    let mut fbanks = vec![vec![0.0; n_mels]; n_freqs];

    for mel_index in 0..n_mels {
        let left = hz_points[mel_index];
        let center = hz_points[mel_index + 1];
        let right = hz_points[mel_index + 2];

        let down_denom = (center - left).max(f64::EPSILON);
        let up_denom = (right - center).max(f64::EPSILON);

        for (freq_index, frequency) in all_freqs.iter().enumerate() {
            let lower = (*frequency - left) / down_denom;
            let upper = (right - *frequency) / up_denom;
            let value = lower.min(upper).max(0.0);
            fbanks[freq_index][mel_index] = value;
        }

        if norm == Some("slaney") {
            let enorm = 2.0 / (right - left).max(f64::EPSILON);
            for row in &mut fbanks {
                row[mel_index] *= enorm;
            }
        }
    }

    fbanks
}
