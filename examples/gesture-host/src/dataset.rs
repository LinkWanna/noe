use std::{
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

#[derive(Debug)]
pub struct GestureData {
    pub samples: Vec<u8>, // 展平后的手势数据 [N * 3 * 150]，按 CHW 排列
    pub labels: Vec<u8>,  // 标签 [N]
}

const INPUT_CHANNELS: usize = 3;
const INPUT_STEPS: usize = 150;
const INPUT_SIZE: usize = INPUT_CHANNELS * INPUT_STEPS;
const INPUT_SCALE: f32 = 16.0;

fn gesture_label(name: &str) -> Option<u8> {
    match name {
        "RightAngle" => Some(0),
        "SharpAngle" => Some(1),
        "Lightning" => Some(2),
        "Triangle" => Some(3),
        "Letter_h" => Some(4),
        "letter_R" => Some(5),
        "letter_W" => Some(6),
        "letter_phi" => Some(7),
        "Circle" => Some(8),
        "UpAndDown" => Some(9),
        "Horn" => Some(10),
        "Wave" => Some(11),
        "NoMotion" => Some(12),
        _ => None,
    }
}

fn parse_label_from_filename(path: &Path) -> Option<u8> {
    if path.extension().and_then(|s| s.to_str()) != Some("txt") {
        return None;
    }

    let stem = path.file_stem()?.to_str()?;
    let (motion_name, id) = stem.rsplit_once('_')?;
    if !id.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }

    gesture_label(motion_name)
}

fn load_gesture_sample(path: &Path) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut steps: Vec<[f32; INPUT_CHANNELS]> = Vec::with_capacity(INPUT_STEPS);

    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 6 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid column count in {} at line {}: expected >= 6, got {}",
                    path.display(),
                    line_no + 1,
                    cols.len()
                ),
            ));
        }

        let gx = cols[3].parse::<f32>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Failed to parse gx in {} at line {}: {e}",
                    path.display(),
                    line_no + 1
                ),
            )
        })?;
        let gy = cols[4].parse::<f32>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Failed to parse gy in {} at line {}: {e}",
                    path.display(),
                    line_no + 1
                ),
            )
        })?;
        let gz = cols[5].parse::<f32>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Failed to parse gz in {} at line {}: {e}",
                    path.display(),
                    line_no + 1
                ),
            )
        })?;

        steps.push([gx, gy, gz]);
    }

    if steps.len() != INPUT_STEPS {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "Invalid sequence length in {}: expected {}, got {}",
                path.display(),
                INPUT_STEPS,
                steps.len()
            ),
        ));
    }

    let mut sample = Vec::with_capacity(INPUT_SIZE);
    for ch in 0..INPUT_CHANNELS {
        for step in steps.iter().take(INPUT_STEPS) {
            let data = (step[ch] * INPUT_SCALE).round().clamp(-127.0, 127.0) as i8 as u8;
            sample.push(data);
        }
    }

    Ok(sample)
}

pub fn load_gesture_data(root: &str, num: u32) -> std::io::Result<GestureData> {
    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.path());

    let mut samples = Vec::new();
    let mut labels = Vec::new();

    for entry in entries {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(label) = parse_label_from_filename(&path) else {
            continue;
        };

        let sample = load_gesture_sample(&path)?;
        samples.extend_from_slice(&sample);
        labels.push(label);

        if num > 0 && labels.len() >= num as usize {
            break;
        }
    }

    if labels.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("No valid gesture samples found in {}", root),
        ));
    }

    Ok(GestureData { samples, labels })
}
