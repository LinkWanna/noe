use std::{
    fs::File,
    io::{self, Read},
    path::Path,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WavFormat {
    pub audio_format: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub bits_per_sample: u16,
}

#[derive(Debug, Clone)]
pub struct WavData {
    pub format: WavFormat,
    pub samples: Vec<f32>, // Interleaved channel-major samples normalized to [-1.0, 1.0]
}

fn invalid_data(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn read_u16_le(input: &[u8], offset: usize) -> io::Result<u16> {
    if offset + 2 > input.len() {
        return Err(invalid_data("Unexpected EOF while reading u16"));
    }
    Ok(u16::from_le_bytes([input[offset], input[offset + 1]]))
}

fn read_u32_le(input: &[u8], offset: usize) -> io::Result<u32> {
    if offset + 4 > input.len() {
        return Err(invalid_data("Unexpected EOF while reading u32"));
    }
    Ok(u32::from_le_bytes([
        input[offset],
        input[offset + 1],
        input[offset + 2],
        input[offset + 3],
    ]))
}

fn decode_pcm_sample(bytes: &[u8], bits_per_sample: u16) -> io::Result<f32> {
    match bits_per_sample {
        8 => {
            let v = bytes[0] as f32;
            Ok((v - 128.0) / 128.0)
        }
        16 => {
            let v = i16::from_le_bytes([bytes[0], bytes[1]]) as f32;
            Ok(v / 32768.0)
        }
        24 => {
            let raw =
                ((bytes[2] as i32) << 24) | ((bytes[1] as i32) << 16) | ((bytes[0] as i32) << 8);
            Ok((raw >> 8) as f32 / 8_388_608.0)
        }
        32 => {
            let v = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f32;
            Ok(v / 2_147_483_648.0)
        }
        _ => Err(invalid_data(format!(
            "Unsupported PCM bit depth: {bits_per_sample}"
        ))),
    }
}

fn decode_float_sample(bytes: &[u8], bits_per_sample: u16) -> io::Result<f32> {
    match bits_per_sample {
        32 => Ok(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])),
        64 => Ok(f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]) as f32),
        _ => Err(invalid_data(format!(
            "Unsupported float bit depth: {bits_per_sample}"
        ))),
    }
}

pub fn decode_wav_bytes(bytes: &[u8]) -> io::Result<WavData> {
    if bytes.len() < 12 {
        return Err(invalid_data("Input too short for RIFF/WAV header"));
    }

    if &bytes[0..4] != b"RIFF" {
        return Err(invalid_data("Missing RIFF header"));
    }
    if &bytes[8..12] != b"WAVE" {
        return Err(invalid_data("Missing WAVE signature"));
    }

    let mut format: Option<WavFormat> = None;
    let mut data_chunk: Option<&[u8]> = None;
    let mut offset = 12usize;

    while offset + 8 <= bytes.len() {
        let chunk_id = &bytes[offset..offset + 4];
        let chunk_size = read_u32_le(bytes, offset + 4)? as usize;
        let data_start = offset + 8;
        let data_end = data_start
            .checked_add(chunk_size)
            .ok_or_else(|| invalid_data("Chunk size overflow"))?;

        if data_end > bytes.len() {
            return Err(invalid_data("Chunk exceeds file length"));
        }

        match chunk_id {
            b"fmt " => {
                if chunk_size < 16 {
                    return Err(invalid_data("fmt chunk too small"));
                }
                let audio_format = read_u16_le(bytes, data_start)?;
                let channels = read_u16_le(bytes, data_start + 2)?;
                let sample_rate = read_u32_le(bytes, data_start + 4)?;
                let bits_per_sample = read_u16_le(bytes, data_start + 14)?;

                if channels == 0 {
                    return Err(invalid_data("Invalid channel count: 0"));
                }

                format = Some(WavFormat {
                    audio_format,
                    channels,
                    sample_rate,
                    bits_per_sample,
                });
            }
            b"data" => {
                data_chunk = Some(&bytes[data_start..data_end]);
            }
            _ => {}
        }

        let padded = chunk_size + (chunk_size & 1);
        offset = data_start
            .checked_add(padded)
            .ok_or_else(|| invalid_data("Chunk offset overflow"))?;
    }

    let fmt = format.ok_or_else(|| invalid_data("Missing fmt chunk"))?;
    let data = data_chunk.ok_or_else(|| invalid_data("Missing data chunk"))?;

    let bytes_per_sample = (fmt.bits_per_sample as usize)
        .checked_div(8)
        .ok_or_else(|| invalid_data("Invalid bits_per_sample"))?;
    if bytes_per_sample == 0 {
        return Err(invalid_data("Invalid bits_per_sample: 0"));
    }

    let frame_bytes = bytes_per_sample
        .checked_mul(fmt.channels as usize)
        .ok_or_else(|| invalid_data("Frame byte size overflow"))?;
    if frame_bytes == 0 || data.len() % frame_bytes != 0 {
        return Err(invalid_data(
            "Data chunk size is not aligned with channel/sample format",
        ));
    }

    let sample_count = data.len() / bytes_per_sample;
    let mut samples = Vec::with_capacity(sample_count);

    for raw in data.chunks_exact(bytes_per_sample) {
        let sample = match fmt.audio_format {
            1 => decode_pcm_sample(raw, fmt.bits_per_sample)?,
            3 => decode_float_sample(raw, fmt.bits_per_sample)?,
            other => {
                return Err(invalid_data(format!(
                    "Unsupported WAV audio_format: {other} (only PCM=1 or IEEE float=3 supported)"
                )));
            }
        };
        samples.push(sample.clamp(-1.0, 1.0));
    }

    Ok(WavData {
        format: fmt,
        samples,
    })
}

pub fn decode_wav_file<P: AsRef<Path>>(path: P) -> io::Result<WavData> {
    let path_ref = path.as_ref();
    let mut file = File::open(path_ref)?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    decode_wav_bytes(&bytes).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("Failed to decode WAV file {}: {e}", path_ref.display()),
        )
    })
}

pub fn decode_wav_mono_f64<P: AsRef<Path>>(path: P) -> io::Result<(u32, Vec<f64>)> {
    let wav = decode_wav_file(path)?;
    let channels = wav.format.channels as usize;
    let mut mono = Vec::with_capacity(wav.samples.len() / channels.max(1));

    for frame in wav.samples.chunks_exact(channels) {
        let sum: f32 = frame.iter().copied().sum();
        mono.push((sum / channels as f32) as f64);
    }

    Ok((wav.format.sample_rate, mono))
}
