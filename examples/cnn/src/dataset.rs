use std::{
    fs::File,
    io::{BufReader, Read},
};

#[derive(Debug)]
pub struct MnistData {
    pub images: Vec<u8>, // 展平的图像数据 [N * 28 * 28]
    pub labels: Vec<u8>, // 标签 [N]
}

fn read_magic_number<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn read_dimension<R: Read>(reader: &mut R) -> std::io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

fn load_fashion_mnist_images(path: &str, num: u32) -> std::io::Result<MnistData> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic = read_magic_number(&mut reader)?;
    if magic != 2051 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic number for images",
        ));
    }

    let num_images = read_dimension(&mut reader)?;
    let rows = read_dimension(&mut reader)?;
    let cols = read_dimension(&mut reader)?;

    let load_num = num.min(num_images);
    let mut images = vec![0u8; (load_num * rows * cols) as usize];
    reader.read_exact(&mut images)?;

    // 数据归一化
    let mean = 0.2860;
    let std = 0.3530;
    let scale = 32.0; // 0.03125
    let images_norm = images
        .iter()
        .map(|&p| {
            let p = p as f32;
            let normalized = (p / 255.0 - mean) / std;
            (normalized * scale).clamp(-127.0, 127.0) as i8 as u8
        })
        .collect::<Vec<_>>();

    Ok(MnistData {
        images: images_norm,
        labels: vec![],
    })
}

fn load_fashion_mnist_labels(path: &str, num: u32) -> std::io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let magic = read_magic_number(&mut reader)?;
    if magic != 2049 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid magic number for labels",
        ));
    }

    let num_items = read_dimension(&mut reader)?;
    let load_num = num.min(num_items);
    let mut labels = vec![0u8; load_num as usize];
    reader.read_exact(&mut labels)?;

    Ok(labels)
}

pub fn load_fashion_mnist(path: &str, num: u32) -> std::io::Result<MnistData> {
    let image_path = format!("{path}/raw/t10k-images-idx3-ubyte");
    let label_path = format!("{path}/raw/t10k-labels-idx1-ubyte");

    let mut data = load_fashion_mnist_images(&image_path, num)?;
    data.labels = load_fashion_mnist_labels(&label_path, num)?;

    Ok(data)
}
