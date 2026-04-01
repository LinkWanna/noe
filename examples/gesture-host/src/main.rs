mod dataset;
mod generated;

use crate::generated::model_run;

fn argmax(input: &[i8]) -> usize {
    let mut max_index = 0;
    let mut max_value = input[0];
    for (i, &value) in input.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }

    max_index
}

fn eval(path: &str, num: u32) -> f32 {
    let data = dataset::load_gesture_data(path, num).expect("Failed to load dataset");
    let samples = data.samples;
    let labels = data.labels;
    let output = &mut [0i8; 10];

    let mut acc = 0;
    for (i, image) in samples.chunks(1 * 3 * 150).enumerate() {
        let input: &[i8] =
            unsafe { core::slice::from_raw_parts(image.as_ptr().cast(), 1 * 3 * 150) };
        model_run(input, output);

        if argmax(output) as u8 == labels[i] {
            acc += 1;
        }
    }
    return acc as f32 / labels.len() as f32;
}

fn main() {
    let gesture_data_path = "/path/to/GestureData";
    let accuracy = eval(gesture_data_path, 10000);
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}
