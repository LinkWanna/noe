mod common;

use common::{leak_i8_slice, make_i8_data};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::layer::{MaxPool1d, Module};

fn bench_maxpool1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxpool1d");
    for (channel, input_shape, kernel_size, stride, padding) in [
        (8usize, 128usize, 2usize, 2usize, 0usize),
        (16, 256, 3, 2, 1),
    ] {
        let output_shape = (input_shape + 2 * padding - kernel_size) / stride + 1;
        let input = leak_i8_slice(make_i8_data(channel * input_shape, 37));
        let output = leak_i8_slice(vec![0; channel * output_shape]);
        let module = MaxPool1d::new(
            input_shape,
            output_shape,
            channel,
            kernel_size,
            stride,
            (padding, padding),
            1,
            1,
            input.as_ptr(),
            output.as_mut_ptr(),
        );

        group.throughput(Throughput::Elements((channel * output_shape) as u64));
        group.bench_with_input(
            BenchmarkId::new("shape", format!("{}x{}", channel, output_shape)),
            &(channel, output_shape),
            |bench, _| {
                bench.iter(|| black_box(module.forward_chw()));
            },
        );
    }
    group.finish();
}

criterion_group!(maxpool1d_benches, bench_maxpool1d);
criterion_main!(maxpool1d_benches);