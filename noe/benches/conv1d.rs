mod common;

use common::{ACT_MAX, ACT_MIN, i8_bytes, leak_i8_slice, make_i8_data};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::layer::{Conv1d, Module};

fn bench_conv1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv1d");
    for (in_c, in_len, out_c, kernel_size, stride, padding) in [
        (8usize, 64usize, 16usize, 3usize, 1usize, 1usize),
        (16, 128, 32, 5, 1, 2),
    ] {
        let out_len = (in_len + 2 * padding - kernel_size) / stride + 1;
        let input = leak_i8_slice(make_i8_data(in_c * in_len, 13));
        let weight = leak_i8_slice(make_i8_data(out_c * in_c * kernel_size, 17));
        let bias = leak_i8_slice(make_i8_data(out_c, 19));
        let output = leak_i8_slice(vec![0; out_c * out_len]);
        let module = Conv1d::new(
            i8_bytes(weight),
            Some(i8_bytes(bias)),
            (in_c, in_len),
            (out_c, out_len),
            kernel_size,
            stride,
            (padding, padding),
            1,
            1,
            1,
            input.as_ptr(),
            output.as_mut_ptr(),
            ACT_MIN,
            ACT_MAX,
        );

        group.throughput(Throughput::Elements((out_c * out_len) as u64));
        group.bench_with_input(
            BenchmarkId::new("shape", format!("{}x{}", in_c, out_len)),
            &(in_c, out_len),
            |bench, _| {
                bench.iter(|| black_box(module.forward_chw()));
            },
        );
    }
    group.finish();
}

criterion_group!(conv1d_benches, bench_conv1d);
criterion_main!(conv1d_benches);