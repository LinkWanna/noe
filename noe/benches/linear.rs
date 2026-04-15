mod common;

use common::{
    ACT_MAX, ACT_MIN, i8_bytes, i16_bytes, leak_i8_slice, leak_i16_slice, make_i8_data,
    make_i16_data,
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::layer::{Linear, Module};

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");
    for (in_features, out_features) in [(128usize, 64usize), (256, 128), (1024, 512)] {
        let input = leak_i8_slice(make_i8_data(in_features, 3));
        let weight = leak_i8_slice(make_i8_data(in_features * out_features, 5));
        let bias = leak_i16_slice(make_i16_data(out_features, 9));
        let output = leak_i8_slice(vec![0; out_features]);
        let mut module = Linear::new(
            i8_bytes(weight),
            Some(i16_bytes(bias)),
            in_features,
            out_features,
            1,
            input.as_ptr(),
            output.as_mut_ptr(),
            ACT_MIN,
            ACT_MAX,
        );

        group.throughput(Throughput::Elements(out_features as u64));
        group.bench_with_input(
            BenchmarkId::new("shape", format!("{}x{}", in_features, out_features)),
            &(in_features, out_features),
            |bench, _| {
                bench.iter(|| black_box(module.forward_chw()));
            },
        );
    }
    group.finish();
}

criterion_group!(linear_benches, bench_linear);
criterion_main!(linear_benches);
