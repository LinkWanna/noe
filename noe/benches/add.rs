mod common;

use common::{ACT_MAX, ACT_MIN, leak_i8_slice, make_i8_data};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::layer::{Add, Module};

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    for size in [1_024usize, 16_384, 65_536] {
        let a = leak_i8_slice(make_i8_data(size, 1));
        let b = leak_i8_slice(make_i8_data(size, 7));
        let output = leak_i8_slice(vec![0; size]);
        let mut module = Add::new(
            (1, 1, size),
            (1, 1, size),
            (1, 1, size),
            1,
            1,
            a.as_ptr(),
            b.as_ptr(),
            output.as_mut_ptr(),
            ACT_MIN,
            ACT_MAX,
        );

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bench, _| {
            bench.iter(|| black_box(module.forward_chw()));
        });
    }
    group.finish();
}

criterion_group!(add_benches, bench_add);
criterion_main!(add_benches);
