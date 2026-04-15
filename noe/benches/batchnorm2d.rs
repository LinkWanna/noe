mod common;

use common::{ACT_MAX, ACT_MIN, i8_bytes, i16_bytes, leak_i8_slice, leak_i16_slice, make_i8_data, make_i16_data};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::DataLayout;
use noe::layer::{BatchNorm2d, Module};

fn bench_batchnorm2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("batchnorm2d");
    for layout in [DataLayout::CHW, DataLayout::HWC] {
        let shape = (16usize, 16usize, 16usize);
        let data_len = shape.0 * shape.1 * shape.2;
        let channel = match layout {
            DataLayout::CHW => shape.0,
            DataLayout::HWC => shape.2,
        };

        let data = leak_i8_slice(make_i8_data(data_len, 47));
        let mul = leak_i8_slice(make_i8_data(channel, 53));
        let add = leak_i16_slice(make_i16_data(channel, 59));

        let module = BatchNorm2d::new(
            shape,
            i8_bytes(mul),
            i16_bytes(add),
            1,
            data.as_mut_ptr(),
            ACT_MIN,
            ACT_MAX,
            layout,
        );

        group.throughput(Throughput::Elements(data_len as u64));
        group.bench_with_input(
            BenchmarkId::new("layout", format!("{:?}", layout)),
            &layout,
            |bench, _| {
                bench.iter(|| {
                    if matches!(layout, DataLayout::CHW) {
                        black_box(module.forward_chw())
                    } else {
                        black_box(module.forward_hwc())
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(batchnorm2d_benches, bench_batchnorm2d);
criterion_main!(batchnorm2d_benches);