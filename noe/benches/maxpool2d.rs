mod common;

use common::{leak_i8_slice, make_i8_data};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::DataLayout;
use noe::layer::{MaxPool2d, Module};

fn bench_maxpool2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("maxpool2d");
    for layout in [DataLayout::CHW, DataLayout::HWC] {
        let (channel, in_h, in_w, kernel_size, stride, padding, out_shift) = (
            8usize,
            32usize,
            32usize,
            (2usize, 2usize),
            (2usize, 2usize),
            (0usize, 0usize),
            1isize,
        );
        let out_h = (in_h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
        let out_w = (in_w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;
        let input_len = channel * in_h * in_w;
        let output_len = channel * out_h * out_w;

        let input = leak_i8_slice(make_i8_data(input_len, 41));
        let output = leak_i8_slice(vec![0; output_len]);

        let module = MaxPool2d::new(
            (in_h, in_w),
            (out_h, out_w),
            channel,
            kernel_size,
            stride,
            (padding.0, padding.1, padding.0, padding.1),
            (1, 1),
            out_shift,
            input.as_ptr(),
            output.as_mut_ptr(),
            layout,
        );

        group.throughput(Throughput::Elements(output_len as u64));
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

criterion_group!(maxpool2d_benches, bench_maxpool2d);
criterion_main!(maxpool2d_benches);