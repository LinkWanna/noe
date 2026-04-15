mod common;

use common::{
    ACT_MAX, ACT_MIN, i8_bytes, i16_bytes, leak_i8_slice, leak_i16_slice, make_i8_data,
    make_i16_data,
};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use noe::DataLayout;
use noe::layer::{Conv2d, Module};

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");

    for layout in [DataLayout::CHW, DataLayout::HWC] {
        let (input_shape, output_shape, kernel_size, stride, padding, dilation, groups) =
            match layout {
                DataLayout::CHW => (
                    (8usize, 16usize, 16usize),
                    (16usize, 16usize, 16usize),
                    (3usize, 3usize),
                    (1usize, 1usize),
                    (1usize, 1usize, 1usize, 1usize),
                    (1usize, 1usize),
                    1usize,
                ),
                DataLayout::HWC => (
                    (16usize, 16usize, 8usize),
                    (16usize, 16usize, 16usize),
                    (3usize, 3usize),
                    (1usize, 1usize),
                    (1usize, 1usize, 1usize, 1usize),
                    (1usize, 1usize),
                    1usize,
                ),
            };

        let (input_len, output_len, weight_len, bias_len, tmp_len) = match layout {
            DataLayout::CHW => {
                let (ic, ih, iw) = input_shape;
                let (oc, oh, ow) = output_shape;
                let (kh, kw) = kernel_size;
                (
                    ic * ih * iw,
                    oc * oh * ow,
                    oc * ic * kh * kw / groups,
                    oc,
                    kh * kw * ic / groups,
                )
            }
            DataLayout::HWC => {
                let (ih, iw, ic) = input_shape;
                let (oh, ow, oc) = output_shape;
                let (kh, kw) = kernel_size;
                (
                    ih * iw * ic,
                    oc * oh * ow,
                    oc * ic * kh * kw / groups,
                    oc,
                    kh * kw * ic / groups,
                )
            }
        };

        let input = leak_i8_slice(make_i8_data(input_len, 23));
        let weight = leak_i8_slice(make_i8_data(weight_len, 29));
        let bias = leak_i16_slice(make_i16_data(bias_len, 31));
        let output = leak_i8_slice(vec![0; output_len]);
        let tmp = leak_i8_slice(vec![0; tmp_len]);

        let mut module = Conv2d::new(
            i8_bytes(weight),
            Some(i16_bytes(bias)),
            input_shape,
            output_shape,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            1,
            input.as_ptr(),
            output.as_mut_ptr(),
            tmp.as_mut_ptr(),
            ACT_MIN,
            ACT_MAX,
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

criterion_group!(conv2d_benches, bench_conv2d);
criterion_main!(conv2d_benches);
