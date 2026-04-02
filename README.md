# NOE

NOE is a lightweight neural network operator library for **microprocessors and embedded systems**.

It targets **int8 inference**, works with `no_std`, and uses a generated static memory layout so models can run with minimal runtime overhead.

NOE is tightly coupled with **Brevitas**: the intended workflow is quantization-aware training/export in Brevitas, then conversion into NOE runtime artifacts.

**Highlights**

- User-friendly interfaces.
- Onboard pre-compiling - zero interpreter performance loss at runtime. 
- best memory efficiency for static models - use [ortools](https://github.com/google/or-tools) to optimize memory layout and operator fusion.
- low accuracy loss from **quantization-aware training**(QAT) with Brevitas.

## Workspace Layout

- `noe/`: core operator implementations and runtime API.
- `noe-util/`: reads `model.json` and generates Rust inference code.
- `examples/`: end-to-end examples (MLP, LeNet, CNN, gesture).
- `tools/`: utilities that process **Brevitas-exported QCDQ ONNX**.

## Model Generation Flow

NOE uses an offline codegen flow:

1. Train/export a quantized model with Brevitas (example scripts in `examples/*/main.py`, using `export_onnx_qcdq`).
2. Parse/fuse/plan and dump parameters into `*_params/`:
   - `model.json`
   - quantized `.bin` weights/bias files
3. During Rust build, `build.rs` runs:
   - `noe_util::process_model("<params_dir>", "src/generated.rs")`
4. Generated file exposes `model_run(input: &[i8], output: &mut [i8])`.

This avoids dynamic graph interpretation at runtime.

## `model.json` (Schema Overview)

The generator expects:

- `memory`: total static scratch memory size.
- `format`: `CHW` or `HWC`.
- `layers`: ordered layer definitions with tensor shapes, quantization shifts, activation, and memory offsets.

See:

- `noe-util/src/types.rs` for the full schema.
- `examples/*/*_params/model.json` for real examples.

## Examples

The `examples/` folder contains end-to-end examples built on top of NOE, demonstrating the full pipeline from Brevitas training/export to Rust inference. You can follow the instructions in `examples/README.md` to get started with these examples.

1. **Set up Python environment**
2. **Train/export model and generate `*_params/`**
3. **Run Rust example**

## Embedded Target Example

The `noe` and `noe-util` crates are designed to be `no_std` compatible, so they can be used in embedded Rust projects. You can use the generated `model_run` function in any Rust environment, including embedded targets.

`examples/gesture-stm32f103c8` demonstrates deployment on STM32F103C8 using Embassy and MPU6050 input, with NOE inference running directly on-device, which comes from [CyberryPotter_ElectromagicWand_Basic_Project](https://github.com/lyg09270/CyberryPotter_ElectromagicWand_Basic_Project).
