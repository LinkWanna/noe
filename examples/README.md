# NOE Examples

This folder contains end-to-end examples built on top of NOE.

Each example follows the same pipeline:

1. Train/export a quantized model with **Brevitas** (`main.py`, exported as QCDQ ONNX).
2. Use `tools/` to parse/fuse/plan and dump runtime parameters into `*_params/`.
3. Run Rust build; `build.rs` calls `noe_util::process_model(...)` to generate `src/generated.rs`.
4. Execute inference via `model_run(input, output)`.

## Example List

### `mlp`

- Task: FashionMNIST classification
- Input shape: `1 x 28 x 28`
- Classes: 10
- Host inference binary: `cargo run -p mlp`
- Expected accuracy in sample code comment: around **87.5%**

### `lenet`

- Task: FashionMNIST classification
- Input shape: `1 x 28 x 28`
- Classes: 10
- Host inference binary: `cargo run -p lenet`
- Expected accuracy in sample code comment: around **86.6%**

### `cnn`

- Task: FashionMNIST classification
- Input shape: `1 x 28 x 28`
- Classes: 10
- Host inference binary: `cargo run -p cnn`
- Expected accuracy in sample code comment: around **90.1%**

### `gesture-host`

- Task: gesture classification on host (desktop)
- Input shape: `3 x 150`
- Classes: 13
- Host inference binary: `cargo run -p gesture-host`

### `gesture-stm32f103c8`

- Task: on-device gesture inference with MPU6050 sensor
- Target: `thumbv7m-none-eabi` (STM32F103C8 + Embassy)
- Uses the same gesture model parameters as `gesture-host`

## Dataset Paths

Before running host examples, update dataset path strings in each `src/main.rs`:

- FashionMNIST examples: `/path/to/FashionMNIST`
- Gesture host example: `/path/to/GestureData`

## Required Workflow Order

For all host examples, use this order:

1. Run `uv sync` at repository root to install Python dependencies.
2. Train/export with Python to generate `*_params/`.
3. Run `cargo run -p <example>`.

## Python Training / Export

From repository root, you can run for example:

```bash
uv sync
uv run python examples/mlp/main.py --dataset /path/to/FashionMNIST
uv run python examples/cnn/main.py --dataset /path/to/FashionMNIST
uv run python examples/lenet/main.py --dataset /path/to/FashionMNIST
uv run python examples/gesture-host/main.py --dataset /path/to/GestureDataset
```

These scripts use Brevitas `export_onnx_qcdq`, then run NOE tools to generate `*_params/` used by Rust `build.rs`.

After parameters are generated, run Rust examples:

```bash
cargo run -p mlp
cargo run -p lenet
cargo run -p cnn
cargo run -p gesture-host
```

## Gesture Model Source

The gesture model and related dataset workflow are derived from:

- https://github.com/lyg09270/CyberryPotter_ElectromagicWand_Basic_Project

## Notes

- `src/generated.rs` is auto-generated during build from `*_params/model.json`.
- If you re-export a model, rebuild the corresponding Rust example to refresh generated code.
