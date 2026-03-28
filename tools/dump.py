import json
import logging
from pathlib import Path

import numpy as np
from onnx import ModelProto, NodeProto

from tools.planner import Planner


def dump_linear(
    node: NodeProto,
    shape_map: dict[str, list[int | None]],
    init_map: dict[str, np.ndarray],
    planner: Planner,
    path: Path,
) -> dict:
    in_features = shape_map[node.input[0]][1]
    out_features = shape_map[node.output[0]][1]

    weight = node.input[1].strip("/").replace(".", "_").replace("/", "_") + ".bin"
    if len(node.input) > 2:
        bias = node.input[2].strip("/").replace(".", "_").replace("/", "_") + ".bin"

    output_scale = None
    for attr in node.attribute:
        if attr.name == "input_scale":
            input_scale = attr.f
        elif attr.name == "weight_scale":
            weight_scale = attr.f
        elif attr.name == "output_scale":
            output_scale = attr.f
        elif attr.name == "activation":
            activation = attr.s.decode("utf-8")
    out_shift = int(np.log2(output_scale / (input_scale * weight_scale))) if output_scale else int(np.log2(1 / weight_scale))

    weight_data = init_map[node.input[1]]
    weight_data.tofile(path / weight)
    if len(node.input) > 2:
        bias_data = init_map[node.input[2]].astype(np.int16)
        bias_data.tofile(path / bias)

    return {
        "type": "linear",
        "in_features": in_features,
        "out_features": out_features,
        "weight": weight,
        "bias": bias if len(node.input) > 2 else None,
        "out_shift": out_shift,
        "activation": activation if activation else None,
        "input_off": planner.get_offset(node.input[0]),
        "output_off": planner.get_offset(node.output[0]),
    }


def dump_conv(
    node: NodeProto,
    shape_map: dict[str, list[int | None]],
    init_map: dict[str, np.ndarray],
    planner: Planner,
    path: Path,
) -> dict:
    input_shape = shape_map[node.input[0]][1:]  # NCHW -> CHW
    output_shape = shape_map[node.output[0]][1:]  # NCHW -> CHW

    weight = node.input[1].strip("/").replace(".", "_").replace("/", "_") + ".bin"
    if len(node.input) > 2:
        bias = node.input[2].strip("/").replace(".", "_").replace("/", "_") + ".bin"

    # get conv type based on input shape
    if len(input_shape) == 2:
        type = "conv1d"
    elif len(input_shape) == 3:
        type = "conv2d"
    elif len(input_shape) == 4:
        type = "conv3d"
    else:
        raise ValueError(f"Unsupported convolution input shape: {input_shape}")

    output_scale = None
    for attr in node.attribute:
        if attr.name == "input_scale":
            input_scale = attr.f
        elif attr.name == "weight_scale":
            weight_scale = attr.f
        elif attr.name == "output_scale":
            output_scale = attr.f
        elif attr.name == "activation":
            activation = attr.s.decode("utf-8")
        elif attr.name == "kernel_shape":
            kernel_size = list(attr.ints)
        elif attr.name == "strides":
            stride = list(attr.ints)
        elif attr.name == "pads":
            padding = list(attr.ints)
        elif attr.name == "dilations":
            dilation = list(attr.ints)
        elif attr.name == "group":
            groups = attr.i
    out_shift = int(np.log2(output_scale / (input_scale * weight_scale))) if output_scale else int(np.log2(1 / weight_scale))

    weight_data = init_map[node.input[1]]
    weight_data.tofile(path / weight)
    if len(node.input) > 2:
        bias_data = init_map[node.input[2]].astype(np.int16)
        bias_data.tofile(path / bias)

    return {
        "type": type,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "weight": weight,
        "bias": bias if len(node.input) > 2 else None,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "out_shift": out_shift,
        "activation": activation if activation else None,
        "input_off": planner.get_offset(node.input[0]),
        "output_off": planner.get_offset(node.output[0]),
    }


def dump_maxpool(
    node: NodeProto,
    shape_map: dict[str, list[int | None]],
    planner: Planner,
) -> dict:
    input_shape = shape_map[node.input[0]][1:]  # NCHW -> CHW
    output_shape = shape_map[node.output[0]][1:]  # NCHW -> CHW

    if len(input_shape) == 2:
        type = "maxpool1d"
    elif len(input_shape) == 3:
        type = "maxpool2d"
    elif len(input_shape) == 4:
        type = "maxpool3d"
    else:
        raise ValueError(f"Unsupported maxpool input shape: {input_shape}")

    for attr in node.attribute:
        if attr.name == "kernel_shape":
            kernel_size = list(attr.ints)
        elif attr.name == "strides":
            stride = list(attr.ints)
        elif attr.name == "pads":
            padding = list(attr.ints)
        elif attr.name == "dilations":
            dilation = list(attr.ints)
        elif attr.name == "input_scale":
            input_scale = attr.f
        elif attr.name == "output_scale":
            output_scale = attr.f
    out_shift = int(np.log2(input_scale / output_scale)) if output_scale else 0

    return {
        "type": type,
        "input_shape": input_shape,
        "output_shape": output_shape,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "out_shift": out_shift,
        "input_off": planner.get_offset(node.input[0]),
        "output_off": planner.get_offset(node.output[0]),
    }


def dump_batchnorm(
    node: NodeProto,
    shape_map: dict[str, list[int | None]],
    init_map: dict[str, np.ndarray],
    planner: Planner,
    path: Path,
) -> dict:
    input_shape = shape_map[node.input[0]][1:]  # NCHW -> CHW
    output_shape = shape_map[node.output[0]][1:]  # NCHW -> CHW
    assert input_shape == output_shape, "BatchNorm input and output shapes must match"
    assert planner.get_offset(node.input[0]) == planner.get_offset(node.output[0]), "BatchNorm input and output must share the same memory offset"

    mul = node.input[1].strip("/").replace(".", "_").replace("/", "_") + ".bin"
    add = node.input[2].strip("/").replace(".", "_").replace("/", "_") + ".bin"

    if len(input_shape) == 2:
        type = "batchnorm1d"
    elif len(input_shape) == 3:
        type = "batchnorm2d"
    elif len(input_shape) == 4:
        type = "batchnorm3d"
    else:
        raise ValueError(f"Unsupported batchnorm input shape: {input_shape}")

    for attr in node.attribute:
        if attr.name == "input_scale":
            input_scale = attr.f
        elif attr.name == "mul_scale":
            mul_scale = attr.f
        elif attr.name == "output_scale":
            output_scale = attr.f
        elif attr.name == "activation":
            activation = attr.s.decode("utf-8")
    out_shift = int(np.log2(output_scale / (input_scale * mul_scale))) if output_scale else int(np.log2(1 / mul_scale))

    mul_data = init_map[node.input[1]]
    mul_data.tofile(path / mul)
    add_data = init_map[node.input[2]].astype(np.int16)
    add_data.tofile(path / add)

    return {
        "type": type,
        "shape": input_shape,
        "mul": mul,
        "add": add,
        "out_shift": out_shift,
        "activation": activation if activation else None,
        "off": planner.get_offset(node.input[0]),
    }


def dump_add(
    node: NodeProto,
    shape_map: dict[str, list[int | None]],
    init_map: dict[str, np.ndarray],
    planner: Planner,
) -> dict:
    input_A_shape = shape_map[node.input[0]][1:]  # NCHW -> CHW
    input_B_shape = shape_map[node.input[1]][1:]  # NCHW -> CHW
    output_shape = shape_map[node.output[0]][1:]  # NCHW -> CHW
    assert input_A_shape == input_B_shape, "Add inputs shapes must match"

    for attr in node.attribute:
        if attr.name == "input_A_scale":
            input_A_scale = attr.f
        elif attr.name == "input_B_scale":
            input_B_scale = attr.f
        elif attr.name == "output_scale":
            output_scale = attr.f
        elif attr.name == "activation":
            activation = attr.s.decode("utf-8")
    B_shift = int(np.log2(input_B_scale / input_A_scale))
    out_shift = int(np.log2(output_scale / input_A_scale))

    # out = (A + B << B_shift) >> out_shift
    return {
        "type": "add",
        "A_shape": input_A_shape,
        "B_shape": input_B_shape,
        "output_shape": output_shape,
        "B_shift": B_shift,
        "out_shift": out_shift,
        "activation": activation if activation else None,
        "A_off": planner.get_offset(node.input[0]),
        "B_off": planner.get_offset(node.input[1]),
        "output_off": planner.get_offset(node.output[0]),
    }


def dump_globalavgpool(
    node: NodeProto,
    shape_map: dict[str, list[int | None]],
    planner: Planner,
) -> dict:
    input_shape = shape_map[node.input[0]][1:]  # NCHW -> CHW
    output_shape = shape_map[node.output[0]][1:]  # NCHW -> CHW
    assert input_shape[0] == output_shape[0], "GlobalAveragePool input and output batch size must match"

    if len(input_shape) == 2:
        type = "globalavgpool1d"
    elif len(input_shape) == 3:
        type = "globalavgpool2d"
    elif len(input_shape) == 4:
        type = "globalavgpool3d"
    else:
        raise ValueError(f"Unsupported global average pool input shape: {input_shape}")

    return {
        "type": type,
        "input_shape": input_shape,
        "input_off": planner.get_offset(node.input[0]),
        "output_off": planner.get_offset(node.output[0]),
    }


def dump(
    model: ModelProto,
    shape_map: dict[str, list[int | None]],
    init_map: dict[str, np.ndarray],
    planner: Planner,
    dir: str = "params",
):
    path = Path(dir)
    if not path.exists():
        path.mkdir(parents=True)

    input: list[str] = [input.name for input in model.graph.input]
    output: list[str] = [output.name for output in model.graph.output]

    layers: list[dict] = []
    layers.append({"type": "input", "size": planner.get_size(input[0]), "off": planner.get_offset(input[0])})
    for node in model.graph.node:
        if node.op_type == "FusedQuantLinear":
            info = dump_linear(node, shape_map, init_map, planner, path)
        elif node.op_type == "FusedQuantConv":
            info = dump_conv(node, shape_map, init_map, planner, path)
        elif node.op_type == "FusedQuantBatchNorm":
            info = dump_batchnorm(node, shape_map, init_map, planner, path)
        elif node.op_type == "FusedQuantAdd":
            info = dump_add(node, shape_map, init_map, planner)
        elif node.op_type == "MaxPool":
            info = dump_maxpool(node, shape_map, planner)
        elif node.op_type == "GlobalAveragePool":
            info = dump_globalavgpool(node, shape_map, planner)
        elif node.op_type in ["Flatten"]:
            continue  # Skip
        else:
            raise ValueError(f"Unsupported node type: {node.op_type}")
        layers.append(info)

    layers.append({"type": "output", "size": planner.get_size(output[0]), "off": planner.get_offset(output[0])})

    # Save layer information to a JSON file
    with open(path / "model.json", "w") as f:
        json.dump(
            {
                "memory": planner.memory,
                "format": "CHW",
                "layers": layers,
            },
            f,
            indent=4,
        )
        logging.info(f"Dumped model parameters and structure to {path / 'model.json'}")
