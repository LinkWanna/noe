from pathlib import Path

import onnx

from tools.dump import dump
from tools.fusion import fusion
from tools.parser import onnx_parse
from tools.planner import Planner


def quantize(path: Path, output: Path, layout: str):
    assert layout in ["CHW", "HWC"], "Unsupported layout, only 'CHW' and 'HWC' are supported."

    net = path.stem
    dir = path.parent

    model = onnx.load(dir / f"{net}.onnx")

    # get shape info and initializers from the model, which may be used for fusion and dumping
    shape_map, init_map = onnx_parse(model)

    # perform operator fusion to optimize the model, which may modify the graph structure and initializers
    model = fusion(model, shape_map, init_map)

    # create a memory planner to analyze tensor lifetimes and compute memory offsets, which may be used for dumping
    planner = Planner(model, shape_map, layout)

    # dump the optimized model and memory plan to files, which may be used for code generation and execution
    dump(model, shape_map, init_map, planner, output, layout)
    onnx.save(model, dir / f"{net}_fused.onnx")  # save the fused model for Netron visualization


__all__ = ["quantize"]
