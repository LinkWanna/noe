import logging

import onnx
from rich.logging import RichHandler

from tools.dump import dump
from tools.fusion import fusion
from tools.parser import onnx_parse
from tools.planner import Planner

# configure logging to use RichHandler for better console output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_path=True,
            show_time=False,
        )
    ],
)


def main():
    # get the current working directory of this script
    cwd = __file__.rsplit("/", 1)[0]

    net = "cnn"
    model = onnx.load(f"{cwd}/model/{net}.onnx")

    # get shape info and initializers from the model, which may be used for fusion and dumping
    shape_map, init_map = onnx_parse(model)

    # perform operator fusion to optimize the model, which may modify the graph structure and initializers
    model = fusion(model, init_map)

    # create a memory planner to analyze tensor lifetimes and compute memory offsets, which may be used for dumping
    planner = Planner(model, shape_map)

    # dump the optimized model and memory plan to files, which may be used for code generation and execution
    dump(model, shape_map, init_map, planner, f"{cwd}/{net}_params")
    onnx.save(model, f"{cwd}/model/{net}_fused.onnx")  # save the fused model for Netron visualization


if __name__ == "__main__":
    main()
