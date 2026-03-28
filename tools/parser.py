import numpy as np
import onnx
from onnx import ModelProto, numpy_helper, shape_inference


def collect_shapes(model: onnx.ModelProto) -> dict[str, list[int | None]]:
    """Collect tensor shape info from graph value_info/input/output."""
    shape_map: dict[str, list[int | None]] = {}

    def _update(vi: onnx.ValueInfoProto):
        if not vi.type.HasField("tensor_type"):
            return
        ttype = vi.type.tensor_type
        if not ttype.HasField("shape"):
            return
        dims: list[int | None] = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims.append(int(d.dim_value))
            else:
                # symbolic dim or unknown
                dims.append(None)
        shape_map[vi.name] = dims

    for vi in model.graph.value_info:
        _update(vi)
    for vi in model.graph.input:
        _update(vi)
    for vi in model.graph.output:
        _update(vi)

    return shape_map


def onnx_parse(model: ModelProto) -> tuple[dict[str, list[int | None]], dict[str, np.ndarray]]:

    # 1. get inferred shapes, which may be incomplete but better than nothing
    inferred = shape_inference.infer_shapes(model)
    shape_map: dict[str, list[int | None]] = collect_shapes(inferred)

    # 2. get initializers, which may be used for fusion and dumping
    init_map: dict[str, np.ndarray] = {}
    for init in inferred.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    for node in inferred.graph.node:
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR:
                init_map[attr.name] = numpy_helper.to_array(attr.t)

    return shape_map, init_map
