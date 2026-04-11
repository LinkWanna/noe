import logging
from collections import defaultdict, deque

import numpy as np
from onnx import GraphProto, ModelProto, NodeProto, helper


def toposort(graph: GraphProto):
    nodes = list(graph.node)
    if len(nodes) <= 1:
        return

    # producer: tensor_name -> node_index
    producer = {}
    for idx, node in enumerate(nodes):
        for out_name in node.output:
            if out_name:  # Ignore empty string tensor names
                producer[out_name] = idx

    # build adjacency list and indegree count
    adj = defaultdict(list)
    indegree = [0] * len(nodes)

    for v, node in enumerate(nodes):
        deps = set()
        for in_name in node.input:
            if not in_name:
                continue
            u = producer.get(in_name)
            if u is None or u == v:
                continue
            deps.add(u)

        indegree[v] = len(deps)
        for u in deps:
            adj[u].append(v)

    # if there are nodes with indegree 0, add them to the queue
    q = deque(i for i, d in enumerate(indegree) if d == 0)
    order = []

    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        raise ValueError("Graph contains a cycle or unresolved dependency; topological sort failed.")

    # inplace reorder nodes in graph according to topological order
    sorted_nodes = [nodes[i] for i in order]
    del graph.node[:]
    graph.node.extend(sorted_nodes)


def find_qdq_front(
    start: str,
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
) -> tuple[NodeProto | None, NodeProto | None]:
    """find the nearest DQ -> Q pattern in the forward direction, starting from the given tensor name"""
    dq_node = out_map.get(start, None)
    if dq_node is None or dq_node.op_type != "DequantizeLinear":
        return None, None

    q_node = out_map.get(dq_node.input[0], None)
    if q_node is None or q_node.op_type != "QuantizeLinear":
        return None, None

    return dq_node, q_node


def find_qdq_back(
    start: str,
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
) -> tuple[NodeProto | None, NodeProto | None]:
    """find the nearest Q -> DQ pattern in the backward direction, starting from the given tensor name"""
    q_node = in_map.get(start, None)
    if q_node is None or q_node.op_type != "QuantizeLinear":
        return None, None

    dq_node = in_map.get(q_node.output[0], None)
    if dq_node is None or dq_node.op_type != "DequantizeLinear":
        return None, None

    return q_node, dq_node


def attach_scale(
    graph: GraphProto,
    init_map: dict[str, np.ndarray],
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
):
    """find the nearest DQ and Q for each Pool/Mul/Conv/Gemm node, and attach the input/output scale as attributes to the node"""
    count = 0
    for node in graph.node:
        if node.op_type not in ["MaxPool", "Mul", "Conv", "Gemm"]:
            continue
        # find the nearest DQ in the forward direction
        dq_front = out_map.get(node.input[0], None)
        while dq_front and dq_front.op_type != "DequantizeLinear":
            dq_front = out_map.get(dq_front.input[0], None)
        assert dq_front is not None, f"Expected to find DequantizeLinear before {node.op_type}"

        # find the nearest Q in the backward direction
        q_back = in_map.get(node.output[0], None)
        while q_back and q_back.op_type != "QuantizeLinear":
            q_back = in_map.get(q_back.output[0], None)

        # attach scale attributes to the node
        # if no Q found, assume output scale is the same as input scale
        input_scale = init_map[dq_front.input[1]].item()
        output_scale = init_map[q_back.input[1]].item() if q_back else input_scale
        node.attribute.extend(
            [
                helper.make_attribute("input_scale", input_scale),
                helper.make_attribute("output_scale", output_scale),
            ]
        )
        count += 1
    logging.info(f"Attached scale attributes to {count} nodes.")


def fusion_linear(
    graph: GraphProto,
    init_map: dict[str, np.ndarray],
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
):
    to_remove = []
    to_create = []

    # 1. find (Flatten) -> Q -> DQ -> Gemm -> Act? -> (Q -> DQ)?
    #                       Weight ->
    #                        Bias? ->
    for node in graph.node:
        gemm = node
        if gemm.op_type != "Gemm":
            continue

        dq_front, q_front = find_qdq_front(gemm.input[0], in_map, out_map)
        if dq_front is None or q_front is None:
            continue
        weight = out_map[gemm.input[1]]
        weight_q = out_map[weight.input[0]]
        bias = out_map.get(gemm.input[2], None) if len(gemm.input) > 2 else None

        # catch the Flatten pattern before Gemm, which have the input shape
        f_node = out_map.get(q_front.input[0], None)
        flatten_input = None
        if f_node is not None and f_node.op_type == "Flatten":
            flatten_input = f_node.input[0]

        # mark these nodes for removal
        to_remove.extend([gemm, dq_front, q_front, weight, weight_q])
        if bias:
            to_remove.append(bias)
        last_node = gemm

        # optional activation
        act = in_map.get(last_node.output[0], None)
        if act and act.op_type in ["Relu"]:
            q_back, dq_back = find_qdq_back(act.output[0], in_map, out_map)
            to_remove.extend([act, q_back, dq_back])
            last_node = dq_back
        else:
            act = None

        assert last_node, "Expected to find last node after optional activation"

        # construct inputs for the fused node
        inputs = [
            q_front.input[0],
            weight_q.input[0],
        ]

        # if there is a bias input, also add it to the fused node's inputs
        if bias:
            inputs.append(bias.input[0])

        for attr in gemm.attribute:
            if attr.name == "input_scale":
                input_scale = attr.f
            elif attr.name == "output_scale":
                output_scale = attr.f
        weight_scale = init_map[weight.input[1]].item()
        bias_scale = init_map[bias.input[1]].item() if bias else None
        fused_node = helper.make_node(
            op_type="FusedQuantLinear",
            inputs=inputs,
            outputs=last_node.output,
            name=f"{gemm.name}_fused",
            # attrs
            activation=act.op_type if act else "",
            input_scale=input_scale,
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            output_scale=output_scale,
            flatten_input=flatten_input,
        )

        # create a new node to replace these nodes
        to_create.append(fused_node)

    logging.info(f"Fused {len(to_create)} linear patterns into {len(to_create)} fused nodes.")

    # 2. remove original nodes from the graph
    for node in to_remove:
        graph.node.remove(node)

    # 3. add new nodes to the graph
    graph.node.extend(to_create)


def fusion_conv(
    graph: GraphProto,
    init_map: dict[str, np.ndarray],
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
):
    to_remove = []
    to_create = []

    # 1. find Q -> DQ -> Conv -> Act? -> (Q -> DQ)?
    #          Weight ->
    #           Bias? ->
    for node in graph.node:
        conv = node
        if conv.op_type != "Conv":
            continue
        dq_front, q_front = find_qdq_front(conv.input[0], in_map, out_map)
        if dq_front is None or q_front is None:
            continue

        weight = out_map[conv.input[1]]
        weight_q = out_map[weight.input[0]]
        bias = out_map.get(conv.input[2], None) if len(conv.input) > 2 else None

        to_remove.extend([conv, dq_front, q_front, weight, weight_q])
        if bias:
            to_remove.append(bias)
        last_node = conv

        act = in_map.get(last_node.output[0], None)
        if act and act.op_type in ["Relu"]:
            q_back, dq_back = find_qdq_back(act.output[0], in_map, out_map)
            to_remove.extend([act, q_back, dq_back])
            last_node = dq_back
        else:
            act = None

        assert last_node, "Expected to find last node after optional activation"

        # Build inputs for the fused node
        inputs = [
            q_front.input[0],  # Q 的输入
            weight_q.input[0],  # Conv 的权重
        ]

        # If there is a bias input, also add it to the fused node's inputs
        if bias:
            inputs.append(bias.input[0])  # Conv 的 bias

        # Create a node to replace these nodes, assuming a custom op "FusedQuantConv" implements this behavior
        weight_scale = init_map[weight.input[1]].item()
        bias_scale = init_map[bias.input[1]].item() if bias else None

        conv_attrs = {attr.name: helper.get_attribute_value(attr) for attr in conv.attribute}
        fused_node = helper.make_node(
            op_type="FusedQuantConv",
            inputs=inputs,
            outputs=last_node.output,
            name=f"{conv.name}_fused",
            # attrs
            activation=act.op_type if act else "",
            weight_scale=weight_scale,
            bias_scale=bias_scale,
            **conv_attrs,
        )

        # Create a new node to replace these nodes
        to_create.append(fused_node)

    logging.info(f"Fused {len(to_create)} conv patterns into {len(to_create)} fused nodes.")

    # 2. Remove original nodes from the graph
    for node in to_remove:
        graph.node.remove(node)

    # 3. Add new nodes to the graph
    graph.node.extend(to_create)


def fusion_bn(
    graph: GraphProto,
    init_map: dict[str, np.ndarray],
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
):
    to_remove = []
    to_create = []

    # 1. Find the Q -> DQ -> Mul -> Add pattern
    for node in graph.node:
        mul = node
        if mul.op_type != "Mul":
            continue
        dq_front, q_front = find_qdq_front(mul.input[0], in_map, out_map)
        if dq_front is None or q_front is None:
            continue
        mul_reshape = out_map.get(mul.input[1], None)
        if mul_reshape is None or mul_reshape.op_type != "Reshape":
            continue
        mul_de = out_map.get(mul_reshape.input[0], None)
        if mul_de is None or mul_de.op_type != "DequantizeLinear":
            continue
        mul_clip = out_map.get(mul_de.input[0], None)
        if mul_clip is None or mul_clip.op_type != "Clip":
            continue

        add = in_map.get(mul.output[0], None)
        if add is None or add.op_type != "Add":
            continue
        add_reshape = out_map.get(add.input[1], None)
        if add_reshape is None or add_reshape.op_type != "Reshape":
            continue
        add_de = out_map.get(add_reshape.input[0], None)
        if add_de is None or add_de.op_type != "DequantizeLinear":
            continue

        to_remove.extend([dq_front, q_front, mul, mul_reshape, mul_de, mul_clip, add, add_reshape, add_de])
        last_node = add

        # Optional activation
        act = in_map.get(last_node.output[0], None)
        if act and act.op_type in ["Relu"]:
            q_back, dq_back = find_qdq_back(act.output[0], in_map, out_map)
            to_remove.extend([act, q_back, dq_back])
            last_node = dq_back
        else:
            act = None

        assert last_node, "Expected to find last node after optional activation"

        inputs = [
            q_front.input[0],  # Input of Mul
            mul_clip.input[0],  # Value tensor for Mul
            add_de.input[0],  # Value tensor for Add
        ]

        # Extract input_scale and output_scale from Mul attributes
        for attr in mul.attribute:
            if attr.name == "input_scale":
                input_scale = attr.f
            elif attr.name == "output_scale":
                output_scale = attr.f
        mul_scale = init_map[mul_de.input[1]].item()
        add_scale = init_map[add_de.input[1]].item()
        fused_node = helper.make_node(
            op_type="FusedQuantBatchNorm",
            inputs=inputs,
            outputs=last_node.output,
            name=f"{mul.name}_fused",
            # attrs
            activation=act.op_type if act else "",
            input_scale=input_scale,
            mul_scale=mul_scale,
            add_scale=add_scale,
            output_scale=output_scale,
        )

        to_create.append(fused_node)

    logging.info(f"Fused {len(to_create)} BatchNorm patterns into {len(to_create)} fused nodes.")

    # 2. Remove original nodes from the graph
    for node in to_remove:
        graph.node.remove(node)

    # 3. Add new nodes to the graph
    graph.node.extend(to_create)


def fusion_add(
    graph: GraphProto,
    init_map: dict[str, np.ndarray],
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
):
    to_remove = []
    to_create = []

    for node in graph.node:
        add = node
        if add.op_type != "Add":
            continue
        # Trace backward to find DQ and Q for input A
        dq_front_A, q_front_A = find_qdq_front(add.input[0], in_map, out_map)
        if dq_front_A is None or q_front_A is None:
            continue
        input_A_scale = init_map[dq_front_A.input[1]].item()

        # Continue tracing backward to ensure the correct scale is used
        dq_front_front_A, q_front_front_A = find_qdq_front(q_front_A.input[0], in_map, out_map)
        if dq_front_front_A and q_front_front_A:
            input_A_scale = init_map[dq_front_front_A.input[1]].item()

        # Trace backward to find DQ and Q for input B
        dq_front_B, q_front_B = find_qdq_front(add.input[1], in_map, out_map)
        if dq_front_B is None or q_front_B is None:
            continue
        input_B_scale = init_map[dq_front_B.input[1]].item()

        # Continue tracing backward to ensure the correct scale is used
        dq_front_front_B, q_front_front_B = find_qdq_front(q_front_B.input[0], in_map, out_map)
        if dq_front_front_B and q_front_front_B:
            input_B_scale = init_map[dq_front_front_B.input[1]].item()

        # Trace forward to find output Q and DQ
        q_back, dq_back = find_qdq_back(add.output[0], in_map, out_map)
        if q_back is None or dq_back is None:
            continue

        to_remove.extend([add, dq_front_A, q_front_A, dq_front_B, q_front_B, q_back, dq_back])
        last_node = dq_back

        # Optional activation
        act = in_map.get(last_node.output[0], None)
        if act and act.op_type in ["Relu"]:
            q_back, dq_back = find_qdq_back(act.output[0], in_map, out_map)
            to_remove.extend([act, q_back, dq_back])
            last_node = dq_back
        else:
            act = None
        assert last_node, "Expected to find last node after optional activation"

        # out = (A + (B << b_shift)) >> out_shift
        # If B has a larger scale, shift B left; otherwise shift A left.
        # The shift amount is determined by the scale difference.
        if input_A_scale > input_B_scale:
            input_A_scale, input_B_scale = input_B_scale, input_A_scale  # Ensure A's scale <= B's scale
            q_front_A, q_front_B = q_front_B, q_front_A  # Swap A and B inputs

        inputs = [
            q_front_A.input[0],  # Input A of Add
            q_front_B.input[0],  # Input B of Add
        ]
        output_scale = init_map[last_node.input[1]].item()
        fused_node = helper.make_node(
            op_type="FusedQuantAdd",
            inputs=inputs,
            outputs=last_node.output,
            name=f"{add.name}_fused",
            # attrs
            activation=act.op_type if act else "",
            input_A_scale=input_A_scale,
            input_B_scale=input_B_scale,
            output_scale=output_scale,
        )

        to_create.append(fused_node)

    logging.info(f"Fused {len(to_create)} Add patterns into {len(to_create)} fused nodes.")

    # 2. Remove original nodes from the graph
    for node in to_remove:
        graph.node.remove(node)

    # 3. Add new nodes to the graph
    graph.node.extend(to_create)


def fusion_gloabal_avgpool(
    graph: GraphProto,
    init_map: dict[str, np.ndarray],
    in_map: dict[str, NodeProto],
    out_map: dict[str, NodeProto],
):
    to_remove = []
    to_create = []

    for node in graph.node:
        avgpool = node
        if avgpool.op_type != "GlobalAveragePool":
            continue
        dq_front, q_front = find_qdq_front(avgpool.input[0], in_map, out_map)
        if dq_front is None or q_front is None:
            continue
        # Trace forward to find output Q and DQ
        q_back, dq_back = find_qdq_back(avgpool.output[0], in_map, out_map)
        if q_back is None or dq_back is None:
            continue

    logging.info(f"Fused {len(to_create)} GlobalAvgPool patterns into {len(to_create)} fused nodes.")

    # 2. 从图中删除原始节点
    for node in to_remove:
        graph.node.remove(node)

    # 3. 将新的节点添加到图中
    graph.node.extend(to_create)


def fusion(model: ModelProto, init_map: dict[str, np.ndarray]) -> ModelProto:
    graph = model.graph

    # 建立索引：输出名/输入名 -> 节点 (用于快速向上回溯)
    in_map = {inp: node for node in graph.node for inp in node.input}
    out_map = {out: node for node in graph.node for out in node.output}

    attach_scale(graph, init_map, in_map, out_map)
    fusion_linear(graph, init_map, in_map, out_map)
    fusion_conv(graph, init_map, in_map, out_map)
    fusion_bn(graph, init_map, in_map, out_map)
    fusion_add(graph, init_map, in_map, out_map)

    # Perform a final topological sort to ensure graph correctness
    toposort(graph)
    return model
