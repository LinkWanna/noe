import logging
import math

from onnx import ModelProto
from ortools.linear_solver import pywraplp

in_place_ops = {
    "Flatten",
    "FusedQuantBatchNorm",
}


def shading(model: ModelProto, shape_map: dict[str, list[int | None]], layout: str) -> dict[str, str]:
    """Assign colors to tensors in the model so that the same tensor shares the same color."""
    idx = 0
    colored: dict[str, str] = {}

    # Color the remaining nodes, ensuring the same tensor keeps the same color
    for node in model.graph.node:
        input = node.input[0]
        output = node.output[0]
        if node.op_type in in_place_ops:
            if input not in colored and output not in colored:
                colored[output] = colored[input] = f"Tensor_{idx}"
                idx += 1
            else:
                if input in colored and output not in colored:
                    colored[output] = colored[input]
                elif output in colored and input not in colored:
                    colored[input] = colored[output]
                # If both are already colored but with different colors, the model is inconsistent
                elif colored[input] != colored[output]:
                    raise ValueError(f"Conflict in coloring for {input} and {output}")

        if input not in colored:
            colored[input] = f"Tensor_{idx}"
            idx += 1

        if output not in colored:
            colored[output] = f"Tensor_{idx}"
            idx += 1

        if node.name in shape_map:  # some nodes have cache
            if node.op_type == "FusedQuantConv" and layout == "CHW":
                shape_map[node.name] = [0]  # when layout is CHW, Conv doesn't need cache
            colored[node.name] = f"Tensor_{idx}"
            idx += 1

    return colored


def lifetime_analysis(model: ModelProto, colored: dict[str, str]) -> dict[str, tuple[int, int]]:
    lifetimes: dict[str, tuple[int, int]] = {}

    for idx, node in enumerate(model.graph.node):
        for inp in node.input:
            if inp in colored:
                color = colored[inp]
                if color not in lifetimes:
                    lifetimes[color] = (idx, idx)
                else:
                    start, end = lifetimes[color]
                    lifetimes[color] = (min(start, idx), max(end, idx))

        for out in node.output:
            if out in colored:
                color = colored[out]
                if color not in lifetimes:
                    lifetimes[color] = (idx, idx)
                else:
                    start, end = lifetimes[color]
                    lifetimes[color] = (min(start, idx), max(end, idx))

        if node.name in colored:
            lifetimes[colored[node.name]] = (idx, idx)

    return lifetimes


def memory_solve(lifetimes: dict[str, tuple[int, int]], size_map: dict[str, int], align: int = 4) -> tuple[int, dict[str, int]]:
    """
    Solve the optimal memory allocation plan using integer linear programming.

    :param lifetimes: Lifetime of each buffer, formatted as tensor_name -> (start, end)
    :param size_map: Size of each buffer, formatted as tensor_name -> size
    :param align: Memory alignment size

    :return: Total memory size and per-buffer offsets, formatted as
             (total_memory, {buffer_name: offset})
    """

    def overlap(life1: tuple[int, int], life2: tuple[int, int]) -> bool:
        """Check whether two lifetimes overlap."""
        return not (life1[1] < life2[0] or life2[1] < life1[0])

    n = len(lifetimes)

    # 1. Create solver
    solver = pywraplp.Solver.CreateSolver("CP-SAT")
    if not solver:
        raise RuntimeError("CP-SAT solver not available")

    # 2. Define variables
    offsets = {}
    for name in lifetimes.keys():
        offsets[name] = solver.IntVar(0, solver.infinity(), f"offset_{name}") * align

    M = solver.IntVar(0, solver.infinity(), "total_memory")

    # 3. Add constraints: M >= offsets[name] + size_map[name] for all buffers
    for name in lifetimes.keys():
        solver.Add(offsets[name] + size_map[name] <= M)

    # 4. Add non-overlap constraints
    big_M = sum(size_map.values())  # Safe upper bound
    keys = list(lifetimes.keys())
    for i in range(n):
        for j in range(i + 1, n):
            buf1, buf2 = keys[i], keys[j]
            if not overlap(lifetimes[buf1], lifetimes[buf2]):
                continue  # No overlap, no separation needed

            # Binary variable: y = 1 => buf1 before buf2; y = 0 => buf2 before buf1
            y = solver.IntVar(0, 1, f"y_{buf1}_{buf2}")

            # Constraint 1: offsets[buf1] + size_map[buf1] <= offsets[buf2] + big_M * (1 - y)
            solver.Add(offsets[buf1] + size_map[buf1] <= offsets[buf2] + big_M * (1 - y))

            # Constraint 2: offsets[buf2] + size_map[buf2] <= offsets[buf1] + big_M * y
            solver.Add(offsets[buf2] + size_map[buf2] <= offsets[buf1] + big_M * y)

    # 5. Objective: minimize total memory
    solver.Minimize(M)

    # 6. Solve
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution found")

    total_memory = int(M.solution_value())
    offsets_result = {name: int(offsets[name].solution_value()) for name in lifetimes.keys()}

    return total_memory, offsets_result


class Planner:
    def __init__(
        self,
        model: ModelProto,
        shape_map: dict[str, list[int | None]],
        layout: str,
        align: int = 4,
    ):
        logging.info("Memory planning started")
        # 1. Color tensors
        self.colored = shading(model, shape_map, layout)
        logging.debug("Colored tensors:")
        for name, color in self.colored.items():
            logging.debug(f"\t{name} -> {color}")

        # 2. Compute size of each tensor
        self.size_map = {tensor: math.prod(shape_map.get(name, [0])) for name, tensor in self.colored.items()}

        # 3. Perform lifetime analysis
        self.lifetimes = lifetime_analysis(model, self.colored)

        # 4. Solve memory allocation
        self.memory, self.allocs = memory_solve(self.lifetimes, self.size_map, align)
        logging.debug(f"Total memory needed: {self.memory} bytes")
        logging.debug("Tensor info:")
        for tensor, (start, end) in self.lifetimes.items():
            logging.debug(f"\t{tensor:<10}: ({start:3}, {end:3}), size: {self.size_map[tensor]:8} bytes, offset: {self.allocs[tensor]:8} bytes")

        logging.info("Memory planning completed")

    def get_size(self, tensor: str) -> int:
        """Get the size of the specified tensor."""
        color = self.colored.get(tensor)
        if color is None:
            raise ValueError(f"Tensor {tensor} not found in colored map")
        return self.size_map[color]

    def get_offset(self, tensor: str) -> int:
        """Get the memory offset of the specified tensor."""
        color = self.colored.get(tensor)
        if color is None:
            raise ValueError(f"Tensor {tensor} not found in colored map")
        return self.allocs[color]
