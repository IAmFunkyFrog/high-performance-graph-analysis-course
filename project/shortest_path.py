import math

import pygraphblas as pgb
from pygraphblas import types, Matrix

from project import Graph

__all__ = ["bellman_ford", "bellman_ford_multi_source", "floyd_warshall"]


def bellman_ford(graph: Graph, start_node: int) -> list[int]:
    return bellman_ford_multi_source(graph, [start_node])[0][1]


def bellman_ford_multi_source(
    graph: Graph, start_nodes: list[int]
) -> list[tuple[int, list[int]]]:
    if len(graph.nodes) == 0 or len(start_nodes) == 0:
        return []

    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.FP64, zero_diag=True)
    front = Matrix.sparse(types.FP64, len(start_nodes), len(graph.nodes))

    for i in range(len(start_nodes)):
        front[i, start_nodes[i]] = 0

    result = bellman_ford_multi_source_matrix(adj_matrix, front)

    if result.isne(result.mxm(adj_matrix, semiring=pgb.FP64.MIN_PLUS)):
        raise ValueError("Negative weight cycle detected")

    return [
        (
            start_nodes[i],
            [
                result.get(start_nodes[i], col, default=math.inf)
                for col in range(len(graph.nodes))
            ],
        )
        for i in range(len(start_nodes))
    ]


def bellman_ford_multi_source_matrix(graph: Matrix, front: Matrix) -> Matrix:
    for _ in range(front.ncols):
        front.mxm(
            graph,
            out=front,
            semiring=pgb.FP64.MIN_PLUS,
        )
    return front


def floyd_warshall(graph: Graph) -> list[tuple[int, list[int]]]:
    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.FP64, zero_diag=True)
    front = adj_matrix.dup()

    for k in range(len(graph.nodes)):
        step = front.extract_matrix(col_index=k).mxm(
            front.extract_matrix(row_index=k), semiring=pgb.FP64.MIN_PLUS
        )
        front.eadd(step, add_op=pgb.FP64.MIN, out=front)

    for k in range(len(graph.nodes)):
        step = front.extract_matrix(col_index=k).mxm(
            front.extract_matrix(row_index=k), semiring=pgb.FP64.MIN_PLUS
        )
        if front.isne(front.eadd(step, add_op=pgb.FP64.MIN)):
            raise ValueError("Negative weight cycle detected")

    return [
        (
            row,
            [front.get(row, col, default=math.inf) for col in range(len(graph.nodes))],
        )
        for row in range(len(graph.nodes))
    ]
