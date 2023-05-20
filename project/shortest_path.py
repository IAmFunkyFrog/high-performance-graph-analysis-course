import math

import pygraphblas as pgb
from pygraphblas import types, Matrix

from project import Graph

__all__ = ["bellman_ford", "bellman_ford_multi_source", "floyd_warshall"]


def bellman_ford(graph: Graph, start_node: int) -> list[int]:
    """
    Make shortest path search with Bellman-Ford algorithm

    @param graph: graph to make search
    @param start_node: one start node
    @return: list of distances to each node
    """
    return bellman_ford_multi_source(graph, [start_node])[0][1]


def bellman_ford_multi_source(
    graph: Graph, start_nodes: list[int]
) -> list[tuple[int, list[int]]]:
    """
    Make shortest path search with Bellman-Ford algorithm

    @param graph: graph to make search
    @param start_nodes: list of start nodes
    @return: list of 2-element tuples (first is node order, second is list of distances to each node)
    """
    if len(graph.nodes) == 0 or len(start_nodes) == 0:
        return []

    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.FP64, zero_diag=True)
    front = Matrix.sparse(types.FP64, len(start_nodes), len(graph.nodes))

    for i in range(len(start_nodes)):
        front[i, start_nodes[i]] = 0

    result = bellman_ford_multi_source_matrix(adj_matrix, front)

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

    if front.isne(front.mxm(graph, semiring=pgb.FP64.MIN_PLUS)):
        raise ValueError("Negative weight cycle detected")

    return front


def floyd_warshall(graph: Graph) -> list[tuple[int, list[int]]]:
    """
    Make shortest path search with Floyd-Warshall algorithm

    @param graph: graph to make search
    @return: list of 2-element tuples (first is node order, second is list of distances to each node)
    """
    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.FP64, zero_diag=True)
    front = floyd_warshall_matrix(adj_matrix)

    return [
        (
            row,
            [front.get(row, col, default=math.inf) for col in range(len(graph.nodes))],
        )
        for row in range(len(graph.nodes))
    ]


def floyd_warshall_matrix(graph: Matrix) -> Matrix:
    front = graph.dup()

    for k in range(graph.ncols):
        step = front.extract_matrix(col_index=k).mxm(
            front.extract_matrix(row_index=k), semiring=pgb.FP64.MIN_PLUS
        )
        front.eadd(step, add_op=pgb.FP64.MIN, out=front)

    for k in range(graph.ncols):
        step = front.extract_matrix(col_index=k).mxm(
            front.extract_matrix(row_index=k), semiring=pgb.FP64.MIN_PLUS
        )
        if front.isne(front.eadd(step, add_op=pgb.FP64.MIN)):
            raise ValueError("Negative weight cycle detected")

    return front
