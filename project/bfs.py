import pygraphblas
from pygraphblas import Matrix, types, Vector
import pygraphblas as pgb

from .graph import *

__all__ = ["bfs", "bfs_multi_source_parents"]


def bfs_multi_source_parents(
    graph: Graph, start_node_orders: list[int]
) -> list[tuple[int, list[int]]]:
    """
    Make bfs on given graph with given start nodes

    @param graph: graph to make bfs
    @param start_node_orders: indexes of start nodes inside node list in graph
    @return: list, which contains info about reachability to each node from given
    """
    if len(graph.nodes) == 0 or len(start_node_orders) == 0:
        return []

    adj_matrix = graph.as_adjacency_matrix()
    front = Matrix.sparse(types.INT32, len(start_node_orders), len(graph.nodes))

    for i in range(len(start_node_orders)):
        front[i, start_node_orders[i]] = start_node_orders[i]

    result = bfs_matrix_multi_source_parents(adj_matrix, front)
    return [
        (start_node_orders[i], list(result[i, :].vals))
        for i in range(len(start_node_orders))
    ]


def bfs(graph: Graph, start_node_order: int) -> list[int]:
    """
    Make bfs on given graph with given start node

    @param graph: graph to make bfs
    @param start_node_order: index of start node inside node list in graph
    @return: list, which contains info (number of hopes need to reach) about reachability to each node from given
    """
    if len(graph.nodes) == 0:
        return []

    if start_node_order is None:
        return [-1] * len(graph.nodes)

    adj_matrix = graph.as_adjacency_matrix()

    front = Vector.sparse(types.BOOL, len(graph.nodes))
    front[start_node_order] = True
    return list(bfs_matrix(adj_matrix, front).vals)


def bfs_matrix(adj_matrix: Matrix, front: Vector) -> Vector:
    """
    Make bfs on given adjacency matrix with given front

    @param adj_matrix: graph to make bfs
    @param front: front from which start bfs
    @return: list, which contains info (number of hopes need to reach) about reachability to each node from given
    """
    result = Vector.sparse(types.INT32, front.size, fill=0, mask=front)

    step = 0
    while True:
        step += 1
        front = front.vxm(
            adj_matrix,
            mask=result,
            desc=pygraphblas.descriptor.S & pygraphblas.descriptor.C,
        )

        if front.nvals == 0:
            break

        result[front] = step

    result.assign_scalar(-1, mask=result, desc=pgb.descriptor.S & pgb.descriptor.C)
    return result


def bfs_matrix_multi_source_parents(adj_matrix: Matrix, front: Matrix) -> Matrix:
    """
    Make bfs on given adjacency matrix with given front

    @param adj_matrix: graph to make bfs
    @param front: front from which start bfs
    @return: list, which contains info about reachability to each node from given
    """
    result = Matrix.sparse(pgb.INT32, front.nrows, front.ncols)

    result.assign_scalar(-1, mask=front, desc=pgb.descriptor.S)
    while front.nvals > 0:
        front.mxm(
            adj_matrix,
            out=front,
            semiring=pgb.INT32.MIN_FIRST,
            mask=result,
            desc=pgb.descriptor.S & pgb.descriptor.RC,
        )
        result.eadd(front, out=result)
        front.apply(
            pgb.INT32.POSITIONJ,
            out=front,
        )
    result.assign_scalar(-2, mask=result, desc=pgb.descriptor.S & pgb.descriptor.C)
    return result
