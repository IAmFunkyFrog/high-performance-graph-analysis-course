import pygraphblas
from pygraphblas import Matrix, types, Vector

from .graph import *

__all__ = ["bfs"]


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

    adj_matrix = Matrix.sparse(types.BOOL, len(graph.nodes), len(graph.nodes))
    for node in graph.nodes:
        connected_nodes = graph.get_connected_nodes(node)
        for connected in connected_nodes:
            adj_matrix[graph.order(node), graph.order(connected)] = True

    front = Vector.sparse(types.BOOL, len(graph.nodes))
    front[start_node_order] = True
    return bfs_matrix(adj_matrix, front)


def bfs_matrix(adj_matrix: Matrix, front: Vector) -> list[int]:
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

    result.assign_scalar(
        -1, mask=result, desc=pygraphblas.descriptor.S & pygraphblas.descriptor.C
    )
    return list(result.vals)
