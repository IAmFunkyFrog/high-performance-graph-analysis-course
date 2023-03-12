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

    result = [-1 for i in range(len(graph.nodes))]
    if start_node_order is None:
        return result

    adj_matrix = Matrix.sparse(types.BOOL, len(graph.nodes), len(graph.nodes))
    for node in graph.nodes:
        connected_nodes = graph.get_connected_nodes(node)
        for connected in connected_nodes:
            adj_matrix[graph.order(node), graph.order(connected)] = True

    result[start_node_order] = 0
    front = Vector.sparse(types.BOOL, len(graph.nodes))
    front[start_node_order] = True

    last_nnz = len(front.nonzero())
    step = 0
    while True:
        step += 1
        front = front + front @ adj_matrix

        cur_nnz = len(front.nonzero())
        if last_nnz == cur_nnz:
            break
        else:
            last_nnz = cur_nnz

        for (node_order, _) in front.nonzero():
            if result[node_order] == -1:
                result[node_order] = step

    return result
