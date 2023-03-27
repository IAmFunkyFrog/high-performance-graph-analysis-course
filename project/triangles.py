from pygraphblas import Matrix, Vector, types

from project import Graph

import pygraphblas as pgb

__all__ = [
    "triangles_count_cohen",
    "triangles_count_sandia",
    "triangles_count_for_each_vertex",
]


def triangles_count_for_each_vertex(graph: Graph) -> list[int]:
    """
    Returns count of triangles for each node

    @param graph: graph to compute count of triangles
    @return: list where for each vertex computed count of triangles in which this node participates
    """
    if len(graph.nodes) == 0:
        return []

    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.INT32)
    result_vector = triangles_count_for_each_vertex_matrix(adj_matrix)

    return [result_vector.get(i, default=0) // 2 for i in range(result_vector.size)]


def triangles_count_for_each_vertex_matrix(graph: Matrix) -> Vector:
    """
    Returns count of triangles for each node

    @param graph: adjacency matrix of graph to compute count of triangles
    @return: vector where for each vertex computed count of triangles in which this node participates
    """
    squared = graph.mxm(graph, mask=graph, desc=pgb.descriptor.S)

    result = squared.reduce_vector()
    return result


def triangles_count_cohen(graph: Graph) -> int:
    """
    Returns count of triangles in graph

    @param graph: graph to compute count of triangles
    @return: count of triangles in graph
    """
    if len(graph.nodes) == 0:
        return 0

    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.INT32)
    return triangles_count_cohen_matrix(adj_matrix)


def triangles_count_cohen_matrix(graph: Matrix) -> int:
    """
    Returns count of triangles in graph

    @param graph: adjacency matrix of graph to compute count of triangles
    @return: count of triangles in graph
    """
    result = graph.tril().mxm(graph.triu(), mask=graph, desc=pgb.descriptor.S)
    return result.reduce() // 2


def triangles_count_sandia(graph: Graph) -> int:
    """
    Returns count of triangles in graph

    @param graph: graph to compute count of triangles
    @return: count of triangles in graph
    """
    if len(graph.nodes) == 0:
        return 0

    adj_matrix = graph.as_adjacency_matrix(matrix_type=types.INT32)
    return triangles_count_sandia_matrix(adj_matrix)


def triangles_count_sandia_matrix(graph: Matrix) -> int:
    """
    Returns count of triangles in graph

    @param graph: adjacency matrix of graph to compute count of triangles
    @return: count of triangles in graph
    """
    tril = graph.tril()
    result = tril.mxm(tril, mask=tril, desc=pgb.descriptor.S)
    return result.reduce()
