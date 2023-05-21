import math
from dataclasses import dataclass, field
from queue import PriorityQueue

import pygraphblas as pgb
from pygraphblas import types, Matrix

from project import Graph, Edge, Node

__all__ = [
    "bellman_ford",
    "dijkstra_one_source",
    "bellman_ford_multi_source",
    "floyd_warshall",
    "dijkstra",
]


@dataclass(order=True)
class PriorityQueueItem:
    dist: int
    node: any = field(compare=False)


def dijkstra(graph: Graph, start_nodes: list[int]) -> list[tuple[int, list[int]]]:
    """
    Make shortest path search with Dijkstra algorithm

    @param graph: graph to make search
    @param start_nodes: start nodes
    @return: list of distances to each node
    """
    answer = list()

    for node in start_nodes:
        answer.append([node, dijkstra_one_source(graph, node)])

    return answer


def dijkstra_one_source(graph: Graph, start_node: int) -> list[int]:
    """
    Make shortest path search with Dijkstra algorithm

    @param graph: graph to make search
    @param start_node: one start nodes
    @return: list of distances to each node
    """
    dists = list([math.inf for _ in graph.nodes])

    queue = PriorityQueue()
    queue.put(PriorityQueueItem(0, graph.nodes[start_node]))
    dists[start_node] = 0

    while queue._qsize() > 0:
        item = queue._get()
        dist = item.dist
        current_node = item.node

        if dist > dists[graph.order(current_node)]:
            continue

        for edge in graph.get_connected_edges(current_node):
            weight = edge.weight
            new_dist = dists[graph.order(current_node)] + weight
            neighbour = edge.nodes[1]

            if new_dist < dists[graph.order(neighbour)]:
                dists[graph.order(neighbour)] = new_dist
                queue._put(PriorityQueueItem(new_dist, neighbour))

    return dists


class DijkstraDynamic:
    def __init__(self, graph: Graph, start_node: int):
        self._graph = graph
        self._start_node_order = start_node
        self._start_node = graph.nodes[start_node]
        dists = dijkstra_one_source(graph, start_node)
        self._dists_with_node = dict()
        for n in graph.nodes:
            self._dists_with_node[n] = dists[self._graph.order(n)]
        self._not_consistent = set()

    def add_edge(self, edge: Edge):
        self._graph.add_edge(edge)
        self._not_consistent.add(edge.nodes[1])
        return

    def remove_edge(self, edge: Edge):
        self._graph.remove_edge(edge)
        self._not_consistent.add(edge.nodes[1])
        return

    def dists(self):
        self._update()
        return self._dists_with_node

    def _rhs(self, node: Node):
        if node == self._start_node:
            return 0

        rhs = math.inf
        for edge in self._graph.get_connected_edges(node, False):
            rhs = min(self._dists_with_node[edge.nodes[0]] + edge.weight, rhs)

        return rhs

    def _update(self):
        queue = PriorityQueue()
        rhs_cache = dict()

        def rhs(node, recompute=False):
            if recompute:
                rhs_cache[node] = self._rhs(node)
            return rhs_cache[node]

        def put_node_in_queue(node: Node):
            queue._put(
                PriorityQueueItem(min(rhs(node), self._dists_with_node[node]), node)
            )

        for node in self._not_consistent:
            rhs_cache[node] = self._rhs(node)
            put_node_in_queue(node)

        self._not_consistent.clear()

        while queue._qsize() > 0:
            to_update_rhs = []
            item = queue._get()
            current_node = item.node
            current_node_rhs = rhs(current_node)

            if current_node_rhs < self._dists_with_node[current_node]:
                self._dists_with_node[current_node] = current_node_rhs
                to_update_rhs = self._graph.get_connected_edges(current_node)
            elif current_node_rhs > self._dists_with_node[current_node]:
                self._dists_with_node[current_node] = math.inf
                to_update_rhs = self._graph.get_connected_edges(current_node)
                if current_node_rhs != self._dists_with_node[current_node]:
                    put_node_in_queue(current_node)

            for edge in to_update_rhs:
                node_to = edge.nodes[1]
                if rhs(node_to, True) != self._dists_with_node[node_to]:
                    put_node_in_queue(node_to)


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
