__all__ = [
    "convert_to_graph",
    "convert_to_undirected_graph",
    "Node",
    "Edge",
    "Graph",
    "convert_to_weighted_graph",
]

import networkx
from pygraphblas import Matrix, types


class Node:
    def __init__(self, value, order=-1):
        self._value = value
        self._order = order

    def __eq__(self, other):
        if self._order >= 0 and other._order >= 0:
            return self._order == other._order
        return self._value == other._value

    def __hash__(self):
        if self._order >= 0:
            return self._order
        return self._value.__hash__()

    def order(self):
        return self._order


class Edge:
    def __init__(self, nodes: tuple[Node, Node], weight: float = 1.0):
        self._nodes = nodes
        self._weight = weight

    def __eq__(self, other):
        return self._nodes == other._nodes

    def inverted(self):
        return Edge((self._nodes[1], self._nodes[0]), self._weight)

    @property
    def nodes(self):
        return self._nodes

    @property
    def weight(self):
        return self._weight

    def __hash__(self):
        return self._nodes[0].__hash__() ^ self._nodes[1].__hash__()


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self._nodes = nodes
        self._orders = dict()
        self._adj = dict()
        self._adj_rev = dict()
        for i in range(len(nodes)):
            self._adj[nodes[i]] = list()
            self._adj_rev[nodes[i]] = list()
            self._orders[nodes[i]] = i
        for e in edges:
            self._adj[e.nodes[0]].append(e)
            self._adj_rev[e.nodes[1]].append(e)

    def add_edge(self, edge: Edge):
        if edge not in self._adj[edge.nodes[0]]:
            self._adj[edge.nodes[0]].append(edge)
            self._adj_rev[edge.nodes[1]].append(edge)
        return

    def remove_edge(self, edge: Edge):
        if edge in self._adj[edge.nodes[0]]:
            self._adj[edge.nodes[0]].remove(edge)
            self._adj_rev[edge.nodes[1]].remove(edge)
        return

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        result = []
        for node in self._nodes:
            result += self._adj[node]
        return result

    def get_connected_edges(self, node_from: any, successors: bool = True):
        """
        Get list of nodes which are reachable from given node

        @param successors: if true, return successors of given node, else return predecessors
        @param node_from: node from graph
        @return: set of nodes which are reachable and adjacent from given node
        """
        if successors:
            for edge in self._adj[node_from]:
                yield edge
        else:
            for edge in self._adj_rev[node_from]:
                yield edge

    def order(self, node: Node) -> int:
        """
        Get position of node in nodes list inside graph

        @param node: node form graph
        @return: index of node
        """
        if node.order() >= 0:
            return node.order()
        return self._orders[node]

    def as_adjacency_matrix(
        self, matrix_type=types.BOOL, zero_diag: bool = False
    ) -> Matrix:
        adj_matrix = Matrix.sparse(matrix_type, len(self.nodes), len(self.nodes))
        for edge in self.edges:
            node, connected = edge.nodes
            if matrix_type == types.BOOL:
                adj_matrix[self.order(node), self.order(connected)] = True
            elif matrix_type == types.INT32:
                adj_matrix[self.order(node), self.order(connected)] = int(edge.weight)
            else:
                adj_matrix[self.order(node), self.order(connected)] = edge.weight

        if zero_diag:
            for node in self._nodes:
                adj_matrix[self.order(node), self.order(node)] = 0

        return adj_matrix


def convert_to_graph(
    nodes: list[any], edges: list[tuple[any, any]], weights: list[float] = None
) -> Graph:
    """
    Converts given list of nodes and edges to Graph

    @param nodes: list of any objects
    @param edges: list of tuples, which contain objects from nodes list
    @return: graph as Graph class object
    """
    boxed_nodes = [Node(value) for value in nodes]
    boxed_edges = []
    for (value1, value2) in edges:
        node1 = Node(value1)
        node2 = Node(value2)
        # sanity check
        try:
            boxed_nodes.index(node1)
            boxed_nodes.index(node2)
        except ValueError:
            raise TypeError("One of edges contains node which is not in node list")

        boxed_edges.append(Edge((node1, node2)))

    if weights is not None:
        for i in range(len(weights)):
            boxed_edges[i]._weight = weights[i]

    return Graph(boxed_nodes, boxed_edges)


def convert_to_undirected_graph(
    nodes: list[any], edges: list[tuple[any, any]]
) -> Graph:
    """
    Converts given list of nodes and edges to undirected Graph

    @param nodes: list of any objects
    @param edges: list of tuples, which contain objects from nodes list
    @return: graph as Graph class object
    """
    new_edges = set()
    for edge in edges:
        new_edges.add((edge[1], edge[0]))
        new_edges.add((edge[0], edge[1]))

    return convert_to_graph(nodes, list(new_edges))


def convert_to_weighted_graph(nodes: list[any], edges: list[tuple[any, float, any]]):
    new_edges = []
    weights = []
    for edge in edges:
        new_edges.append((edge[0], edge[2]))
        weights.append(edge[1])

    return convert_to_graph(nodes, list(new_edges), weights)


def convert_from_networkx_graph(graph: networkx.DiGraph) -> Graph:
    boxed_nodes = []
    boxed_edges = []

    i = 0
    orders = dict()
    for node in graph.nodes:
        boxed_nodes.append(Node(node, i))
        orders[node] = i
        i += 1

    for node_from, node_to in graph.edges:
        boxed_edges.append(
            Edge((Node(node_from, orders[node_from]), Node(node_to, orders[node_from])))
        )

    return Graph(boxed_nodes, boxed_edges)
