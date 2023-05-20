__all__ = [
    "convert_to_graph",
    "convert_to_undirected_graph",
    "Node",
    "Edge",
    "Graph",
    "convert_to_weighted_graph",
]

from pygraphblas import Matrix, types


class Node:
    def __init__(self, value):
        self._value = value

    def __eq__(self, other):
        if not isinstance(other, Node):
            return self._value == other
        return self._value == other._value

    def __hash__(self):
        return self._value.__hash__()


class Edge:
    def __init__(self, nodes: tuple[Node, Node], weight: float = 1.0):
        self._nodes = nodes
        self._weight = weight

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self._nodes == other._nodes

    def inverted(self):
        return Edge((self._nodes[1], self._nodes[0]), self._weight)

    @property
    def nodes(self):
        return self._nodes

    @property
    def weight(self):
        return self._weight


class Graph:
    def __init__(self, nodes: list[Node], edges: list[Edge]):
        self._nodes = nodes
        self._edges = edges

    @property
    def nodes(self):
        return self._nodes

    def get_connected_nodes(self, node_from: any) -> set[Node]:
        """
        Get list of nodes which are reachable from given node

        @param node_from: node from graph
        @return: set of nodes which are reachable and adjacent from given node
        """
        result = set()
        for edge in self._edges:
            nodes = edge.nodes
            if nodes[0] == node_from:
                result.add(nodes[1])
        return result

    def order(self, node: any) -> int:
        """
        Get position of node in nodes list inside graph

        @param node: node form graph
        @return: index of node
        """
        for i in range(0, len(self._nodes)):
            if self._nodes[i] == node:
                return i
        raise ValueError("Graph does not contain given node")

    def as_adjacency_matrix(
        self, matrix_type=types.BOOL, zero_diag: bool = False
    ) -> Matrix:
        adj_matrix = Matrix.sparse(matrix_type, len(self.nodes), len(self.nodes))
        for edge in self._edges:
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
