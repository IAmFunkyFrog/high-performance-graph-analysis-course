__all__ = ["convert_to_graph", "Node", "Edge", "Graph"]

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
    def __init__(self, nodes: tuple[Node, Node]):
        self._nodes = nodes

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self._nodes == other._nodes

    @property
    def nodes(self):
        return self._nodes


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

    def as_adjacency_matrix(self) -> Matrix:
        adj_matrix = Matrix.sparse(types.BOOL, len(self.nodes), len(self.nodes))
        for node in self.nodes:
            connected_nodes = self.get_connected_nodes(node)
            for connected in connected_nodes:
                adj_matrix[self.order(node), self.order(connected)] = True
        return adj_matrix


def convert_to_graph(nodes: list[any], edges: list[tuple[any, any]]) -> Graph:
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

    return Graph(boxed_nodes, boxed_edges)
