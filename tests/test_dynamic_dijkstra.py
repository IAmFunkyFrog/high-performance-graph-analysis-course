import math

import pytest

from project import Graph
from project.graph import convert_to_weighted_graph, Edge, Node
from project.shortest_path import DijkstraDynamic

testdata = [
    ("Linear graph",
     convert_to_weighted_graph([0, 1, 2], [(0, 1.0, 1), (1, 1.0, 2)]),
     0,
     [
         ("remove", (0, 1.0, 1), [0.0, math.inf, math.inf]),
         ("add", (0, 1.0, 1), [0.0, 1.0, 2.0])
     ]
    ),
    ("Linear graph 2",
     convert_to_weighted_graph([0, 1, 2, 3], [(0, 1.0, 1), (1, 1.0, 2), (2, 1.0, 3)]),
     0,
     [
         ("remove", (0, 1.0, 1), [0.0, math.inf, math.inf, math.inf]),
         ("add", (0, 1.0, 1), [0.0, 1.0, 2.0, 3.0]),
         ("add", (0, 1.0, 3), [0.0, 1.0, 2.0, 1.0]),
         ("remove", (0, 1.0, 1), [0.0, math.inf, math.inf, 1.0]),
     ]
    ),
    ("Cycle graph",
     convert_to_weighted_graph([0, 1, 2, 3], [(0, 1.0, 1), (1, 1.0, 2), (2, 1.0, 3), (3, 1.0, 0)]),
     0,
     [
         ("remove", (0, 1.0, 1), [0.0, math.inf, math.inf, math.inf]),
         ("add", (0, 1.0, 1), [0.0, 1.0, 2.0, 3.0]),
         ("add", (0, 1.0, 3), [0.0, 1.0, 2.0, 1.0]),
         ("remove", (0, 1.0, 1), [0.0, math.inf, math.inf, 1.0]),
         ("add", (3, 1.0, 1), [0.0, 2.0, 3.0, 1.0]),
     ]
    ),
]


@pytest.mark.parametrize("name, graph, start_node, expected", testdata)
def test_dynamic_dijkstra(name: str, graph: Graph, start_node: int, expected):
    dynamic_dijksta = DijkstraDynamic(graph, start_node)
    for (op, edge, expected_answer) in expected:
        if op == "remove":
            dynamic_dijksta.remove_edge(Edge((Node(edge[0]), Node(edge[2])), edge[1]))
        elif op == "add":
            dynamic_dijksta.add_edge(Edge((Node(edge[0]), Node(edge[2])), edge[1]))
        dists_with_node = dynamic_dijksta.dists()
        dists = list([dists_with_node[graph.nodes[i]] for i in range(len(graph.nodes))])
        assert dists == expected_answer
