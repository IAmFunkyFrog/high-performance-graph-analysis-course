import math

import pytest

from project import Graph
from project.graph import convert_to_weighted_graph
from project.shortest_path import bellman_ford_multi_source, floyd_warshall

testdata = [
    ("Linear graph", convert_to_weighted_graph([0, 1, 2], [(0, 1.0, 1), (1, 1.0, 2)]), [0], [(0, [0.0, 1.0, 2.0])]),
    (
        "Graph with multiple edges from one node",
        convert_to_weighted_graph([0, 1, 2, 3], [(0, 0.5, 1), (1, 0.5, 2), (0, 0.5, 3)]),
        [0],
        [(0, [0.0, 0.5, 1.0, 0.5])],
    ),
    (
        "Graph with cycle",
        convert_to_weighted_graph(
            [0, 1, 2],
            [
                (0, 1.0, 1),
                (1, 1.0, 2),
                (2, 1.0, 0),
            ],
        ),
        [0, 1],
        [
            (0, [0.0, 1.0, 2.0]),
            (1, [2.0, 0.0, 1.0])
        ],
    ),
    (
        "Start graph",
        convert_to_weighted_graph(
            [0, 1, 2, 3],
            [
                (0, 1.0, 1),
                (0, 1.0, 2),
                (0, 1.0, 3),
            ],
        ),
        [0, 1],
        [
            (0, [0.0, 1.0, 1.0, 1.0]),
            (1, [math.inf, 0.0, math.inf, math.inf])
        ],
    )
]


@pytest.mark.parametrize("name, graph, start_nodes, expected", testdata)
def test_bellman_ford(name: str, graph: Graph, start_nodes: list[int], expected: list[tuple[int, list[int]]]):
    actual = bellman_ford_multi_source(graph, start_nodes)
    assert actual == expected


@pytest.mark.parametrize("name, graph, start_nodes, expected", testdata)
def test_floyd_warshall(name: str, graph: Graph, start_nodes: list[int], expected: list[tuple[int, list[int]]]):
    actual = floyd_warshall(graph)
    for (node, actual_answer) in expected:
        expected_answer = expected[node]
        assert actual_answer == expected_answer[1]
