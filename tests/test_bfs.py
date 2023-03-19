import pytest

from project import Graph, convert_to_graph, bfs

testdata = [
    ("Linear graph", convert_to_graph([0, 1, 2], [(0, 1), (1, 2)]), 0, [0, 1, 2]),
    (
        "Graph with multiple edges from one node",
        convert_to_graph([0, 1, 2, 3], [(0, 1), (1, 2), (0, 3)]),
        0,
        [0, 1, 2, 1],
    ),
    ("Graph without edges", convert_to_graph([0, 1, 2], []), 0, [0, -1, -1]),
    (
        "Graph with cycle",
        convert_to_graph(
            [0, 1, 2],
            [
                (0, 1),
                (1, 2),
                (2, 0),
            ],
        ),
        0,
        [0, 1, 2],
    ),
    ("Empty graph", convert_to_graph([], []), None, []),
    (
        "Start node is None",
        convert_to_graph(
            [0, 1, 2],
            [
                (0, 1),
                (1, 2),
                (2, 0),
            ],
        ),
        None,
        [-1, -1, -1],
    ),
]


@pytest.mark.parametrize("name, graph, start_node, expected", testdata)
def test_bfs(name: str, graph: Graph, start_node: int, expected: list[int]):
    actual = bfs(graph, start_node)
    assert actual == expected
