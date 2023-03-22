import pytest

from project import Graph, convert_to_graph, bfs_multi_source_parents

testdata = [
    ("Linear graph", convert_to_graph([0, 1, 2], [(0, 1), (1, 2)]), [0], [(0, [-1, 0, 1])]),
    (
        "Graph with multiple edges from one node",
        convert_to_graph([0, 1, 2, 3], [(0, 1), (1, 2), (0, 3)]),
        [0],
        [(0, [-1, 0, 1, 0])],
    ),
    ("Graph without edges", convert_to_graph([0, 1, 2], []), [0], [(0, [-1, -2, -2])]),
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
        [0],
        [(0, [-1, 0, 1])],
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
        [],
        [],
    ),
    (
        "BFS with multiple start nodes",
        convert_to_graph([0, 1, 2, 3], [(0, 1), (2, 3)]),
        [0, 2],
        [(0, [-1, 0, -2, -2]), (2, [-2, -2, -1, 2])],
    ),
]


@pytest.mark.parametrize("name, graph, start_nodes, expected", testdata)
def test_bfs_multi_source_parents(name: str, graph: Graph, start_nodes: list[int], expected: list[int]):
    actual = bfs_multi_source_parents(graph, start_nodes)
    assert actual == expected
