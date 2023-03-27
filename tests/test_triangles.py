import pytest

from project import Graph, convert_to_undirected_graph
from project.triangles import triangles_count_for_each_vertex, triangles_count_cohen, triangles_count_sandia

testdata = [
    ("Linear graph", convert_to_undirected_graph([0, 1, 2], [(0, 1), (1, 2)]), [0, 0, 0]),
    (
        "Graph with 1 cycle and 3 nodes",
        convert_to_undirected_graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)]),
        [1, 1, 1],
    ),
    ("Graph without edges", convert_to_undirected_graph([0, 1, 2], []), [0, 0, 0]),
    ("Empty graph", convert_to_undirected_graph([], []), []),
    (
        "Big graph",
        convert_to_undirected_graph(
            [0, 1, 2, 3, 4, 5, 6],
            [
                (0, 1),
                (0, 3),
                (1, 3),
                (1, 4),
                (1, 6),
                (2, 3),
                (2, 5),
                (2, 6),
                (3, 5),
                (3, 6),
                (4, 5),
                (4, 6),
            ]
        ),
        [1, 3, 2, 4, 1, 1, 3]
    )
]


@pytest.mark.parametrize("name, graph, triangles_count", testdata)
def test_triangles_count_for_each_vertex(name: str, graph: Graph, triangles_count: list[int]):
    assert triangles_count_for_each_vertex(graph) == triangles_count


@pytest.mark.parametrize("name, graph, triangles_count", testdata)
def test_triangles_count_cohen(name: str, graph: Graph, triangles_count: list[int]):
    assert triangles_count_cohen(graph) == sum(triangles_count) // 3


@pytest.mark.parametrize("name, graph, triangles_count", testdata)
def test_triangles_count_cohen(name: str, graph: Graph, triangles_count: list[int]):
    assert triangles_count_sandia(graph) == sum(triangles_count) // 3
