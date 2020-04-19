import math
from typing import List
from typing import Tuple
from unittest.mock import Mock

import numpy
import pytest

from optuna.importance._fanova._stats import _WeightedRunningStats
from optuna.importance._fanova._tree import _FanovaTree


@pytest.fixture
def tree() -> _FanovaTree:
    sklearn_tree = Mock()
    sklearn_tree.n_features = 3
    sklearn_tree.node_count = 5
    sklearn_tree.feature = [1, 2, -1, -1, -1]
    sklearn_tree.children_left = [1, 2, -1, -1, -1]
    sklearn_tree.children_right = [4, 3, -1, -1, -1]
    sklearn_tree.value = [-1.0, -1.0, 0.1, 0.2, 0.5]
    sklearn_tree.threshold = [0.5, 1.5, -1.0, -1.0, -1.0]

    tree = _FanovaTree(
        tree=sklearn_tree,
        search_spaces=numpy.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]], dtype=numpy.float64),
    )

    return tree


@pytest.fixture
def expected_marginal_prediction_stats() -> Tuple[_WeightedRunningStats, ...]:
    stat4 = _WeightedRunningStats()
    stat4.push(0.5, 1.0)
    stat3 = _WeightedRunningStats()
    stat3.push(0.2, 0.25)
    stat2 = _WeightedRunningStats()
    stat2.push(0.1, 0.75)
    stat1 = _WeightedRunningStats()
    stat1 += stat2
    stat1 += stat3
    stat0 = _WeightedRunningStats()
    stat0 += stat1
    stat0 += stat4
    return (stat0, stat1, stat2, stat3, stat4)


def test_tree_variance(
    tree: _FanovaTree, expected_marginal_prediction_stats: Tuple[_WeightedRunningStats]
) -> None:
    assert math.isclose(tree.variance, expected_marginal_prediction_stats[0].variance_population())


@pytest.mark.parametrize(
    "features,expected",
    [
        ([0], [([1.0], [(0, 1.0)]),]),
        ([1], [([0.5], [(1, 0.5)]), ([0.5], [(4, 0.5)])]),
        ([2], [([1.5], [(2, 1.5), (4, 2.0)]), ([0.5], [(3, 0.5), (4, 2.0)])]),
        ([0, 1], [([1.0, 0.5], [(1, 0.5)]), ([1.0, 0.5], [(4, 0.5)]),]),
        ([0, 2], [([1.0, 1.5], [(2, 1.5), (4, 2.0)]), ([1.0, 0.5], [(3, 0.5), (4, 2.0)]),]),
        (
            [1, 2],
            [
                ([0.5, 1.5], [(2, 0.5 * 1.5)]),
                ([0.5, 1.5], [(4, 0.5 * 2.0)]),
                ([0.5, 0.5], [(3, 0.5 * 0.5)]),
                ([0.5, 0.5], [(4, 0.5 * 2.0)]),
            ],
        ),
        (
            [0, 1, 2],
            [
                ([1.0, 0.5, 1.5], [(2, 1.0 * 0.5 * 1.5)]),
                ([1.0, 0.5, 1.5], [(4, 1.0 * 0.5 * 2.0)]),
                ([1.0, 0.5, 0.5], [(3, 1.0 * 0.5 * 0.5)]),
                ([1.0, 0.5, 0.5], [(4, 1.0 * 0.5 * 2.0)]),
            ],
        ),
    ],
)
def test_tree_marginalized_prediction_stat_for_features(
    tree: _FanovaTree,
    expected_marginal_prediction_stats: Tuple[_WeightedRunningStats],
    features: List[int],
    expected: List[Tuple[List[float], List[Tuple[int, float]]]],
) -> None:
    expected_stat = _WeightedRunningStats()
    for sizes, node_indices_and_corrections in expected:
        stat = _WeightedRunningStats()
        for node_index, correction in node_indices_and_corrections:
            stat += expected_marginal_prediction_stats[node_index].multiply_weights_by(
                1.0 / correction
            )
        sizes_array = numpy.array(sizes, numpy.float64)
        expected_stat.push(stat.mean(), numpy.prod(sizes_array * stat.sum_of_weights()))

    stat = tree.marginalized_prediction_stat_for_features(numpy.array(features, dtype=numpy.int32))
    assert math.isclose(stat.mean(), expected_stat.mean())
    assert math.isclose(stat.sum_of_weights(), expected_stat.sum_of_weights())
    assert math.isclose(stat.variance_population(), expected_stat.variance_population())


def test_tree_attrs(tree: _FanovaTree) -> None:
    assert tree._n_features == 3

    assert tree._n_nodes == 5

    assert not tree._is_node_leaf(0)
    assert not tree._is_node_leaf(1)
    assert tree._is_node_leaf(2)
    assert tree._is_node_leaf(3)
    assert tree._is_node_leaf(4)

    assert tree._get_node_left_child(0) == 1
    assert tree._get_node_left_child(1) == 2
    assert tree._get_node_left_child(2) == -1
    assert tree._get_node_left_child(3) == -1
    assert tree._get_node_left_child(4) == -1

    assert tree._get_node_right_child(0) == 4
    assert tree._get_node_right_child(1) == 3
    assert tree._get_node_right_child(2) == -1
    assert tree._get_node_right_child(3) == -1
    assert tree._get_node_right_child(4) == -1

    assert tree._get_node_children(0) == (1, 4)
    assert tree._get_node_children(1) == (2, 3)
    assert tree._get_node_children(2) == (-1, -1)
    assert tree._get_node_children(3) == (-1, -1)
    assert tree._get_node_children(4) == (-1, -1)

    assert tree._get_node_value(0) == -1.0
    assert tree._get_node_value(1) == -1.0
    assert tree._get_node_value(2) == 0.1
    assert tree._get_node_value(3) == 0.2
    assert tree._get_node_value(4) == 0.5

    assert tree._get_node_split_threshold(0) == 0.5
    assert tree._get_node_split_threshold(1) == 1.5
    assert tree._get_node_split_threshold(2) == -1.0
    assert tree._get_node_split_threshold(3) == -1.0
    assert tree._get_node_split_threshold(4) == -1.0

    assert tree._get_node_split_feature(0) == 1
    assert tree._get_node_split_feature(1) == 2


def test_tree_get_node_subspaces(tree: _FanovaTree) -> None:
    search_spaces = numpy.array([[0.0, 1.0], [0.0, 1.0], [0.0, 2.0]], dtype=numpy.float64)
    search_spaces_copy = search_spaces.copy()

    # Test splitting on second feature, first node.
    expected_left_child_subspace = numpy.array(
        [[0.0, 1.0], [0.0, 0.5], [0.0, 2.0]], dtype=numpy.float64
    )
    expected_right_child_subspace = numpy.array(
        [[0.0, 1.0], [0.5, 1.0], [0.0, 2.0]], dtype=numpy.float64
    )
    numpy.testing.assert_array_equal(
        tree._get_node_left_child_subspaces(0, search_spaces), expected_left_child_subspace,
    )
    numpy.testing.assert_array_equal(
        tree._get_node_right_child_subspaces(0, search_spaces), expected_right_child_subspace,
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(0, search_spaces)[0], expected_left_child_subspace,
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(0, search_spaces)[1], expected_right_child_subspace,
    )
    numpy.testing.assert_array_equal(search_spaces, search_spaces_copy)

    # Test splitting on third feature, second node.
    expected_left_child_subspace = numpy.array(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.5]], dtype=numpy.float64
    )
    expected_right_child_subspace = numpy.array(
        [[0.0, 1.0], [0.0, 1.0], [1.5, 2.0]], dtype=numpy.float64
    )
    numpy.testing.assert_array_equal(
        tree._get_node_left_child_subspaces(1, search_spaces), expected_left_child_subspace,
    )
    numpy.testing.assert_array_equal(
        tree._get_node_right_child_subspaces(1, search_spaces), expected_right_child_subspace,
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(1, search_spaces)[0], expected_left_child_subspace,
    )
    numpy.testing.assert_array_equal(
        tree._get_node_children_subspaces(1, search_spaces)[1], expected_right_child_subspace,
    )
    numpy.testing.assert_array_equal(search_spaces, search_spaces_copy)


def test_tree_split_midpoints_and_sizes(tree: _FanovaTree) -> None:
    split_midpoints = tree._split_midpoints
    numpy.testing.assert_equal(split_midpoints[0], numpy.array([0.5], dtype=numpy.float64))
    numpy.testing.assert_equal(split_midpoints[1], numpy.array([0.25, 0.75], dtype=numpy.float64))
    numpy.testing.assert_equal(split_midpoints[2], numpy.array([0.75, 1.75], dtype=numpy.float64))

    split_sizes = tree._split_sizes
    numpy.testing.assert_equal(split_sizes[0], numpy.array([1.0], dtype=numpy.float64))
    numpy.testing.assert_equal(split_sizes[1], numpy.array([0.5, 0.5], dtype=numpy.float64))
    numpy.testing.assert_equal(split_sizes[2], numpy.array([1.5, 0.5], dtype=numpy.float64))


def test_tree_marginal_prediction_stat(
    tree: _FanovaTree, expected_marginal_prediction_stats: Tuple[_WeightedRunningStats, ...]
) -> None:
    marginal_prediction_stats = tree._marginal_prediction_stats
    for stat, expected_stat in zip(marginal_prediction_stats, expected_marginal_prediction_stats):
        assert math.isclose(stat.mean(), expected_stat.mean())
        assert math.isclose(stat.sum_of_weights(), expected_stat.sum_of_weights())
        assert math.isclose(stat.variance_population(), expected_stat.variance_population())


def test_tree_subtree_active_features(tree: _FanovaTree) -> None:
    subtree_active_features = tree._subtree_active_features
    subtree_active_features[0] == set([0, 1])
    subtree_active_features[1] == set([1])
    subtree_active_features[2] == set()
    subtree_active_features[3] == set()
    subtree_active_features[4] == set()


@pytest.mark.parametrize(
    "feature_vector,expected",
    [
        ([0.5, float("nan"), float("nan")], [(0, 1.0)]),
        ([float("nan"), 0.25, float("nan")], [(1, 0.5)]),
        ([float("nan"), 0.75, float("nan")], [(4, 0.5)]),
        ([float("nan"), float("nan"), 0.75], [(2, 1.5), (4, 2.0)]),
        ([float("nan"), float("nan"), 1.75], [(3, 0.5), (4, 2.0)]),
        ([0.5, 0.25, float("nan")], [(1, 1.0 * 0.5)]),
        ([0.5, 0.75, float("nan")], [(4, 1.0 * 0.5)]),
        ([0.5, float("nan"), 0.75], [(2, 1.0 * 1.5), (4, 1.0 * 2.0)]),
        ([0.5, float("nan"), 1.75], [(3, 1.0 * 0.5), (4, 1.0 * 2.0)]),
        ([float("nan"), 0.25, 0.75], [(2, 0.5 * 1.5)]),
        ([float("nan"), 0.25, 1.75], [(3, 0.5 * 0.5)]),
        ([float("nan"), 0.75, 0.75], [(4, 0.5 * 2.0)]),
        ([float("nan"), 0.75, 1.75], [(4, 0.5 * 2.0)]),
        ([0.5, 0.25, 0.75], [(2, 1.0 * 0.5 * 1.5)]),
        ([0.5, 0.25, 1.75], [(3, 1.0 * 0.5 * 0.5)]),
        ([0.5, 0.75, 0.75], [(4, 1.0 * 0.5 * 2.0)]),
        ([0.5, 0.75, 1.75], [(4, 1.0 * 0.5 * 2.0)]),
    ],
)
def test_tree_marginalized_prediction_stat(
    tree: _FanovaTree,
    expected_marginal_prediction_stats: Tuple[_WeightedRunningStats, ...],
    feature_vector: List[float],
    expected: List[Tuple[int, float]],
) -> None:
    expected_stat = _WeightedRunningStats()
    for node_index, correction in expected:
        expected_stat += expected_marginal_prediction_stats[node_index].multiply_weights_by(
            1.0 / correction
        )

    feature_vector_array = numpy.array(feature_vector, dtype=numpy.float64)
    stat = tree._marginalized_prediction_stat(feature_vector_array)
    assert math.isclose(stat.mean(), expected_stat.mean())
    assert math.isclose(stat.sum_of_weights(), expected_stat.sum_of_weights())
    assert math.isclose(stat.variance_population(), expected_stat.variance_population())
