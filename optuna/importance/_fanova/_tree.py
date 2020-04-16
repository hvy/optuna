import itertools
from typing import List
from typing import Set
from typing import Tuple

import numpy

from optuna.importance._fanova._stats import _WeightedRunningStats
from optuna import type_checking

if type_checking.TYPE_CHECKING:
    import sklearn.tree


class _FanovaTree(object):
    def __init__(self, tree: "sklearn.tree._tree.Tree", search_spaces: numpy.ndarray) -> None:
        assert search_spaces.shape[0] == tree.n_features
        assert search_spaces.shape[1] == 2

        self._tree = tree
        self._search_spaces = search_spaces

        # To compute marginalized predictions, we will need data points for each leaf partition.
        # Precompute them once when constructing the tree since they will not change.
        self._split_midpoints, self._split_sizes = self._precompute_split_midpoints_and_sizes()

        self._marginal_prediction_stats = self._precompute_marginal_prediction_stats()
        self._variance = self._marginal_prediction_stats[0].variance_population()

        self._subtree_active_features = self._precompute_subtree_active_features()

    @property
    def variance(self) -> float:
        return self._variance

    def marginalized_prediction_stat_for_features(
        self, features: numpy.ndarray
    ) -> _WeightedRunningStats:
        midpoints = [self._split_midpoints[f] for f in features]
        sizes = [self._split_sizes[f] for f in features]

        stat = _WeightedRunningStats()

        # For each midpoint along the given dimensions, traverse this tree to compute the
        # marginal predictions.
        prod_midpoints = itertools.product(*midpoints)
        prod_sizes = itertools.product(*sizes)

        sample = numpy.full(self._n_features, numpy.nan, dtype=numpy.float64)

        for midpoints, sizes in zip(prod_midpoints, prod_sizes):
            sample[features] = numpy.array(midpoints, dtype=numpy.float64)

            marginal_prediction_stat = self._marginalized_prediction_stat(sample)

            mean = marginal_prediction_stat.mean()
            assert not numpy.isnan(mean)

            weight = numpy.prod(numpy.array(sizes) * marginal_prediction_stat.sum_of_weights())
            stat.push(mean, weight)

        return stat

    def _marginalized_prediction_stat(
        self, feature_vector: numpy.ndarray
    ) -> _WeightedRunningStats:
        assert feature_vector.size == self._n_features

        marginalized_features = numpy.isnan(feature_vector)
        active_features = set(numpy.argwhere(~marginalized_features).ravel())

        # Reduce search space cardinalities to 1 for non-active features.
        search_spaces = self._search_spaces.copy()
        search_spaces[marginalized_features] = [0.0, 1.0]

        # Start from the root and traverse towards the leafs.
        active_nodes = [0]
        active_search_spaces = [search_spaces]

        stat = _WeightedRunningStats()

        while len(active_nodes) > 0:
            node_index = active_nodes.pop()
            search_spaces = active_search_spaces.pop()

            if not self._is_node_leaf(node_index):
                feature = self._get_node_split_feature(node_index)

                # If node splits on an active feature, push the child node that we end up in.
                response = feature_vector[feature]
                if not numpy.isnan(response):
                    if response <= self._get_node_split_threshold(node_index):
                        next_node_index = self._get_node_left_child(node_index)
                        next_subspace = self._get_node_left_child_subspaces(
                            node_index, search_spaces
                        )
                    else:
                        next_node_index = self._get_node_right_child(node_index)
                        next_subspace = self._get_node_right_child_subspaces(
                            node_index, search_spaces
                        )

                    active_nodes.append(next_node_index)
                    active_search_spaces.append(next_subspace)
                    continue

                # If subtree starting from node splits on an active feature, push both child nodes.
                # if active_feature in self._subtree_active_features[node_index]:
                if (
                    len(active_features.intersection(self._subtree_active_features[node_index]))
                    > 0
                ):
                    for child_node_index in self._get_node_children(node_index):
                        active_nodes.append(child_node_index)
                        active_search_spaces.append(search_spaces)
                    continue

            # Here, the node is either a leaf, or the subtree does not split on any active feature
            # so statistics are collected.
            correction = 1.0 / _get_cardinality(search_spaces)
            stat += self._marginal_prediction_stats[node_index].multiply_weights_by(correction)

        return stat

    def _precompute_split_midpoints_and_sizes(
        self,
    ) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
        midpoints = []
        sizes = []

        search_spaces = self._search_spaces
        for feature, feature_split_values in enumerate(self._compute_features_split_values()):
            feature_split_values = numpy.concatenate(
                (
                    numpy.atleast_1d(search_spaces[feature, 0]),
                    feature_split_values,
                    numpy.atleast_1d(search_spaces[feature, 1]),
                )
            )
            midpoint = 0.5 * (feature_split_values[1:] + feature_split_values[:-1])
            size = feature_split_values[1:] - feature_split_values[:-1]

            assert midpoint.dtype == numpy.float64
            assert size.dtype == numpy.float64

            midpoints.append(midpoint)
            sizes.append(size)

        return midpoints, sizes

    def _compute_features_split_values(self) -> List[numpy.ndarray]:
        all_split_values = [set() for _ in range(self._n_features)]  # type: List[Set[float]]

        for node_index in range(self._n_nodes):
            if not self._is_node_leaf(node_index):
                feature = self._get_node_split_feature(node_index)
                threshold = self._get_node_split_threshold(node_index)
                all_split_values[feature].add(threshold)

        sorted_all_split_values = []  # type: List[numpy.ndarray]

        for split_values in all_split_values:
            split_values_array = numpy.array(list(split_values), dtype=numpy.float64)
            split_values_array.sort()
            sorted_all_split_values.append(split_values_array)

        return sorted_all_split_values

    def _precompute_marginal_prediction_stats(self) -> List[_WeightedRunningStats]:
        n_nodes = self._n_nodes

        marginal_prediction_stats = [_WeightedRunningStats() for _ in range(n_nodes)]
        subspaces = [None for _ in range(n_nodes)]
        subspaces[0] = self._search_spaces

        # Compute marginals for leaf nodes.
        for node_index in range(n_nodes):
            subspace = subspaces[node_index]

            if self._is_node_leaf(node_index):
                value = self._get_node_value(node_index)
                subspace_cardinality = _get_cardinality(subspace)
                marginal_prediction_stats[node_index].push(value, subspace_cardinality)
            else:
                for child_node_index, child_subspace in zip(
                    self._get_node_children(node_index),
                    self._get_node_children_subspaces(node_index, subspace),
                ):
                    assert subspaces[child_node_index] is None
                    subspaces[child_node_index] = child_subspace

        # Compute marginals for internal nodes.
        for node_index in reversed(range(n_nodes)):
            if not self._is_node_leaf(node_index):
                for child_node_index in self._get_node_children(node_index):
                    marginal_prediction_stats[node_index] += marginal_prediction_stats[
                        child_node_index
                    ]

        return marginal_prediction_stats

    def _precompute_subtree_active_features(self) -> List[Set[int]]:
        # Compute for each node, a set of active features in the subtree starting from that node.
        subtree_active_features = [set() for _ in range(self._n_nodes)]  # type: List[Set[int]]

        for node_index in reversed(range(self._n_nodes)):
            if not self._is_node_leaf(node_index):
                feature = self._get_node_split_feature(node_index)
                subtree_active_features[node_index].add(feature)
                for child_node_index in self._get_node_children(node_index):
                    subtree_active_features[node_index].update(
                        subtree_active_features[child_node_index]
                    )

        return subtree_active_features

    @property
    def _n_features(self) -> int:
        return len(self._search_spaces)

    @property
    def _n_nodes(self) -> int:
        return self._tree.node_count

    def _is_node_leaf(self, node_index: int) -> bool:
        return self._tree.feature[node_index] < 0

    def _get_node_left_child(self, node_index: int) -> int:
        return self._tree.children_left[node_index]

    def _get_node_right_child(self, node_index: int) -> int:
        return self._tree.children_right[node_index]

    def _get_node_children(self, node_index: int) -> Tuple[int, int]:
        return self._get_node_left_child(node_index), self._get_node_right_child(node_index)

    def _get_node_value(self, node_index: int) -> float:
        return self._tree.value[node_index]

    def _get_node_split_threshold(self, node_index: int) -> float:
        return self._tree.threshold[node_index]

    def _get_node_split_feature(self, node_index: int) -> int:
        feature = self._tree.feature[node_index]
        assert feature >= 0  # This method is never called with a leaf `node_index`.
        return feature

    def _get_node_left_child_subspaces(
        self, node_index: int, search_spaces: numpy.ndarray
    ) -> numpy.ndarray:
        return _get_subspaces(
            search_spaces,
            search_spaces_column=1,
            feature=self._get_node_split_feature(node_index),
            threshold=self._get_node_split_threshold(node_index),
        )

    def _get_node_right_child_subspaces(
        self, node_index: int, search_spaces: numpy.ndarray
    ) -> numpy.ndarray:
        return _get_subspaces(
            search_spaces,
            search_spaces_column=0,
            feature=self._get_node_split_feature(node_index),
            threshold=self._get_node_split_threshold(node_index),
        )

    def _get_node_children_subspaces(
        self, node_index: int, search_spaces: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        return (
            self._get_node_left_child_subspaces(node_index, search_spaces),
            self._get_node_right_child_subspaces(node_index, search_spaces),
        )


def _get_cardinality(search_spaces: numpy.ndarray) -> float:
    return numpy.prod(search_spaces[:, 1] - search_spaces[:, 0])


def _get_subspaces(
    search_spaces: numpy.ndarray, *, search_spaces_column: int, feature: int, threshold: float
) -> numpy.ndarray:
    search_spaces_subspace = numpy.copy(search_spaces)
    search_spaces_subspace[feature, search_spaces_column] = threshold

    # Sanity check.
    assert _get_cardinality(search_spaces_subspace) < _get_cardinality(search_spaces)

    return search_spaces_subspace
