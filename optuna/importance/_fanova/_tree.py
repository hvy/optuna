import itertools
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy
import sklearn

from optuna.importance._fanova._stats import _WeightedRunningStats


class _FanovaTree(object):
    def __init__(
        self,
        tree: sklearn.tree._tree.Tree,
        search_spaces: numpy.ndarray,
        search_spaces_is_categorical: List[bool],
        raw_features_to_features: List[int],
    ) -> None:
        assert len(search_spaces) == len(search_spaces_is_categorical)
        assert len(search_spaces) - 1 == max(raw_features_to_features)

        self._tree = tree
        self._search_spaces = search_spaces
        self._search_spaces_is_categorical = search_spaces_is_categorical
        self._raw_features_to_features = raw_features_to_features

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
        self, features: Tuple[int, ...]
    ) -> _WeightedRunningStats:
        # For each midpoint along the given dimensions, traverse this tree to compute the
        # marginal predictions.
        midpoints = [self._split_midpoints[feature] for feature in features]
        sizes = [self._split_sizes[feature] for feature in features]

        # TODO(hvy): Keep objects `numpy.ndarray`s.
        prod_midpoints = itertools.product(*midpoints)
        prod_sizes = itertools.product(*sizes)

        stat = _WeightedRunningStats()

        # For each midpoint along the given dimensions, traverse this tree to compute the
        # marginal predictions.
        sample = numpy.full(self._n_features, numpy.nan, dtype=numpy.float64)
        for i, (midpoints, sizes) in enumerate(zip(prod_midpoints, prod_sizes)):
            sample[list(features)] = numpy.array(midpoints, dtype=numpy.float64)
            marginal_prediction_stat = self._marginalized_prediction_stat(sample)
            mean = marginal_prediction_stat.mean()
            weight = numpy.prod(numpy.array(sizes) * marginal_prediction_stat.sum_of_weights())
            assert not numpy.isnan(mean)
            stat.push(mean, weight)

        return stat

    def _precompute_split_midpoints_and_sizes(
        self,
    ) -> Tuple[List[numpy.ndarray], List[numpy.ndarray]]:
        midpoints = []
        sizes = []

        search_spaces = self._search_spaces
        for feature, feature_split_values in enumerate(self._compute_features_split_values()):
            if self._is_categorical(feature):
                if feature_split_values.size == 0:
                    # This tree does not split on this feature.
                    # The feature will therefore have 0 importance in this tree.
                    midpoint = numpy.zeros(1)
                    size = numpy.atleast_1d(search_spaces[feature, 1])  # Number of choices.
                else:
                    midpoint = feature_split_values
                    size = numpy.ones(feature_split_values.size)
            else:
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

    def _marginalized_prediction_stat(
        self, feature_vector: numpy.ndarray
    ) -> _WeightedRunningStats:
        assert len(feature_vector) == self._n_features

        search_spaces = numpy.copy(self._search_spaces)

        marginalized_features = numpy.isnan(feature_vector)
        # Alter search space cardinalities of marginalized features to be 1.
        # Both numerical and categorical features require the same modifications.
        search_spaces[marginalized_features] = numpy.array([0, 1], dtype=search_spaces.dtype)
        active_features = set(numpy.argwhere(~marginalized_features).ravel())

        # Start from the root and traverse towards the leafs.
        active_nodes = [0]
        active_search_spaces = [search_spaces]

        stat = _WeightedRunningStats()

        while len(active_nodes) > 0:
            node_index = active_nodes.pop()
            search_spaces = active_search_spaces.pop()

            raw_feature = self._get_node_split_raw_feature(node_index)
            if raw_feature is not None:  # Not leaf.
                feature = self._get_feature(raw_feature)

                # If node splits on an active feature, push the child node to which we end up in.
                response = feature_vector[feature]
                if not numpy.isnan(response):
                    left_subspaces, right_subspaces = self._compute_child_subspaces(
                        node_index, search_spaces
                    )

                    if response <= self._get_node_split_threshold(node_index):
                        next_node_index = self._get_node_left_child(node_index)
                        next_subspace = left_subspaces
                    else:
                        next_node_index = self._get_node_right_child(node_index)
                        next_subspace = right_subspaces

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

            # Here, the node is either a leaf, or the subtree does not split on any active
            # feature. Record statistics.
            size_correction = _get_subspace_cardinality(search_spaces)
            stat += self._marginal_prediction_stats[node_index].multiply_weights_by(
                1.0 / size_correction
            )

        return stat

    @property
    def _n_features(self) -> int:
        return len(self._search_spaces)

    @property
    def _n_nodes(self) -> int:
        return self._tree.node_count

    def _is_node_leaf(self, node_index: int) -> bool:
        return self._get_node_split_raw_feature(node_index) is None

    def _get_node_left_child(self, node_index: int) -> int:
        return self._tree.children_left[node_index]

    def _get_node_right_child(self, node_index: int) -> int:
        return self._tree.children_right[node_index]

    def _get_node_value(self, node_index: int) -> float:
        return self._tree.value[node_index]

    def _get_node_split_threshold(self, node_index: int) -> float:
        return self._tree.threshold[node_index]

    def _get_node_split_raw_feature(self, node_index: int) -> Optional[int]:
        raw_feature = self._tree.feature[node_index]
        return None if raw_feature < 0 else raw_feature

    def _get_node_children(self, node_index: int) -> Tuple[int, int]:
        return self._get_node_left_child(node_index), self._get_node_right_child(node_index)

    def _is_categorical(self, feature: int) -> bool:
        return self._search_spaces_is_categorical[feature]

    def _get_feature(self, raw_feature: int) -> int:
        # Convert one-hot encoded feature index to original feature index.
        return self._raw_features_to_features[raw_feature]

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
                subspace_cardinality = _get_subspace_cardinality(subspace)
                marginal_prediction_stats[node_index].push(value, subspace_cardinality)
            else:
                for child_node_index, child_subspace in zip(
                    self._get_node_children(node_index),
                    self._compute_child_subspaces(node_index, subspace),
                ):
                    assert subspaces[child_node_index] is None
                    subspaces[child_node_index] = child_subspace

        # Compute marginals for internal nodes.
        for node_index in reversed(range(n_nodes)):
            if self._is_node_leaf(node_index):
                continue

            for child_node_index in self._get_node_children(node_index):
                marginal_prediction_stats[node_index] += marginal_prediction_stats[
                    child_node_index
                ]

        return marginal_prediction_stats

    def _precompute_subtree_active_features(self) -> List[Set[int]]:
        # Precompute for each node, a set of active features in the subtree starting from that
        # node.
        n_nodes = self._n_nodes

        subtree_active_features = [set() for _ in range(n_nodes)]  # type: List[Set[int]]

        for node_index in reversed(range(n_nodes)):
            if self._is_node_leaf(node_index):
                continue

            raw_feature = self._get_node_split_raw_feature(node_index)
            assert raw_feature is not None
            feature = self._get_feature(raw_feature)

            subtree_active_features[node_index].add(feature)

            for child_node_index in self._get_node_children(node_index):
                subtree_active_features[node_index].update(
                    subtree_active_features[child_node_index]
                )

        return subtree_active_features

    def _compute_features_split_values(self) -> List[numpy.ndarray]:
        all_split_values = [set() for _ in range(self._n_features)]  # type: List[Set[float]]

        for node_index in range(self._n_nodes):
            if self._is_node_leaf(node_index):
                continue

            raw_feature = self._get_node_split_raw_feature(node_index)
            assert raw_feature is not None
            feature = self._get_feature(raw_feature)

            if self._is_categorical(feature):
                if len(all_split_values[feature]) == 0:
                    n_choices = int(self._search_spaces[feature, 1])
                    all_split_values[feature].update(range(n_choices))
            else:
                threshold = self._get_node_split_threshold(node_index)
                all_split_values[feature].add(threshold)

        for i in range(len(all_split_values)):
            split_values = all_split_values[i]
            split_values = numpy.array(list(split_values), dtype=numpy.float64)
            # TODO(hvy): Do not shadow variable.
            assert isinstance(
                split_values, numpy.ndarray
            )  # Required for mypy since variable name is reused.
            split_values.sort()
            all_split_values[i] = split_values

        return all_split_values

    def _compute_child_subspaces(
        self, node_index: int, subspace: numpy.ndarray
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        left_subspace = numpy.copy(subspace)
        right_subspace = numpy.copy(subspace)

        raw_feature = self._get_node_split_raw_feature(node_index)
        assert raw_feature is not None
        feature = self._get_feature(raw_feature)

        if self._is_categorical(feature):
            n_choices = subspace[feature, 1]
            n_choices -= 1
            left_subspace[feature, 1] = n_choices
            right_subspace[feature, 1] = n_choices
        else:
            threshold = self._get_node_split_threshold(node_index)
            left_subspace[feature, 1] = threshold
            right_subspace[feature, 0] = threshold

        # Sanity check.
        subspace_cardinality = _get_subspace_cardinality(subspace)
        left_subspace_cardinality = _get_subspace_cardinality(left_subspace)
        right_subspace_cardinality = _get_subspace_cardinality(right_subspace)
        assert left_subspace_cardinality < subspace_cardinality
        assert right_subspace_cardinality < subspace_cardinality

        return left_subspace, right_subspace


def _get_subspace_cardinality(search_spaces: numpy.ndarray) -> float:
    return numpy.prod(search_spaces[:, 1] - search_spaces[:, 0])
