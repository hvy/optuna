import itertools
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy

from optuna.importance._fanova._tree import _FanovaTree

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import OneHotEncoder

    _available = True
except ImportError as e:
    _import_error = e
    _available = False


class _Fanova(object):
    # Implements the fANOVA algorithm in https://github.com/automl/fanova using scikit-learn.

    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: Union[int, float],
        min_samples_leaf: Union[int, float],
        random_state: Optional[int],
    ) -> None:
        _check_sklearn_availability()

        self._forest = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._trees = None  # type: Optional[List[_FanovaTree]]
        self._v_u_total = None  # type: Optional[Dict[Tuple[int, ...], numpy.ndarray]]
        self._v_u_individual = None  # type: Optional[Dict[Tuple[int, ...], numpy.ndarray]]
        self._features_to_raw_features = None  # type: Optional[List[numpy.ndarray]]

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        search_spaces: numpy.ndarray,
        search_spaces_is_categorical: List[bool],
    ) -> None:
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == search_spaces.shape[0]
        assert X.shape[1] == len(search_spaces_is_categorical)
        assert search_spaces.shape[1] == 2

        encoder = _CategoricalFeaturesOneHotEncoder()
        X, search_spaces = encoder.fit_transform(X, search_spaces, search_spaces_is_categorical)

        self._forest.fit(X, y)

        self._trees = [_FanovaTree(e.tree_, search_spaces) for e in self._forest.estimators_]
        self._v_u_total = {}
        self._v_u_individual = {}
        self._features_to_raw_features = encoder.features_to_raw_features

    def quantify_importance(
        self, features: Tuple[int, ...]
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        assert self._trees is not None
        assert self._v_u_total is not None
        assert self._v_u_individual is not None

        if all(tree.variance == 0 for tree in self._trees):
            # If all trees have 0 variance, we cannot assess any importances.
            # This could occur if for instance `X.shape[0] == 1`.
            raise RuntimeError("Encountered zero total variance in all trees.")

        self._compute_marginals(features)

        importance_dict = {}  # type: Dict[Tuple[int, ...], Dict[str, float]]

        for k in range(1, len(features) + 1):
            for sub_features in itertools.combinations(features, k):
                importance_dict[sub_features] = {}

                fractions_total = []
                fractions_individual = []
                for tree_index, tree in enumerate(self._trees):
                    tree_variance = tree.variance
                    if tree_variance == 0:
                        continue
                    fractions_individual.append(
                        self._v_u_individual[sub_features][tree_index] / tree_variance
                    )
                    fractions_total.append(
                        self._v_u_total[sub_features][tree_index] / tree_variance
                    )
                fractions_individual = numpy.array(fractions_individual, dtype=numpy.float64)
                fractions_total = numpy.array(fractions_total, dtype=numpy.float64)

                importance_dict[sub_features]["individual importance"] = numpy.mean(
                    fractions_individual
                )
                importance_dict[sub_features]["total importance"] = numpy.mean(fractions_total)
                importance_dict[sub_features]["individual std"] = numpy.std(fractions_individual)
                importance_dict[sub_features]["total std"] = numpy.std(fractions_total)

        return importance_dict

    def _compute_marginals(self, features: Tuple[int, ...]) -> None:
        assert self._trees is not None
        assert self._v_u_total is not None
        assert self._v_u_individual is not None
        assert self._features_to_raw_features is not None

        if features in self._v_u_individual:
            return

        for k in range(1, len(features)):
            for sub_features in itertools.combinations(features, k):
                if sub_features not in self._v_u_total:
                    self._compute_marginals(sub_features)

        n_trees = len(self._trees)
        self._v_u_individual[features] = numpy.empty(n_trees, dtype=numpy.float64)
        self._v_u_total[features] = numpy.empty(n_trees, dtype=numpy.float64)

        raw_features = numpy.concatenate([self._features_to_raw_features[f] for f in features])

        for tree_index, tree in enumerate(self._trees):
            stat = tree.marginalized_prediction_stat_for_features(raw_features)

            if stat.sum_of_weights() > 0:
                variance_population = stat.variance_population()
                v_u_total = variance_population
                v_u_individual = variance_population

                for k in range(1, len(features)):
                    for sub_features in itertools.combinations(features, k):
                        v_u_individual -= self._v_u_individual[sub_features][tree_index]
                v_u_individual = numpy.clip(v_u_individual, 0, numpy.inf)
            else:
                v_u_total = numpy.nan
                v_u_individual = numpy.nan

            self._v_u_individual[features][tree_index] = v_u_individual
            self._v_u_total[features][tree_index] = v_u_total


class _CategoricalFeaturesOneHotEncoder(object):
    def __init__(self) -> None:
        # `features_to_raw_features["column index in original matrix"]
        #     == "numpy.ndarray with corresponding columns in the transformed matrix"`
        self.features_to_raw_features = None  # type: Optional[List[numpy.ndarray]]

    def fit_transform(
        self,
        X: numpy.ndarray,
        search_spaces: numpy.ndarray,
        search_spaces_is_categorical: List[bool],
    ) -> Tuple[numpy.ndarray, numpy.ndarray]:
        # Transform the `X` matrix by expanding categorical integer-valued columns to one-hot
        # encoding matrices and search spaces `search_spaces` similarly.
        # Note that the resulting matrices are sparse and potentially very big.

        n_features = X.shape[1]
        assert n_features == len(search_spaces)
        assert n_features == len(search_spaces_is_categorical)

        categories = []
        categorical_features = []
        categorical_features_n_uniques = {}
        numerical_features = []

        for feature, is_categorical in enumerate(search_spaces_is_categorical):
            if is_categorical:
                n_unique = search_spaces[feature][1].astype(numpy.int32)
                categories.append(numpy.arange(n_unique))
                categorical_features.append(feature)
                categorical_features_n_uniques[feature] = n_unique
            else:
                numerical_features.append(feature)

        transformer = ColumnTransformer(
            [
                (
                    "_categorical",
                    OneHotEncoder(categories=categories, sparse=False),
                    categorical_features,
                )
            ],
            remainder="passthrough",
        )

        # All categorical one-hot features will be placed before the numerical features in
        # `ColumnTransformer.fit_transform`.
        X = transformer.fit_transform(X)

        features_to_raw_features = [None for _ in range(n_features)]  # type: List[numpy.ndarray]
        i = 0
        if len(categorical_features) > 0:
            categories = transformer.transformers_[0][1].categories_
            assert len(categories) == len(categorical_features)

            for j, (feature, category) in enumerate(zip(categorical_features, categories)):
                categorical_raw_features = category.astype(numpy.int32)
                if i > 0:
                    # Adjust offset.
                    previous_categorical_feature = categorical_features[j - 1]
                    previous_categorical_raw_features = features_to_raw_features[
                        previous_categorical_feature
                    ]
                    categorical_raw_features += previous_categorical_raw_features[-1] + 1
                assert features_to_raw_features[feature] is None
                features_to_raw_features[feature] = categorical_raw_features
                i = categorical_raw_features[-1] + 1
        for feature in numerical_features:
            features_to_raw_features[feature] = numpy.atleast_1d(i)
            i += 1
        assert i == X.shape[1]

        # `raw_features_to_features["column index in transformed matrix"]
        #     == "column in the original matrix"`
        raw_features_to_features = numpy.empty((X.shape[1],), dtype=numpy.int32)
        i = 0
        for categorical_feature in categorical_features:
            for _ in range(categorical_features_n_uniques[categorical_feature]):
                raw_features_to_features[i] = categorical_feature
                i += 1
        for numerical_col in numerical_features:
            raw_features_to_features[i] = numerical_col
            i += 1
        assert i == X.shape[1]

        # Transform search spaces.
        n_raw_features = raw_features_to_features.size
        raw_search_spaces = numpy.empty((n_raw_features, 2), dtype=numpy.float64)

        for raw_feature, feature in enumerate(raw_features_to_features):
            if search_spaces_is_categorical[feature]:
                raw_search_spaces[raw_feature] = [0.0, 1.0]
            else:
                raw_search_spaces[raw_feature] = search_spaces[feature]

        self.features_to_raw_features = features_to_raw_features

        return X, raw_search_spaces


def _check_sklearn_availability() -> None:
    if not _available:
        raise ImportError(
            "scikit-learn is not available. Please install scikit-learn to "
            "use this feature. scikit-learn can be installed by executing "
            "`$ pip install scikit-learn>=0.19.0`. For further information, "
            "please refer to the installation guide of scikit-learn. (The "
            "actual import error is as follows: " + str(_import_error) + ")"
        )
