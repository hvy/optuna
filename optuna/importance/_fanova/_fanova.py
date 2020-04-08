import itertools
from typing import Dict
from typing import List
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
    def __init__(
        self,
        n_estimators: int,
        max_depth: int,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        random_state: int = None,
    ) -> None:
        _check_sklearn_availability()

        self._forest = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        self._trees = None
        self._V_U_total = None
        self._V_U_individual = None

    def fit(
        self,
        X: numpy.ndarray,
        y: numpy.ndarray,
        search_spaces: numpy.ndarray,
        search_spaces_is_categorical: List[bool],
    ) -> None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "Parameter data length does not match value data length. {} != {}.".format(
                    X.shape[0], y.shape[0]
                )
            )
        if X.shape[1] != search_spaces.shape[0]:
            raise ValueError(
                "Parameter data width does not match the search space length. "
                "{} != {}.".format(X.shape[1], search_spaces.shape[0])
            )
        if search_spaces.shape[1] != 2:
            raise ValueError(
                "Each search space must be represented by a lower and upper bound. Categorical "
                "must be represented with a 0 followed by the number of choices. "
                "{} != 2 (expected).".format(search_spaces.shape[1])
            )
        if len(search_spaces_is_categorical) != search_spaces.shape[0]:
            raise ValueError(
                "Number of categorical flags does not match the search space length. "
                "{} != {}.".format(len(search_spaces_is_categorical), search_spaces.shape[0])
            )

        encoder = _CategoricalFeaturesOneHotEncoder(search_spaces_is_categorical)
        X = encoder.fit_transform(X)
        self._forest.fit(X, y)

        trees = [
            _FanovaTree(
                estimator.tree_,
                search_spaces,
                search_spaces_is_categorical,
                encoder.raw_features_to_features,
            )
            for estimator in self._forest.estimators_
        ]

        # If all trees have 0 variance, we cannot assess any importances.
        # This could occur if for instance `X.shape[0] == 1`.
        if all(tree.variance == 0 for tree in trees):
            raise RuntimeError("Encountered zero total variance in all trees.")

        self._trees = trees
        self._V_U_total = {}
        self._V_U_individual = {}

    def quantify_importance(
        self, features: Tuple[int, ...]
    ) -> Dict[Tuple[int, ...], Dict[str, float]]:
        self._compute_marginals(features)

        importance_dict = {}

        for k in range(1, len(features) + 1):
            for sub_features in itertools.combinations(features, k):
                importance_dict[sub_features] = {}

                fractions_total = []
                fractions_individual = []
                for tree_index, tree in enumerate(self._trees):
                    tree_variance = tree.variance
                    if tree_variance == 0:
                        continue
                    fractions_total.append(
                        self._V_U_total[sub_features][tree_index] / tree_variance
                    )
                    fractions_individual.append(
                        self._V_U_individual[sub_features][tree_index] / tree_variance
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
        assert isinstance(features, tuple)

        if features in self._V_U_individual:
            return

        for k in range(1, len(features)):
            for sub_features in itertools.combinations(features, k):
                if sub_features not in self._V_U_total:
                    self._compute_marginals(sub_features)

        n_trees = self._forest.n_estimators
        self._V_U_individual[features] = numpy.empty(n_trees, dtype=numpy.float64)
        self._V_U_total[features] = numpy.empty(n_trees, dtype=numpy.float64)

        for tree_index, tree in enumerate(self._trees):
            stat = tree.marginalized_prediction_stat_for_features(features)

            if stat.sum_of_weights() > 0:
                variance_population = stat.variance_population()
                V_U_total = variance_population
                V_U_individual = variance_population

                for k in range(1, len(features)):
                    for sub_features in itertools.combinations(features, k):
                        V_U_individual -= self._V_U_individual[sub_features][tree_index]
                V_U_individual = numpy.clip(V_U_individual, 0, numpy.inf)
            else:
                V_U_total = numpy.nan
                V_U_individual = numpy.nan

            self._V_U_individual[features][tree_index] = V_U_individual
            self._V_U_total[features][tree_index] = V_U_total


class _CategoricalFeaturesOneHotEncoder(object):
    def __init__(self, is_categorical: List[bool]) -> None:
        self._is_categorical = is_categorical

    @property
    def n_features(self) -> int:
        return len(self._is_categorical)

    def fit_transform(self, X: numpy.ndarray) -> numpy.ndarray:
        # Transform the `X` matrix by expanding categorical integer-valued columns to one-hot
        # encoding matrices. Note that the resulting matrix is sparse and potentially very big.

        if X.shape[1] != self.n_features:
            raise ValueError("Number of columns do not match.")

        numerical_features = []
        categorical_features = []
        categorical_features_n_uniques = {}

        for i, is_categ in enumerate(self._is_categorical):
            if is_categ:
                categorical_features_n_uniques[i] = numpy.unique(X[:, i]).size
                categorical_features.append(i)
            else:
                numerical_features.append(i)

        transformer = ColumnTransformer(
            [("_categorical", OneHotEncoder(sparse=False), categorical_features)],
            remainder="passthrough",
        )

        # All categorical one-hot features will be placed before the numerical features in
        # `ColumnTransformer.fit_transform`.
        X = transformer.fit_transform(X)

        # `raw_features_to_features["column index in transformed matrix"]
        #     == "column index in original matrix"`
        raw_features_to_features = numpy.empty((X.shape[1],), dtype=numpy.int32)
        i = 0
        for categorical_feature in categorical_features:
            for _ in range(categorical_features_n_uniques[categorical_feature]):
                raw_features_to_features[i] = categorical_feature
                i += 1
        for numerical_col in numerical_features:
            raw_features_to_features[i] = numerical_col
            i += 1
        assert i == raw_features_to_features.size

        self.raw_features_to_features = raw_features_to_features

        return X


def _check_sklearn_availability():
    # type: () -> None

    if not _available:
        raise ImportError(
            "scikit-learn is not available. Please install scikit-learn to "
            "use this feature. scikit-learn can be installed by executing "
            "`$ pip install scikit-learn>=0.19.0`. For further information, "
            "please refer to the installation guide of scikit-learn. (The "
            "actual import error is as follows: " + str(_import_error) + ")"
        )
