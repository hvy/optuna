from collections import OrderedDict
from typing import Dict
from typing import List
from typing import Optional

import numpy

from optuna._experimental import experimental
from optuna.distributions import CategoricalDistribution
from optuna.distributions import DiscreteUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution
from optuna.importance._base import _get_distributions
from optuna.importance._base import _get_study_data
from optuna.importance._base import BaseImportanceEvaluator
from optuna.importance._fanova._fanova import _Fanova
from optuna.study import Study


@experimental("1.3.0")
class FanovaImportanceEvaluator(BaseImportanceEvaluator):
    """fANOVA parameter importance evaluator.

    .. note::

        Requires the `sklearn <https://github.com/scikit-learn/scikit-learn>`_ Python package.

    .. seealso::

        `An Efficient Approach for Assessing Hyperparameter Importance
        <http://proceedings.mlr.press/v32/hutter14.html>`_.

    # TODO(hvy): Document arguments.

    """

    def __init__(
        self, n_estimators: int = 16, max_depth: int = 64, random_state: Optional[int] = None
    ) -> None:
        self._evaluator = _Fanova(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=random_state,
        )

    def evaluate(self, study: Study, params: Optional[List[str]]) -> Dict[str, float]:
        distributions = _get_distributions(study, params)
        params_data, values_data = _get_study_data(study, distributions)

        if params_data.size == 0:  # `params` were given but as an empty list.
            return OrderedDict()

        search_spaces = numpy.empty((len(distributions), 2), dtype=numpy.float64)
        search_spaces_is_categorical = []

        for i, distribution in enumerate(distributions.values()):
            if isinstance(distribution, CategoricalDistribution):
                search_spaces[i, 0] = 0
                search_spaces[i, 1] = float(len(distribution.choices))
                search_spaces_is_categorical.append(True)
            elif isinstance(
                distribution,
                (
                    DiscreteUniformDistribution,
                    IntUniformDistribution,
                    LogUniformDistribution,
                    UniformDistribution,
                ),
            ):
                search_spaces[i, 0] = distribution.low
                search_spaces[i, 1] = distribution.high
                search_spaces_is_categorical.append(False)
            else:
                assert False

        evaluator = self._evaluator
        evaluator.fit(
            X=params_data,
            y=values_data,
            search_spaces=search_spaces,
            search_spaces_is_categorical=search_spaces_is_categorical,
        )

        individual_importances = {}
        for i, name in enumerate(distributions.keys()):
            importance = evaluator.quantify_importance((i,))
            individual_importance = importance[(i,)]["individual importance"]
            individual_importances[name] = individual_importance

        tot_importance = sum(v for v in individual_importances.values())
        for name in individual_importances.keys():
            individual_importances[name] /= tot_importance

        param_importances = OrderedDict(
            reversed(
                sorted(
                    individual_importances.items(),
                    key=lambda name_and_importance: name_and_importance[1],
                )
            )
        )
        return param_importances
