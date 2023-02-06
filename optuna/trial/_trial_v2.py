import copy
import datetime
from typing import Any
from typing import Dict
from typing import Optional
from typing import overload
from typing import Sequence
import warnings

import optuna
from optuna import distributions
from optuna import logging
from optuna import pruners
from optuna._deprecated import deprecated_func
from optuna.distributions import BaseDistribution
from optuna.distributions import CategoricalChoiceType
from optuna.distributions import CategoricalDistribution
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
from optuna.trial._frozen import FrozenTrial
from optuna.trial._base import BaseTrial
from optuna.trial._context import TrialContext


_logger = logging.get_logger(__name__)
_suggest_deprecated_msg = "Use :func:`~optuna.trial.Trial.suggest_float` instead."


class TrialV2(BaseTrial):
    def __init__(self, study: "optuna.study.Study", trial_id: int) -> None:
        self._study = study
        self._trial_id = trial_id
        self._ctx = TrialContext(study=study, trial=self)
        self._joint_params = None

    def suggest_float(
        self,
        name: str,
        low: float,
        high: float,
        *,
        step: Optional[float] = None,
        log: bool = False,
    ) -> float:
        distribution = FloatDistribution(low, high, log=log, step=step)
        suggested_value = self._suggest(name, distribution)
        self._check_distribution(name, distribution)
        return suggested_value

    def suggest_int(self, name: str, low: int, high: int, step: int = 1, log: bool = False) -> int:
        distribution = IntDistribution(low=low, high=high, log=log, step=step)
        suggested_value = int(self._suggest(name, distribution))
        self._check_distribution(name, distribution)
        return suggested_value

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[None]) -> None:
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[bool]) -> bool:
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[int]) -> int:
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[float]) -> float:
        ...

    @overload
    def suggest_categorical(self, name: str, choices: Sequence[str]) -> str:
        ...

    @overload
    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        ...

    def suggest_categorical(
        self, name: str, choices: Sequence[CategoricalChoiceType]
    ) -> CategoricalChoiceType:
        # There is no need to call self._check_distribution because
        # CategoricalDistribution does not support dynamic value space.

        return self._suggest(name, CategoricalDistribution(choices=choices))

    def report(self, value: float, step: int) -> None:
        raise NotImplementedError

    def should_prune(self) -> bool:
        raise NotImplementedError

    def set_user_attr(self, key: str, value: Any) -> None:
        self.storage.set_trial_user_attr(self._trial_id, key, value)

    def _suggest(self, name: str, dist: BaseDistribution) -> Any:
        if self._joint_params is None:
            self._joint_params = self._study.sampler.sample_joint(self._ctx)
            # TODO(hvy): Validate joint params.

        # TODO(hvy): Consider optimizing with cached local trial
        trial = self._ctx._get_current_trial()
        if name in trial.params:
            distributions.check_distribution_compatibility(trial.distributions[name], dist)
            value = trial.params[name]
        else:
            # TODO(hvy): Support fixed params.
            if dist.single():
                value = distributions._get_single_value(dist)
            elif name in self._joint_params:  # TODO(hvy): Check similar to _is_delative_param
                value = self._joint_params[name]
            else:
                value = self._study.sampler.sample_independent(
                    self._ctx, name, dist
                )

            # `param_value` is validated here (invalid value like `np.nan` raises ValueError).
            value_in_internal_repr = dist.to_internal_repr(value)
            self._study._storage.set_trial_param(
                self._trial_id, name, value_in_internal_repr, dist)

            # TODO(hvy): Update local trial cache with params and dist info.
            # self._cached_frozen_trial.distributions[name] = distribution
            #self._cached_frozen_trial.params[name] = param_value
        return value


    def _check_distribution(self, name: str, distribution: BaseDistribution) -> None:

        # old_distribution = self._cached_frozen_trial.distributions.get(name, distribution)
        old_distribution = self._ctx._get_current_trial().distributions.get(name, distribution)
        if old_distribution != distribution:
            warnings.warn(
                'Inconsistent parameter values for distribution with name "{}"! '
                "This might be a configuration mistake. "
                "Optuna allows to call the same distribution with the same "
                "name more than once in a trial. "
                "When the parameter values are inconsistent optuna only "
                "uses the values of the first call and ignores all following. "
                "Using these values: {}".format(name, old_distribution._asdict()),
                RuntimeWarning,
            )

    @property
    def params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def distributions(self) -> Dict[str, BaseDistribution]:
        raise NotImplementedError

    @property
    def user_attrs(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def datetime_start(self) -> Optional[datetime.datetime]:
        raise NotImplementedError

    @property
    def number(self) -> int:
        raise NotImplementedError

    def suggest_uniform(self, name: str, low: float, high: float) -> float:
        raise NotImplementedError

    def suggest_loguniform(self, name: str, low: float, high: float) -> float:
        raise NotImplementedError

    def suggest_discrete_uniform(self, name: str, low: float, high: float, q: float) -> float:
        raise NotImplementedError

    def set_system_attr(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def system_attrs(self) -> Dict[str, Any]:
        raise NotImplementedError
