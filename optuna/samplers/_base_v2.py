import abc
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
import warnings

import numpy as np

from optuna.distributions import BaseDistribution
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.trial import TrialState
from optuna.trial._context import TrialContext


class BaseSamplerV2(abc.ABC):

    @abc.abstractmethod
    def sample_joint(
        self,
        ctx: TrialContext,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_independent(
        self,
        ctx: TrialContext,
        param_name: str,
        param_dist: BaseDistribution,
    ) -> Any:
        raise NotImplementedError

    def after_trial(
        self,
        ctx: TrialContext,
    ) -> None:
        pass

    def reseed_rng(self) -> None:
        pass
