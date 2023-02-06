# TODO
# 1. Pass with TPE
# 2. Fix attribute setters and getter. better design
# 3. Optimize storage access
# 4. Fix snapshot only once, incl. study and trial system attr
# 5. More hip names get_observations -> observations()
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List

import optuna


@dataclass
class TrialObservation:
    number: int
    state: "optuna.trial.TrialSte"
    params: Dict[str, Any]
    distributions: Dict[str, Any]
    values: List[float]
    intermediate_values: Dict[str, float]

    _trial_id: int


@dataclass
class TrialAttributes:
    number: int
    attrs: Dict[str, Any]


'''
@dataclass
class _StudySnapshot:
    study: "optuna.study.FrozenStudy"
    trials: "optuna.trials.FrozenTrial"
'''


class TrialContext:
    def __init__(
        self,
        study,
        trial,
    ):
        self._study = study
        self._trial = trial
        self._trials = None
        self._observations = None

    def get_study_directions(self):
        return self._study.directions

    def get_study_attrs(self):
        # TODO(hvy): Do not query storage. Get snapshot once!
        self._study._storage.get_study_system_attr(self._study._study_id)

    def get_trial_number(self):
        return self._trial.number

    def get_trial_attrs(self, number=None):
        if number is None:
            number = self._trial.number
        return self._study._storage.get_trial_system_attrs(self._observations[number]._trial_id)

    def update_study_attr(self, key, value):
        self._study._storage.set_study_system_attr(self._study._study_id, key, value)

    def update_trial_attr(self, key, value):
        self._study._storage.set_trial_system_attr(self._trial._trial_id, key, value)

    # TODO(hvy): Reconsider if this is really necessary.
    # def stop_study(self):
    #     self._study.stop()

    def get_observations(self):
        # TODO(hvy): Filter with Hyperband.

        if self._observations is None:
            self._trials = self._study.get_trials(deepcopy=False)
            obss = []
            for t in self._trials:
                obs = TrialObservation(
                    t.number,
                    t.state,
                    t.params,
                    t.distributions,
                    t.values,
                    t.intermediate_values,
                    t._trial_id,
                )
                obss.append(obs)
            self._observations = obss
        return self._observations

    def _get_current_trial(self):
        # Used by TrialV2 and temporarily by some callers such as TPESamplerV2
        # TODO(hvy): Optimize.
        return self._study._storage.get_trial(self._trial._trial_id)

