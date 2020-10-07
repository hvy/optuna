import optuna
from optuna import core


class DeterministicPruner(core.pruners.BasePruner):
    def __init__(self, is_pruning: bool) -> None:

        self.is_pruning = is_pruning

    def prune(self, study: "core.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:

        return self.is_pruning


def create_running_trial(study: "optuna.study.Study", value: float) -> core.trial.Trial:

    trial_id = study._storage.create_new_trial(study._study_id)
    study._storage.set_trial_value(trial_id, value)
    return optuna.trial.Trial(study, trial_id)
