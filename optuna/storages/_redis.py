import copy
from datetime import datetime
import pickle
from typing import Any  # NOQA
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Union

import optuna
from optuna import distributions
from optuna import exceptions
from optuna._experimental import experimental
from optuna._imports import try_import
from optuna._study_direction import StudyDirection
from optuna._study_summary import StudySummary
from optuna.storages import BaseStorage
from optuna.storages._base import DEFAULT_STUDY_NAME_PREFIX
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


_logger = optuna.logging.get_logger(__name__)


with try_import() as _imports:
    import redis


@experimental("1.4.0")
class RedisStorage(BaseStorage):
    """Storage class for Redis backend.

    Note that library users can instantiate this class, but the attributes
    provided by this class are not supposed to be directly accessed by them.

    Example:

        We create an :class:`~optuna.storages.RedisStorage` instance using
        the given redis database URL.

        .. code::

            import optuna


            def objective(trial):
                ...


            storage = optuna.storages.RedisStorage(
                url='redis://passwd@localhost:port/db',
            )

            study = optuna.create_study(storage=storage)
            study.optimize(objective)

    Args:
        url: URL of the redis storage, password and db are optional. (ie: redis://localhost:6379)

    .. note::
        If you use plan to use Redis as a storage mechanism for optuna,
        make sure Redis in installed and running.
        Please execute ``$ pip install -U redis`` to install redis python library.
    """

    # Study ID to trial ID of best trial.
    _STUDY_ID_BEST_TRIAL_ID_PKL_KEY = "study_id:{study_id:010d}:best_trial_id"

    # Study ID to study direction.
    _STUDY_ID_STUDY_DIRECTION_PKL_KEY = "study_id:{study_id:010d}:direction"

    _STUDY_ID_STUDY_NAME_KEY = "study_id:{study_id:010d}:study_name"
    _STUDY_ID_STUDY_SUMMARY_PKL_KEY = "study_id:{study_id:010d}:study_summary"
    _STUDY_ID_TRIAL_ID_PKL_LIST_KEY = "study_id:{study_id:010d}:trial_list"
    _TRIAL_ID_TRIAL_PKL_KEY = "trial_id:{trial_id:010d}:frozentrial"
    _TRIAL_ID_STUDY_ID_PKL_KEY = "trial_id:{trial_id:010d}:study_id"

    def __init__(self, url: str) -> None:

        _imports.check()

        self._url = url
        self._redis = redis.Redis.from_url(url)

    def create_new_study(self, study_name: Optional[str] = None) -> int:

        if study_name is not None and self._redis.exists(self._key_study_name(study_name)):
            raise exceptions.DuplicatedStudyError

        if not self._redis.exists("study_counter"):
            # We need the counter to start with 0.
            self._redis.set("study_counter", -1)
        study_id = self._redis.incr("study_counter", 1)
        # We need the trial_number counter to start with 0.
        self._redis.set("study_id:{:010d}:trial_number".format(study_id), -1)

        if study_name is None:
            study_name = "{}{:010d}".format(DEFAULT_STUDY_NAME_PREFIX, study_id)

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_study_name(study_name), pickle.dumps(study_id))
            pipe.set("study_id:{:010d}:study_name".format(study_id), pickle.dumps(study_name))
            pipe.set(
                "study_id:{:010d}:direction".format(study_id),
                pickle.dumps(StudyDirection.NOT_SET),
            )

            study_summary = StudySummary(
                study_name=study_name,
                direction=StudyDirection.NOT_SET,
                best_trial=None,
                user_attrs={},
                system_attrs={},
                n_trials=0,
                datetime_start=None,
                study_id=study_id,
            )
            pipe.rpush("study_list", pickle.dumps(study_id))
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

        _logger.info("A new study created in Redis with name: {}".format(study_name))

        return study_id

    def delete_study(self, study_id: int) -> None:

        self._check_study_id(study_id)

        with self._redis.pipeline() as pipe:
            pipe.multi()
            # Sumaries
            pipe.delete(self._key_study_summary(study_id))
            pipe.lrem("study_list", 0, pickle.dumps(study_id))
            # Trials
            trial_ids = self._get_study_trials(study_id)
            for trial_id in trial_ids:
                pipe.delete("trial_id:{:010d}:frozentrial".format(trial_id))
                pipe.delete("trial_id:{:010d}:study_id".format(trial_id))
            pipe.delete("study_id:{:010d}:trial_list".format(study_id))
            pipe.delete("study_id:{:010d}:trial_number".format(study_id))
            # Study
            study_name = self.get_study_name_from_id(study_id)
            pipe.delete("study_name:{}:study_id".format(study_name))
            pipe.delete("study_id:{:010d}:study_name".format(study_id))
            pipe.delete("study_id:{:010d}:direction".format(study_id))
            pipe.delete("study_id:{:010d}:best_trial_id".format(study_id))
            pipe.delete("study_id:{:010d}:params_distribution".format(study_id))
            pipe.execute()

    @staticmethod
    def _key_study_name(study_name: str) -> str:

        return "study_name:{}:study_id".format(study_name)

    @staticmethod
    def _key_study_summary(study_id: int) -> str:

        return "study_id:{:010d}:study_summary".format(study_id)

    def _set_study_summary(self, study_id: int, study_summary: StudySummary) -> None:

        self._redis.set(self._key_study_summary(study_id), pickle.dumps(study_summary))

    def _get_study_summary(self, study_id: int) -> StudySummary:

        summary_pkl = self._redis.get(self._key_study_summary(study_id))
        assert summary_pkl is not None
        return pickle.loads(summary_pkl)

    def _del_study_summary(self, study_id: int) -> None:

        self._redis.delete(self._key_study_summary(study_id))

    @staticmethod
    def _key_study_direction(study_id: int) -> str:

        return "study_id:{:010d}:direction".format(study_id)

    def set_study_direction(self, study_id: int, direction: StudyDirection) -> None:

        self._check_study_id(study_id)

        if self._redis.exists(self._key_study_direction(study_id)):
            direction_pkl = self._redis.get(self._key_study_direction(study_id))
            assert direction_pkl is not None
            current_direction = pickle.loads(direction_pkl)
            if current_direction != StudyDirection.NOT_SET and current_direction != direction:
                raise ValueError(
                    "Cannot overwrite study direction from {} to {}.".format(
                        current_direction, direction
                    )
                )

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_study_direction(study_id), pickle.dumps(direction))
            study_summary = self._get_study_summary(study_id)
            study_summary.direction = direction
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

    def set_study_user_attr(self, study_id: int, key: str, value: Any) -> None:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.user_attrs[key] = value
        self._set_study_summary(study_id, study_summary)

    def set_study_system_attr(self, study_id: int, key: str, value: Any) -> None:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        study_summary.system_attrs[key] = value
        self._set_study_summary(study_id, study_summary)

    def get_study_id_from_name(self, study_name: str) -> int:

        if not self._redis.exists(self._key_study_name(study_name)):
            raise KeyError("No such study {}.".format(study_name))
        study_id_pkl = self._redis.get(self._key_study_name(study_name))
        assert study_id_pkl is not None
        return pickle.loads(study_id_pkl)

    def get_study_id_from_trial_id(self, trial_id: int) -> int:

        study_id_pkl = self._redis.get("trial_id:{:010d}:study_id".format(trial_id))
        if study_id_pkl is None:
            raise KeyError("No such trial: {}.".format(trial_id))
        return pickle.loads(study_id_pkl)

    def get_study_name_from_id(self, study_id: int) -> str:

        self._check_study_id(study_id)

        study_name_pkl = self._redis.get("study_id:{:010d}:study_name".format(study_id))
        if study_name_pkl is None:
            raise KeyError("No such study: {}.".format(study_id))
        return pickle.loads(study_name_pkl)

    def get_study_direction(self, study_id: int) -> StudyDirection:

        direction_pkl = self._redis.get("study_id:{:010d}:direction".format(study_id))
        if direction_pkl is None:
            raise KeyError("No such study: {}.".format(study_id))
        return pickle.loads(direction_pkl)

    def get_study_user_attrs(self, study_id: int) -> Dict[str, Any]:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        return copy.deepcopy(study_summary.user_attrs)

    def get_study_system_attrs(self, study_id: int) -> Dict[str, Any]:

        self._check_study_id(study_id)

        study_summary = self._get_study_summary(study_id)
        return copy.deepcopy(study_summary.system_attrs)

    @staticmethod
    def _key_study_param_distribution(study_id: int) -> str:

        return "study_id:{:010d}:params_distribution".format(study_id)

    def _get_study_param_distribution(self, study_id: int) -> Dict:

        if self._redis.exists(self._key_study_param_distribution(study_id)):
            param_distribution_pkl = self._redis.get(self._key_study_param_distribution(study_id))
            assert param_distribution_pkl is not None
            return pickle.loads(param_distribution_pkl)
        else:
            return {}

    def _set_study_param_distribution(self, study_id: int, param_distribution: Dict) -> None:

        self._redis.set(
            self._key_study_param_distribution(study_id), pickle.dumps(param_distribution)
        )

    def get_all_study_summaries(self) -> List[StudySummary]:

        study_summaries = []
        study_ids = [pickle.loads(sid) for sid in self._redis.lrange("study_list", 0, -1)]
        for study_id in study_ids:
            study_summary = self._get_study_summary(study_id)
            study_summaries.append(study_summary)

        return study_summaries

    def create_new_trial(self, study_id: int, template_trial: Optional[FrozenTrial] = None) -> int:
        study_id_number_key = "study_id:{:010d}:trial_number".format(study_id)
        study_id_trial_list_key = "study_id:{:010d}:trial_list".format(study_id)
        trial_counter_key = "trial_counter"
        trial_id_study_id_key = "trial_id:{:010d}:study_id"

        if template_trial is not None:
            trial = copy.deepcopy(template_trial)
        else:
            trial = FrozenTrial(
                trial_id=-1,  # dummy value.
                number=-1,  # dummy value.
                state=TrialState.RUNNING,
                params={},
                distributions={},
                user_attrs={},
                system_attrs={},
                value=None,
                intermediate_values={},
                datetime_start=datetime.now(),
                datetime_complete=None,
            )

        import threading

        def transaction(pipe: redis.client.Pipeline) -> None:
            _check_study_exists(pipe, study_id)

            trial_counter = pipe.get(trial_counter_key)
            if trial_counter is None:
                print('New trial', threading.get_ident())
                trial_counter = 0
            else:
                print('Increment trial from', trial_counter, threading.get_ident())
                trial_counter = int(trial_counter)
                trial_counter += 1

            trial_id = trial_counter

            number = pipe.get(study_id_number_key)
            print('Number', number, threading.get_ident())
            number = int(number)
            number += 1

            trial._trial_id = trial_id
            trial.number = number

            pipe.multi()
            pipe.set(trial_counter_key, trial_counter)
            pipe.set(study_id_number_key, number)
            pipe.set(self._key_trial(trial_id), pickle.dumps(trial))
            pipe.set(trial_id_study_id_key.format(trial_id), pickle.dumps(study_id))
            pipe.rpush(study_id_trial_list_key, trial_id)

        self._redis.transaction(
            transaction,
            trial_counter_key,
            study_id_number_key,
            study_id_trial_list_key,
        )

        self._update_study_summary(study_id)

        if trial.state.is_finished():
            self._update_best_trial(trial._trial_id)

        return trial._trial_id

    def _update_study_summary(self, study_id: int) -> None:
        study_summary_key = self._key_study_summary(study_id)
        study_trial_list_key = "study_id:{:010d}:trial_list".format(study_id)

        def transaction(pipe: redis.client.Pipeline) -> None:
            _check_study_exists(pipe, study_id)

            study_summary_pkl = pipe.get(study_summary_key)
            study_summary = pickle.loads(study_summary_pkl)

            study_trial_list = pipe.lrange(study_trial_list_key, 0, -1)
            study_summary.n_trials = len(study_trial_list)
            study_summary.datetime_start = min(
                pickle.loads(pipe.get(self._key_trial(int(trial_id)))).datetime_start
                for trial_id in study_trial_list
            )

            pipe.multi()
            pipe.set(study_summary_key, pickle.dumps(study_summary))

        self._redis.transaction(transaction, study_summary_key)

    def set_trial_state(self, trial_id: int, state: TrialState) -> bool:

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        if state == TrialState.RUNNING and trial.state != TrialState.WAITING:
            return False

        trial.state = state
        if state.is_finished():
            trial.datetime_complete = datetime.now()
            self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))
            self._update_best_trial(trial_id)
        else:
            self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

        return True

    def set_trial_param(
        self,
        trial_id: int,
        param_name: str,
        param_value_internal: float,
        distribution: distributions.BaseDistribution,
    ) -> None:

        self._check_trial_id(trial_id)
        self.check_trial_is_updatable(trial_id, self.get_trial(trial_id).state)

        # Check param distribution compatibility with previous trial(s).
        study_id = self.get_study_id_from_trial_id(trial_id)
        param_distribution = self._get_study_param_distribution(study_id)
        if param_name in param_distribution:
            distributions.check_distribution_compatibility(
                param_distribution[param_name], distribution
            )

        trial = self.get_trial(trial_id)

        with self._redis.pipeline() as pipe:
            pipe.multi()
            # Set study param distribution.
            param_distribution[param_name] = distribution
            pipe.set(
                self._key_study_param_distribution(study_id), pickle.dumps(param_distribution)
            )

            # Set params.
            trial.params[param_name] = distribution.to_external_repr(param_value_internal)
            trial.distributions[param_name] = distribution
            pipe.set(self._key_trial(trial_id), pickle.dumps(trial))
            pipe.execute()

    def get_trial_number_from_id(self, trial_id: int) -> int:

        return self.get_trial(trial_id).number

    @staticmethod
    def _key_best_trial(study_id: int) -> str:

        return "study_id:{:010d}:best_trial_id".format(study_id)

    def get_best_trial(self, study_id: int) -> FrozenTrial:

        if not self._redis.exists(self._key_best_trial(study_id)):
            all_trials = self.get_all_trials(study_id, deepcopy=False)
            all_trials = [t for t in all_trials if t.state is TrialState.COMPLETE]

            if len(all_trials) == 0:
                raise ValueError("No trials are completed yet.")

            if self.get_study_direction(study_id) == StudyDirection.MAXIMIZE:
                best_trial = max(all_trials, key=lambda t: t.value)
            else:
                best_trial = min(all_trials, key=lambda t: t.value)

            self._set_best_trial(study_id, best_trial.number)
        else:
            best_trial_id_pkl = self._redis.get(self._key_best_trial(study_id))
            assert best_trial_id_pkl is not None
            best_trial_id = pickle.loads(best_trial_id_pkl)
            best_trial = self.get_trial(best_trial_id)

        return best_trial

    def _set_best_trial(self, study_id: int, trial_id: int) -> None:

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.set(self._key_best_trial(study_id), pickle.dumps(trial_id))

            study_summary = self._get_study_summary(study_id)
            study_summary.best_trial = self.get_trial(trial_id)
            pipe.set(self._key_study_summary(study_id), pickle.dumps(study_summary))
            pipe.execute()

    def get_trial_param(self, trial_id: int, param_name: str) -> float:

        distribution = self.get_trial(trial_id).distributions[param_name]
        return distribution.to_internal_repr(self.get_trial(trial_id).params[param_name])

    def set_trial_value(self, trial_id: int, value: float) -> None:

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)

        trial.value = value
        self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _update_best_trial(self, trial_id: int) -> None:
        study_id_pkl = self._redis.get(
            RedisStorage._TRIAL_ID_STUDY_ID_PKL_KEY.format(trial_id=trial_id)
        )
        study_id = pickle.loads(study_id_pkl)

        def transaction(pipe: redis.client.Pipeline) -> None:
            _check_study_exists(pipe, study_id)
            _check_trial_exists(pipe, trial_id)

            trial_pkl = pipe.get(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=trial_id))
            trial = pickle.loads(trial_pkl)
            if trial.state != TrialState.COMPLETE:
                return

            is_new_better = False

            if not pipe.exists(
                RedisStorage._STUDY_ID_BEST_TRIAL_ID_PKL_KEY.format(study_id=study_id)
            ):
                is_new_better = True
            else:
                best_trial = _get_best_trial_with_pipe(pipe, study_id)

                # TODO(hvy): Do we need to cast? Skip casting if not necessary.
                best_value = float(best_trial.value)
                new_value = float(trial.value)

                if _get_study_direction_with_pipe(pipe, study_id) == StudyDirection.MAXIMIZE:
                    if new_value > best_value:
                        is_new_better = True
                else:
                    if new_value < best_value:
                        is_new_better = True

            if is_new_better:
                # No best trial is associated with this study. Set the current trial.
                study_summary_pkl = pipe.get(
                    RedisStorage._STUDY_ID_STUDY_SUMMARY_PKL_KEY.format(study_id=study_id),
                )
                study_summary = pickle.loads(study_summary_pkl)
                study_summary.best_trial = trial

                # Write.
                pipe.multi()
                pipe.set(
                    RedisStorage._STUDY_ID_BEST_TRIAL_ID_PKL_KEY.format(study_id=study_id),
                    pickle.dumps(trial_id),
                )
                pipe.set(
                    RedisStorage._STUDY_ID_STUDY_SUMMARY_PKL_KEY.format(study_id=study_id),
                    pickle.dumps(study_summary),
                )

        self._redis.transaction(
            transaction,
            RedisStorage._TRIAL_ID_STUDY_ID_PKL_KEY.format(trial_id=trial_id),
            RedisStorage._STUDY_ID_BEST_TRIAL_ID_PKL_KEY.format(study_id=study_id),
            RedisStorage._STUDY_ID_STUDY_SUMMARY_PKL_KEY.format(study_id=study_id),
        )

    def set_trial_intermediate_value(
        self, trial_id: int, step: int, intermediate_value: float
    ) -> None:

        self._check_trial_id(trial_id)
        frozen_trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, frozen_trial.state)
        frozen_trial.intermediate_values[step] = intermediate_value
        self._set_trial(trial_id, frozen_trial)

    def set_trial_user_attr(self, trial_id: int, key: str, value: Any) -> None:

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)
        trial.user_attrs[key] = value
        self._set_trial(trial_id, trial)

    def set_trial_system_attr(self, trial_id: int, key: str, value: Any) -> None:

        self._check_trial_id(trial_id)
        trial = self.get_trial(trial_id)
        self.check_trial_is_updatable(trial_id, trial.state)
        trial.system_attrs[key] = value
        self._set_trial(trial_id, trial)

    @staticmethod
    def _key_trial(trial_id: int) -> str:

        return "trial_id:{:010d}:frozentrial".format(trial_id)

    def get_trial(self, trial_id: int) -> FrozenTrial:
        frozen_trial_pkl = self._redis.transaction(
            lambda pipe: self._get_trial_with_pipe(pipe, trial_id),
            self._key_trial(trial_id),
        )[0]

        assert frozen_trial_pkl is not None
        return pickle.loads(frozen_trial_pkl)

    def _get_trial_with_pipe(self, pipe: "redis.client.Pipeline", trial_id: int) -> FrozenTrial:
        if not pipe.exists(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=trial_id)):
            raise KeyError("Trial with id {} does not exist.".format(trial_id))

        pipe.multi()
        pipe.get(self._key_trial(trial_id))

    def _set_trial(self, trial_id: int, trial: FrozenTrial) -> None:

        self._redis.set(self._key_trial(trial_id), pickle.dumps(trial))

    def _del_trial(self, trial_id: int) -> None:

        with self._redis.pipeline() as pipe:
            pipe.multi()
            pipe.delete(self._key_trial(trial_id))
            pipe.delete("trial_id:{:010d}:study_id".format(trial_id))
            pipe.execute()

    def _get_study_trials(self, study_id: int) -> List[int]:
        study_list_key = "study_list"
        study_trial_list_key = "study_id:{:010d}:trial_list".format(study_id)

        def transaction(pipe: redis.client.Pipeline) -> None:
            self._check_study_id(study_id)

            pipe.multi()
            pipe.lrange(study_trial_list_key, 0, -1)

        ret = self._redis.transaction(transaction, study_list_key, study_trial_list_key)

        study_trial_list = ret[0]

        return [int(tid) for tid in study_trial_list]

    def get_all_trials(self, study_id: int, deepcopy: bool = True) -> List[FrozenTrial]:

        self._check_study_id(study_id)

        trials = []
        trial_ids = self._get_study_trials(study_id)
        for trial_id in trial_ids:
            frozen_trial = self.get_trial(trial_id)
            trials.append(frozen_trial)

        if deepcopy:
            return copy.deepcopy(trials)
        else:
            return trials

    def read_trials_from_remote_storage(self, study_id: int) -> None:
        self._check_study_id(study_id)

    def _check_study_id(self, study_id: int) -> None:
        if not self._redis.exists("study_id:{:010d}:study_name".format(study_id)):
            raise KeyError("study_id {} does not exist.".format(study_id))

    def _check_trial_id(self, trial_id: int) -> None:
        if not self._redis.exists(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=trial_id)):
            raise KeyError("study_id {} does not exist.".format(trial_id))

    @staticmethod
    def _get_study_id_from_trial_id_with_pipe(pipe: "redis.client.Pipeline", trial_id: int) -> int:
        assert not pipe.explicit_transaction
        return pickle.loads(pipe.get(RedisStorage._TRIAL_ID_STUDY_ID_PKL_KEY.format(trial_id)))

    def _check_study_exists(study_id: int) -> None:
        _check_study_exists(self._redis, study_id)

    def _check_trial_exists(trial_id: int) -> None:
        _check_trial_exists(self._redis, study_id)


def _check_study_exists(
    client: Union["redis.client.Redis", "redis.client.Pipeline"], study_id: int
) -> None:
    if isinstance(client, redis.client.Pipeline):
        assert not client.explicit_transaction  # Must be immediate and not MULTI.

    if not client.exists(RedisStorage._STUDY_ID_STUDY_NAME_KEY.format(study_id=study_id)):
        raise KeyError("Study with id {} does not exist.".format(study_id))


def _check_trial_exists(
    client: Union["redis.client.Redis", "redis.client.Pipeline"], trial_id: int
) -> None:
    if isinstance(client, redis.client.Pipeline):
        assert not client.explicit_transaction  # Must be immediate and not MULTI.

    if not client.exists(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=trial_id)):
        raise KeyError("Trial with {} does not exist.".format(trial_id))


def _get_best_trial_with_pipe(pipe: "redis.client.Pipeline", study_id: int) -> FrozenTrial:
    assert not pipe.explicit_transaction  # Must be immediate and not MULTI.

    best_trial_id_pkl = pipe.get(
        RedisStorage._STUDY_ID_BEST_TRIAL_ID_PKL_KEY.format(study_id=study_id)
    )
    best_trial_id = pickle.loads(best_trial_id_pkl)

    best_trial_pkl = pipe.get(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=best_trial_id))
    best_trial = pickle.loads(best_trial_pkl)

    return best_trial

    """
    if not pipe.exists(_STUDY_ID_BEST_TRIAL_ID_PKL_KEY.format(study_id)):
        trials = _get_trials_with_pipe(pipe, study_id)
        trials = [t for t in trials if t.state is TrialState.COMPLETE]

        if len(trials) == 0:
            raise ValueError("No trials are completed yet.")

        if _get_study_direction_with_pipe(pipe, study_id) == StudyDirection.MAXIMIZE:
            best_trial = max(trials, key=lambda t: t.value)
        else:
            best_trial = min(trials, key=lambda t: t.value)

        # TODO(hvy): Set at caller after MULTI.
        # self._set_best_trial(study_id, best_trial.number)
    else:
        best_trial_id_pkl = pipe.get(RedisStorage._STUDY_ID_BEST_TRIAL_ID_PKL_KEY.format((study_id)))
        assert best_trial_id_pkl is not None
        best_trial_id = pickle.loads(best_trial_id_pkl)
        best_trial_pkl = pipe.get(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(best_trial_id))
        best_trial = pickle.loads(best_trial_pkl)

    return best_trial
    """


def _get_study_direction_with_pipe(pipe, study_id):
    assert not pipe.explicit_transaction  # Must be immediate and not MULTI.

    direction_pkl = pipe.get(
        RedisStorage._STUDY_ID_STUDY_DIRECTION_PKL_KEY.format(study_id=study_id)
    )
    direction = pickle.loads(direction_pkl)

    return direction


def _get_trials_with_pipe(pipe: "redis.client.Pipeline", study_id: int) -> List[FrozenTrial]:
    assert not pipe.explicit_transaction  # Must be immediate and not MULTI.

    trial_ids = pipe.lrange(
        RedistStorage._STUDY_ID_TRIAL_ID_PKL_LIST_KEY.format(study_id=study_id), 0, -1
    )
    trials = []
    for trial_id in trial_ids:
        trial_pkl = pipe.get(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=int(trial_id)))
        trial = pickle.loads(trial_pkl)
        trials.append(trial)

    return trials


def _get_trial_with_pipe(pipe: "redis.client.Pipeline", trial_id: int) -> List[FrozenTrial]:
    assert not pipe.explicit_transaction  # Must be immediate and not MULTI.

    trial_ids = pipe.lrange(
        RedistStorage._STUDY_ID_TRIAL_ID_PKL_LIST_KEY.format(study_id=study_id), 0, -1
    )
    trials = []
    for trial_id in trial_ids:
        trial_pkl = pipe.get(RedisStorage._TRIAL_ID_TRIAL_PKL_KEY.format(trial_id=int(trial_id)))
        trial = pickle.loads(trial_pkl)
        trials.append(trial)

    return trials
