import math

from optuna.importance._fanova._stats import _RunningStats
from optuna.importance._fanova._stats import _WeightedRunningStats


def test_running_stats() -> None:
    stats = _RunningStats(2, 3.0, 4.0)
    assert stats.N == 2
    assert stats.avg == 3.0
    assert stats.sdm == 4.0

    stats = stats * 2.0
    assert stats.N == 2
    assert stats.avg == 6.0
    assert stats.sdm == 16.0

    stats += _RunningStats(1, 3.0, 1.0)
    assert stats.N == 3
    assert stats.avg == 5.0
    assert stats.sdm == 23.0

    stats.push(9.0)
    assert stats.N == 4
    assert stats.avg == 6.0
    assert stats.sdm == 35.0

    assert stats.sum() == 24.0


def test_weighted_running_stats() -> None:
    stats = _WeightedRunningStats(0.0, 0.0, _RunningStats())
    assert stats.avg == 0.0
    assert stats.sdm == 0.0
    assert math.isnan(stats.mean())
    assert stats.sum_of_weights() == 0.0
    assert math.isnan(stats.variance_population())

    stats.push(2.0, 3.0)
    assert stats.avg == 2.0
    assert stats.sdm == 0.0
    assert stats.mean() == 2.0
    assert stats.sum_of_weights() == 3.0
    assert stats.variance_population() == 0.0

    stats.push(6.0, 1.0)
    assert stats.avg == 3.0
    assert stats.sdm == 12.0
    assert stats.mean() == 3.0
    assert stats.sum_of_weights() == 4.0
    assert stats.variance_population() == 3.0

    stats = stats.multiply_weights_by(2.0)
    assert stats.avg == 3.0
    assert stats.sdm == 24.0
    assert stats.mean() == 3.0
    assert stats.sum_of_weights() == 8.0
    assert stats.variance_population() == 3.0

    assert stats.divide_sdm_by(2.0, 0.0) == 12.0

    other = _WeightedRunningStats()
    other.push(4.0, 2.0)
    stats += other
    assert stats.avg == 3.2
    assert stats.sdm == 25.6
    assert stats.mean() == 3.2
    assert stats.sum_of_weights() == 10.0
    assert stats.variance_population() == 2.56
