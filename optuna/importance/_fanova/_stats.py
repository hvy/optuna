class _RunningStats(object):
    def __init__(self, N: int = 0, avg: float = 0.0, sdm: float = 0.0) -> None:
        self.N = N
        self.avg = avg
        self.sdm = sdm  # Squared distance from the mean.

    def __mul__(self, other: float) -> "_RunningStats":
        if not isinstance(other, float):
            raise NotImplementedError
        return _RunningStats(self.N, self.avg * other, self.sdm * other * other)

    def __iadd__(self, other: "_RunningStats") -> "_RunningStats":
        N_total = self.N + other.N
        n1 = self.N
        n2 = other.N
        nt = N_total

        avg_total = self.avg * (n1 / nt) + other.avg * (n2 / nt)
        sdm_total = (
            self.sdm
            + other.sdm
            + n1 * (self.avg - avg_total) ** 2
            + n2 * (other.avg - avg_total) ** 2
        )

        self.avg = avg_total
        self.sdm = sdm_total
        self.N = N_total
        return self

    def push(self, x: float) -> None:
        self.N += 1
        delta = x - self.avg
        self.avg += delta / self.N
        self.sdm += delta * (x - self.avg)

    def sum(self) -> float:
        return self.avg * self.N


class _WeightedRunningStats(object):
    def __init__(
        self, avg: float = 0.0, sdm: float = 0.0, weight_stat: _RunningStats = None
    ) -> None:
        self.avg = avg
        self.sdm = sdm
        if weight_stat is None:
            weight_stat = _RunningStats()
        self.weight_stat = weight_stat

    def __iadd__(self, other: "_WeightedRunningStats") -> "_WeightedRunningStats":
        sw1 = self.weight_stat.sum()
        sw2 = other.weight_stat.sum()
        swt = sw1 + sw2
        avg_total = (self.avg * (sw1 / swt)) + (other.avg * (sw2 / swt))
        sdm_total = (
            self.sdm
            + other.sdm
            + sw1 * ((self.avg - avg_total) ** 2)
            + sw2 * ((other.avg - avg_total) ** 2)
        )
        self.avg = avg_total
        self.sdm = sdm_total
        self.weight_stat += other.weight_stat
        return self

    def push(self, x: float, weight: float) -> None:
        assert weight > 0.0

        delta = x - self.avg

        self.weight_stat.push(weight)

        self.avg += delta * weight / self.weight_stat.sum()
        self.sdm += delta * weight * (x - self.avg)

    def mean(self) -> float:
        return self.avg if self.weight_stat.sum() > 0.0 else float("nan")

    def multiply_weights_by(self, a: float) -> "_WeightedRunningStats":
        return _WeightedRunningStats(self.avg, a * self.sdm, self.weight_stat * a)

    def sum_of_weights(self) -> float:
        return self.weight_stat.sum()

    def variance_population(self) -> float:
        weight_stat_sum = self.weight_stat.sum()
        assert weight_stat_sum > 0.0
        return self.sdm / weight_stat_sum
