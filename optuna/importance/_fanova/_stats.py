class _RunningStats(object):
    def __init__(self, N=0, avg=0, sdm=0):
        self.N = N
        self.avg = avg
        self.sdm = sdm

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            raise NotImplementedError
        return _RunningStats(self.N, self.avg * other, self.sdm * other * other)

    def __iadd__(self, other):
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

    def push(self, x):
        self.N += 1
        delta = x - self.avg
        self.avg += delta / self.N
        self.sdm += delta * (x - self.avg)

    def sum(self):
        return self.avg * self.N


class _WeightedRunningStats(object):
    def __init__(self, avg=0, sdm=0, weight_stat=None):
        self.avg = avg
        self.sdm = sdm
        if weight_stat is None:
            weight_stat = _RunningStats()
        self.weight_stat = weight_stat

    def __iadd__(self, other):
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

    def push(self, x, weight):
        assert weight > 0

        delta = x - self.avg

        self.weight_stat.push(weight)

        self.avg += delta * weight / self.weight_stat.sum()
        self.sdm += delta * weight * (x - self.avg)

    def mean(self):
        if self.weight_stat.sum() > 0:
            return self.avg
        return float("nan")

    def multiply_weights_by(self, a):
        return _WeightedRunningStats(self.avg, a * self.sdm, self.weight_stat * a)

    def sum_of_weights(self):
        return self.weight_stat.sum()

    def variance_population(self):
        return self.divide_sdm_by(self.weight_stat.sum(), 0.0)

    def divide_sdm_by(self, fraction, min_weight):
        return (
            max(0.0, self.sdm / fraction) if self.weight_stat.sum() > min_weight else float("nan")
        )
