from functools import partial

from distance import levenshtein
from mmengine.evaluator import BaseMetric
from mmengine.registry import METRICS


@METRICS.register_module()
class CER(BaseMetric):
    OUTPUTS = "outputs"
    TARGETS = "targets"

    def process(self, data_batch, data_samples):
        self.results.extend(data_samples)

    def compute_metrics(self, results: list):
        # CER
        dist = sum(map(self._dist, results))
        norm = sum(map(self._norm, results))

        # ExpRate
        exp0 = sum(map(partial(self._rate, dist=0), results))
        exp1 = sum(map(partial(self._rate, dist=1), results))
        exp2 = sum(map(partial(self._rate, dist=2), results))

        # determine scores
        cer = dist / max(1, norm)
        er0 = exp0 / max(1, len(results))
        er1 = exp1 / max(1, len(results))
        er2 = exp2 / max(1, len(results))

        return dict(CER=cer, EM=er0, ExpRate1=er1, ExpRate2=er2)

    def _dist(self, result):
        y = result[self.OUTPUTS][self.prefix]
        t = result[self.TARGETS][self.prefix]

        return levenshtein(y, t)

    def _norm(self, result):
        return len(result[self.TARGETS][self.prefix])

    def _rate(self, result, dist: int):
        return self._dist(result) <= dist


if __name__ == "__main__":
    import json
    from pathlib import Path

    data = json.loads(Path("sample_data.json").read_text())
    test = json.loads(Path("sample_test.json").read_text())

    assert test == CER(prefix="tex").compute_metrics(data).get("CER")
