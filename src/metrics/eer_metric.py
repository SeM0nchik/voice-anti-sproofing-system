import numpy as np

from src.metrics.eer_utils import compute_eer


class EERMetric:
    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__
        self._scores = []
        self._labels = []

    def reset(self):
        self._scores = []
        self._labels = []

    def update(self, logits, labels, **batch):
        scores = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        self._scores.append(scores)
        self._labels.append(labels.detach().cpu().numpy())

    def compute(self):
        scores = np.concatenate(self._scores)
        labels = np.concatenate(self._labels)

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]

        eer, _ = compute_eer(bonafide_scores, spoof_scores)
        return eer * 100
