import numpy as np


class Accuracy:
    def __init__(self, name=None):
        self.name = name if name is not None else type(self).__name__
        self._preds = []
        self._labels = []

    def reset(self):
        self._preds = []
        self._labels = []

    def update(self, logits, labels, **batch):
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        self._preds.append(preds)
        self._labels.append(labels.detach().cpu().numpy())

    def compute(self):
        preds = np.concatenate(self._preds)
        labels = np.concatenate(self._labels)
        return (preds == labels).mean() * 100
