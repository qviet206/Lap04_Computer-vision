"""Khoảng cách Hamming, ngưỡng, độ chính xác / độ nhạy / độ đặc hiệu, ROC."""
from dataclasses import dataclass, asdict

import numpy as np


class HammingComparator:
    @staticmethod
    def distance(h1, h2) -> int:
        return int(np.sum(h1 != h2))

    @staticmethod
    def matrix(B: np.ndarray) -> np.ndarray:
        """Ma trận n x n khoảng cách Hamming (nhân ma trận thay vòng lặp lồng)."""
        B = B.astype(np.int32)
        return B @ (1 - B).T + (1 - B) @ B.T

    @staticmethod
    def to_query(codes_packed, nbits, q_bits) -> np.ndarray:
        q = np.packbits(q_bits)
        return np.unpackbits(codes_packed ^ q, axis=1)[:, :nbits].sum(1)


@dataclass
class BinaryMetrics:
    TP: int
    FP: int
    TN: int
    FN: int

    @classmethod
    def at_threshold(cls, d, y, T):
        pred = d <= T
        return cls(int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & ~y).sum()), int((~pred & y).sum()))

    @property
    def accuracy(self): return (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
    @property
    def sensitivity(self): return self.TP / (self.TP + self.FN)           # độ nhạy / recall
    @property
    def specificity(self): return self.TN / (self.TN + self.FP)           # độ đặc hiệu
    @property
    def precision(self): return self.TP / (self.TP + self.FP) if self.TP + self.FP else float("nan")

    def to_dict(self):
        d = asdict(self)
        d.update(accuracy=self.accuracy, sensitivity=self.sensitivity,
                 specificity=self.specificity, precision=self.precision)
        return d


class RocAnalyzer:
    """Quét ngưỡng d <= T để dựng đường cong ROC, AUC và ngưỡng Youden."""

    def __init__(self, d, y):
        self.d, self.y = d, y
        self.thresholds = np.r_[-1, np.unique(d)]
        P, N = y.sum(), (~y).sum()
        self.tpr = np.array([((d <= t) & y).sum() / P for t in self.thresholds])
        self.fpr = np.array([((d <= t) & ~y).sum() / N for t in self.thresholds])

    @property
    def auc(self) -> float:
        o = np.argsort(self.fpr, kind="stable")
        f, t = self.fpr[o], self.tpr[o]
        return float(np.sum((f[1:] - f[:-1]) * (t[1:] + t[:-1]) / 2))      # hình thang

    @property
    def youden_threshold(self):
        return self.thresholds[np.argmax(self.tpr - self.fpr)]
