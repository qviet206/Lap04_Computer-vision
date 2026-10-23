"""Ứng dụng tìm kiếm ảnh dựa trên hàm băm wavelet."""
import json

import numpy as np
import pandas as pd

from dataset import ImageDataset
from hashers import create_hasher, Hasher
from metrics import HammingComparator


class WaveletImageSearch:
    """Chỉ mục mã băm (đóng gói 8 bit/byte) + truy vấn k ảnh gần nhất theo Hamming."""

    def __init__(self, hasher: Hasher, size=64, spec=None):
        self.hasher, self.size, self.spec = hasher, size, spec or {}
        self.paths, self.codes, self.nbits = [], None, None

    def _bits(self, path):
        return self.hasher.hash(ImageDataset.load_gray(path, self.size))

    def add(self, paths):
        new = np.array([self._bits(p) for p in paths])
        self.nbits = new.shape[1]
        packed = np.packbits(new, axis=1)
        self.codes = packed if self.codes is None else np.vstack([self.codes, packed])
        self.paths += [str(p) for p in paths]

    def distances(self, query_path):
        return HammingComparator.to_query(self.codes, self.nbits, self._bits(query_path))

    def search(self, query_path, k=5):
        d = self.distances(query_path)
        order = np.argsort(d, kind="stable")[:k]
        return [(self.paths[i], int(d[i])) for i in order]

    def save(self, f):
        np.savez_compressed(f, paths=np.array(self.paths), codes=self.codes, nbits=self.nbits,
                            spec=json.dumps(dict(self.spec, size=self.size)))

    @classmethod
    def load(cls, f):
        z = np.load(f, allow_pickle=False)
        spec = json.loads(str(z["spec"]))
        e = cls(create_hasher(spec["method"], spec.get("wavelet", "haar"), spec.get("level", 3)), spec["size"], spec)
        e.paths, e.codes, e.nbits = list(z["paths"]), z["codes"], int(z["nbits"])
        return e


class RetrievalEvaluator:
    """Truy vấn lần lượt từng ảnh (bỏ chính nó): Top-1, Top-5, Precision@8, mAP."""

    @staticmethod
    def evaluate(H: np.ndarray, groups: np.ndarray) -> dict:
        D = HammingComparator.matrix(H).astype(float)
        np.fill_diagonal(D, np.inf)
        n = len(groups)
        order = np.argsort(D, axis=1, kind="stable")[:, : n - 1]
        rel = groups[order] == groups[:, None]
        prec = np.cumsum(rel, 1) / np.arange(1, n)
        ap = (prec * rel).sum(1) / rel.sum(1)
        return {"Top-1": rel[:, 0].mean(), "Top-5": rel[:, :5].any(1).mean(),
                "Precision@8": rel[:, :8].mean(), "mAP": ap.mean()}
