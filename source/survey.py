"""Khảo sát các phương pháp băm wavelet."""
import numpy as np
import pandas as pd

from config import Config
from dataset import ImageDataset
from hashers import create_hasher
from metrics import HammingComparator, RocAnalyzer


class HashSurvey:
    METHODS = ("slide_parity", "ll_median", "detail_sign", "quant_gray")

    def __init__(self, ds: ImageDataset, cfg: Config):
        self.ds, self.cfg = ds, cfg
        self._cache = {}

    def codes(self, method, wavelet="haar", level=3) -> np.ndarray:
        key = (method, wavelet, level)
        if key not in self._cache:
            self._cache[key] = create_hasher(method, wavelet, level).hash_many(self.ds.grays)
        return self._cache[key]

    def pair_distances(self, method, wavelet="haar", level=3) -> np.ndarray:
        return HammingComparator.matrix(self.codes(method, wavelet, level))[self.ds.iu]

    def auc(self, method, wavelet="haar", level=3) -> float:
        return RocAnalyzer(self.pair_distances(method, wavelet, level), self.ds.y).auc

    def run(self) -> pd.DataFrame:
        rows = []
        for m in self.METHODS:
            for w in self.cfg.wavelets:
                for lv in self.cfg.levels:
                    rows.append(dict(method=m, wavelet=w, level=lv,
                                     bits=self.codes(m, w, lv).shape[1], AUC=self.auc(m, w, lv)))
        rows.append(dict(method="ahash", wavelet="-", level=0, bits=64, AUC=self.auc("ahash")))
        return pd.DataFrame(rows)

    def per_variant_sensitivity(self, configs: dict) -> pd.DataFrame:
        """Độ nhạy theo từng biến thể (cặp gốc–biến thể), ngưỡng Youden kiểm định chéo 2 phần."""
        train, m_tr, m_te = self.ds.split_groups()
        out = {}
        for label, kw in configs.items():
            d = self.pair_distances(**kw)
            T_tr = RocAnalyzer(d[m_tr], self.ds.y[m_tr]).youden_threshold
            T_te = RocAnalyzer(d[m_te], self.ds.y[m_te]).youden_threshold
            Dm = HammingComparator.matrix(self.codes(**kw))
            col = {}
            for v in self.ds.meta.variant.unique():
                if v == "orig":
                    continue
                hit = []
                for g in self.ds.meta.group.unique():
                    i, j = self.ds.index_of(g, "orig"), self.ds.index_of(g, v)
                    hit.append(Dm[i, j] <= (T_te if g in train else T_tr))   # ngưỡng từ nửa còn lại
                col[v] = float(np.mean(hit))
            out[label] = col
        return pd.DataFrame(out)
