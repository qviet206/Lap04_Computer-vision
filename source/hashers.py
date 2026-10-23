"""Biến đổi wavelet, lượng tử hoá và tạo mã băm.
Lớp cơ sở trừu tượng WaveletHasher + các lớp con cho từng phương pháp băm."""
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pywt

warnings.filterwarnings("ignore")


class Hasher(ABC):
    """Giao diện chung: ảnh xám (2D, [0,1]) -> vector bit uint8."""
    name = "base"

    @abstractmethod
    def hash(self, gray: np.ndarray) -> np.ndarray: ...

    def hash_many(self, grays) -> np.ndarray:
        return np.array([self.hash(g) for g in grays])

    def describe(self) -> str:
        return self.name


class WaveletHasher(Hasher):
    """Băm dựa trên wavedec2: biến đổi -> (lớp con) lượng tử hoá -> bit."""

    def __init__(self, wavelet="haar", level=3):
        self.wavelet, self.level = wavelet, level

    def decompose(self, gray):
        """[cA_n, (cH_n,cV_n,cD_n), ..., (cH_1,cV_1,cD_1)]"""
        return pywt.wavedec2(gray, self.wavelet, level=self.level)

    def hash(self, gray):
        return self.quantize(self.decompose(gray)).astype(np.uint8)

    @abstractmethod
    def quantize(self, coeffs) -> np.ndarray: ...

    def describe(self):
        return f"{self.name} {self.wavelet} L{self.level}"


class LLMedianHasher(WaveletHasher):
    """[Phương pháp chính] băng LL so với trung vị -> 1 bit/hệ số."""
    name = "ll_median"

    def quantize(self, coeffs):
        ll = coeffs[0]
        return (ll > np.median(ll)).ravel()


class DetailSignHasher(WaveletHasher):
    """Dấu của hệ số chi tiết cH, cV, cD ở mức sâu nhất (mã hoá hướng cạnh)."""
    name = "detail_sign"

    def quantize(self, coeffs):
        return np.concatenate([(d > 0).ravel() for d in coeffs[1]])


class QuantGrayHasher(WaveletHasher):
    """LL lượng tử hoá 2**bits mức theo phân vị, mã hoá Gray (mức kề nhau chỉ khác 1 bit)."""
    name = "quant_gray"

    def __init__(self, wavelet="db2", level=2, bits=2):
        super().__init__(wavelet, level)
        self.bits = bits

    def quantize(self, coeffs):
        c = coeffs[0].ravel()
        edges = np.quantile(c, np.linspace(0, 1, 2 ** self.bits + 1)[1:-1])
        idx = np.searchsorted(edges, c)
        gray = idx ^ (idx >> 1)
        sh = np.arange(self.bits - 1, -1, -1)
        return ((gray[:, None] >> sh) & 1).ravel()


class SlideParityHasher(WaveletHasher):
    """Làm đúng ý code slide 17: mọi hệ số, làm tròn 2 chữ số, bit = chẵn/lẻ (phương pháp đối chứng yếu)."""
    name = "slide_parity"

    def quantize(self, coeffs):
        flat = np.concatenate([coeffs[0].ravel()] + [d.ravel() for t in coeffs[1:] for d in t])
        return np.round(flat * 100).astype(np.int64) % 2


class AverageHasher(Hasher):
    """Đối chứng không dùng wavelet: ảnh 8x8 so với trung bình."""
    name = "ahash"

    def hash(self, gray):
        k = gray.shape[0] // 8
        small = gray.reshape(8, k, 8, k).mean((1, 3))
        return (small > small.mean()).ravel().astype(np.uint8)


_REGISTRY = {c.name: c for c in (LLMedianHasher, DetailSignHasher, QuantGrayHasher, SlideParityHasher, AverageHasher)}


def create_hasher(method: str, wavelet="haar", level=3) -> Hasher:
    """Factory: tạo hasher theo tên phương pháp."""
    cls = _REGISTRY[method]
    return cls() if cls is AverageHasher else cls(wavelet=wavelet, level=level)
