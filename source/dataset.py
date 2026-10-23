"""Chuẩn bị dữ liệu: tạo ảnh (DatasetBuilder) và nạp/ghép cặp (ImageDataset)."""
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from skimage import data as skdata

from config import Config


class DatasetBuilder:
    """Sinh bộ ảnh: mỗi ảnh gốc + 8 biến thể (cùng nhóm = 'tương tự')."""
    VARIANTS = ("noise10", "noise25", "blur", "bright", "dark", "rot5", "crop90", "jpeg20")

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _variants(self, img: Image.Image) -> dict:
        a = np.asarray(img, dtype=np.float64)
        out = {}
        for s in (10, 25):
            out[f"noise{s}"] = Image.fromarray(np.clip(a + self.rng.normal(0, s, a.shape), 0, 255).astype(np.uint8))
        out["blur"] = img.filter(ImageFilter.GaussianBlur(2))
        out["bright"] = ImageEnhance.Brightness(img).enhance(1.4)
        out["dark"] = ImageEnhance.Brightness(img).enhance(0.7)
        out["rot5"] = img.rotate(5, resample=Image.BICUBIC)
        w, h = img.size
        dw, dh = int(w * 0.05), int(h * 0.05)
        out["crop90"] = img.crop((dw, dh, w - dw, h - dh)).resize((w, h), Image.BICUBIC)
        out["jpeg20"] = img                       # nén JPEG q=20 khi lưu
        return out

    def build(self) -> None:
        d = self.cfg.data_dir
        d.mkdir(parents=True, exist_ok=True)
        for name in self.cfg.sources:
            im = getattr(skdata, name)()
            im = im[0] if isinstance(im, tuple) else im
            im = Image.fromarray(im).convert("RGB")
            im.thumbnail((256, 256))
            im.save(d / f"{name}__orig.png")
            for v, vi in self._variants(im).items():
                if v == "jpeg20":
                    vi.save(d / f"{name}__{v}.jpg", quality=20)
                else:
                    vi.save(d / f"{name}__{v}.png")

    def build_queries(self, groups_info) -> list:
        """Tạo các ảnh truy vấn kết hợp nhiều biến đổi (không nằm trong chỉ mục)."""
        qd = self.cfg.query_dir
        qd.mkdir(parents=True, exist_ok=True)
        rq = np.random.default_rng(7)
        src = lambda n: Image.open(self.cfg.data_dir / f"{n}__orig.png").convert("RGB")

        def noisy(im, s):
            a = np.asarray(im, float)
            return Image.fromarray(np.clip(a + rq.normal(0, s, a.shape), 0, 255).astype(np.uint8))

        q1 = noisy(ImageEnhance.Brightness(src("astronaut")).enhance(1.2), 15)
        q1.save(qd / "q1_astronaut_noise_bright_jpeg40.jpg", quality=40)
        q2 = src("horse").filter(ImageFilter.GaussianBlur(1))
        w, h = q2.size
        q2.crop((int(.07 * w), int(.07 * h), int(.93 * w), int(.93 * h))).resize((w, h)).save(qd / "q2_horse_blur_crop.png")
        noisy(src("coffee").rotate(-3, resample=Image.BICUBIC), 15).save(qd / "q3_coffee_rot_noise.png")
        return [(qd / "q1_astronaut_noise_bright_jpeg40.jpg", "astronaut"),
                (qd / "q2_horse_blur_crop.png", "horse"),
                (qd / "q3_coffee_rot_noise.png", "coffee")]


class ImageDataset:
    """Tập ảnh + nhãn nhóm; cung cấp ảnh xám chuẩn hoá, danh sách cặp và nhãn ground truth."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        rows = []
        for f in sorted(cfg.data_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                g, v = f.stem.split("__")
                rows.append((str(f), g, v))
        self.meta = pd.DataFrame(rows, columns=["path", "group", "variant"])
        self.groups = np.array(self.meta.group.tolist())
        self.grays = [self.load_gray(p, cfg.size) for p in self.meta.path]
        self.iu = np.triu_indices(len(self.meta), 1)           # mọi cặp i < j
        self.y = self.groups[self.iu[0]] == self.groups[self.iu[1]]   # True = tương tự

    @staticmethod
    def load_gray(path, size) -> np.ndarray:
        im = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
        return np.asarray(im, dtype=np.float64) / 255.0

    def __len__(self):
        return len(self.meta)

    def index_of(self, group, variant) -> int:
        return int(self.meta.index[(self.meta.group == group) & (self.meta.variant == variant)][0])

    def split_groups(self):
        """Chia nhóm thành huấn luyện / kiểm tra; trả về (tập nhóm huấn luyện, mask cặp huấn luyện, mask cặp kiểm tra)."""
        rng = np.random.default_rng(self.cfg.seed)
        gl = np.array(sorted(self.meta.group.unique()))
        rng.shuffle(gl)
        train = set(map(str, gl[: len(gl) // 2]))
        is_tr = np.array([g in train for g in self.groups])
        m_tr = is_tr[self.iu[0]] & is_tr[self.iu[1]]
        m_te = ~is_tr[self.iu[0]] & ~is_tr[self.iu[1]]
        return train, m_tr, m_te
