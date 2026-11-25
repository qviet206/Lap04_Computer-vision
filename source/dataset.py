
import io
import re
import zipfile
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from config import Config

URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
POSES = (0, 4, 8, 12, 16, 20, 24, 28, 32)
QUERY_POSE = 6

class DatasetBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

    def build(self):
        archive = self.cfg.root / "coil-20-proc.zip"
        if not archive.exists():
            print("Đang tải COIL-20 từ Columbia University...")
            try:
                with urllib.request.urlopen(URL, timeout=60) as response:
                    payload = response.read()
                with zipfile.ZipFile(io.BytesIO(payload)) as z:
                    if not z.namelist():
                        raise ValueError("ZIP rỗng")
                archive.write_bytes(payload)
            except Exception as exc:
                raise RuntimeError("Không tải được COIL-20. Tải coil-20-proc.zip theo README và đặt cạnh requirements.txt.") from exc
        with zipfile.ZipFile(archive) as z:
            files = {}
            for entry in z.namelist():
                match = re.fullmatch(r"obj(\d+)__(\d+)\.png", Path(entry).name, re.I)
                if match:
                    files[tuple(map(int, match.groups()))] = entry
            tasks = []
            for obj in range(1, 17):
                for pose in POSES:
                    tasks.append((obj, pose, self.cfg.data_dir / f"obj{obj:02d}__view{pose*5:03d}.png"))
            for obj in (1, 2, 3):
                tasks.append((obj, QUERY_POSE, self.cfg.query_dir / f"obj{obj:02d}__view{QUERY_POSE*5:03d}.png"))
            missing = [(o,p) for o,p,_ in tasks if (o,p) not in files]
            if missing:
                raise ValueError(f"ZIP thiếu ảnh COIL-20: {missing[:5]}")
            for o,p,target in tasks:
                target.parent.mkdir(parents=True, exist_ok=True)
                with Image.open(io.BytesIO(z.read(files[o,p]))) as im:
                    im.load()
                    im.save(target)

    def build_queries(self, groups_info=None):
        queries = [(self.cfg.query_dir / f"obj{o:02d}__view030.png", f"obj{o:02d}") for o in (1,2,3)]
        if not all(p.is_file() for p,_ in queries):
            self.build()
        return queries


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
