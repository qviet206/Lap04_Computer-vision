"""Cấu hình chung"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    root: Path = Path(__file__).resolve().parent.parent
    seed: int = 42
    size: int = 64                      # ảnh chuẩn hoá về size x size (luỹ thừa của 2)
    sources: tuple = ("astronaut", "coffee", "chelsea", "rocket", "camera", "coins", "moon",
                      "hubble_deep_field", "immunohistochemistry", "brick", "gravel", "horse",
                      "clock", "colorwheel", "page", "cell")
    wavelets: tuple = ("haar", "db2", "db4", "sym4", "bior1.3")
    levels: tuple = (2, 3, 4)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def query_dir(self) -> Path:
        return self.root / "data" / "queries"

    @property
    def output_dir(self) -> Path:
        d = self.root / "output"
        d.mkdir(exist_ok=True)
        return d

    @property
    def fig_dir(self) -> Path:
        d = self.output_dir / "figures"
        d.mkdir(exist_ok=True)
        return d
