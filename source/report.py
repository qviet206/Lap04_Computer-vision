"""Vẽ hình và ghi kết quả ra thư mục output/."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from PIL import Image

from metrics import RocAnalyzer


class Reporter:
    def __init__(self, fig_dir, out_dir):
        self.fig_dir, self.out_dir = fig_dir, out_dir
        plt.rcParams.update({"figure.dpi": 110})

    def _save(self, name):
        plt.tight_layout()
        plt.savefig(self.fig_dir / name, dpi=130, bbox_inches="tight")
        plt.close()

    def gallery(self, ds, group="astronaut"):
        g0 = ds.meta[ds.meta.group == group]
        fig, axes = plt.subplots(2, 5, figsize=(12, 5.2))
        for ax, (_, r) in zip(axes.ravel(), g0.iterrows()):
            ax.imshow(Image.open(r.path)); ax.set_title(r.variant); ax.axis("off")
        axes.ravel()[-1].axis("off")
        fig.suptitle(f"Một nhóm ảnh tương tự: '{group}' (gốc + 8 biến thể)")
        self._save("01_nhom_anh_tuong_tu.png")

        names = list(ds.meta.group.unique())[:16]
        fig, axes = plt.subplots(2, 8, figsize=(15, 4.6))
        for ax, n in zip(axes.ravel(), names):
            ax.imshow(Image.open(ds.meta[(ds.meta.group == n) & (ds.meta.variant == "orig")].path.iloc[0])); ax.set_title(n, fontsize=7); ax.axis("off")
        fig.suptitle("Các ảnh gốc khác nhau (cặp không tương tự)")
        self._save("02_cac_anh_goc.png")

    def decomposition(self, gray, wavelet="db4", level=3):
        c = pywt.wavedec2(gray, wavelet, level=level)
        fig, axes = plt.subplots(1, 5, figsize=(14, 3))
        axes[0].imshow(gray, cmap="gray"); axes[0].set_title("Ảnh xám 64x64")
        axes[1].imshow(c[0], cmap="gray"); axes[1].set_title(f"LL (cA{level})")
        for ax, d, t in zip(axes[2:], c[1], ["LH (cH)", "HL (cV)", "HH (cD)"]):
            ax.imshow(np.abs(d), cmap="magma"); ax.set_title(t + " |chi tiết|")
        for ax in axes: ax.axis("off")
        self._save("03_phan_ra_wavelet.png")

    def histogram(self, dist, y, L, title):
        plt.figure(figsize=(7, 3.6))
        bins = np.linspace(0, L, 50)
        plt.hist(dist[~y], bins, alpha=.6, density=True, label="không tương tự")
        plt.hist(dist[y], bins, alpha=.6, density=True, label="tương tự")
        plt.xlabel("khoảng cách Hamming"); plt.ylabel("mật độ"); plt.title(title); plt.legend()
        self._save("04_phan_bo_khoang_cach.png")

    def roc(self, roc: RocAnalyzer, T, op_point, label):
        plt.figure(figsize=(5, 4.6))
        plt.plot(roc.fpr, roc.tpr, lw=2, label=f"{label} – AUC = {roc.auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="đoán ngẫu nhiên")
        plt.scatter(*op_point, c="r", zorder=5, label=f"ngưỡng T = {T:.0f}")
        plt.xlabel("1 − độ đặc hiệu (FPR)"); plt.ylabel("Độ nhạy (TPR)")
        plt.title("Đường cong ROC – tập kiểm tra"); plt.legend(loc="lower right", fontsize=8); plt.grid(alpha=.3)
        self._save("05_roc_phuong_phap_chinh.png")

    def survey_heatmap(self, survey: pd.DataFrame, methods, wavelets):
        fig, axes = plt.subplots(1, len(methods), figsize=(15, 3.6), sharey=True)
        for ax, m in zip(axes, methods):
            pv = survey[survey.method == m].pivot(index="wavelet", columns="level", values="AUC").loc[list(wavelets)]
            im = ax.imshow(pv.values, vmin=.65, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(pv.shape[1])); ax.set_xticklabels(pv.columns)
            ax.set_yticks(range(len(wavelets))); ax.set_yticklabels(pv.index)
            ax.set_xlabel("mức phân rã"); ax.set_title(m)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center",
                            color="w" if pv.values[i, j] < .93 else "k", fontsize=8)
        fig.colorbar(im, ax=axes, shrink=.8, label="AUC")
        plt.savefig(self.fig_dir / "06_khao_sat_AUC.png", dpi=130, bbox_inches="tight"); plt.close()

    def search_demo(self, engine, queries, group_of_path, k=5):
        fig, axes = plt.subplots(len(queries), k + 1, figsize=(2.2 * (k + 1), 2.4 * len(queries)))
        for r, (qf, truth) in enumerate(queries):
            axes[r, 0].imshow(Image.open(qf)); axes[r, 0].set_title("TRUY VẤN", fontsize=9, color="navy"); axes[r, 0].axis("off")
            for c, (p, d) in enumerate(engine.search(qf, k), start=1):
                ok = group_of_path[p] == truth
                axes[r, c].imshow(Image.open(p)); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                axes[r, c].set_title(f"d = {d}\n{p.split('/')[-1].rsplit('.', 1)[0][:20]}", fontsize=7)
                for s in axes[r, c].spines.values():
                    s.set_edgecolor("green" if ok else "red"); s.set_linewidth(3)
        self._save("07_demo_tim_kiem_anh.png")
