import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config
from dataset import DatasetBuilder, ImageDataset
from hashers import create_hasher
from metrics import BinaryMetrics, HammingComparator, RocAnalyzer
from report import Reporter
from search import RetrievalEvaluator, WaveletImageSearch
from survey import HashSurvey

MAIN = dict(method="ll_median", wavelet="db4", level=3)       # phương pháp chính 
APP = dict(method="detail_sign", wavelet="haar", level=3)     # cấu hình dùng cho ứng dụng tìm kiếm


class Lab4Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.builder = DatasetBuilder(cfg)
        if not any(cfg.data_dir.glob("*__*.*")):
            print("Chưa có dữ liệu -> tạo thư mục data/ ...")
            self.builder.build()
        self.ds = ImageDataset(cfg)
        self.rep = Reporter(cfg.fig_dir, cfg.output_dir)
        self.survey = HashSurvey(self.ds, cfg)
        self.results = {}

    # ---------------- Phần II ----------------
    def part2(self):
        ds, out = self.ds, self.cfg.output_dir
        print(f"[Bước 1] {len(ds)} ảnh, {ds.meta.group.nunique()} nhóm | {len(ds.y)} cặp: "
              f"{ds.y.sum()} tương tự, {(~ds.y).sum()} không tương tự")
        self.rep.gallery(ds)
        self.rep.decomposition(ds.grays[ds.index_of("astronaut", "orig")])

        hasher = create_hasher(**MAIN)                                   # Bước 2-3
        H = hasher.hash_many(ds.grays)
        L = H.shape[1]
        dist = HammingComparator.matrix(H)[ds.iu]                        # Bước 4
        print(f"[Bước 3-4] mã {L} bit | d TB tương tự = {dist[ds.y].mean():.1f}, không tương tự = {dist[~ds.y].mean():.1f}")
        self.rep.histogram(dist, ds.y, L, "Phân bố khoảng cách – db4, mức 3, LL median")

        train, m_tr, m_te = ds.split_groups()                            # Bước 5
        T = RocAnalyzer(dist[m_tr], ds.y[m_tr]).youden_threshold
        bm = BinaryMetrics.at_threshold(dist[m_te], ds.y[m_te], T)
        roc = RocAnalyzer(dist[m_te], ds.y[m_te])
        self.rep.roc(roc, T, (bm.FP / (bm.FP + bm.TN), bm.sensitivity), "wavelet hash (db4, mức 3, LL)")
        res = dict(config=hasher.describe(), bits=L, threshold=float(T), AUC=roc.auc, **bm.to_dict(),
                   train_groups=sorted(train))
        self.results["part2"] = res
        self.results["n_pairs"] = (len(ds.y), int(ds.y.sum()))
        self.results["mean_d"] = (float(dist[ds.y].mean()), float(dist[~ds.y].mean()), L)
        (out / "ket_qua_phan_II.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
        print(f"[Bước 5] T = {T:.0f} | accuracy {bm.accuracy:.4f} | độ nhạy {bm.sensitivity:.4f} | "
              f"độ đặc hiệu {bm.specificity:.4f} | precision {bm.precision:.4f} | AUC {roc.auc:.4f}")

    # ---------------- Phần III.1 ----------------
    def part3_survey(self):
        out = self.cfg.output_dir
        df = self.survey.run()
        df.to_csv(out / "khao_sat_wavelet_hash.csv", index=False)
        self.results["survey"] = df
        self.rep.survey_heatmap(df, HashSurvey.METHODS, self.cfg.wavelets)
        print("[III.1] AUC theo phương pháp:\n", df.groupby("method").AUC.agg(["min", "mean", "max"]).round(4))
        print(df.sort_values("AUC", ascending=False).head(10).round(4).to_string(index=False))

        configs = {"chinh: ll_median db4 L3": MAIN, "detail_sign haar L3 (ung dung)": APP,
                   "quant_gray db2 L2": dict(method="quant_gray", wavelet="db2", level=2),
                   "aHash (doi chung)": dict(method="ahash")}
        tab = self.survey.per_variant_sensitivity(configs).round(2)
        tab.to_csv(out / "do_nhay_theo_bien_the.csv")
        self.results["per_variant"] = tab
        print("[III.1] Độ nhạy theo biến thể:\n", tab.to_string())

    # ---------------- Phần III.2 ----------------
    def part3_search(self):
        ds, cfg = self.ds, self.cfg
        hasher = create_hasher(**APP)
        engine = WaveletImageSearch(hasher, cfg.size, APP)
        engine.add(list(ds.meta.path))
        engine.save(cfg.output_dir / "wavelet_index.npz")
        engine = WaveletImageSearch.load(cfg.output_dir / "wavelet_index.npz")
        print(f"[III.2] chỉ mục {len(engine.paths)} ảnh, {engine.nbits} bit, {engine.codes.nbytes / 1024:.1f} KB")

        rows = {}
        for name, kw in {"chinh: ll_median db4 L3": MAIN, "ung dung: detail_sign haar L3": APP,
                         "quant_gray db2 L2": dict(method="quant_gray", wavelet="db2", level=2),
                         "aHash (doi chung)": dict(method="ahash")}.items():
            rows[name] = RetrievalEvaluator.evaluate(self.survey.codes(**kw), ds.groups)
        ev = pd.DataFrame(rows).T.round(3)
        ev.to_csv(cfg.output_dir / "danh_gia_tim_kiem.csv")
        self.results["retrieval"] = ev
        self.results["index_info"] = (len(engine.paths), engine.nbits, engine.codes.nbytes / 1024)
        print(ev.to_string())

        queries = self.builder.build_queries(None)
        self.rep.search_demo(engine, queries, dict(zip(ds.meta.path, ds.meta.group)))
        with open(cfg.output_dir / "ket_qua_tim_kiem_demo.txt", "w", encoding="utf-8") as f:
            for qf, truth in queries:
                f.write(f"Truy vấn {Path(qf).name} (đúng: {truth})\n")
                for p, d in engine.search(qf, 5):
                    f.write(f"   d={d:3d}  {Path(p).name}\n")

    def write_report(self):
        """Sinh output/bao_cao.md từ số liệu của chính lần chạy này."""
        r = self.results
        p, sv = r["part2"], r["survey"]
        best = sv.sort_values("AUC", ascending=False).head(5)
        compact = sv[(sv.bits <= 256) & (sv.method != "ahash")].sort_values("AUC", ascending=False).iloc[0]
        main_row = sv[(sv.method == MAIN["method"]) & (sv.wavelet == MAIN["wavelet"]) & (sv.level == MAIN["level"])].iloc[0]
        ah = sv[sv.method == "ahash"].AUC.iloc[0]
        parity = sv[sv.method == "slide_parity"].AUC
        md = lambda df: df.round(4).to_markdown() if hasattr(df, "to_markdown") else df.round(4).to_string()
        lines = [
            "# Báo cáo tự động – Bài thực hành 4 (Wavelet hash)", "",
            f"Tạo lúc chạy chương trình. Dữ liệu: {len(self.ds)} ảnh, {self.ds.meta.group.nunique()} nhóm; "
            f"{r['n_pairs'][0]} cặp, trong đó {r['n_pairs'][1]} cặp tương tự.", "",
            "## Phần II – phương pháp chính", f"Cấu hình: `{p['config']}`, mã {p['bits']} bit.",
            f"- Khoảng cách Hamming trung bình: cặp tương tự {r['mean_d'][0]:.1f} bit, không tương tự {r['mean_d'][1]:.1f} bit (độ dài mã {r['mean_d'][2]}).",
            f"- Ngưỡng chọn trên nhóm huấn luyện: T = {p['threshold']:.0f} bit.",
            f"- Tập kiểm tra: TP={p['TP']}, FN={p['FN']}, FP={p['FP']}, TN={p['TN']}.",
            f"- Độ chính xác {p['accuracy']:.4f} | độ nhạy {p['sensitivity']:.4f} | độ đặc hiệu {p['specificity']:.4f} | precision {p['precision']:.4f} | AUC {p['AUC']:.4f}.",
            "- Lưu ý: cặp tương tự là thiểu số nên accuracy một mình dễ gây hiểu lầm; xem thêm độ nhạy, độ đặc hiệu, AUC.", "",
            "## Phần III.1 – khảo sát phương pháp băm", "5 cấu hình AUC cao nhất:", "", "```", best.round(4).to_string(index=False), "```", "",
            f"- `slide_parity` (chẵn/lẻ hệ số theo ý slide): AUC từ {parity.min():.3f} đến {parity.max():.3f}.",
            f"- Cấu hình mặc định của slide (`{MAIN['wavelet']}` mức {MAIN['level']}, `{MAIN['method']}`): AUC {main_row.AUC:.4f}; aHash (không dùng wavelet): AUC {ah:.4f}.",
            f"- Cấu hình gọn (≤ 256 bit) tốt nhất: `{compact.method} {compact.wavelet} L{compact.level}` – AUC {compact.AUC:.4f}, {int(compact.bits)} bit.", "",
            "Độ nhạy theo từng biến thể (ngưỡng kiểm định chéo 2 phần):", "", "```", r["per_variant"].to_string(), "```", "",
            "## Phần III.2 – tìm kiếm ảnh",
            f"Chỉ mục: {r['index_info'][0]} ảnh, {r['index_info'][1]} bit/ảnh, {r['index_info'][2]:.1f} KB. Truy vấn lần lượt từng ảnh (bỏ chính nó):", "",
            "```", r["retrieval"].to_string(), "```", "",
            "Kết quả demo 3 ảnh truy vấn: xem `ket_qua_tim_kiem_demo.txt` và `figures/07_demo_tim_kiem_anh.png`.", "",
            "## Lưu ý",
            "- Dữ liệu tự tạo từ ảnh mẫu với 8 biến đổi đơn giản nên kết quả lạc quan hơn ảnh thật; chưa kiểm tra xoay lớn, đổi tỉ lệ, đổi góc nhìn.",
            "- Cấu hình tốt nhất được chọn trên chính tập dữ liệu này; ngưỡng phân loại thì chọn trên nhóm huấn luyện và đánh giá trên nhóm kiểm tra.", ""]
        (self.cfg.output_dir / "bao_cao.md").write_text("\n".join(lines), encoding="utf-8")

    def run_all(self):
        self.part2()
        self.part3_survey()
        self.part3_search()
        self.write_report()
        print(f"\nXong. Kết quả nằm trong: {self.cfg.output_dir}")


def search_cli(cfg: Config, query: str, k: int):
    idx = cfg.output_dir / "wavelet_index.npz"
    if not idx.exists():
        sys.exit("Chưa có chỉ mục. Hãy chạy `python source/main.py` trước.")
    engine = WaveletImageSearch.load(idx)
    for rank, (p, d) in enumerate(engine.search(query, k), 1):
        print(f"{rank:2d}. d={d:3d}  {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", help="ảnh truy vấn (dùng chỉ mục đã tạo)")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args()
    cfg = Config()
    if a.search:
        search_cli(cfg, a.search, a.k)
    else:
        class Tee:                                   # in ra màn hình đồng thời ghi vào output/log.txt
            def __init__(self, *fs): self.fs = fs
            def write(self, x): [f.write(x) for f in self.fs]
            def flush(self): [f.flush() for f in self.fs]
        with open(cfg.output_dir / "log.txt", "w", encoding="utf-8") as log:
            sys.stdout = Tee(sys.__stdout__, log)
            try:
                Lab4Pipeline(cfg).run_all()
            finally:
                sys.stdout = sys.__stdout__
