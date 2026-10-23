# Bài thực hành 4 – So sánh sự tương đồng của ảnh bằng Wavelet hash

## Cấu trúc thư mục

```
wavelet_lab/
├── data/
├── output/
├── source/
│   ├── config.py     Config
│   ├── dataset.py    DatasetBuilder
│   │                 ImageDataset
│   ├── hashers.py    Hasher (ABC) -> WaveletHasher -> LLMedianHasher, DetailSignHasher,
│   │                 QuantGrayHasher, SlideParityHasher; AverageHasher; create_hasher()
│   ├── metrics.py    HammingComparator, BinaryMetrics, RocAnalyzer
│   ├── survey.py     HashSurvey
│   ├── search.py     WaveletImageSearch, RetrievalEvaluator
│   ├── report.py     Reporter
│   └── main.py       Lab4Pipeline
├── requirements.txt
└── README.md
```

## Cách chạy

```
pip install -r requirements.txt
python source/main.py                               # chạy toàn bộ, ghi kết quả vào output/
python source/main.py --search data/queries/q2_horse_blur_crop.png -k 5   # tìm ảnh giống
```

Muốn dùng ảnh riêng: thay nội dung `data/` bằng ảnh đặt tên `<nhóm>__<biến thể>.jpg`.
