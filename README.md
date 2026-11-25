# Lab04 – So sánh ảnh bằng Wavelet hash (COIL-20)

Dữ liệu mới: Columbia Object Image Library (COIL-20), ảnh chụp vật thể từ nhiều góc, không xoay/làm mờ ảnh gốc để tạo dữ liệu. Nguồn: https://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php — S. A. Nene, S. K. Nayar, H. Murase, 1996.

Chọn 16 vật thể đầu tiên; mỗi vật thể có 9 ảnh ở góc 0, 20, 40, 60, 80, 100, 120, 140, 160 độ. Tổng 144 ảnh, 10.296 cặp, 576 cặp cùng vật thể, 9.720 cặp khác vật thể. Ba truy vấn ở góc 30 độ của vật thể 1–3, không nằm trong chỉ mục. Không phải dữ liệu khuôn mặt theo độ tuổi.

## Chạy trên Windows (CMD hoặc terminal VS Code)

```cmd
python -m pip install -r requirements.txt
python source/main.py
python source/main.py --search data/queries/obj01__view030.png -k 5
```

Lần đầu chương trình tự tải COIL-20. Nếu tải lỗi, tải file `coil-20-proc.zip` từ trang nguồn trên (hoặc https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip), đặt cạnh `requirements.txt` (không cần giải nén), chạy lại `python source/main.py`.

`data/` chứa ảnh nhập từ ZIP. `output/` chứa báo cáo, biểu đồ ROC, khảo sát 61 cấu hình và chỉ mục. Kết quả cũ đã được xóa vì không còn hợp lệ. Bộ ảnh và kết quả mới chỉ xuất hiện sau khi chạy thành công. Không phát hành lại ảnh nguồn trong ZIP dự án; kiểm tra điều khoản nguồn trước khi đăng ảnh lên GitHub.

Nhãn: cùng vật thể = tương đồng, khác vật thể = không tương đồng. Wavelet hash không phải mô hình nhận dạng ngữ nghĩa; độ chính xác có thể giảm khi góc nhìn thay đổi. Không sử dụng số liệu cũ cho báo cáo mới.
