# Báo cáo tự động – Bài thực hành 4 (Wavelet hash)

Tạo lúc chạy chương trình. Dữ liệu: 144 ảnh, 16 nhóm; 10296 cặp, trong đó 576 cặp tương tự.

## Phần II – phương pháp chính
Cấu hình: `ll_median db4 L3`, mã 196 bit.
- Khoảng cách Hamming trung bình: cặp tương tự 23.5 bit, không tương tự 99.1 bit (độ dài mã 196).
- Ngưỡng chọn trên nhóm huấn luyện: T = 64 bit.
- Tập kiểm tra: TP=241, FN=47, FP=144, TN=2124.
- Độ chính xác 0.9253 | độ nhạy 0.8368 | độ đặc hiệu 0.9365 | precision 0.6260 | AUC 0.9475.
- Lưu ý: cặp tương tự là thiểu số nên accuracy một mình dễ gây hiểu lầm; xem thêm độ nhạy, độ đặc hiệu, AUC.

## Phần III.1 – khảo sát phương pháp băm
5 cấu hình AUC cao nhất:

```
     method wavelet  level  bits    AUC
 quant_gray     db2      2   648 0.9989
 quant_gray bior1.3      2   722 0.9979
detail_sign    haar      3   192 0.9979
 quant_gray    haar      2   512 0.9976
  ll_median     db2      2   324 0.9971
```

- `slide_parity` (chẵn/lẻ hệ số theo ý slide): AUC từ 0.653 đến 0.710.
- Cấu hình mặc định của slide (`db4` mức 3, `ll_median`): AUC 0.9434; aHash (không dùng wavelet): AUC 0.9826.
- Cấu hình gọn (≤ 256 bit) tốt nhất: `detail_sign haar L3` – AUC 0.9979, 192 bit.

Độ nhạy theo từng biến thể (ngưỡng kiểm định chéo 2 phần):

```
         chinh: ll_median db4 L3  detail_sign haar L3 (ung dung)  quant_gray db2 L2  aHash (doi chung)
blur                        1.00                            1.00               1.00               1.00
bright                      1.00                            1.00               1.00               1.00
crop90                      0.81                            0.88               0.94               0.94
dark                        1.00                            1.00               1.00               1.00
jpeg20                      1.00                            1.00               1.00               1.00
noise10                     1.00                            1.00               1.00               1.00
noise25                     1.00                            1.00               1.00               1.00
rot5                        0.44                            1.00               1.00               0.81
```

## Phần III.2 – tìm kiếm ảnh
Chỉ mục: 144 ảnh, 192 bit/ảnh, 3.4 KB. Truy vấn lần lượt từng ảnh (bỏ chính nó):

```
                               Top-1  Top-5  Precision@8    mAP
chinh: ll_median db4 L3        0.889  0.917        0.817  0.842
ung dung: detail_sign haar L3  0.993  1.000        0.982  0.990
quant_gray db2 L2              0.993  1.000        0.990  0.994
aHash (doi chung)              0.965  0.979        0.947  0.955
```

Kết quả demo 3 ảnh truy vấn: xem `ket_qua_tim_kiem_demo.txt` và `figures/07_demo_tim_kiem_anh.png`.

## Lưu ý
- Dữ liệu tự tạo từ ảnh mẫu với 8 biến đổi đơn giản nên kết quả lạc quan hơn ảnh thật; chưa kiểm tra xoay lớn, đổi tỉ lệ, đổi góc nhìn.
- Cấu hình tốt nhất được chọn trên chính tập dữ liệu này; ngưỡng phân loại thì chọn trên nhóm huấn luyện và đánh giá trên nhóm kiểm tra.
