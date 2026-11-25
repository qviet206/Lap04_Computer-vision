# Báo cáo tự động – Bài thực hành 4 (Wavelet hash)

Tạo lúc chạy chương trình. Dữ liệu: 144 ảnh, 16 nhóm; 10296 cặp, trong đó 576 cặp tương tự.

## Phần II – phương pháp chính
Cấu hình: `ll_median db4 L3`, mã 196 bit.
- Khoảng cách Hamming trung bình: cặp tương tự 42.2 bit, không tương tự 63.0 bit (độ dài mã 196).
- Ngưỡng chọn trên nhóm huấn luyện: T = 40 bit.
- Tập kiểm tra: TP=99, FN=189, FP=150, TN=2118.
- Độ chính xác 0.8674 | độ nhạy 0.3438 | độ đặc hiệu 0.9339 | precision 0.3976 | AUC 0.6469.
- Lưu ý: cặp tương tự là thiểu số nên accuracy một mình dễ gây hiểu lầm; xem thêm độ nhạy, độ đặc hiệu, AUC.

## Phần III.1 – khảo sát phương pháp băm
5 cấu hình AUC cao nhất:

```
     method wavelet  level  bits    AUC
 quant_gray    haar      2   512 0.8169
 quant_gray     db2      2   648 0.8086
detail_sign    haar      2   768 0.8046
 quant_gray    haar      3   128 0.8027
detail_sign    haar      3   192 0.8018
```

- `slide_parity` (chẵn/lẻ hệ số theo ý slide): AUC từ 0.680 đến 0.705.
- Cấu hình mặc định của slide (`db4` mức 3, `ll_median`): AUC 0.7336; aHash (không dùng wavelet): AUC 0.7883.
- Cấu hình gọn (≤ 256 bit) tốt nhất: `quant_gray haar L3` – AUC 0.8027, 128 bit.

Độ nhạy theo từng góc chụp (ngưỡng kiểm định chéo 2 phần):

```
         chinh: ll_median db4 L3  detail_sign haar L3 (ung dung)  quant_gray db2 L2  aHash (doi chung)
view020                     0.81                            0.94               0.94               0.94
view040                     0.44                            0.69               0.81               0.75
view060                     0.38                            0.50               0.50               0.50
view080                     0.25                            0.56               0.50               0.56
view100                     0.25                            0.56               0.50               0.44
view120                     0.38                            0.44               0.44               0.38
view140                     0.56                            0.50               0.62               0.56
view160                     0.56                            0.62               0.75               0.62
```

## Phần III.2 – tìm kiếm ảnh
Chỉ mục: 144 ảnh, 192 bit/ảnh, 3.4 KB. Truy vấn lần lượt từng ảnh (bỏ chính nó):

```
                               Top-1  Top-5  Precision@8    mAP
chinh: ll_median db4 L3        0.667  0.806        0.445  0.478
ung dung: detail_sign haar L3  0.826  0.931        0.527  0.571
quant_gray db2 L2              0.722  0.917        0.528  0.560
aHash (doi chung)              0.750  0.910        0.488  0.521
```

Kết quả demo 3 ảnh truy vấn: xem `ket_qua_tim_kiem_demo.txt` và `figures/07_demo_tim_kiem_anh.png`.

## Lưu ý
- COIL-20 gồm ảnh chụp cùng vật thể ở các góc khác nhau. Wavelet hash đo tương đồng thị giác, không đảm bảo nhận dạng vật thể khi góc nhìn thay đổi lớn. Chưa đánh giá khuôn mặt theo độ tuổi.
- Cấu hình tốt nhất được chọn trên chính tập dữ liệu này; ngưỡng phân loại thì chọn trên nhóm huấn luyện và đánh giá trên nhóm kiểm tra.
