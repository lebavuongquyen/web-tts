# 📊 BENCHMARK & HIỆU NĂNG PHẦN CỨNG

Đo đạc trên đoạn văn 300 từ tiếng Việt (tương đương 85 giây audio chuẩn 48kHz).

---

## 1. Bảng so sánh tổng thể

| Cấu hình | Thời gian xử lý | Tốc độ FPS | Hệ số RTF | Mức chiếm RAM / VRAM | Ghi chú |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ZeroTTS GPU FP32 Baseline** | 39.30 s | 26.5 FPS | 0.459x | 4.8 GB VRAM | Chạy trên RTX 3060 12GB |
| **ZeroTTS CPU FP32** | 76.24 s | 13.5 FPS | 0.904x | 4.4 GB RAM | CPU x86_64 đa luồng |
| **ZeroTTS CPU INT8 (mimalloc)** | 54.00 s | 19.0 FPS | 0.630x | 2.8 GB RAM | Đã bật mimalloc chống nghẽn |
| **ZeroTTS GPU + IoBinding** | ~29.50 s | ~35.0 FPS | 0.345x | 4.8 GB VRAM | Triệt tiêu 5.4 GB copy PCIe |
| **★ VITS ONNX Fine-tuned (98%)** | **4.20 s** | **~240.0 FPS** | **0.050x** | **180 MB VRAM** | 1-step forward pass siêu tốc |

---

## 2. Phân tích chi tiết nút thắt cổ chai

### ① Vì sao ZeroTTS AR tốn ~39s?
* ZeroTTS là mô hình **Autoregressive (AR)**: phải sinh từng frame âm thanh một (1.040 bước lặp cho 300 từ).
* Mỗi bước lặp gửi token và cập nhật KV Cache.

### ② Vì sao VITS ONNX chỉ tốn 4.2s (Nhanh gấp 9 lần)?
* VITS là mô hình **Non-Autoregressive**: Toàn bộ chuỗi âm vị (phonemes) được xử lý đồng thời trong **1 bước tính toán song song duy nhất**.
* Không có vòng lặp, không có KV Cache, GPU RTX 3060 giải ma trận chỉ trong vài chục mili-giây cho mỗi câu.
