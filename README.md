# ZeroTTS Standalone Rust Core Runtime 🚀

Trình thực thi lõi (Core Runtime) hiệu năng cao viết bằng **Rust** cho mô hình **ZeroTTS** (Vietnamese Zero-Shot Text-to-Speech). Dự án được thiết kế chuyên biệt để kiểm thử hiệu năng (benchmarking), tối ưu hóa thời gian sinh (latency / TTFF / RTF) và nâng cấp đồ thị nơ-ron (ONNX Runtime, CUDA 12, INT8 Quantization).

---

## 🌟 Đặc Điểm Thiết Kế

- ⚡ **Pure Runtime**: Không chứa logic xử lý văn bản hay encoding phụ; tập trung 100% vào tốc độ suy luận của mô hình ONNX qua CUDA Execution Provider.
- ⏱️ **Đo lường chi tiết**: Phân tách và đo chính xác thời gian từng giai đoạn:
  1. Nạp đồ thị ONNX & Tokenizer
  2. Mã hóa Text Tokens
  3. Text Encoder & Cross-KV Attention
  4. Khởi tạo KV-Cache (Prefix Step Init)
  5. Vòng lặp Autoregressive (AR Loop)
  6. MossCodec Audio Decoding
- 🔊 **Âm thanh gốc**: Xuất file định dạng WAV 48kHz mono 16-bit PCM không nén để đảm bảo độ trung thực cao nhất của tín hiệu âm thanh phục vụ kiểm định.
- 🗜️ **Hỗ trợ INT8 Quantized**: Tương thích hoàn toàn với cả weights gốc FP32 lẫn weights INT8 (giảm kích thước model còn 1/4).

---

## 🛠️ Biên Dịch & Chạy

```bash
# Biên dịch bản tối ưu
cargo build --release

# Chạy kiểm thử hiệu năng
./target/release/zerotts "Xin chào, đây là bài kiểm tra hiệu năng runtime thuần túy." \
  --voice huuduc \
  -o output.wav
```

### Sử dụng với weights INT8 Quantized:
```bash
./target/release/zerotts "Đo kiểm hiệu năng trên mô hình INT8." \
  -m /path/to/model_weights_quant \
  -o output_quant.wav
```

---

## 🗜️ Lượng Tử Hóa Mô Hình (FP32 -> INT8)
Trong thư mục `scripts/` có script `quantize.py` để nén các đồ thị ONNX sang INT8:
```bash
python scripts/quantize.py \
  --src /path/to/model_weights \
  --dst /path/to/model_weights_quant
```
- `text_encoder.onnx`: 309 MB ➔ 80 MB
- `prefix_step.onnx`: 333 MB ➔ 85 MB
- `local_frame_decode.onnx`: 179 MB ➔ 45 MB
- Tổng dung lượng: **820 MB ➔ 230 MB** (Giảm ~75%).

---

## 📜 Giấy Phép
Dự án được phân phối dưới giấy phép MIT License.
