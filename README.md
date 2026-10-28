# ZeroTTS Standalone Rust Runner & INT8 Quantization 🚀

Trình thực thi độc lập hiệu năng cao viết bằng **Rust** cho mô hình **ZeroTTS** (Mô hình Text-to-Speech Tiếng Việt Zero-Shot). Hỗ trợ tăng tốc GPU NVIDIA CUDA, mô hình nén **INT8 Quantized**, tự động bóc tách chương truyện dài, chuẩn hóa số tiếng Việt, xuất trực tiếp file **MP3** và sinh file phụ đề **SRT**.

---

## 🌟 Tính Năng Nổi Bật

- ⚡ **Hiệu năng vượt trội**: Viết bằng Rust thuần túy, kết hợp ONNX Runtime C API và CUDA Execution Provider. Hệ số thời gian thực (RTF) đạt tới **0.3x - 0.5x** trên GPU (nhanh gấp 2 - 3 lần thời gian thực).
- 📖 **Bóc tách đoạn văn dài (Chapter Chunking)**: Hỗ trợ đọc cả chương truyện hoặc văn bản dài thông qua cơ chế phân rã câu thông minh theo dấu ngắt (`.`, `!`, `?`, `…`, `\n`) và chèn khoảng lặng tự nhiên giữa các câu.
- 🔢 **Chuẩn hóa số tiếng Việt (Text Normalization)**:
  - Số nguyên: `100` -> *"một trăm"*, `24` -> *"hai mươi tư"*, `105` -> *"một trăm linh năm"*, `2026` -> *"hai nghìn không trăm hai mươi sáu"*.
  - Ngày tháng: `28/10/2026` -> *"ngày hai mươi tám tháng mười năm hai nghìn không trăm hai mươi sáu"*.
  - Phần trăm: `99%` -> *"chín mươi chín phần trăm"*.
- 🎵 **Xuất trực tiếp MP3 & WAV**: Tự động nhận diện định dạng theo đuôi file (`.mp3` hoặc `.wav`). Mã hóa MP3 chất lượng cao (mặc định 192 kbps, tùy chọn lên đến 320 kbps), dung lượng nén giảm hơn 4 lần so với WAV.
- ⏱️ **Tự động sinh phụ đề SRT**: Sinh file `.srt` với timestamp khớp chính xác từng mili-giây theo từng câu của văn bản, sẵn sàng dùng làm phụ đề video hoặc audiobook tương tác.
- 🗜️ **Tương thích INT8 Quantized**: Chạy mượt mà cả mô hình gốc FP32 lẫn mô hình lượng tử hóa động INT8 (dung lượng giảm từ 820MB xuống 230MB).

---

## 🛠️ Yêu Cầu Hệ Thống & Cài Đặt

### 1. Yêu cầu tiên quyết
- **Rust Toolchain**: `cargo` và `rustc` (phiên bản 1.75+).
- **FFmpeg**: Cần có `ffmpeg` trên hệ thống để mã hóa file MP3 (`sudo apt install -y ffmpeg`).
- **NVIDIA Driver & CUDA** (nếu chạy GPU): Hỗ trợ CUDA 11/12.
- **ONNX Runtime**: Thư viện động `libonnxruntime.so` (hỗ trợ version 1.20+).

### 2. Biên dịch (Build Release)
```bash
git clone https://github.com/lebavuongquyen/zerotts_quantized_int8.git
cd zerotts_quantized_int8

# Biên dịch bản tối ưu release
cargo build --release
```
File thực thi nằm tại: `target/release/zerotts`.

---

## 📖 Hướng Dẫn Sử Dụng

### 1. Đọc văn bản ngắn xuất ra MP3 và SRT
```bash
./target/release/zerotts "Hôm nay ngày 28/10/2026, tôi đã vượt qua 100 tầng tháp thử thách. Độ chính xác đạt 99%." \
  -o output.mp3 \
  --voice huuduc \
  --bitrate 192k \
  --generate-srt
```
- Kết quả tạo ra:
  - `output.mp3`: File âm thanh giọng đọc Hữu Đức 48kHz mono chuẩn 192 kbps.
  - `output.srt`: File phụ đề đồng bộ từng câu.

### 2. Đọc cả chương truyện từ file văn bản (.txt)
```bash
./target/release/zerotts --file chapter_1.txt \
  -o chapter_1.mp3 \
  --voice huuduc \
  --pause-dur 0.35 \
  --generate-srt
```

### 3. Xuất file WAV gốc 16-bit PCM
Nếu muốn xuất định dạng WAV thay vì MP3:
```bash
./target/release/zerotts "Văn bản mẫu cần đọc" -o output.wav
```

### 4. Sử dụng mô hình nén INT8 Quantized
Chỉ định thư mục weights INT8 qua cờ `--model-dir`:
```bash
./target/release/zerotts "Đọc với mô hình lượng tử hóa INT8 siêu nhẹ." \
  -m /path/to/model_weights_quant \
  -o test_quant.mp3
```

---

## 🗜️ Hướng Dẫn Lượng Tử Hóa Mô Hình (FP32 -> INT8)

Trong thư mục `scripts/` có sẵn kịch bản `quantize.py` để chuyển đổi mô hình ZeroTTS sang INT8:
```bash
python scripts/quantize.py \
  --src /path/to/model_weights \
  --dst /path/to/model_weights_quant
```
- `text_encoder.onnx`: 309 MB ➔ ~80 MB
- `prefix_step.onnx`: 333 MB ➔ ~85 MB
- `local_frame_decode.onnx`: 179 MB ➔ ~45 MB
- Tổng kích thước weights: **~820 MB ➔ ~230 MB** (Tiết kiệm gần 75% dung lượng lưu trữ và RAM).

---

## 📊 Bảng Tham Số Dòng Lệnh (CLI Options)

| Tham số | Viết tắt | Mặc định | Mô tả |
|---|---|---|---|
| `<TEXT>` | *(Vị trí)* | `None` | Chuỗi văn bản tiếng Việt cần đọc (hoặc `-` để đọc stdin) |
| `--file` | `-f` | `None` | Đường dẫn file `.txt` chứa nội dung truyện hoặc văn bản dài |
| `--output` | `-o` | `output.mp3` | Đường dẫn file âm thanh xuất ra (`.mp3` hoặc `.wav`) |
| `--voice` | `-v` | `huuduc` | Tên giọng đọc (ví dụ: `huuduc`, `maichi`,...) |
| `--bitrate` | `-b` | `192k` | Bitrate khi xuất file MP3 (`128k`, `192k`, `256k`, `320k`) |
| `--generate-srt` | | `true` | Tự động xuất file phụ đề `.srt` cùng tên với audio |
| `--srt` | | `None` | Chỉ định đường dẫn file `.srt` tùy chỉnh |
| `--pause-dur` | | `0.35` | Thời gian khoảng lặng nghỉ giữa các câu (giây) |
| `--no-norm` | | `false` | Bỏ qua bước chuẩn hóa số tiếng Việt |
| `--model-dir` | `-m` | `.../model_weights` | Thư mục chứa các tệp mô hình ONNX và tokenizer |
| `--cuda` | | `true` | Sử dụng card đồ họa GPU NVIDIA CUDA |
| `--max-frames` | | `1500` | Giới hạn số frame âm thanh sinh tối đa mỗi câu |

---

## 📜 Giấy Phép
Dự án được phân phối dưới giấy phép MIT License.
