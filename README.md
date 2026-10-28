# 🦀 rust_tts_runtime

> **High-Performance Standalone Rust TTS Runtime with Smart Chunking, Subtitle SRT Generation, Voice Cloning & Quantization.**

Repository: [https://github.com/lebavuongquyen/rust_tts_runtime](https://github.com/lebavuongquyen/rust_tts_runtime)

---

## 🌟 Điểm nổi bật (Key Features)

* 🚀 **Tối ưu hóa hiệu năng cao**: Tích hợp bộ cấp phát bộ nhớ siêu tốc **`mimalloc`**, hỗ trợ tăng tốc phần cứng **NVIDIA CUDA (GPU)** và đa luồng **CPU**.
* 🧩 **Kiến trúc tách rời (Decoupled Architecture)**: 
  * **Lõi Engine (Rust)**: Tính toán ma trận ONNX, quản lý bộ nhớ đệm, tối ưu tốc độ và độ trễ.
  * **Bộ vỏ SDK (Python / CLI)**: Tự động chia đoạn (chunking), chuẩn hóa văn bản tiếng Việt, xuất file phụ đề `.srt`, API tiện lợi.
* 🗣️ **Hệ thống Quản lý & Clone Giọng (Voice Cloning)**:
  * Nạp 8 giọng chuẩn có sẵn (`huuduc`, `kimoanh`, `maichi`, `baotrang`, `giahuy`, `hamy`, `quangminh`, `tiendat`).
  * Nạp file giọng clone tùy chỉnh (`.bin` 30KB) từ bất kỳ đường dẫn nào.
  * **Pha trộn giọng lai độc bản (Voice Blending)**: Tạo giọng mới bằng cách nội suy không gian vector latent (`huuduc:0.7,kimoanh:0.3`).
* 📜 **Đồng bộ phụ đề SRT tự động**: Đo đạc độ dài từng câu chính xác đến từng mili-giây, xuất file `.srt` chuẩn chỉnh cùng lúc với file âm thanh (`.mp3` / `.wav`).
* 🇻🇳 **Chuẩn hóa tiếng Việt chuyên sâu**: Đọc số nguyên, phần trăm (`100%`), ngày tháng năm (`28/10/2026`) thành chữ tiếng Việt chuẩn xác trước khi tổng hợp giọng nói.

---

## 🏗️ Cài đặt & Biên dịch

### Yêu cầu hệ thống:
* Rust toolchain (Cargo) 1.80+
* `ffmpeg` (để xuất file `.mp3`)
* `onnxruntime` CUDA hoặc CPU

```bash
cd /home/quyen/08_TTS/rust_tts_runtime
cargo build --release
```
Binary sau khi biên dịch nằm tại: `target/release/rust_tts_runtime`.

---

## 💻 Sử dụng qua dòng lệnh (CLI)

### 1. Xem danh sách giọng đọc có sẵn
```bash
./target/release/rust_tts_runtime --list-voices
```

### 2. Đọc một câu ngắn hoặc đoạn văn
```bash
./target/release/rust_tts_runtime "Tôi là một siêu cấp lão làng. Hôm nay là ngày 28/10/2026 với 100% sức mạnh." \
  --voice kimoanh \
  --output story.mp3 \
  --generate-srt
```
*(Hệ thống sẽ tạo ra đồng thời cả `story.mp3` và file phụ đề `story.srt`)*.

### 3. Đọc cả chương truyện dài từ file text
```bash
./target/release/rust_tts_runtime --file chapter_1.txt \
  --voice huuduc \
  --output chapter_1.mp3 \
  --pause 0.35
```

### 4. Clone giọng & Pha trộn giọng lai (Voice Blending)
```bash
# Nạp file clone riêng
./target/release/rust_tts_runtime "Xin chào các bạn." --voice /path/to/my_clone.bin

# Pha trộn 70% Hữu Đức + 30% Kim Oanh
./target/release/rust_tts_runtime "Đây là chất giọng mới độc bản." \
  --blend "huuduc:0.7,kimoanh:0.3" \
  --output hybrid_voice.mp3
```

---

## 🐍 Sử dụng qua Bộ vỏ Python (Python SDK)

```python
from rust_tts_runtime import RustTTS

tts = RustTTS(
    binary_path="target/release/rust_tts_runtime",
    model_dir="/home/quyen/08_TTS/ZeroTTS/model_weights",
    cuda=True  # Sử dụng GPU RTX 3060
)

# 1. Liệt kê các giọng có sẵn
print(tts.list_voices())

# 2. Tổng hợp văn bản dài kèm phụ đề SRT
res = tts.speak(
    text="Đêm đã về khuya trên thung lũng sương mù. Tiếng suối chảy róc rách qua từng khe đá...",
    voice="kimoanh",
    output="output.mp3",
    generate_srt=True,
    pause=0.35
)

print("Audio file:", res["audio"])
print("SRT file:  ", res["srt"])

# 3. Đọc cả file truyện
res = tts.speak_file("chapter_1.txt", voice="huuduc", output="chapter_1.mp3")
```

---

## 🔬 Nghiên cứu cơ chế Clone Giọng (Voice Latents Architecture)

Trong hệ thống ZeroTTS:
1. Giọng nói được nén trong ma trận **Acoustic Prefix Latents** kích thước **`(1, 10, 768)` Float32** (tương đương 7.680 con số thực hoặc đúng **30.720 bytes**).
2. Khi suy luận, ma trận này đóng vai trò là "mỏ neo âm sắc" (timbre anchor) mở đầu chuỗi AR Attention.
3. Không gian vector $10 \times 768$ này là **không gian liên tục**:
   $$\mathbf{v}_{\text{blend}} = \sum_i w_i \cdot \mathbf{v}_i \quad \text{với } \sum w_i = 1$$
   Cho phép pha trộn tỷ lệ giữa các giọng nói để tạo ra các giọng đọc độc bản hoàn toàn mới mà không cần train lại mô hình.
