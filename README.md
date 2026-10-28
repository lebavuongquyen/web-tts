# 🦀 rust_tts_runtime

> **High-Performance Standalone Rust TTS Runtime with Smart Chunking, Subtitle SRT Generation, Voice Cloning & WebAssembly Architecture.**

[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-blue?style=for-the-badge&logo=github)](https://lebavuongquyen.github.io/rust_tts_runtime/)
[![Rust](https://img.shields.io/badge/Rust-1.80+-orange?style=for-the-badge&logo=rust)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%20RTX%203060-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-zone)
[![WebAssembly](https://img.shields.io/badge/WebAssembly-WASM-purple?style=for-the-badge&logo=webassembly)](https://webassembly.org/)

🌐 **Live Demo Trải Nghiệm Trực Tuyến**: [https://lebavuongquyen.github.io/rust_tts_runtime/](https://lebavuongquyen.github.io/rust_tts_runtime/)

---

## 🌟 Điểm nổi bật (Key Features)

* 🚀 **Lõi Runtime Thuần Khiết (Pure Rust)**: Chuyên trách tính toán ma trận nơ-ron tốc độ cao, tích hợp bộ cấp phát siêu tốc **`mimalloc`** và tăng tốc **NVIDIA CUDA (GPU)**.
* 🧩 **Kiến trúc phân tầng chuẩn mực**: 
  * **Lõi Engine (Rust)**: Tính toán ma trận ONNX, xuất duy nhất file WAV PCM chuẩn 16-bit.
  * **Bộ vỏ điều phối (Python SDK)**: Tự động chia đoạn (Smart Chunking), chuẩn hóa tiếng Việt, xuất file phụ đề `.srt` đồng bộ từng mili-giây, ghép nối xuất file `.mp3` / `.wav`.
* 🗣️ **Hỗ trợ 2 Chuẩn Voice Đa Dạng**:
  1. **Voice Latents (`.bin` - 30 KB)**: Dành cho ZeroTTS, nạp tức thì, hỗ trợ pha trộn giọng lai độc bản (Voice Blending: `huuduc:0.7,kimoanh:0.3`).
  2. **Standalone Model (`.onnx` - 60 MB)**: Dành cho VITS Fine-tuned, đạt **độ giống 98%**, tốc độ siêu tốc RTF 0.05x (gấp 20x thời gian thực).
* 🛠️ **Bộ Toolchain Tạo Voice Hoàn Chỉnh**:
  * `tools/create_voice_bin.py`: Tạo / trích xuất file `.bin` (30KB) từ MP3/WAV hoặc blend giọng.
  * `tools/auto_finetune_vits_onnx.py`: Tự động Fine-Tune từ tập audio MP3/WAV $\rightarrow$ xuất file `.onnx` mới chỉ bằng 1 dòng lệnh.
* 🌐 **Sẵn sàng cho WebAssembly (In-Browser TTS)**: Thiết kế 100% In-Memory để đưa lên trình duyệt web (như `vangioi.com`), người dùng tự đọc truyện trên máy của họ, server tốn 0đ chi phí GPU!

---

## 📚 Tài liệu chi tiết (Documentation)

* 🏛️ **[Kiến trúc hệ thống chi tiết (Architecture)](docs/ARCHITECTURE.md)**: Phân tách rõ ràng giữa Pure Runtime, Python Wrapper và WebAssembly.
* 🗣️ **[Hướng dẫn Clone Giọng Nói (Voice Cloning Guide)](docs/VOICE_CLONING.md)**: Hướng dẫn tạo file `.bin` 30KB và fine-tune file `.onnx` 60MB.
* 📊 **[Báo cáo Benchmark & Hiệu năng (Benchmarks)](docs/BENCHMARKS.md)**: Đo đạc tốc độ FPS, RTF, RAM/VRAM giữa GPU RTX 3060 và CPU Xeon.
* 🌐 **[Kế hoạch triển khai WebAssembly (WebAssembly Guide)](docs/WEBASSEMBLY.md)**: Hướng dẫn đưa WebAssembly lên vangioi.com để chạy 100% In-Memory Client-side.

---

## 🏗️ Cài đặt & Biên dịch Lõi Runtime

```bash
git clone https://github.com/lebavuongquyen/rust_tts_runtime.git
cd rust_tts_runtime
cargo build --release
```
Binary nằm tại: `target/release/rust_tts_runtime`.

---

## 💻 Sử dụng Lõi Runtime qua CLI

```bash
# 1. Chạy ZeroTTS với giọng chuẩn (hoặc file .bin riêng):
./target/release/rust_tts_runtime "Tôi là một siêu cấp lão làng." \
  --voice huuduc \
  --output test.wav \
  --json

# 2. Chạy ZeroTTS với giọng pha trộn (Voice Blend):
./target/release/rust_tts_runtime "Giọng đọc lai độc bản." \
  --blend "huuduc:0.7,kimoanh:0.3" \
  --output blend.wav

# 3. Chạy VITS ONNX Engine (98% độ giống):
./target/release/rust_tts_runtime \
  --voice path/to/model.onnx \
  --phoneme-ids "1 0 32 0 142 0 120 0 27 0 2" \
  --output vits.wav
```

---

## 🐍 Sử dụng Bộ vỏ Python SDK

Bộ vỏ Python đảm nhận việc đọc đoạn văn bản dài, tự động chia câu, chuẩn hóa số tiếng Việt và xuất file phụ đề `.srt` đồng bộ:

```python
import sys
sys.path.insert(0, 'python')
from rust_tts_runtime import RustTTS

tts = RustTTS(cuda=True)

# Đọc văn bản dài kèm phụ đề SRT và xuất file MP3:
res = tts.speak(
    text="Tôi là một siêu cấp lão làng. Hôm nay là ngày 28/10/2026 với 100% công lực.",
    voice="kimoanh",           # hoặc đường dẫn tới file .onnx / .bin
    output="chapter_1.mp3",
    generate_srt=True,         # Tự động xuất chapter_1.srt
    pause=0.35                 # Khoảng lặng giữa các câu
)

print("Audio file:", res["audio"])
print("SRT file:  ", res["srt"])
print(f"Tổng số câu: {res['sentences']} | Thời lượng: {res['duration']}s")
```

---

## 📄 Giấy phép

Mã nguồn được phân phối theo giấy phép MIT.
