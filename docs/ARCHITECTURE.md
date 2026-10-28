# 🏛️ KIẾN TRÚC HỆ THỐNG: RUST TTS RUNTIME

Dự án tuân thủ nghiêm ngặt **Nguyên lý phân tách trách nhiệm (Separation of Concerns)** và **Triết lý Unix (Unix Philosophy)**:

```
┌─────────────────────────────────────────────────────────────┐
│               BỘ VỎ ĐIỀU PHỐI (PYTHON SDK / CLI)            │
│   • Smart Chunking: Cắt câu dài 10-25 từ giữ nhịp thở      │
│   • Chuẩn hóa tiếng Việt: 100% -> một trăm phần trăm        │
│   • Đồng bộ phụ đề: Đo mili-giây từng câu -> sinh file .srt │
│   • Nối ghép âm thanh: Nối các đoạn WAV kèm pause 0.35s     │
│   • Xuất file: .mp3 (ffmpeg 192k/320k) hoặc .wav           │
└──────────────────────────────┬──────────────────────────────┘
                               │ Gọi lệnh JSON/WAV
┌──────────────────────────────▼──────────────────────────────┐
│                  LÕI RUNTIME (RUST ENGINE)                  │
│   • Nhận: Text 1 câu + voice (.bin hoặc .onnx)              │
│   • Bộ cấp phát bộ nhớ siêu tốc: mimalloc (chống nghẽn RAM) │
│   • Tối ưu phần cứng: NVIDIA CUDA EP + GraphOptimization All│
│   • Động cơ kép: ZeroTTS (48kHz) + VITS ONNX (22kHz)        │
│   • Đầu ra: DUY NHẤT file WAV PCM chuẩn 16-bit              │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Lõi Runtime (`rust_tts_runtime`)
* Viết bằng 100% Rust, biên dịch thành binary độc lập.
* **Không phụ thuộc FFmpeg, không chứa logic chunking truyện, không sinh phụ đề SRT**.
* Tập trung 100% tài nguyên CPU/GPU vào việc giải ma trận nơ-ron và ghi ra sóng âm PCM WAV.
* Hỗ trợ 2 động cơ:
  * **ZeroTTS Engine**: Nạp file vector `voice.bin` 30KB.
  * **VITS Engine**: Nạp trực tiếp file `ten_giong.onnx` 60MB và chuỗi phoneme IDs.

---

## 2. Bộ vỏ điều phối (`python/rust_tts_runtime/client.py`)
* Là lớp giao tiếp thân thiện với lập trình viên (Developer Experience).
* Nhận vào văn bản dài (hàng nghìn từ, chương truyện).
* Tự động tiền xử lý ngôn ngữ tiếng Việt (vi_normalizer).
* Gọi lõi Runtime sinh từng câu, đo đạc thời gian chính xác, chèn khoảng lặng tự nhiên.
* Xuất ra file âm thanh tổng hợp và file phụ đề `.srt` hoàn hảo.
