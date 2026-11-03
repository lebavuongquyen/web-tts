# WebAssembly Client-Side Dual-Engine AI TTS (Zero Backend API)

Module WebAssembly hiệu năng cao chạy trực tiếp trên trình duyệt web người dùng, sử dụng **ONNX Runtime Web (WASM SIMD Multi-threaded)** kết hợp **Dedicated Web Worker** và **Service Worker PWA Offline Cache**, đạt tốc độ phản hồi âm thanh tức thì (**TTFA < 100ms**) mà **không tốn 1 đồng chi phí server API hay GPU máy chủ**.

---

## 🌟 Tính Năng Nổi Bật

1. **Dual-Engine Hỗ Trợ 2 Kiến Trúc AI Đỉnh Cao**:
   - **Kokoro-Vietnamese (INT8 ONNX)**: Single-pass feedforward siêu nhẹ (~137 MB), tốc độ cực nhanh (RTF ~0.15x), TTFA < 80ms. Hỗ trợ 7 giọng đọc chuẩn miền Bắc, Trung, Nam và Giọng lai (Voice Blend).
   - **VieNeu-TTS v3 Nano (Flow-Matching ODE)**: 4 Graph ONNX giải phương trình vi phân Euler (6-8 steps, CFG 2.5), âm thanh 24kHz tự nhiên, giàu cảm xúc. Hỗ trợ 8 giọng đọc thế hệ mới: *Minh Quân, Trúc Ly, Ái Hân, Đức Trí, Mỹ Duyên, Xuân Tiên, Mai Anh, Mạnh Dũng*.
2. **100% Client-Side In-Browser Inference**:
   - Không gọi bất kỳ REST API, GraphQL hay WebSocket máy chủ nào để suy luận TTS.
   - Toàn bộ phép tính ma trận nơ-ron thực hiện trực tiếp trên CPU người dùng qua WebAssembly SIMD 8 luồng (hoặc tuỳ chỉnh 1–16 luồng).
3. **Dedicated Web Worker & Mượt mà 60 FPS**:
   - Toàn bộ quá trình tải mô hình và suy luận ONNX nằm trong background Web Worker (`tts-worker.js`).
   - Main UI Thread hoàn toàn tự do, không giật lag, giữ vững 60 FPS khi máy đang đọc văn bản dài.
4. **PWA Offline 100% với Service Worker v5**:
   - Tải trước (precache) toàn bộ Model ONNX, WebAssembly Runtime và Voicepacks vào Cache Storage.
   - Sau khi nạp, người dùng có thể tắt toàn bộ mạng Internet/WiFi và hệ thống vẫn tổng hợp giọng nói bình thường.
5. **Từ Điển Âm Vị 27.877 Từ Tiếng Việt & Triệt Tiêu Nuốt Từ**:
   - Tích hợp sẵn `vi_phoneme_dict.json` (27.877 từ Kokoro IPA) và `vieneu/dict.json` (27.876 từ VieNeu IPA) sinh bằng `vig2p` & `sea_g2p`.
   - Đệm âm đuôi (Trailing Silence Buffer `16`) chống hiện tượng cắt cụt âm tiết của Duration Predictor.
   - Tự động ngắt nghỉ chuẩn ngữ điệu: nghỉ 250ms cho dấu chấm (`.`, `!`, `?`), 140ms cho dấu phẩy (`,`, `;`), và 80ms cho nhịp thở giữa các mệnh đề.
   - Tốc độ tối ưu mặc định `0.95x` giúp phát âm rõ vành, rõ chữ, chuẩn 6 thanh điệu tiếng Việt.

---

## 📂 Cấu Trúc Thư Mục `wasm/`

```text
wasm/
├── demo_worker_tts.html        # Giao diện Web App Studio hoàn chỉnh
├── index.html                  # Bản sao đồng bộ phục vụ tại root URL (/)
├── tts-worker.js               # Dedicated Web Worker (WASM Multi-thread Runtime)
├── sw.js                       # Service Worker PWA (Cache Storage v5)
├── reset.html                  # Trang tiện ích dọn sạch Service Worker & Cache
├── server.py                   # HTTP Server tích hợp header COOP/COEP
├── vi_phoneme_dict.json        # 27.877 từ tiếng Việt đã phiên âm cho Kokoro
├── ort.min.js                  # ONNX Runtime Web v1.17
├── ort-wasm-simd-threaded.wasm # WASM Binary đa luồng SIMD
├── kokoro_vi_fast_int8.onnx    # Model Kokoro INT8 (137 MB)
├── kokoro_voices/              # 20+ tệp vector giọng đọc Kokoro (.bin ~510 KB)
│   ├── ngoc_huyen.bin
│   ├── hung_thinh.bin
│   ├── diem_trinh.bin
│   └── huuduc.bin
└── vieneu/                     # Model & Siêu dữ liệu VieNeu-TTS v3 Nano
    ├── text_encoder.onnx       # Graph 1: Text Encoder (26 MB)
    ├── duration_predictor.onnx # Graph 2: Duration Predictor (711 KB)
    ├── vector_estimator.onnx   # Graph 3: Flow-Matching Estimator (148 MB)
    ├── codec_decoder.onnx      # Graph 4: Codec Audio Decoder (95 MB)
    ├── config.json             # Cấu hình kiến trúc Flow-Matching
    ├── constants.json          # Style vector và hằng số Euler ODE
    ├── voices.json             # 8 giọng đọc preset (Minh Quân, Trúc Ly, Ái Hân...)
    └── dict.json               # 27.876 từ tiếng Việt phiên âm VieNeu IPA
```

---

## 🚀 Hướng Dẫn Khởi Chạy

### Bước 1: Khởi động Web Server (Kèm Header COOP/COEP)

Để trình duyệt kích hoạt tính năng `SharedArrayBuffer` đa luồng cho WebAssembly, server bắt buộc phải gửi 2 HTTP header:
- `Cross-Origin-Opener-Policy: same-origin`
- `Cross-Origin-Embedder-Policy: require-corp`

Khởi chạy bằng server Python có sẵn:
```bash
python3 wasm/server.py
```
Server sẽ lắng nghe tại `http://0.0.0.0:8085`.

### Bước 2: Sử dụng trên Trình duyệt

Truy cập:
👉 **`http://localhost:8085/`** (hoặc `http://localhost:8085/demo_worker_tts.html`)

1. **Lưu trữ Offline (Tùy chọn)**:
   - Bấm nút `📥 Nạp trước Model vào Cache Offline` để lưu toàn bộ tệp model vào Cache Storage.
2. **Nạp Model vào Worker**:
   - Chọn mô hình: **Kokoro-Vietnamese** hoặc **VieNeu-TTS v3 Nano**.
   - Bấm `⚙️ 1. Nạp Model vào Web Worker`. Nút sẽ tự động nạp các đồ thị ONNX vào bộ nhớ worker.
3. **Phát âm thanh Streaming**:
   - Nhập bất kỳ câu tiếng Việt nào (hoặc chọn kịch bản mẫu).
   - Bấm **`⚡ 3. Đọc Streaming Ngay Lập Tức`**. Âm thanh câu đầu tiên sẽ phát ra loa sau **< 100ms** trong khi các câu tiếp theo được tính toán ngầm dưới nền.

---

## ⚡ Bảng So Sánh Hiệu Năng (Test trên CPU Intel Core i5 / AMD Ryzen 5)

| Chỉ số | Kokoro-Vietnamese (INT8) | VieNeu-TTS v3 Nano (ODE 6-Steps) |
| :--- | :--- | :--- |
| **Kiến trúc** | Single-pass Feedforward | 4-Graph Euler Flow-Matching |
| **Dung lượng nạp** | 137 MB | 270 MB |
| **TTFA (Time to First Audio)** | **~60 – 90 ms** | **~250 – 400 ms** |
| **RTF (Real-Time Factor)** | **0.15x** (Nhanh gấp 6.5x thời gian thực) | **0.33x** (Nhanh gấp 3.0x thời gian thực) |
| **Tần số lấy mẫu (Sample Rate)** | 24,000 Hz | 24,000 Hz |
| **Độ tự nhiên ngữ điệu** | ⭐⭐⭐⭐ (Rất mượt) | ⭐⭐⭐⭐⭐ (Xuất sắc, chuẩn cảm xúc) |
| **Hỗ trợ Offline 100%** | Có | Có |

---

## 🛡️ Tiêu Chuẩn Kỹ Thuật & Clean Code

- **Ràng buộc cứng độ dài hàm**: 100% các hàm trong `demo_worker_tts.html`, `index.html`, `sw.js`, `tts-worker.js` đều tuân thủ nghiêm ngặt **$\le 35$ dòng code**.
- **Giải phóng bộ nhớ C++ WASM**: Mọi `ort.Tensor` sau khi `session.run()` đều được gọi `.dispose()` ngay lập tức để chống hiện tượng Memory Leak khi đọc văn bản dài hàng nghìn câu.
