# 🌐 HƯỚNG DẪN TRIỂN KHAI WEBASSEMBLY (IN-BROWSER TTS) CHO VANGIOI.COM

> Mục tiêu: Đưa toàn bộ tính năng tổng hợp giọng nói tiếng Việt chạy trực tiếp trên trình duyệt của người dùng (Client-side In-Memory), **server tốn 0 đồng chi phí GPU/VPS**.

---

## 1. Kiến trúc Client-Side In-Browser

```
Trình duyệt người dùng (Chrome / Safari / Edge / Mobile)
  ├── 1. Giao diện Web (HTML/JS/React): Nhận text từ người đọc
  ├── 2. rtts_core.wasm (~500 KB):
  │       • Tách câu thông minh (Chunking)
  │       • Chuẩn hóa số, ngày tháng, phần trăm tiếng Việt
  │       • Tính toán Voice Blending (30KB Float32Array)
  │       • Xuất timeline phụ đề SRT
  ├── 3. onnxruntime-web:
  │       • Tận dụng WebGPU / WebAssembly SIMD của card máy khách
  │       • Chạy đồ thị ONNX trực tiếp trong Web Worker
  └── 4. Web Audio API:
          • Phát âm thanh trực tiếp qua loa mà không cần tải file về ổ cứng
```

---

## 2. Các bước triển khai cho Web `vangioi.com`

### Bước 1: Build mã nguồn Rust sang WebAssembly
Cài đặt `wasm-pack`:
```bash
cargo install wasm-pack
```

Build target WebAssembly:
```bash
cd rust_tts_runtime
wasm-pack build --target web --out-dir pkg
```
Kết quả xuất ra thư mục `pkg/` gồm:
* `rtts_core_bg.wasm` (~500 KB)
* `rtts_core.js` (JavaScript glue code)

### Bước 2: Tích hợp vào Frontend
```html
<script type="module">
  import init, { normalize_vietnamese, split_sentences } from './pkg/rtts_core.js';

  async function run() {
    await init();
    
    const rawText = "Hôm nay 28/10/2026 với 100% công lực.";
    const cleanText = normalize_vietnamese(rawText);
    console.log("Đã chuẩn hóa:", cleanText);

    const chunks = split_sentences(cleanText);
    console.log("Danh sách câu:", chunks);
  }

  run();
</script>
```

### Bước 3: Nạp ONNX Runtime Web
Sử dụng CDN chính thức của Microsoft:
```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>
```
Mô hình `.onnx` và file `voice.bin` được đặt trên CDN tĩnh (Cloudflare / GitHub Pages). Trình duyệt tải về và cache trên IndexedDB, lần sau mở lên đọc offline ngay lập tức!
