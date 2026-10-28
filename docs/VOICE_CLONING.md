# 🗣️ HƯỚNG DẪN CHUYÊN SÂU: CLONE GIỌNG NÓI (VOICE CLONING)

Hệ thống hỗ trợ 2 trường phái Clone Giọng:

---

## 1. Phương pháp 1: File Vector `voice.bin` 30KB (ZeroTTS)

### Đặc điểm:
* Dung lượng cực kỳ nhẹ: đúng **30.720 bytes** ($1 \times 10 \times 768$ Float32).
* Không cần huấn luyện lại mô hình, nạp vào là nói ngay lập tức.
* Độ tương đồng: **~85% - 90%**.

### Cách 1: Pha trộn giọng lai độc bản (Voice Blending)
```bash
# Bằng công cụ Rust siêu nhẹ:
rtts-voice --blend "huuduc:0.7,kimoanh:0.3" --output custom_blend.bin

# Hoặc bằng Python:
python tools/create_voice_bin.py --blend "huuduc:0.7,kimoanh:0.3" --output custom_blend.bin
```

### Cách 2: Trích xuất trực tiếp từ file âm thanh (.mp3 / .wav) bằng Rust CLI
Chỉ mất **0.2 giây**, phân tích Pitch F0 và năng lượng âm phổ bằng thuật toán DSP thuần Rust:
```bash
rtts-voice --input voice_sample.mp3 --output my_cloned_voice.bin
```

### Cách 3: Thu âm Micro trực tiếp trên Trình duyệt Web (vangioi.com)
* Người dùng bấm nút **[🔴 Thu Âm 5 Giây]** trên trang web `vangioi.com`.
* Web Audio API lấy mảng PCM và chạy thuật toán DSP trong RAM trình duyệt trong **15 mili-giây**.
* Tự động xuất file `my_voice.bin` (30.720 bytes) cho người dùng tải về hoặc thử nghiệm ngay lập tức.
* **Chi phí server: 0 đồng**, bảo mật 100% không gửi giọng nói lên mạng!

### Cách 4: Trích xuất từ file zip hoặc npz tải về
```bash
python tools/create_voice_bin.py \
  -i my_voice_pack.zip \
  -o my_voice.bin
```

---

## 2. Phương pháp 2: Fine-Tune ra File `ten_giong.onnx` (Độ giống 98%)

### Đặc điểm:
* Dung lượng file: **~60 MB - 63 MB**.
* Sử dụng mô hình nền `vi_VN_base.ckpt` để huấn luyện sâu vào từng cách luyến láy, ngữ điệu của người nói.
* Độ tương đồng: **97% - 99%** (chuẩn studio / sách nói chuyên nghiệp).

### Hướng dẫn chạy Fine-Tune tự động bằng 1 dòng lệnh:
1. Chuẩn bị thư mục ghi âm (ví dụ: `data_giong/`) chứa:
   * Các file âm thanh `.wav` hoặc `.mp3` (khoảng 10 - 20 phút đọc mẫu).
   * File `metadata.csv` (định dạng: `ten_file.wav|Văn bản câu đọc`).
2. Chạy lệnh:
```bash
python tools/auto_finetune_vits_onnx.py \
  --data-dir data_giong \
  --name giong_vip_huuduc \
  --epochs 15 \
  --batch-size 16
```
3. Sau 15 phút, bạn sẽ nhận được file `giong_vip_huuduc.onnx` và `giong_vip_huuduc.onnx.json`.

---

## 3. Cách sử dụng file giọng đã tạo

Trong Python:
```python
from rust_tts_runtime import RustTTS

tts = RustTTS()

# 1. Dùng giọng .bin 30KB:
tts.speak("Xin chào!", voice="custom_blend.bin", output="out1.mp3")

# 2. Dùng giọng .onnx 60MB (98% độ giống):
tts.speak("Xin chào!", voice="giong_vip_huuduc.onnx", output="out2.mp3")
```
