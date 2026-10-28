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
python tools/create_voice_bin.py \
  --blend "huuduc:0.7,kimoanh:0.3" \
  --output custom_blend.bin
```

### Cách 2: Trích xuất từ file zip hoặc npz tải về
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
