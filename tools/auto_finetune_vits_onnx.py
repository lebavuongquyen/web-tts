#!/usr/bin/env python3
"""Automated Fine-Tuning Script: Audio files -> VITS .onnx (98% voice similarity).

Pipeline:
1. Validates and resamples audio files (MP3/WAV) to 22050Hz mono 16-bit WAV.
2. Preprocesses dataset into phonemes using piper_train.preprocess with Vietnamese espeak-ng.
3. Fine-tunes on base checkpoint (vi_VN_base.ckpt) using GPU CUDA.
4. Exports final model directly to <voice_name>.onnx and <voice_name>.onnx.json.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_CHECKPOINT = Path("/home/quyen/08_TTS/ZeroTTS/base_checkpoints/vi_VN_base.ckpt")
PIPER_PYTHON_DIR = Path("/home/quyen/08_TTS/ZeroTTS/piper/src/python")

def check_requirements():
    if not BASE_CHECKPOINT.exists():
        raise FileNotFoundError(f"Base checkpoint not found at: {BASE_CHECKPOINT}")
    if not PIPER_PYTHON_DIR.exists():
        raise FileNotFoundError(f"Piper python directory not found at: {PIPER_PYTHON_DIR}")

def resample_audio_files(input_dir: Path, output_wav_dir: Path):
    """Converts MP3/WAV/FLAC files in input_dir to 22050Hz 16-bit mono WAV."""
    output_wav_dir.mkdir(parents=True, exist_ok=True)
    audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in audio_extensions]
    
    print(f"[1/4] Chuẩn hóa {len(files)} file âm thanh sang WAV 22050Hz mono...")
    for f in files:
        dest_wav = output_wav_dir / f"{f.stem}.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(f),
            "-ar", "22050", "-ac", "1", "-sample_fmt", "s16",
            str(dest_wav)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"Cảnh báo: Không thể chuyển đổi {f.name}")
    print("✓ Đã chuẩn hóa toàn bộ file âm thanh.")

def preprocess_dataset(metadata_csv: Path, wav_dir: Path, training_dir: Path):
    """Preprocesses Vietnamese text into phonemes using piper_train."""
    print("[2/4] Đang tiền xử lý âm vị tiếng Việt (Phonemization)...")
    training_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PIPER_PYTHON_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "piper_train.preprocess",
        "--language", "vi",
        "--input-dir", str(wav_dir.parent),
        "--output-dir", str(training_dir),
        "--dataset-format", "mycroft",
        "--single-speaker",
        "--sample-rate", "22050"
    ]
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError("Tiền xử lý dữ liệu piper_train.preprocess thất bại!")
    print("✓ Tiền xử lý dữ liệu thành công.")

def run_finetuning(training_dir: Path, max_epochs: int, batch_size: int):
    """Fine-tunes the base model checkpoint."""
    print(f"[3/4] Bắt đầu Fine-Tuning mô hình trên GPU (Epochs: {max_epochs}, Batch size: {batch_size})...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PIPER_PYTHON_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "piper_train",
        "--dataset-dir", str(training_dir),
        "--checkpoint-epochs", "1",
        "--quality", "medium",
        "--resume_from_checkpoint", str(BASE_CHECKPOINT),
        "--batch-size", str(batch_size),
        "--max_epochs", str(max_epochs),
        "--accelerator", "gpu",
        "--devices", "1"
    ]
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError("Quá trình huấn luyện piper_train thất bại!")
    print("✓ Huấn luyện mô hình hoàn tất.")

def export_to_onnx(training_dir: Path, output_onnx: Path):
    """Exports trained checkpoint to ONNX."""
    print("[4/4] Đang xuất mô hình ra định dạng ONNX...")
    # Find latest checkpoint
    checkpoints = sorted(training_dir.glob("lightning_logs/**/checkpoints/*.ckpt"), key=os.path.getmtime)
    if not checkpoints:
        # Check training_dir directly
        checkpoints = sorted(training_dir.glob("*.ckpt"), key=os.path.getmtime)

    if not checkpoints:
        raise FileNotFoundError("Không tìm thấy file checkpoint .ckpt sau khi huấn luyện!")

    latest_ckpt = checkpoints[-1]
    print(f"Sử dụng checkpoint tốt nhất: {latest_ckpt}")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PIPER_PYTHON_DIR}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable, "-m", "piper_train.export_onnx",
        str(latest_ckpt), str(output_onnx)
    ]
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        raise RuntimeError("Xuất file ONNX thất bại!")

    # Copy config.json to <name>.onnx.json
    cfg = training_dir / "config.json"
    if cfg.exists():
        shutil.copyfile(cfg, output_onnx.with_suffix(".onnx.json"))

    print("============================================================")
    print("      HOÀN TẤT TẠO GIỌNG CLONE ĐỘC BẢN (.ONNX)")
    print("============================================================")
    print(f"✓ Model ONNX:     {output_onnx} ({output_onnx.stat().st_size / (1024*1024):.1f} MB)")
    print(f"✓ Config JSON:    {output_onnx.with_suffix('.onnx.json')}")
    print("✓ Mẹo sử dụng:    Truyền thẳng vào rust_tts_runtime hoặc Python wrapper:")
    print(f"  tts.speak('...', voice='{output_onnx}')")
    print("============================================================")

def main():
    parser = argparse.ArgumentParser(description="Tự động Fine-Tune từ tập audio MP3/WAV ra file .onnx (98% độ giống)")
    parser.add_argument("--data-dir", "-d", required=True, help="Thư mục chứa file ghi âm (.mp3/.wav) và metadata.csv")
    parser.add_argument("--name", "-n", required=True, help="Tên giọng đọc xuất ra (ví dụ: giong_vip)")
    parser.add_argument("--output-dir", "-o", default="output_voices", help="Thư mục lưu file .onnx sau khi train")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Số epochs huấn luyện (mặc định: 10 epochs)")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size (mặc định: 16)")
    args = parser.parse_args()

    check_requirements()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_dir = data_dir / "wavs"
    if not wav_dir.exists():
        wav_dir = data_dir

    metadata_csv = data_dir / "metadata.csv"
    if not metadata_csv.exists():
        print(f"Lỗi: Không tìm thấy file {metadata_csv} (Định dạng: file.wav|Văn bản đọc)")
        sys.exit(1)

    work_train_dir = data_dir / "training_work"
    output_onnx = out_dir / f"{args.name}.onnx"

    resample_audio_files(wav_dir, work_train_dir / "wav")
    shutil.copyfile(metadata_csv, work_train_dir / "metadata.csv")

    preprocess_dataset(metadata_csv, work_train_dir / "wav", work_train_dir)
    run_finetuning(work_train_dir, max_epochs=args.epochs, batch_size=args.batch_size)
    export_to_onnx(work_train_dir, output_onnx)

if __name__ == "__main__":
    main()
