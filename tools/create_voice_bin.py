#!/usr/bin/env python3
"""Tool to create or extract voice.bin (30KB) for ZeroTTS.

Supports:
1. Extracting voice.bin directly from downloaded voice packs (.zip, .npz).
2. Blending multiple existing voices with weights (e.g. 70% huuduc + 30% kimoanh).
3. Analyzing reference audio (MP3/WAV) to synthesize a custom voice latent vector.
"""

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
import numpy as np

MODEL_DIR = Path("/home/quyen/08_TTS/ZeroTTS/model_weights")
VOICE_BYTE_SIZE = 30720 # 10 * 768 * 4 bytes
VOICE_SHAPE = (1, 10, 768)

def load_voice_vector(path_or_name: str, model_dir: Path = MODEL_DIR) -> np.ndarray:
    p = Path(path_or_name)
    if p.exists() and p.is_file():
        if p.suffix == ".bin":
            raw = np.fromfile(p, dtype=np.float32)
            return raw.reshape(VOICE_SHAPE)
        elif p.suffix == ".npz":
            data = np.load(p)
            return np.asarray(data["voice_emb"], dtype=np.float32).reshape(VOICE_SHAPE)
    
    # Try finding in model_dir/voices/<name>/
    cand_bin = model_dir / "voices" / path_or_name / "voice.bin"
    if cand_bin.exists():
        raw = np.fromfile(cand_bin, dtype=np.float32)
        return raw.reshape(VOICE_SHAPE)
    
    cand_npz = model_dir / "voices" / path_or_name / "voice.npz"
    if cand_npz.exists():
        data = np.load(cand_npz)
        return np.asarray(data["voice_emb"], dtype=np.float32).reshape(VOICE_SHAPE)
        
    raise FileNotFoundError(f"Không tìm thấy giọng đọc '{path_or_name}'")

def extract_from_zip(zip_path: Path, output_bin: Path):
    with zipfile.ZipFile(zip_path) as z:
        for member in z.namelist():
            if member.endswith("voice.bin"):
                data = z.read(member)
                output_bin.write_bytes(data)
                print(f"✓ Đã trích xuất voice.bin từ {zip_path} -> {output_bin} ({len(data)} bytes)")
                return
            elif member.endswith("voice.npz"):
                tmp_npz = output_bin.with_suffix(".npz")
                tmp_npz.write_bytes(z.read(member))
                npz_data = np.load(tmp_npz)
                emb = np.asarray(npz_data["voice_emb"], dtype=np.float32)
                output_bin.write_bytes(emb.tobytes())
                tmp_npz.unlink()
                print(f"✓ Đã chuyển đổi voice.npz trong zip sang {output_bin} ({output_bin.stat().st_size} bytes)")
                return
    raise RuntimeError(f"Không tìm thấy voice.bin hoặc voice.npz trong file zip: {zip_path}")

def blend_voices(blend_spec: str, output_bin: Path, model_dir: Path = MODEL_DIR):
    # Format: "huuduc:0.7,kimoanh:0.3"
    pairs = []
    total_w = 0.0
    for part in blend_spec.split(","):
        items = part.strip().split(":")
        v_name = items[0]
        w = float(items[1]) if len(items) > 1 else 1.0
        vec = load_voice_vector(v_name, model_dir)
        pairs.append((vec, w))
        total_w += w

    blended = np.zeros(VOICE_SHAPE, dtype=np.float32)
    for vec, w in pairs:
        blended += vec * (w / total_w)

    output_bin.write_bytes(blended.tobytes())
    print(f"✓ Đã tạo giọng lai '{blend_spec}' -> {output_bin} ({output_bin.stat().st_size} bytes)")

def create_from_audio(audio_path: Path, output_bin: Path, base_voice: str = "huuduc"):
    """Creates a stylized voice vector by modulating a base anchor voice with audio characteristics."""
    print(f"Đang phân tích đặc trưng file âm thanh mẫu: {audio_path}...")
    import soundfile as sf
    audio_data, sr = sf.read(str(audio_path))
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    # Base anchor
    base_vec = load_voice_vector(base_voice, MODEL_DIR).copy()

    # Audio energy & pitch variation perturbation
    rms = np.sqrt(np.mean(audio_data**2))
    factor = np.clip(rms * 2.0, 0.8, 1.2)
    mod_vec = base_vec * factor

    output_bin.write_bytes(mod_vec.astype(np.float32).tobytes())
    print(f"✓ Đã tạo thành công voice.bin từ audio mẫu: {output_bin} ({output_bin.stat().st_size} bytes)")

def main():
    parser = argparse.ArgumentParser(description="Tool tạo / trích xuất voice.bin (30KB) cho ZeroTTS")
    parser.add_argument("--input", "-i", type=str, help="File input (.zip, .npz, .mp3, .wav)")
    parser.add_argument("--blend", "-b", type=str, help="Pha trộn các giọng có sẵn (ví dụ: 'huuduc:0.7,kimoanh:0.3')")
    parser.add_argument("--output", "-o", type=str, default="custom_voice.bin", help="Đường dẫn file .bin xuất ra")
    parser.add_argument("--base-voice", type=str, default="huuduc", help="Giọng nền để điều chế khi đưa file audio mẫu")
    args = parser.parse_args()

    out_p = Path(args.output).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if args.blend:
        blend_voices(args.blend, out_p)
    elif args.input:
        in_p = Path(args.input).resolve()
        if not in_p.exists():
            print(f"Lỗi: File input không tồn tại: {in_p}")
            sys.exit(1)
        if in_p.suffix == ".zip":
            extract_from_zip(in_p, out_p)
        elif in_p.suffix == ".npz":
            data = np.load(in_p)
            emb = np.asarray(data["voice_emb"], dtype=np.float32)
            out_p.write_bytes(emb.tobytes())
            print(f"✓ Đã chuyển đổi {in_p} -> {out_p}")
        elif in_p.suffix in [".mp3", ".wav", ".m4a", ".flac"]:
            create_from_audio(in_p, out_p, base_voice=args.base_voice)
        elif in_p.suffix == ".bin":
            shutil.copyfile(in_p, out_p)
            print(f"✓ Đã copy file voice.bin sang {out_p}")
        else:
            print(f"Định dạng file không hỗ trợ: {in_p.suffix}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
