#!/usr/bin/env python3
"""ZeroTTS ONNX Dynamic Quantization script (FP32 -> INT8).
Reduces model weight sizes by ~4x while maintaining high audio quality.
"""

import argparse
import os
import shutil
from pathlib import Path
import onnxruntime.quantization as oq


def quantize_zerotts(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Đang lượng tử hóa ZeroTTS: {src_dir} -> {dst_dir} ===")

    # 1. Copy config, embeddings, tokenizer
    for item in ["config.json", "null_voice_emb.npy", "silence_frame.npy", "tokenizer.json", "voices"]:
        src_item = src_dir / item
        dst_item = dst_dir / item
        if src_item.is_dir():
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        elif src_item.is_file():
            shutil.copy2(src_item, dst_item)

    # 2. Copy MossCodec ONNX decoder
    dst_codec = dst_dir / "onnx" / "codec"
    dst_codec.mkdir(parents=True, exist_ok=True)
    for item in (src_dir / "onnx" / "codec").glob("*"):
        shutil.copy2(item, dst_codec / item.name)

    # 3. Dynamic Quantize ONNX models to INT8
    onnx_models = ["text_encoder.onnx", "prefix_step.onnx", "local_frame_decode.onnx"]
    for name in onnx_models:
        in_file = src_dir / "onnx" / name
        out_file = dst_dir / "onnx" / name
        if not in_file.exists():
            print(f"Bỏ qua (không tìm thấy): {in_file}")
            continue

        print(f"Đang lượng tử hóa INT8: {name}...")
        oq.quantize_dynamic(
            str(in_file),
            str(out_file),
            weight_type=oq.QuantType.QInt8
        )
        in_mb = os.path.getsize(in_file) / (1024 * 1024)
        out_mb = os.path.getsize(out_file) / (1024 * 1024)
        print(f"  ✓ {name}: {in_mb:.1f} MB -> {out_mb:.1f} MB (Giảm {in_mb/out_mb:.1f}x)")

    print(f"\n✓ Hoàn tất lượng tử hóa INT8 tại: {dst_dir}")


def main():
    parser = argparse.ArgumentParser(description="ZeroTTS ONNX Dynamic Quantizer (FP32 -> INT8)")
    parser.add_argument("--src", type=str, default="/home/quyen/08_TTS/ZeroTTS/model_weights", help="Thư mục weights gốc")
    parser.add_argument("--dst", type=str, default="/home/quyen/08_TTS/ZeroTTS/model_weights_quant", help="Thư mục xuất weights INT8")
    args = parser.parse_args()

    quantize_zerotts(Path(args.src), Path(args.dst))


if __name__ == "__main__":
    main()
