#!/usr/bin/env python3
"""
Kokoro Voice Reverse Trainer & Extractor using ZeroTTS
=====================================================
Huấn luyện và trích xuất âm/giọng nói cho Kokoro-Vietnamese từ ZeroTTS (Reverse Synthetic Training & Distillation).

Quy trình:
1. Dùng ZeroTTS (CUDA/CPU) để đọc tập câu mẫu tiếng Việt chuẩn âm vị (Phoneme-rich sentences),
   tạo tập dữ liệu âm thanh 24000Hz WAV chất lượng cao.
2. Hỗ trợ 2 chế độ:
   - FAST (Mặc định, ~5-10 giây): Trích xuất trực tiếp đặc trưng ngữ âm (Acoustic & Prosodic Style Vectors)
     qua Kokoro StyleEncoder từ checkpoint nền, xuất ra Voicepack .pt (512 KB) và WebAssembly .bin (522 KB).
   - TRAIN (Huấn luyện ngược Stage 1, ~1-2 phút): Tiền xử lý âm vị vig2p -> Cấu hình tự động -> Chạy huấn luyện
     fine-tune trên GPU với PyTorch Accelerate -> Trích xuất voicepack chuẩn xác nhất.
3. Tự động kiểm tra (test synthesis) bằng Kokoro inference và đăng ký giọng vào cả Kokoro & Rust/WASM Runtime.

Sử dụng:
    # 1. Trích xuất siêu tốc từ giọng ZeroTTS (kimoanh, maichi, huuduc, giahuy,...)
    python tools/train_kokoro_from_zerotts.py --voice kimoanh

    # 2. Trích xuất từ giọng lai (Voice Blending)
    python tools/train_kokoro_from_zerotts.py --blend "huuduc:0.7,kimoanh:0.3" --name giong_lai_hd_ko

    # 3. Huấn luyện ngược đầy đủ (Stage 1 Fine-Tuning 5 epochs)
    python tools/train_kokoro_from_zerotts.py --voice maichi --mode train --epochs 5
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Tự động chuyển sang venv nếu chạy bằng python hệ thống chưa có ZeroTTS/Torch
VENV_PYTHON = Path("/home/quyen/08_TTS/ZeroTTS/venv/bin/python3")
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    try:
        import zerotts
    except ImportError:
        print(f"[*] Tự động chuyển sang môi trường venv chuẩn: {VENV_PYTHON}")
        cmd = [str(VENV_PYTHON)] + sys.argv
        os.execv(str(VENV_PYTHON), cmd)

import numpy as np
import scipy.signal
import soundfile as sf
import torch

# ── Định nghĩa các đường dẫn mặc định ──────────────────────────────────────
ZEROTTS_DIR = Path("/home/quyen/08_TTS/ZeroTTS")
ZEROTTS_MODELS = ZEROTTS_DIR / "model_weights"
KOKORO_DIR = Path("/home/quyen/08_TTS/Kokoro-Vietnamese")
KOKORO_CHECKPOINTS = KOKORO_DIR / "checkpoints"
KOKORO_VOICEPACKS = KOKORO_CHECKPOINTS / "voicepacks"
KOKORO_STYLETTS2 = KOKORO_DIR / "StyleTTS2"
WASM_VOICES_DIR = Path("/home/quyen/08_TTS/rust_tts_runtime/wasm/kokoro_voices")

# ── Bộ câu mẫu tiếng Việt chuẩn âm vị (Phoneme-Rich Corpus) ───────────────
DEFAULT_CORPUS = [
    "Gió lạnh rít từng hồi qua hẻm núi hoang vu trên đỉnh Trường Sơn.",
    "Bóng đêm mịt mù bao trùm lên ngôi chùa cổ kính tĩnh mịch nơi chân trời xa xăm.",
    "Vị kiếm khách lặng lẽ đứng tựa vào gốc tùng già, tay nắm chặt đốc kiếm, đôi mắt sắc lạnh.",
    "Mười năm phiêu bạt giang hồ, trải qua bao trận sinh tử, chàng chưa từng lùi bước.",
    "Thế nhưng mối thù gia tộc năm xưa vẫn như ngọn lửa âm ỉ cháy mãi trong lồng ngực.",
    "Trăng thượng tuần lấp ló sau tầng mây xám, hắt xuống mặt đất thứ ánh sáng lờ mờ kỳ ảo.",
    "Một tiếng động khẽ vang lên sau rặng trúc đằng xa làm xao động cả màn đêm.",
    "Chàng khẽ nhếch mép cười, mũi kiếm sắc bén từ từ rời khỏi vỏ trong bóng tối tĩnh mịch.",
    "Ma mà má mả mã mạ là sáu thanh điệu đặc sắc trong hệ thống ngữ âm tiếng Việt.",
    "Xin chào quý thính giả, hôm nay chúng ta cùng khám phá thế giới giọng đọc trí tuệ nhân tạo.",
    "Tôi đang kiểm tra độ chuẩn xác của các âm vực, dấu hỏi ngã và độ ngân vang tự nhiên.",
    "Chiếc điện thoại thông minh thế hệ mới tích hợp chip xử lý nơ-ron thần tốc.",
    "Những hạt sương mai đọng trên cánh hoa lan rừng tỏa ngát hương thơm dịu nhẹ.",
    "Tiếng suối reo róc rách hòa cùng tiếng chim hót véo von báo hiệu một ngày mới rạng rỡ.",
    "Đoàn tàu hỏa lướt nhanh qua những cánh đồng lúa chín vàng óng ả bát ngát mênh mông.",
    "Trí tuệ nhân tạo đang thay đổi cách con người sáng tạo nghệ thuật và tiếp cận tri thức.",
    "Dù trải qua ngàn năm thăng trầm, bản sắc văn hóa dân tộc vẫn trường tồn cùng thời gian.",
    "Đêm khuya thanh vắng, ánh đèn le lói hắt qua khung cửa sổ gác nhỏ đơn sơ.",
    "Người lữ khách nâng chén trà ấm, hồi tưởng lại những năm tháng thanh xuân hào hùng.",
    "Bão tố phong ba không thể nào dập tắt được ý chí quật cường của những người tiên phong.",
    "Tiếng sóng biển vỗ rì rào vào vách đá tạo nên bản hòa ca hùng vĩ của đại dương.",
    "Hương cốm mới thơm lừng len lỏi qua từng con ngõ nhỏ của mùa thu Hà Nội.",
    "Đường về quê mẹ quanh co uốn lượn bên dòng sông xanh biếc rợp bóng tre làng.",
    "Ánh mắt rạng ngời và nụ cười rạng rỡ của các em thơ khi đón nhận món quà xuân.",
    "Quy trình huấn luyện ngược tự động trích xuất phong cách ngữ âm với độ chính xác cao.",
    "Chất giọng truyền cảm, nhịp điệu mượt mà và ngữ điệu chân thực như người thật.",
    "Công nghệ chuyển đổi giọng nói thời gian thực đạt tốc độ nhanh hơn nhiều lần thực tế.",
    "Những bí kíp võ học thượng thừa được ghi chép cẩn thận trên những trang sách cổ phong sương.",
    "Bước chân dồn dập trong rừng trúc rậm rạp làm đàn chim đêm giật mình tung cánh bay vút lên bầu trời.",
    "Hẹn gặp lại bạn trong những chương truyện tiếp theo đầy kịch tính và bất ngờ."
]

def sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", name).strip("_")

def load_zerotts_engine(device: str = "cuda"):
    from zerotts import ZeroTTS
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" and torch.cuda.is_available() else ["CPUExecutionProvider"]
    print(f"[*] Đang nạp mô hình ZeroTTS từ {ZEROTTS_MODELS} trên {providers[0]}...")
    return ZeroTTS.from_pretrained(str(ZEROTTS_MODELS), providers=providers)

def resolve_zerotts_voice(tts, voice: str = None, blend: str = None, voice_bin: str = None):
    """Xác định embedding vector giọng nói cho ZeroTTS."""
    if blend:
        print(f"[*] Đang tính toán ma trận pha trộn giọng (Voice Blending): '{blend}'...")
        pairs = []
        total_w = 0.0
        for part in blend.split(","):
            items = part.strip().split(":")
            v_name = items[0]
            w = float(items[1]) if len(items) > 1 else 1.0
            vec = tts.load_voice(v_name).emb
            pairs.append((vec, w))
            total_w += w
        blended = np.zeros_like(pairs[0][0], dtype=np.float32)
        for vec, w in pairs:
            blended += vec * (w / total_w)
        return blended, sanitize_name(blend)

    if voice_bin:
        p = Path(voice_bin)
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy file voice_bin: {voice_bin}")
        raw = np.fromfile(p, dtype=np.float32).reshape((1, 10, 768))
        return raw, sanitize_name(p.stem)

    if voice:
        avail = tts.list_voices()
        if voice not in avail:
            print(f"[!] Cảnh báo: Giọng '{voice}' không có trong ZeroTTS. Có sẵn: {avail}")
            voice = "kimoanh"
        return voice, sanitize_name(voice)

    return "kimoanh", "kimoanh"

def generate_synthetic_audio(tts, voice_obj, sentences: list[str], output_wav_dir: Path):
    """Dùng ZeroTTS đọc các câu mẫu, resample sang 24000Hz WAV cho Kokoro."""
    output_wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = []
    print(f"[*] Bắt đầu tổng hợp {len(sentences)} file âm thanh chuẩn 24kHz bằng ZeroTTS...")
    t0 = time.time()

    for idx, sentence in enumerate(sentences, start=1):
        filename = f"sample_{idx:04d}.wav"
        filepath = output_wav_dir / filename
        
        # ZeroTTS synthesize (48kHz)
        audio = tts.synthesize(sentence, voice=voice_obj).squeeze()
        # Resample poly 48kHz -> 24kHz chuẩn xác
        audio_24k = scipy.signal.resample_poly(audio, 1, 2)
        # Chuẩn hóa âm lượng peak tránh rè
        max_val = np.max(np.abs(audio_24k))
        if max_val > 0.01:
            audio_24k = audio_24k / max_val * 0.95

        sf.write(str(filepath), audio_24k.astype(np.float32), 24000)
        metadata_rows.append(f"audio/{filename}|{sentence}")
        print(f"  [{idx:02d}/{len(sentences):02d}] {filename} ({len(audio_24k)/24000:.1f}s): \"{sentence[:45]}...\"")

    elapsed = time.time() - t0
    print(f"✓ Hoàn tất tạo {len(sentences)} câu mẫu trong {elapsed:.2f}s")
    
    meta_file = output_wav_dir.parent / "metadata.csv"
    meta_file.write_text("\n".join(metadata_rows) + "\n", encoding="utf-8")
    return meta_file

def find_first_stage_checkpoint():
    """Tìm checkpoint có StyleEncoder (first_stage.pth)."""
    candidates = [
        KOKORO_STYLETTS2 / "logs" / "huuduc" / "first_stage.pth",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Quét tất cả logs
    all_ckpts = list((KOKORO_STYLETTS2 / "logs").glob("**/first_stage.pth"))
    if all_ckpts:
        return all_ckpts[0]
    return None

def calibrate_voicepack(raw_pt_path: Path, target_pt_path: Path):
    """Căn chuẩn L2-Norm cho Voicepack Kokoro chống rè."""
    raw = torch.load(raw_pt_path, map_location="cpu", weights_only=True)
    ac = raw[:, :, :128]
    pr = raw[:, :, 128:]
    ac_norm = torch.norm(ac, dim=-1, keepdim=True)
    ac_cal = ac / (ac_norm + 1e-8) * 1.38

    ref_candidates = [
        KOKORO_VOICEPACKS / "diem_trinh.pt",
        KOKORO_VOICEPACKS / "mai_linh.pt",
        KOKORO_VOICEPACKS / "hung_thinh.pt"
    ]
    ref_pt = next((p for p in ref_candidates if p.exists()), None)
    if ref_pt:
        ref = torch.load(ref_pt, map_location="cpu", weights_only=True)
        pr_ref = ref[:, :, 128:]
        pr_norm = torch.norm(pr, dim=-1, keepdim=True)
        pr_unit = pr / (pr_norm + 1e-8) * 1.95
        pr_cal = 0.35 * pr_unit + 0.65 * pr_ref
    else:
        pr_norm = torch.norm(pr, dim=-1, keepdim=True)
        pr_cal = pr / (pr_norm + 1e-8) * 1.95

    calibrated = torch.cat([ac_cal, pr_cal], dim=-1)
    target_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, target_pt_path)
    return calibrated

def extract_kokoro_voicepack(audio_dir: Path, output_pt: Path, ckpt_path: Path, device: str = "cuda"):
    """Chạy trích xuất đặc trưng âm sắc sang Voicepack .pt [510, 1, 256] và calibrate norm."""
    extract_script = KOKORO_DIR / "scripts" / "extract_voicepack.py"
    raw_pt = output_pt.parent / f"{output_pt.stem}_raw.pt"
    cmd = [
        sys.executable,
        str(extract_script),
        "--model", str(ckpt_path),
        "--audio-dir", str(audio_dir),
        "--output", str(raw_pt),
        "--device", device
    ]
    print(f"[*] Đang trích xuất Style Vectors từ {audio_dir} -> {raw_pt}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(res.stderr)
        raise RuntimeError(f"Lỗi khi trích xuất voicepack:\n{res.stderr}\n{res.stdout}")
    
    # Căn chuẩn L2 norm chống rè
    calibrate_voicepack(raw_pt, output_pt)
    if raw_pt.exists():
        raw_pt.unlink()
    print(f"✓ Đã tạo thành công Kokoro Voicepack (chuẩn hóa chống rè): {output_pt} ({output_pt.stat().st_size / 1024:.1f} KB)")

def export_wasm_bin(voicepack_pt: Path, output_bin: Path):
    """Chuyển tensor [510, 1, 256] float32 sang định dạng nhị phân thuần .bin (522,240 bytes)."""
    tensor = torch.load(voicepack_pt, map_location="cpu", weights_only=True)
    raw_bytes = tensor.detach().cpu().numpy().astype(np.float32).tobytes()
    output_bin.parent.mkdir(parents=True, exist_ok=True)
    output_bin.write_bytes(raw_bytes)
    print(f"✓ Đã xuất file WebAssembly / Rust Runtime: {output_bin} ({len(raw_bytes):,} bytes)")

def register_voice(name: str, label: str, voicepack_pt: Path, voicepack_bin: Path):
    """Tự động đăng ký giọng vào voices.json của Kokoro và copy vào wasm_voices."""
    # 1. Copy sang KOKORO_VOICEPACKS
    KOKORO_VOICEPACKS.mkdir(parents=True, exist_ok=True)
    target_pt = KOKORO_VOICEPACKS / f"{name}.pt"
    if voicepack_pt.resolve() != target_pt.resolve():
        shutil.copyfile(voicepack_pt, target_pt)

    # 2. Update voices.json
    v_json = KOKORO_CHECKPOINTS / "voices.json"
    if v_json.exists():
        try:
            data = json.loads(v_json.read_text(encoding="utf-8"))
            data[name] = {
                "label": label or name.replace("_", " ").title(),
                "filename": f"voicepacks/{name}.pt"
            }
            v_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"✓ Đã đăng ký giọng '{name}' vào {v_json}")
        except Exception as e:
            print(f"[!] Không thể cập nhật voices.json: {e}")

    # 3. Copy sang wasm_voices
    if WASM_VOICES_DIR.exists():
        target_bin = WASM_VOICES_DIR / f"{name}.bin"
        if voicepack_bin.resolve() != target_bin.resolve():
            shutil.copyfile(voicepack_bin, target_bin)
            print(f"✓ Đã đồng bộ sang Thư viện Web Studio: {target_bin}")

def run_stage1_training(dataset_name: str, src_dir: Path, meta_file: Path, epochs: int = 5, batch_size: int = 2):
    """Huấn luyện fine-tune ngược Stage 1 trên GPU CUDA bằng PyTorch Accelerate."""
    work_dataset_dir = KOKORO_DIR / f"dataset_{dataset_name}"
    output_training = KOKORO_DIR / "training" / dataset_name
    
    print(f"\n[1/3] Tiền xử lý âm vị học (vig2p frontend)...")
    prep_cmd = [
        sys.executable,
        str(KOKORO_DIR / "scripts" / "prepare_vietnamese_dataset.py"),
        "--source-root", str(src_dir),
        "--metadata", str(meta_file),
        "--output-dataset", str(work_dataset_dir),
        "--output-training", str(output_training),
        "--min-duration", "0.2",
        "--clean"
    ]
    p_res = subprocess.run(prep_cmd, capture_output=True, text=True)
    if p_res.returncode != 0:
        raise RuntimeError(f"Lỗi khi chạy prepare_vietnamese_dataset.py:\n{p_res.stderr}\n{p_res.stdout}")

    # Đảm bảo val_list có dữ liệu
    val_file = output_training / "val_list.txt"
    train_file = output_training / "train_list.txt"
    if val_file.exists() and train_file.exists():
        if not val_file.read_text(encoding="utf-8").strip():
            val_file.write_text(train_file.read_text(encoding="utf-8").strip() + "\n", encoding="utf-8")

    # Tạo file config stage 1
    print(f"[2/3] Sinh file cấu hình huấn luyện Stage 1...")
    configs_dir = KOKORO_DIR / "configs"
    base_config = (configs_dir / "config_vietnamese_30h_stage1.yml").read_text(encoding="utf-8")
    custom_cfg = base_config
    custom_cfg = custom_cfg.replace('log_dir: "logs/kokoro-vi-30h"', f'log_dir: "logs/{dataset_name}"')
    custom_cfg = custom_cfg.replace('train_data: "../training/vi_30h/train_list.txt"', f'train_data: "../training/{dataset_name}/train_list.txt"')
    custom_cfg = custom_cfg.replace('val_data: "../training/vi_30h/val_list.txt"', f'val_data: "../training/{dataset_name}/val_list.txt"')
    custom_cfg = custom_cfg.replace('root_path: "../dataset_vi_30h"', f'root_path: "../dataset_{dataset_name}"')
    custom_cfg = custom_cfg.replace('OOD_data: "../training/vi_30h/OOD_texts.txt"', f'OOD_data: "../training/{dataset_name}/OOD_texts.txt"')
    custom_cfg = custom_cfg.replace('batch_size: 6', f'batch_size: {int(batch_size)}')
    custom_cfg = custom_cfg.replace('epochs_1st: 20', f'epochs_1st: {int(epochs)}')
    custom_cfg = custom_cfg.replace('epochs: 20', f'epochs: {int(epochs)}')
    custom_cfg = custom_cfg.replace('num_workers: 24', 'num_workers: 2')
    custom_cfg = custom_cfg.replace('min_length: 50', 'min_length: 30')

    cfg_path = configs_dir / f"config_{dataset_name}_stage1.yml"
    cfg_path.write_text(custom_cfg, encoding="utf-8")

    # Đảm bảo symlink base checkpoint
    base_pth = KOKORO_DIR / "training" / "kokoro_base.pth"
    if not base_pth.exists():
        base_pth.symlink_to(KOKORO_CHECKPOINTS / "kokoro_vi.pth")

    print(f"[3/3] Khởi động Huấn luyện PyTorch GPU CUDA ({epochs} epochs, batch {batch_size})...")
    cmd_train = [
        sys.executable,
        "-m", "accelerate.commands.launch",
        "train_first.py",
        "--config_path", str(cfg_path)
    ]
    proc = subprocess.Popen(
        cmd_train,
        cwd=str(KOKORO_STYLETTS2),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    for line in iter(proc.stdout.readline, ''):
        l = line.strip()
        if any(k in l for k in ["Epoch", "Validation loss:", "Saving..", "error", "Error"]):
            print(f"  | {l}")
    proc.wait()

    first_stage = KOKORO_STYLETTS2 / "logs" / dataset_name / "first_stage.pth"
    if not first_stage.exists():
        raise RuntimeError(f"Huấn luyện không tạo ra first_stage.pth tại {first_stage}")

    print(f"🎉 Huấn luyện hoàn tất! Checkpoint: {first_stage}")
    return first_stage, work_dataset_dir / "audio"

def test_synthesize_voice(voicepack_pt: Path, test_text: str, output_wav: Path, device: str = "cuda"):
    """Thử nghiệm tổng hợp ngay giọng mới bằng Kokoro engine."""
    print(f"\n[*] Đang tổng hợp kiểm tra âm thanh bằng Kokoro với giọng mới...")
    sys.path.insert(0, str(KOKORO_DIR / "src"))
    from kokoro_vietnamese.core import KokoroVietnamese

    tts = KokoroVietnamese(
        model_path=str(KOKORO_CHECKPOINTS / "kokoro_vi.pth"),
        config_path=str(KOKORO_CHECKPOINTS / "config.json"),
        voicepack_path=str(voicepack_pt),
        device=device
    )
    t0 = time.time()
    audio, _ = tts.synthesize(test_text)
    elapsed = time.time() - t0
    dur = len(audio) / 24000
    rtf = elapsed / dur if dur > 0 else 0

    sf.write(str(output_wav), audio, 24000)
    print(f"✓ Đã lưu file audio nghe thử: {output_wav}")
    print(f"  • Thời lượng: {dur:.2f}s | Thời gian sinh: {elapsed*1000:.1f}ms | RTF: {rtf:.4f}x (Nhanh {1/rtf:.1f}x real-time)")

def main():
    parser = argparse.ArgumentParser(
        description="Huấn luyện và trích xuất âm Kokoro ngược bằng ZeroTTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--voice", "-v", type=str, default="kimoanh", help="Tên giọng ZeroTTS (kimoanh, maichi, huuduc, giahuy,...)")
    parser.add_argument("--blend", "-b", type=str, help="Chuỗi pha trộn giọng ZeroTTS (ví dụ: 'huuduc:0.7,kimoanh:0.3')")
    parser.add_argument("--voice-bin", type=str, help="Đường dẫn trực tiếp tới file .bin (30KB) vector latent của ZeroTTS")
    parser.add_argument("--name", "-n", type=str, help="Tên giọng Kokoro xuất ra (mặc định lấy theo tên giọng ZeroTTS)")
    parser.add_argument("--label", type=str, help="Tên hiển thị người đọc (ví dụ: 'Kim Oanh (ZeroTTS)')")
    parser.add_argument("--mode", "-m", choices=["fast", "train"], default="fast", help="Chế độ: fast (trích xuất nhanh 5-10s) hoặc train (huấn luyện Stage 1 ~1-2 phút)")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Số epochs khi chạy mode train (mặc định: 5)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size khi huấn luyện (mặc định: 2)")
    parser.add_argument("--num-samples", type=int, default=None, help="Số câu audio sinh ra từ ZeroTTS (mặc định: 8 cho fast, 24 cho train)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Thiết bị tính toán (mặc định: cuda)")
    parser.add_argument("--output-dir", "-o", type=str, default=".", help="Thư mục xuất file .pt và .bin")
    parser.add_argument("--test-text", type=str, default="Xin chào, đây là giọng đọc tiếng Việt được chuyển giao âm sắc từ ZeroTTS sang Kokoro.", help="Câu văn thử nghiệm âm thanh sau khi hoàn thành")
    parser.add_argument("--test-output", type=str, help="Đường dẫn file audio test xuất ra (mặc định: test_kokoro_<name>.wav)")
    
    args = parser.parse_args()

    # Kiểm tra phần cứng
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[!] Không phát hiện NVIDIA CUDA GPU. Tự động chuyển sang CPU.")
        args.device = "cpu"

    # 1. Khởi tạo ZeroTTS và nhận diện giọng
    tts_zero = load_zerotts_engine(device=args.device)
    voice_obj, def_name = resolve_zerotts_voice(tts_zero, voice=args.voice, blend=args.blend, voice_bin=args.voice_bin)
    
    voice_name = sanitize_name(args.name) if args.name else def_name
    display_label = args.label or voice_name.replace("_", " ").title()

    num_samples = args.num_samples or (8 if args.mode == "fast" else 24)
    sentences = DEFAULT_CORPUS[:num_samples]

    print("\n" + "="*70)
    print(f"🚀 BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN ÂM KOKORO NGƯỢC BẰNG ZEROTTS")
    print(f"  • Giọng nguồn:       {args.blend or args.voice_bin or args.voice}")
    print(f"  • Tên giọng Kokoro:  {voice_name}")
    print(f"  • Chế độ:            {args.mode.upper()}")
    print(f"  • Số câu mẫu:        {len(sentences)}")
    print(f"  • Thiết bị:          {args.device.upper()}")
    print("="*70 + "\n")

    # 2. Sinh tập dữ liệu audio mẫu từ ZeroTTS
    work_temp_dir = Path(f"/tmp/zerotts_to_kokoro_{voice_name}")
    wav_dir = work_temp_dir / "audio"
    meta_file = generate_synthetic_audio(tts_zero, voice_obj, sentences, wav_dir)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pt = out_dir / f"{voice_name}.pt"
    out_bin = out_dir / f"{voice_name}.bin"

    # 3. Thực hiện chế độ FAST hoặc TRAIN
    if args.mode == "fast":
        ckpt = find_first_stage_checkpoint()
        if not ckpt:
            print("[!] Không tìm thấy first_stage.pth. Chuyển sang huấn luyện Stage 1...")
            ckpt, audio_dir = run_stage1_training(voice_name, work_temp_dir, meta_file, epochs=args.epochs, batch_size=args.batch_size)
        else:
            audio_dir = wav_dir
        extract_kokoro_voicepack(audio_dir, out_pt, ckpt, device=args.device)
    else:
        ckpt, audio_dir = run_stage1_training(voice_name, work_temp_dir, meta_file, epochs=args.epochs, batch_size=args.batch_size)
        extract_kokoro_voicepack(audio_dir, out_pt, ckpt, device=args.device)

    # 4. Xuất file WebAssembly / Rust Runtime (.bin)
    export_wasm_bin(out_pt, out_bin)

    # 5. Tự động đăng ký và đồng bộ
    register_voice(voice_name, display_label, out_pt, out_bin)

    # 6. Test thử nghiệm giọng nói
    test_wav_path = Path(args.test_output) if args.test_output else out_dir / f"test_kokoro_{voice_name}.wav"
    test_synthesize_voice(out_pt, args.test_text, test_wav_path, device=args.device)

    print("\n" + "="*70)
    print("🎉 HOÀN TẤT THÀNH CÔNG!")
    print(f"  • File Voicepack PyTorch:     {out_pt} ({out_pt.stat().st_size / 1024:.1f} KB)")
    print(f"  • File WebAssembly/Rust .bin: {out_bin} ({out_bin.stat().st_size / 1024:.1f} KB)")
    print(f"  • Audio Test kiểm tra:       {test_wav_path}")
    print(f"  • Trạng thái:                 Đã nạp vào Kokoro & Thư viện Web Studio sẵn sàng sử dụng!")
    print("="*70)

if __name__ == "__main__":
    main()
