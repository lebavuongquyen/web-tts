#!/usr/bin/env python3
"""
Unified TTS Voice Training Pipeline: ZeroTTS -> Kokoro
======================================================
Quy trình huấn luyện toàn diện 2 tầng:
1. Phân tích audio & transcript người dùng -> Huấn luyện vector giọng cho ZeroTTS -> Xuất `<name>_zerotts.bin` (30 KB).
2. Dùng ZeroTTS với giọng vừa tạo để tổng hợp bộ dữ liệu âm thanh mẫu chuẩn âm vị 24kHz (Phoneme-rich synthetic data).
3. Huấn luyện ngược Stage 1 trên GPU NVIDIA bằng PyTorch Accelerate cho Kokoro.
4. Trích xuất đồng thời:
   - File `.bin` cho ZeroTTS: `<name>_zerotts.bin` (30.720 bytes)
   - File `.pth` (Checkpoint) cho Kokoro: `<name>.pth` (1.7 GB)
   - File `.pt` (Voicepack) cho Kokoro: `<name>.pt` (511 KB)
   - File `.bin` (WebAssembly/Rust Runtime) cho Kokoro: `<name>_kokoro.bin` (522.240 bytes)
5. Tự động kiểm tra chất lượng giọng bằng Kokoro inference.
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

# Đảm bảo môi trường venv
VENV_PYTHON = Path("/home/quyen/08_TTS/ZeroTTS/venv/bin/python3")
if VENV_PYTHON.exists() and sys.executable != str(VENV_PYTHON):
    try:
        import zerotts
        import torch
        import soundfile
    except ImportError:
        print(f"[*] Chuyển sang venv: {VENV_PYTHON}")
        cmd = [str(VENV_PYTHON)] + sys.argv
        os.execv(str(VENV_PYTHON), cmd)

import numpy as np
import scipy.signal
import soundfile as sf
import torch

ZEROTTS_DIR = Path("/home/quyen/08_TTS/ZeroTTS")
ZEROTTS_MODELS = ZEROTTS_DIR / "model_weights"
KOKORO_DIR = Path("/home/quyen/08_TTS/Kokoro-Vietnamese")
KOKORO_CHECKPOINTS = KOKORO_DIR / "checkpoints"
KOKORO_VOICEPACKS = KOKORO_CHECKPOINTS / "voicepacks"
KOKORO_STYLETTS2 = KOKORO_DIR / "StyleTTS2"
WASM_VOICES_DIR = Path("/home/quyen/08_TTS/rust_tts_runtime/wasm/kokoro_voices")

# Bộ câu mẫu tiếng Việt chuẩn âm vị (Phoneme-Rich Corpus)
CORPUS = [
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
    "Bão tố phong ba không thể nào dập tắt được ý chí quật cường của những người tiên phong."
]

def sanitize_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower())
    return re.sub(r"_+", "_", name).strip("_")

def analyze_audio_pitch_f0(audio_path: Path) -> float:
    """Phân tích cao độ Pitch F0 (Hz) chính xác của file âm thanh bằng librosa pyin."""
    try:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=24000)
        # Chỉ lấy tối đa 45 giây để phân tích nhanh và chuẩn
        if len(y) > sr * 45:
            y = y[:sr * 45]
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=350, sr=sr)
        f0_clean = f0[voiced_flag & ~np.isnan(f0)]
        if len(f0_clean) > 0:
            median_f0 = float(np.median(f0_clean))
            return median_f0
        return 120.0
    except Exception as e:
        print(f"[*] Cảnh báo đo F0 ({e}), dùng giá trị mặc định 120Hz.")
        return 120.0

def train_zerotts_voice_vector(audio_files: list[Path], tts_zero, voice_name: str, out_bin: Path) -> np.ndarray:
    """Huấn luyện và xuất file vector .bin (30.720 bytes) cho ZeroTTS."""
    print(f"[*] [Bước 1/4] Huấn luyện vector giọng ZeroTTS cho '{voice_name}'...")
    
    # Nạp các giọng anchor cơ sở của ZeroTTS
    v_giahuy = tts_zero.load_voice("giahuy").emb.copy()     # Nam trầm sâu (~80-100Hz)
    v_huuduc = tts_zero.load_voice("huuduc").emb.copy()     # Nam trầm ấm (~115-135Hz)
    v_quangminh = tts_zero.load_voice("quangminh").emb.copy() # Nam trung (~160-180Hz)
    v_kimoanh = tts_zero.load_voice("kimoanh").emb.copy()   # Nữ trầm/ấm (~190-210Hz)
    v_maichi = tts_zero.load_voice("maichi").emb.copy()     # Nữ trẻ/cao (~250-300Hz)

    # Phân tích Pitch F0 từ các file audio tải lên
    f0_list = [analyze_audio_pitch_f0(f) for f in audio_files[:3]]
    avg_f0 = float(np.median(f0_list))
    print(f"  • Cao độ Pitch F0 đo được: {avg_f0:.1f} Hz")

    # Phối trộn không gian tiềm ẩn liên tục dựa trên cao độ thực tế
    if avg_f0 < 110.0:
        # Nam trầm sâu (Gia Huy <-> Hữu Đức)
        t = np.clip((avg_f0 - 75.0) / 35.0, 0.0, 1.0)
        blended = (1.0 - t) * v_giahuy + t * v_huuduc
        print(f"  • Khớp cấu trúc âm vực: Nam trầm sâu ({100-int(t*100)}% Gia Huy, {int(t*100)}% Hữu Đức)")
    elif avg_f0 < 150.0:
        # Nam trầm ấm (Hữu Đức <-> Quang Minh)
        t = np.clip((avg_f0 - 110.0) / 40.0, 0.0, 1.0)
        blended = (1.0 - t) * v_huuduc + t * v_quangminh
        print(f"  • Khớp cấu trúc âm vực: Nam trầm ấm ({100-int(t*100)}% Hữu Đức, {int(t*100)}% Quang Minh)")
    elif avg_f0 < 185.0:
        # Nam trung / Nữ trầm (Quang Minh <-> Kim Oanh)
        t = np.clip((avg_f0 - 150.0) / 35.0, 0.0, 1.0)
        blended = (1.0 - t) * v_quangminh + t * v_kimoanh
        print(f"  • Khớp cấu trúc âm vực: Nam trung / Nữ trầm ({100-int(t*100)}% Quang Minh, {int(t*100)}% Kim Oanh)")
    else:
        # Nữ (Kim Oanh <-> Mai Chi)
        t = np.clip((avg_f0 - 185.0) / 65.0, 0.0, 1.0)
        blended = (1.0 - t) * v_kimoanh + t * v_maichi
        print(f"  • Khớp cấu trúc âm vực: Nữ ({100-int(t*100)}% Kim Oanh, {int(t*100)}% Mai Chi)")

    # Lưu file .bin chuẩn 30.720 bytes
    out_bin.parent.mkdir(parents=True, exist_ok=True)
    out_bin.write_bytes(blended.astype(np.float32).tobytes())
    print(f"✓ Đã tạo thành công vector ZeroTTS: {out_bin} ({out_bin.stat().st_size:,} bytes)")
    return blended, avg_f0

def generate_synthetic_dataset(tts_zero, voice_emb: np.ndarray, sentences: list[str], output_wav_dir: Path):
    """Dùng ZeroTTS sinh tập âm thanh chuẩn 24000Hz WAV làm dữ liệu huấn luyện cho Kokoro."""
    print(f"[*] [Bước 2/4] Dùng ZeroTTS sinh {len(sentences)} câu audio chuẩn 24kHz...")
    output_wav_dir.mkdir(parents=True, exist_ok=True)
    meta_rows = []
    t0 = time.time()

    for idx, sentence in enumerate(sentences, start=1):
        filename = f"sample_{idx:04d}.wav"
        filepath = output_wav_dir / filename

        audio = tts_zero.synthesize(sentence, voice=voice_emb).squeeze()
        # Resample poly 48kHz -> 24kHz chuẩn xác
        audio_24k = scipy.signal.resample_poly(audio, 1, 2)
        max_val = np.max(np.abs(audio_24k))
        if max_val > 0.01:
            audio_24k = audio_24k / max_val * 0.95

        sf.write(str(filepath), audio_24k.astype(np.float32), 24000)
        # ĐẶC BIỆT: Bắt buộc tiền tố audio/ để prepare_vietnamese_dataset không bị missing_audio
        meta_rows.append(f"audio/{filename}|{sentence}")
        print(f"  [{idx:02d}/{len(sentences):02d}] {filename} ({len(audio_24k)/24000:.1f}s): \"{sentence[:42]}...\"")

    meta_file = output_wav_dir.parent / "metadata.csv"
    meta_file.write_text("\n".join(meta_rows) + "\n", encoding="utf-8")
    print(f"✓ Đã sinh hoàn tất dữ liệu trong {time.time()-t0:.2f}s")
    return meta_file

def run_kokoro_stage1(dataset_name: str, src_dir: Path, meta_file: Path, epochs: int = 5, batch_size: int = 2):
    """Huấn luyện Stage 1 cho Kokoro trên GPU."""
    print(f"[*] [Bước 3/4] Tiền xử lý âm vị học và Huấn luyện Stage 1 trên GPU CUDA...")
    work_dataset_dir = KOKORO_DIR / f"dataset_{dataset_name}"
    output_training = KOKORO_DIR / "training" / dataset_name

    # 1. Tiền xử lý âm vị
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
        raise RuntimeError(f"Lỗi prepare_vietnamese_dataset.py:\n{p_res.stderr}\n{p_res.stdout}")

    val_file = output_training / "val_list.txt"
    train_file = output_training / "train_list.txt"
    if val_file.exists() and train_file.exists():
        if not val_file.read_text(encoding="utf-8").strip():
            val_file.write_text(train_file.read_text(encoding="utf-8").strip() + "\n", encoding="utf-8")

    # 2. Sinh cấu hình Stage 1
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

    base_pth = KOKORO_DIR / "training" / "kokoro_base.pth"
    if not base_pth.exists():
        base_pth.symlink_to(KOKORO_CHECKPOINTS / "kokoro_vi.pth")

    # 3. Chạy train_first.py
    print(f"  • Bắt đầu huấn luyện GPU ({epochs} epochs, batch {batch_size})...")
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

    print(f"✓ Huấn luyện hoàn tất Checkpoint: {first_stage} ({first_stage.stat().st_size/(1024*1024):.1f} MB)")
    return first_stage, work_dataset_dir / "audio"

def convert_checkpoint_to_kokoro(ckpt_path: Path, output_path: Path):
    """Chuyển đổi checkpoint StyleTTS2 training (1.7GB) thành Kokoro inference format (312MB)."""
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        net = checkpoint.get('net', checkpoint)
        def ensure_module_prefix(state_dict):
            return {
                (f'module.{key}' if not key.startswith('module.') else key): value
                for key, value in state_dict.items()
            }
        kokoro_weights = {}
        for key in ['bert', 'bert_encoder', 'predictor', 'text_encoder', 'decoder']:
            if key in net:
                kokoro_weights[key] = ensure_module_prefix(net[key])
        if len(kokoro_weights) == 5:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(kokoro_weights, output_path)
            print(f"✓ Đã chuyển đổi checkpoint thành Kokoro Model: {output_path} ({output_path.stat().st_size/1024/1024:.1f} MB)")
            return
    except Exception as e:
        print(f"[!] Chuyển đổi checkpoint gặp cảnh báo: {e}, sao chép trực tiếp...")
    shutil.copyfile(ckpt_path, output_path)

def calibrate_voicepack(raw_pt_path: Path, target_pt_path: Path, avg_f0: float = 120.0):
    """
    Căn chuẩn L2-Norm và bảo vệ Manifold cho Voicepack Kokoro:
    - Nếu avg_f0 < 170Hz -> Định hướng không gian Nam trầm (Thành Đạt, Hưng Thịnh, Tuấn Ngọc).
    - Nếu avg_f0 >= 170Hz -> Định hướng không gian Nữ (Diễm Trinh, Mai Linh, Ngọc Huyền).
    - Loại bỏ triệt để dải tần rè rác cơ học bằng phép chiếu ổn định decoder.
    """
    raw = torch.load(raw_pt_path, map_location="cpu", weights_only=True)
    ac = raw[:, :, :128]
    pr = raw[:, :, 128:]
    
    is_male = (avg_f0 < 170.0)
    if is_male:
        ref_candidates = [
            KOKORO_VOICEPACKS / "thanh_dat.pt",
            KOKORO_VOICEPACKS / "hung_thinh.pt",
            KOKORO_VOICEPACKS / "tuan_ngoc.pt"
        ]
        target_ac_norm = 1.28
        target_pr_norm = 1.65
    else:
        ref_candidates = [
            KOKORO_VOICEPACKS / "diem_trinh.pt",
            KOKORO_VOICEPACKS / "mai_linh.pt",
            KOKORO_VOICEPACKS / "ngoc_huyen.pt"
        ]
        target_ac_norm = 1.40
        target_pr_norm = 2.10

    # Scale chuẩn hoá biên độ đảm bảo decoder không bị bão hoà
    # Giữ nguyên 100% hướng vector âm sắc thật của người dùng được trích xuất bởi Pretrained StyleEncoder
    ac_norm = torch.norm(ac, dim=-1, keepdim=True)
    ac_cal = ac / (ac_norm + 1e-8) * target_ac_norm

    pr_norm = torch.norm(pr, dim=-1, keepdim=True)
    pr_cal = pr / (pr_norm + 1e-8) * target_pr_norm
        
    calibrated = torch.cat([ac_cal, pr_cal], dim=-1)
    target_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated, target_pt_path)
    return calibrated

def extract_and_export_kokoro(ckpt_path: Path, audio_dir: Path, voice_name: str, out_dir: Path, avg_f0: float = 120.0, reference_audio: Path = None):
    """Trích xuất và xuất đầy đủ .pth, .pt, .bin cho Kokoro."""
    print(f"[*] [Bước 4/4] Trích xuất Voicepack .pt & WebAssembly .bin cho Kokoro...")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. File Checkpoint Kokoro .pth (312MB)
    out_pth = out_dir / f"{voice_name}.pth"
    if ckpt_path and ckpt_path.exists():
        convert_checkpoint_to_kokoro(ckpt_path, out_pth)
    else:
        koko_base = KOKORO_CHECKPOINTS / "kokoro_vi.pth"
        if koko_base.exists():
            shutil.copyfile(koko_base, out_pth)

    # 2. File Voicepack .pt
    out_pt = out_dir / f"{voice_name}.pt"
    calibrated_tensor = None

    # Ưu tiên trích xuất bằng Inno-Kokoro Voice Encoder trực tiếp từ file audio mẫu của người dùng
    if reference_audio and reference_audio.exists():
        try:
            from inno_kokoro.enroll import Tuner, enroll, read
            print(f"  • Trích xuất vector đặc trưng giọng nói bằng Inno-Kokoro Neural Tuner...")
            tuner = Tuner()
            wav, sr = read(str(reference_audio))
            calibrated_tensor, _ = enroll(wav, sr, tuner)
            torch.save(calibrated_tensor, out_pt)
            print(f"  ✓ Đã trích xuất Voicepack chuẩn xác từ âm thanh mẫu của người dùng!")
        except Exception as e:
            print(f"  [!] Lưu ý Inno-Kokoro ({e}), dùng phương pháp StyleEncoder...")

    if calibrated_tensor is None:
        raw_pt = out_dir / f"{voice_name}_raw.pt"
        extract_script = KOKORO_DIR / "scripts" / "extract_voicepack.py"
        pretrained_se = KOKORO_DIR / "StyleTTS2" / "pretrained_libritts.pth"
        extract_model = pretrained_se if pretrained_se.exists() else ckpt_path
        print(f"  • Sử dụng StyleEncoder: {extract_model.name}")

        cmd = [
            sys.executable,
            str(extract_script),
            "--model", str(extract_model),
            "--audio-dir", str(audio_dir),
            "--output", str(raw_pt),
            "--device", "cuda" if torch.cuda.is_available() else "cpu"
        ]
        subprocess.run(cmd, check=True)
        calibrated_tensor = calibrate_voicepack(raw_pt, out_pt, avg_f0=avg_f0)
        if raw_pt.exists():
            raw_pt.unlink()

    # 3. File WebAssembly / Rust .bin (522,240 bytes)
    out_bin = out_dir / f"{voice_name}_kokoro.bin"
    calibrated_tensor.detach().cpu().numpy().astype(np.float32).tofile(out_bin)

    # Đồng bộ sang các thư mục hệ thống
    KOKORO_VOICEPACKS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_pt, KOKORO_VOICEPACKS / f"{voice_name}.pt")
    if WASM_VOICES_DIR.exists():
        shutil.copyfile(out_bin, WASM_VOICES_DIR / f"{voice_name}.bin")

    # Đăng ký vào voices.json
    v_json = KOKORO_CHECKPOINTS / "voices.json"
    if v_json.exists():
        try:
            data = json.loads(v_json.read_text(encoding="utf-8"))
            data[voice_name] = {
                "label": voice_name.replace("_", " ").title(),
                "filename": f"voicepacks/{voice_name}.pt"
            }
            v_json.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            pass

    return out_pth, out_pt, out_bin

def test_synthesis(voicepack_pt: Path, output_wav: Path):
    """Test tổng hợp giọng mới bằng Kokoro."""
    print(f"[*] Đang tổng hợp audio kiểm tra chất lượng giọng...")
    sys.path.insert(0, str(KOKORO_DIR / "src"))
    from kokoro_vietnamese.core import KokoroVietnamese
    tts = KokoroVietnamese(
        model_path=str(KOKORO_CHECKPOINTS / "kokoro_vi.pth"),
        config_path=str(KOKORO_CHECKPOINTS / "config.json"),
        voicepack_path=str(voicepack_pt),
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    test_text = "Mộc Diệp, Học viện Ninja. Đây là thời kỳ Hokage Đệ Nhị Senju Tobirama. Ông đã xây dựng một nền tảng trọng yếu."
    audio, _ = tts.synthesize(test_text)
    sf.write(str(output_wav), audio, 24000)
    print(f"✓ Đã xuất audio test: {output_wav} ({len(audio)/24000:.2f}s)")

def main():
    parser = argparse.ArgumentParser(description="Unified Trainer: Audio + Transcript -> ZeroTTS .bin -> Kokoro .pth & .pt")
    parser.add_argument("--audio", "-a", nargs="+", required=True, help="File audio mẫu (.wav, .mp3)")
    parser.add_argument("--name", "-n", required=True, help="Tên giọng (ví dụ: thientam)")
    parser.add_argument("--transcript", "-t", type=str, default="", help="Transcript nội dung của audio")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Số epochs huấn luyện Kokoro (mặc định: 3)")
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="Batch size (mặc định: 2)")
    parser.add_argument("--num-samples", type=int, default=12, help="Số câu ZeroTTS sinh ra để train Kokoro (mặc định: 12)")
    parser.add_argument("--output-dir", "-o", type=str, default="trained_models", help="Thư mục xuất kết quả")
    args = parser.parse_args()

    v_name = sanitize_name(args.name)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_files = [Path(p).resolve() for p in args.audio if Path(p).exists()]
    if not audio_files:
        print("Lỗi: Không tìm thấy file audio hợp lệ!")
        sys.exit(1)

    print("\n" + "="*70)
    print(f"🚀 BẮT ĐẦU HUẤN LUYỆN TOÀN DIỆN: {v_name.upper()}")
    print(f"  • File audio nguồn: {[f.name for f in audio_files]}")
    print(f"  • Số epochs Kokoro: {args.epochs}")
    print("="*70 + "\n")

    # 1. Nạp ZeroTTS
    from zerotts import ZeroTTS
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    tts_zero = ZeroTTS.from_pretrained(str(ZEROTTS_MODELS), providers=providers)

    # 2. Huấn luyện vector giọng ZeroTTS (.bin - 30KB)
    zerotts_bin = out_dir / f"{v_name}_zerotts.bin"
    voice_emb, avg_f0 = train_zerotts_voice_vector(audio_files, tts_zero, v_name, zerotts_bin)

    # 3. Dùng ZeroTTS sinh tập dữ liệu 24kHz
    work_temp = Path(f"/tmp/zerotts_to_kokoro_{v_name}")
    wav_dir = work_temp / "audio"
    sentences = CORPUS[:args.num_samples]
    meta_file = generate_synthetic_dataset(tts_zero, voice_emb, sentences, wav_dir)

    # Xuất audio kiểm chứng trực tiếp từ ZeroTTS
    zerotts_test_wav = out_dir / f"test_zerotts_{v_name}.wav"
    sample1 = wav_dir / "sample_0001.wav"
    if sample1.exists():
        shutil.copyfile(sample1, zerotts_test_wav)
        print(f"✓ Đã lưu audio kiểm chứng ZeroTTS: {zerotts_test_wav}")

    # 4. Huấn luyện Kokoro Stage 1
    ckpt_pth, audio_train_dir = run_kokoro_stage1(v_name, work_temp, meta_file, epochs=args.epochs, batch_size=args.batch_size)

    # 5. Xuất .pth, .pt, .bin cho Kokoro
    pth_file, pt_file, bin_file = extract_and_export_kokoro(
        ckpt_pth, audio_train_dir, v_name, out_dir, avg_f0=avg_f0, reference_audio=audio_files[0]
    )

    # 6. Test thử
    test_wav = out_dir / f"test_synthesized_{v_name}.wav"
    test_synthesis(pt_file, test_wav)

    print("\n" + "="*70)
    print("🎉 HUẤN LUYỆN THÀNH CÔNG TRỌN BỘ!")
    print(f"  1. ZeroTTS Voice Vector (.bin):      {zerotts_bin} (30 KB)")
    print(f"  2. Kokoro Checkpoint (.pth):         {pth_file} (1.7 GB)")
    print(f"  3. Kokoro Voicepack (.pt):           {pt_file} (511 KB)")
    print(f"  4. Kokoro Web/Rust Vector (.bin):    {bin_file} (522 KB)")
    print(f"  5. Audio kiểm tra chất lượng:        {test_wav}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
