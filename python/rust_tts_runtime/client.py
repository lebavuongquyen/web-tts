"""Python SDK Wrapper for rust_tts_runtime.

Coordinates text chunking, Vietnamese normalization, audio concatenation,
subtitle SRT generation, and multi-voice dispatching (.bin and .onnx).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

DEFAULT_BINARY_PATH = "/home/quyen/08_TTS/rust_tts_runtime/target/release/rust_tts_runtime"
DEFAULT_MODEL_DIR = "/home/quyen/08_TTS/ZeroTTS/model_weights"

# ── 1. Vietnamese Number & Date Normalization ────────────────────────────────
_DIGITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_SCALES = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ", "triệu tỷ"]

def int_to_vietnamese_words(n: int) -> str:
    if n == 0:
        return "không"
    groups = []
    num = n
    while num > 0:
        groups.append(num % 1000)
        num //= 1000

    result_parts = []
    num_groups = len(groups)

    for i in range(num_groups - 1, -1, -1):
        chunk = groups[i]
        if chunk == 0:
            continue
        is_highest = i == num_groups - 1
        c_hundred = chunk // 100
        c_ten = (chunk % 100) // 10
        c_unit = chunk % 10

        chunk_words = []
        if c_hundred > 0 or not is_highest:
            chunk_words.append(f"{_DIGITS[c_hundred]} trăm")

        if c_ten > 1:
            chunk_words.append(f"{_DIGITS[c_ten]} mươi")
            if c_unit == 1:
                chunk_words.append("mốt")
            elif c_unit == 4:
                chunk_words.append("tư")
            elif c_unit == 5:
                chunk_words.append("lăm")
            elif c_unit > 0:
                chunk_words.append(_DIGITS[c_unit])
        elif c_ten == 1:
            chunk_words.append("mười")
            if c_unit == 5:
                chunk_words.append("lăm")
            elif c_unit > 0:
                chunk_words.append(_DIGITS[c_unit])
        else:
            if c_hundred > 0 or not is_highest:
                if c_unit > 0:
                    chunk_words.append("linh")
                    chunk_words.append(_DIGITS[c_unit])
            elif c_unit > 0:
                chunk_words.append(_DIGITS[c_unit])

        scale = _SCALES[i]
        if scale:
            chunk_words.append(scale)
        result_parts.append(" ".join(chunk_words))

    return " ".join(result_parts)

def normalize_vietnamese_text(text: str) -> str:
    # 1. Dates (dd/mm/yyyy)
    re_date = re.compile(r"(?i)(?:ngày\s+)?\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
    def _repl_date(m):
        d = int(m.group(1))
        mon = int(m.group(2))
        y = int(m.group(3))
        m_str = "tư" if mon == 4 else int_to_vietnamese_words(mon)
        return f"ngày {int_to_vietnamese_words(d)} tháng {m_str} năm {int_to_vietnamese_words(y)}"
    text = re_date.sub(_repl_date, text)

    # 2. Percentages (100% -> một trăm phần trăm)
    re_percent = re.compile(r"(\d+)\s*%")
    text = re_percent.sub(lambda m: f"{int_to_vietnamese_words(int(m.group(1)))} phần trăm", text)

    # 3. Numbers
    re_num = re.compile(r"\b\d+\b")
    text = re_num.sub(lambda m: int_to_vietnamese_words(int(m.group(0))), text)
    return text

# ── 2. Smart Sentence Chunking ───────────────────────────────────────────────
_SENTENCE_END_RE = re.compile(r"(?<=[.!?…])\s+")

def split_sentences(text: str, max_chars: int = 200) -> List[str]:
    text = text.strip()
    if not text:
        return []
    
    sentences = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        for sent in _SENTENCE_END_RE.split(para):
            sent = sent.strip().replace(";", ",")
            if sent:
                sentences.append(sent)

    final_chunks = []
    for s in sentences:
        if len(s) > max_chars:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            cur = ""
            for p in parts:
                if not cur:
                    cur = p
                elif len(cur) + len(p) + 2 <= max_chars:
                    cur += ", " + p
                else:
                    final_chunks.append(cur + ",")
                    cur = p
            if cur:
                final_chunks.append(cur)
        else:
            final_chunks.append(s)

    return final_chunks

# ── 3. Subtitle SRT Formatter ────────────────────────────────────────────────
def format_srt_time(seconds: float) -> str:
    millis = int(round((seconds % 1) * 1000)) % 1000
    total_secs = int(seconds)
    s = total_secs % 60
    m = (total_secs // 60) % 60
    h = total_secs // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

# ── 4. Main Client Class ─────────────────────────────────────────────────────
class RustTTS:
    def __init__(
        self,
        binary_path: Union[str, Path] = DEFAULT_BINARY_PATH,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        cuda: bool = True
    ):
        self.binary_path = Path(binary_path)
        self.model_dir = Path(model_dir)
        self.cuda = cuda

        if not self.binary_path.exists():
            raise FileNotFoundError(f"Binary not found: {self.binary_path}")

    def list_voices(self) -> List[Dict[str, str]]:
        cmd = [str(self.binary_path), "--list-voices", "--model-dir", str(self.model_dir), "--json"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            try:
                data = json.loads(res.stdout.strip())
                return data.get("voices", [])
            except Exception:
                pass
        return []

    def _phonemize_text(self, text: str) -> List[int]:
        try:
            from piper_phonemize import phonemize_espeak, phoneme_ids_espeak
            all_phonemes = phonemize_espeak(text, "vi")
            flat_phonemes = [p for s in all_phonemes for p in s]
            return phoneme_ids_espeak(flat_phonemes)
        except ImportError:
            raise RuntimeError("Cần cài đặt piper_phonemize để dùng động cơ VITS ONNX")

    def _synthesize_single_sentence(
        self,
        text: str,
        voice: str,
        blend: Optional[str],
        output_wav: Path,
        cuda: bool,
    ) -> Dict[str, Union[float, int, str]]:
        cmd = [
            str(self.binary_path),
            "--output", str(output_wav),
            "--model-dir", str(self.model_dir),
            "--json"
        ]
        if not cuda:
            cmd.extend(["--cuda", "false"])

        # Check if voice is VITS .onnx model
        is_onnx = voice.endswith(".onnx") or Path(voice).suffix == ".onnx"

        if is_onnx:
            pids = self._phonemize_text(text)
            pids_str = " ".join(map(str, pids))
            cmd.extend(["--voice", voice, "--phoneme-ids", pids_str])
        else:
            if blend:
                cmd.extend(["--blend", blend])
            else:
                cmd.extend(["--voice", voice])
            cmd.append(text)

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"Runtime failed: {res.stderr}\n{res.stdout}")

        try:
            meta = json.loads(res.stdout.strip())
            return meta
        except Exception:
            return {"status": "success"}

    def speak(
        self,
        text: str,
        voice: str = "huuduc",
        blend: Optional[str] = None,
        output: Union[str, Path] = "output.mp3",
        bitrate: str = "192k",
        generate_srt: bool = True,
        srt: Optional[Union[str, Path]] = None,
        pause: float = 0.35,
        no_norm: bool = False,
        cuda: Optional[bool] = None,
    ) -> Dict[str, Union[str, float, int]]:
        output_path = Path(output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        use_cuda = self.cuda if cuda is None else cuda

        # 1. Normalize
        norm_text = text if no_norm else normalize_vietnamese_text(text)

        # 2. Chunking
        sentences = split_sentences(norm_text)
        if not sentences:
            raise ValueError("Văn bản rỗng sau khi xử lý")

        # 3. Create temp workspace for sentence WAVs
        temp_dir = Path(tempfile.mkdtemp(prefix="rust_tts_"))
        sentence_wavs: List[Path] = []
        subtitles: List[Dict[str, Union[int, float, str]]] = []
        current_time = 0.0

        sample_rate = 48000 if not (voice.endswith(".onnx") or Path(voice).suffix == ".onnx") else 22050

        try:
            for idx, sent in enumerate(sentences):
                piece_wav = temp_dir / f"chunk_{idx:04d}.wav"
                meta = self._synthesize_single_sentence(
                    text=sent,
                    voice=voice,
                    blend=blend,
                    output_wav=piece_wav,
                    cuda=use_cuda
                )
                sentence_wavs.append(piece_wav)

                # Read audio duration
                with wave.open(str(piece_wav), "rb") as wf:
                    n_frames = wf.getnframes()
                    sample_rate = wf.getframerate()
                    dur = n_frames / sample_rate

                start_sec = current_time
                end_sec = current_time + dur
                subtitles.append({
                    "id": idx + 1,
                    "start": start_sec,
                    "end": end_sec,
                    "text": sent
                })
                current_time = end_sec + pause

            # 4. Concatenate WAVs with pause
            pause_samples = int(pause * sample_rate)
            pause_bytes = b"\x00" * (pause_samples * 2) # 16-bit mono

            merged_wav = temp_dir / "merged.wav"
            with wave.open(str(merged_wav), "wb") as out_wf:
                out_wf.setnchannels(1)
                out_wf.setsampwidth(2)
                out_wf.setframerate(sample_rate)

                for idx, sw in enumerate(sentence_wavs):
                    with wave.open(str(sw), "rb") as in_wf:
                        data = in_wf.readframes(in_wf.getnframes())
                        out_wf.writeframes(data)
                    if idx < len(sentence_wavs) - 1:
                        out_wf.writeframes(pause_bytes)

            total_audio_dur = current_time - pause

            # 5. Export to target format (MP3 or WAV)
            ext = output_path.suffix.lower()
            if ext == ".mp3":
                ffmpeg_cmd = [
                    "ffmpeg", "-y", "-i", str(merged_wav),
                    "-b:a", bitrate, str(output_path)
                ]
                ff_res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
                if ff_res.returncode != 0:
                    shutil.copyfile(merged_wav, output_path.with_suffix(".wav"))
                    raise RuntimeError(f"FFmpeg encoding failed: {ff_res.stderr}")
            else:
                shutil.copyfile(merged_wav, output_path)

            # 6. Export SRT subtitles
            srt_path = None
            if generate_srt or srt:
                srt_path = Path(srt).resolve() if srt else output_path.with_suffix(".srt")
                with open(srt_path, "w", encoding="utf-8") as sf:
                    for sub in subtitles:
                        sf.write(f"{sub['id']}\n")
                        sf.write(f"{format_srt_time(sub['start'])} --> {format_srt_time(sub['end'])}\n")
                        sf.write(f"{sub['text']}\n\n")

            return {
                "audio": str(output_path),
                "srt": str(srt_path) if srt_path else None,
                "sentences": len(sentences),
                "duration": round(total_audio_dur, 2),
                "sample_rate": sample_rate,
                "status": "success"
            }

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def speak_file(
        self,
        file_path: Union[str, Path],
        voice: str = "huuduc",
        blend: Optional[str] = None,
        output: Union[str, Path] = "output.mp3",
        bitrate: str = "192k",
        generate_srt: bool = True,
        srt: Optional[Union[str, Path]] = None,
        pause: float = 0.35,
        cuda: Optional[bool] = None,
    ) -> Dict[str, Union[str, float, int]]:
        text = Path(file_path).read_text(encoding="utf-8")
        return self.speak(
            text=text,
            voice=voice,
            blend=blend,
            output=output,
            bitrate=bitrate,
            generate_srt=generate_srt,
            srt=srt,
            pause=pause,
            cuda=cuda
        )
