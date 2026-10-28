import os
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Dict, Union

DEFAULT_BINARY_PATH = "/home/quyen/08_TTS/rust_tts_runtime/target/release/rust_tts_runtime"
DEFAULT_MODEL_DIR = "/home/quyen/08_TTS/ZeroTTS/model_weights"

class RustTTS:
    """Python high-level wrapper for Rust TTS Runtime.
    
    Provides an ergonomic Python API identical to standard TTS libraries,
    while delegating computation to the high-performance standalone Rust engine.
    """

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
            raise FileNotFoundError(
                f"Rust binary not found at {self.binary_path}. Please build it first with `cargo build --release`."
            )
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {self.model_dir}.")

    def list_voices(self) -> List[Dict[str, str]]:
        """List all voices available in the model weights directory."""
        index_file = self.model_dir / "voices" / "index.json"
        if index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data.get("voices", [])
            except Exception:
                pass
        
        voices_dir = self.model_dir / "voices"
        if voices_dir.exists():
            return [
                {"name": d.name, "display_name": d.name}
                for d in voices_dir.iterdir()
                if d.is_dir() and (d / "voice.bin").exists()
            ]
        return []

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
        max_frames: int = 1500,
        cuda: Optional[bool] = None,
    ) -> Dict[str, Union[str, float]]:
        """Synthesize text into speech with auto-chunking, voice cloning, and SRT generation.
        
        Args:
            text: Input Vietnamese text (short sentence or long article).
            voice: Voice name (e.g. 'huuduc', 'kimoanh') or path to custom cloned .bin file.
            blend: Voice blending specification (e.g. 'huuduc:0.7,kimoanh:0.3').
            output: Path to output file (.mp3 or .wav).
            bitrate: MP3 bitrate (e.g. '192k', '256k', '320k').
            generate_srt: Automatically generate synchronized .srt file.
            srt: Custom path for .srt file.
            pause: Pause duration in seconds between sentences.
            no_norm: Skip Vietnamese number/date normalization.
            max_frames: Max frames per sentence.
            cuda: Enable NVIDIA CUDA GPU acceleration (default: self.cuda).
        
        Returns:
            Dictionary containing paths to output audio and srt file.
        """
        output_p = Path(output).resolve()
        output_p.parent.mkdir(parents=True, exist_ok=True)

        use_cuda = self.cuda if cuda is None else cuda

        cmd = [
            str(self.binary_path),
            text,
            "--output", str(output_p),
            "--bitrate", bitrate,
            "--model-dir", str(self.model_dir),
            "--pause", str(pause),
            "--max-frames", str(max_frames),
        ]

        if blend:
            cmd.extend(["--blend", blend])
        else:
            cmd.extend(["--voice", voice])

        if not use_cuda:
            cmd.extend(["--cuda", "false"])

        if no_norm:
            cmd.append("--no-norm")

        if generate_srt:
            cmd.append("--generate-srt")
            if srt:
                cmd.extend(["--srt", str(Path(srt).resolve())])
        else:
            cmd.extend(["--generate-srt", "false"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Rust TTS Runtime failed:\n{result.stderr}\n{result.stdout}")

        srt_out = output_p.with_suffix(".srt") if generate_srt and not srt else (Path(srt) if srt else None)

        return {
            "audio": str(output_p),
            "srt": str(srt_out) if srt_out and srt_out.exists() else None,
            "status": "success",
            "log": result.stdout
        }

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
    ) -> Dict[str, Union[str, float]]:
        """Synthesize text from a file (e.g. story chapter)."""
        input_file = Path(file_path).resolve()
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")

        text = input_file.read_text(encoding="utf-8")
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
