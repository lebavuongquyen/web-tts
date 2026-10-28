//! Audio output utility: high-fidelity WAV and MP3 export.

use std::error::Error;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use hound::{WavSpec, WavWriter};

/// Lưu dữ liệu âm thanh mono f32 ra file WAV hoặc MP3 tùy theo đuôi mở rộng.
pub fn save_audio(
    samples: &[f32],
    sample_rate: u32,
    path: &Path,
    bitrate: &str,
) -> Result<(), Box<dyn Error>> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();

    // Chuyển đổi f32 [-1.0, 1.0] sang i16 PCM
    let mut pcm_bytes = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val_i16 = (clamped * 32767.0) as i16;
        pcm_bytes.extend_from_slice(&val_i16.to_le_bytes());
    }

    if ext == "mp3" {
        // Kiểm tra và sử dụng ffmpeg
        let mut child = match Command::new("ffmpeg")
            .args([
                "-y",
                "-f", "s16le",
                "-ar", &sample_rate.to_string(),
                "-ac", "1",
                "-i", "pipe:0",
                "-b:a", bitrate,
                path.to_str().ok_or("Đường dẫn file không hợp lệ")?,
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Cảnh báo: Không thể khởi chạy ffmpeg ({:?}). Đang lưu tạm dưới dạng WAV...", e);
                let wav_fallback = path.with_extension("wav");
                save_wav(samples, sample_rate, &wav_fallback)?;
                return Err(format!("FFmpeg không khả dụng, đã lưu tạm ra {:?}", wav_fallback).into());
            }
        };

        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(&pcm_bytes)?;
        }

        let output = child.wait_with_output()?;
        if !output.status.success() {
            let err_msg = String::from_utf8_lossy(&output.stderr);
            return Err(format!("FFmpeg mã hóa MP3 thất bại: {}", err_msg).into());
        }

        Ok(())
    } else {
        save_wav(samples, sample_rate, path)
    }
}

/// Ghi file WAV 16-bit PCM tiêu chuẩn bằng hound
pub fn save_wav(samples: &[f32], sample_rate: u32, path: &Path) -> Result<(), Box<dyn Error>> {
    let spec = WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let val_i16 = (clamped * 32767.0) as i16;
        writer.write_sample(val_i16)?;
    }
    writer.finalize()?;
    Ok(())
}
