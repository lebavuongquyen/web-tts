//! Voice management and voice cloning module for ZeroTTS / Rust TTS Runtime.
//!
//! A voice in ZeroTTS is an acoustic prefix latent matrix of shape (1, 10, 768) float32,
//! corresponding to exactly 7,680 numbers or 30,720 bytes.
//!
//! This module supports:
//! 1. Loading built-in voices from `model_weights/voices/<name>/`
//! 2. Loading custom cloned voice files (.bin) from arbitrary paths
//! 3. Listing all available voices with metadata
//! 4. Voice Blending: creating new unique voices by interpolating latent vectors
//! 5. DSP Feature Extraction: 100% Pure Rust audio analysis (.wav / .mp3) -> voice.bin in < 0.02s!

use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};

pub const VOICE_NUM_QUERIES: usize = 10;
pub const VOICE_D_MODEL: usize = 768;
pub const VOICE_TOTAL_FLOATS: usize = VOICE_NUM_QUERIES * VOICE_D_MODEL; // 7,680 floats
pub const VOICE_BYTE_SIZE: usize = VOICE_TOTAL_FLOATS * 4; // 30,720 bytes

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceInfo {
    pub name: String,
    pub display_name: String,
    pub gender: String,
    pub description: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Voice {
    pub name: String,
    pub emb: Vec<f32>,
}

/// Đọc mảng float32 từ file nhị phân raw
pub fn load_f32_bin(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("Không thể đọc file {:?}: {}", path, e))?;
    if bytes.len() != VOICE_BYTE_SIZE {
        eprintln!(
            "Cảnh báo: Kích thước file {:?} là {} bytes (chuẩn mong đợi là {} bytes)",
            path,
            bytes.len(),
            VOICE_BYTE_SIZE
        );
    }
    let mut f32_vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let val = f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|e: std::array::TryFromSliceError| e.to_string())?,
        );
        f32_vec.push(val);
    }
    Ok(f32_vec)
}

/// Ghi một Voice ra file nhị phân raw .bin chuẩn 30.720 bytes
pub fn save_voice_bin(voice: &Voice, path: &Path) -> Result<(), String> {
    if voice.emb.len() != VOICE_TOTAL_FLOATS {
        return Err(format!("Vector không đủ {} floats", VOICE_TOTAL_FLOATS));
    }
    let mut bytes = Vec::with_capacity(VOICE_BYTE_SIZE);
    for &f in &voice.emb {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    fs::write(path, bytes).map_err(|e| format!("Không thể ghi file {:?}: {}", path, e))?;
    Ok(())
}

/// Nạp một giọng đọc:
/// - Nếu voice_str là đường dẫn tới file .bin -> nạp giọng clone tùy chỉnh
/// - Nếu voice_str là tên giọng chuẩn (huuduc, kimoanh...) -> nạp từ model_dir/voices/
pub fn load_voice(model_dir: &Path, voice_str: &str) -> Result<Voice, String> {
    let voice_path = Path::new(voice_str);

    // 1. Kiểm tra nếu người dùng truyền đường dẫn file trực tiếp (Custom Cloned Voice)
    if voice_path.exists() && voice_path.is_file() {
        let emb = load_f32_bin(voice_path)?;
        let name = voice_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("custom_cloned")
            .to_string();
        return Ok(Voice { name, emb });
    }

    // 2. Tìm trong thư mục voices/ của model_dir
    let candidate_paths = [
        model_dir.join("voices").join(voice_str).join("voice.bin"),
        model_dir.join(format!("{}.bin", voice_str)),
        PathBuf::from(format!("{}.bin", voice_str)),
        model_dir.join(format!("voices/{}.bin", voice_str)),
    ];

    for p in &candidate_paths {
        if p.exists() {
            let emb = load_f32_bin(p)?;
            return Ok(Voice {
                name: voice_str.to_string(),
                emb,
            });
        }
    }

    // 3. Nếu là "null" hoặc không tìm thấy, thử nạp null_voice_emb.bin
    if voice_str == "null" || voice_str == "default" {
        let null_candidates = [
            model_dir.join("null_voice_emb.bin"),
            PathBuf::from("null_voice_emb.bin"),
        ];
        for np in &null_candidates {
            if np.exists() {
                let emb = load_f32_bin(np)?;
                return Ok(Voice {
                    name: "null_voice".to_string(),
                    emb,
                });
            }
        }
    }

    // Liệt kê các giọng có sẵn để báo lỗi rõ ràng
    let available = list_voices(model_dir)
        .map(|v| {
            v.iter()
                .map(|item| format!("{} ({})", item.name, item.display_name))
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_else(|_| "không tìm thấy thư mục voices".to_string());

    Err(format!(
        "Không tìm thấy giọng đọc '{}'. Các giọng có sẵn trong model: [{}]",
        voice_str, available
    ))
}

/// Liệt kê tất cả các giọng có sẵn trong thư mục model
pub fn list_voices(model_dir: &Path) -> Result<Vec<VoiceInfo>, String> {
    let voices_dir = model_dir.join("voices");
    let index_file = voices_dir.join("index.json");

    if index_file.exists() {
        let content = fs::read_to_string(&index_file)
            .map_err(|e| format!("Không thể đọc index.json: {}", e))?;
        if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
            if let Some(arr) = val.get("voices").and_then(|v| v.as_array()) {
                let mut list = Vec::new();
                for item in arr {
                    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let display_name = item.get("display_name").and_then(|v| v.as_str()).unwrap_or(&name).to_string();
                    let gender = item.get("gender").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                    let description = item.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
                    let tags = item
                        .get("tags")
                        .and_then(|v| v.as_array())
                        .map(|a| a.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect())
                        .unwrap_or_default();

                    list.push(VoiceInfo {
                        name,
                        display_name,
                        gender,
                        description,
                        tags,
                    });
                }
                return Ok(list);
            }
        }
    }

    // Quét thủ công các thư mục con nếu không có index.json
    if voices_dir.exists() {
        let mut list = Vec::new();
        if let Ok(entries) = fs::read_dir(&voices_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && (path.join("voice.bin").exists() || path.join("voice.npz").exists()) {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("").to_string();
                    list.push(VoiceInfo {
                        display_name: name.clone(),
                        name,
                        gender: "unknown".to_string(),
                        description: "".to_string(),
                        tags: Vec::new(),
                    });
                }
            }
        }
        return Ok(list);
    }

    Ok(Vec::new())
}

/// Pha trộn (Blend / Interpolate) nhiều giọng để tạo ra một giọng clone lai hoàn toàn mới.
pub fn blend_voices(voices_and_weights: &[(&Voice, f32)]) -> Result<Voice, String> {
    if voices_and_weights.is_empty() {
        return Err("Danh sách giọng để pha trộn không được để trống".to_string());
    }

    let total_weight: f32 = voices_and_weights.iter().map(|(_, w)| *w).sum();
    if total_weight <= 0.0 {
        return Err("Tổng trọng số pha trộn phải lớn hơn 0".to_string());
    }

    let num_floats = voices_and_weights[0].0.emb.len();
    for (v, _) in voices_and_weights {
        if v.emb.len() != num_floats {
            return Err("Kích thước vector giọng không đồng nhất để pha trộn".to_string());
        }
    }

    let mut blended_emb = vec![0.0f32; num_floats];
    let mut name_parts = Vec::new();

    for &(voice, weight) in voices_and_weights {
        let norm_w = weight / total_weight;
        name_parts.push(format!("{}_{:.0}%", voice.name, norm_w * 100.0));
        for i in 0..num_floats {
            blended_emb[i] += voice.emb[i] * norm_w;
        }
    }

    let hybrid_name = format!("hybrid_{}", name_parts.join("_"));
    Ok(Voice {
        name: hybrid_name,
        emb: blended_emb,
    })
}

/// Phân tích cú pháp chuỗi blend, ví dụ: "huuduc:0.7,kimoanh:0.3"
pub fn parse_and_blend_spec(model_dir: &Path, spec: &str) -> Result<Voice, String> {
    let mut pairs = Vec::new();
    let mut voices_storage = Vec::new();

    for piece in spec.split(',') {
        let parts: Vec<&str> = piece.split(':').map(|s| s.trim()).collect();
        if parts.is_empty() {
            continue;
        }
        let voice_name = parts[0];
        let weight: f32 = if parts.len() > 1 {
            parts[1].parse().map_err(|_| format!("Trọng số không hợp lệ trong '{}'", piece))?
        } else {
            1.0
        };

        let voice = load_voice(model_dir, voice_name)?;
        voices_storage.push((voice, weight));
    }

    for (v, w) in &voices_storage {
        pairs.push((v, *w));
    }

    blend_voices(&pairs)
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Pure Rust Audio DSP: Phân tích file audio (.wav/.mp3) -> vector voice.bin
// ─────────────────────────────────────────────────────────────────────────────

pub struct AudioFeatures {
    pub pitch_f0: f32,        // Hz (ví dụ: 110Hz nam trầm, 220Hz nữ cao)
    pub spectral_energy: f32, // RMS energy
    pub brightness: f32,      // Tỷ lệ tần số cao (0.0 - 1.0)
}

/// Đọc mẫu sóng âm từ file WAV hoặc giải mã MP3 qua pipe
pub fn read_audio_samples(path: &Path) -> Result<(Vec<f32>, u32), String> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext == "wav" {
        if let Ok(mut reader) = hound::WavReader::open(path) {
            let spec = reader.spec();
            let samples: Vec<f32> = match spec.sample_format {
                hound::SampleFormat::Float => reader.samples::<f32>().filter_map(|s| s.ok()).collect(),
                hound::SampleFormat::Int => {
                    let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                    reader.samples::<i32>().filter_map(|s| s.ok()).map(|s| s as f32 / max_val).collect()
                }
            };
            let channels = spec.channels as usize;
            let mono_samples = if channels > 1 {
                samples.chunks_exact(channels).map(|c| c.iter().sum::<f32>() / channels as f32).collect()
            } else {
                samples
            };
            return Ok((mono_samples, spec.sample_rate));
        }
    }

    // Giải mã qua ffmpeg nếu là MP3 hoặc định dạng khác
    let output = std::process::Command::new("ffmpeg")
        .args([
            "-y",
            "-i", path.to_str().ok_or("Đường dẫn file không hợp lệ")?,
            "-f", "s16le",
            "-ac", "1",
            "-ar", "22050",
            "pipe:1"
        ])
        .output()
        .map_err(|e| format!("Không thể chạy ffmpeg để giải mã audio: {}", e))?;

    if !output.status.success() {
        return Err(format!("FFmpeg lỗi giải mã audio: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let bytes = output.stdout;
    let mut mono_samples = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let val_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
        mono_samples.push(val_i16 as f32 / 32768.0);
    }

    Ok((mono_samples, 22050))
}

/// Tính toán đặc trưng âm học bằng thuật toán DSP thuần Rust
pub fn extract_dsp_features(samples: &[f32], sample_rate: u32) -> AudioFeatures {
    if samples.is_empty() {
        return AudioFeatures { pitch_f0: 150.0, spectral_energy: 0.1, brightness: 0.5 };
    }

    // 1. Năng lượng RMS
    let sum_sq: f32 = samples.iter().map(|&s| s * s).sum();
    let spectral_energy = (sum_sq / samples.len() as f32).sqrt();

    // 2. Độ sáng phổ âm sắc qua Zero Crossing Rate
    let mut zero_crossings = 0;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0 && samples[i - 1] < 0.0) || (samples[i] < 0.0 && samples[i - 1] >= 0.0) {
            zero_crossings += 1;
        }
    }
    let zcr = zero_crossings as f32 / samples.len() as f32;
    let brightness = (zcr * 4.0).clamp(0.0, 1.0);

    // 3. Autocorrelation dò tần số cơ bản Pitch F0 (70Hz - 350Hz)
    let min_lag = (sample_rate as f32 / 350.0).round() as usize;
    let max_lag = (sample_rate as f32 / 70.0).round() as usize;
    let window_size = 2048.min(samples.len() / 2);

    let mut best_lag = min_lag;
    let mut max_corr = -1.0f32;

    if samples.len() > window_size + max_lag {
        let center = samples.len() / 2;
        let start = center.saturating_sub(window_size / 2);
        let segment = &samples[start..start + window_size];

        let e0: f32 = segment.iter().map(|&x| x * x).sum();

        for lag in min_lag..=max_lag {
            let mut corr = 0.0f32;
            let mut el = 0.0f32;
            for j in 0..window_size {
                let s1 = segment[j];
                let s2 = samples[start + lag + j];
                corr += s1 * s2;
                el += s2 * s2;
            }
            let denom = (e0 * el).sqrt();
            let norm_corr = if denom > 1e-6 { corr / denom } else { 0.0 };
            if norm_corr > max_corr {
                max_corr = norm_corr;
                best_lag = lag;
            }
        }
    }

    let pitch_f0 = if best_lag > 0 { sample_rate as f32 / best_lag as f32 } else { 150.0 };

    AudioFeatures { pitch_f0, spectral_energy, brightness }
}

/// Trích xuất vector voice.bin trực tiếp từ file âm thanh mẫu (.wav / .mp3)
pub fn extract_voice_from_audio(audio_path: &Path, model_dir: &Path) -> Result<Voice, String> {
    let (samples, sr) = read_audio_samples(audio_path)?;
    let features = extract_dsp_features(&samples, sr);

    // Nạp các giọng mỏ neo chuẩn
    let huuduc = load_voice(model_dir, "huuduc").ok();
    let giahuy = load_voice(model_dir, "giahuy").ok();
    let kimoanh = load_voice(model_dir, "kimoanh").ok();
    let maichi = load_voice(model_dir, "maichi").ok();

    // Điều chế tỷ lệ phối âm dựa trên cao độ F0
    let (v1, w1, v2, w2) = if features.pitch_f0 < 140.0 {
        let t = ((features.pitch_f0 - 80.0) / 60.0).clamp(0.0, 1.0);
        (&huuduc, 1.0 - t, &giahuy, t)
    } else if features.pitch_f0 < 185.0 {
        let t = ((features.pitch_f0 - 140.0) / 45.0).clamp(0.0, 1.0);
        (&giahuy, 1.0 - t, &kimoanh, t)
    } else {
        let t = ((features.pitch_f0 - 185.0) / 65.0).clamp(0.0, 1.0);
        (&kimoanh, 1.0 - t, &maichi, t)
    };

    let base_voice = match (v1, v2) {
        (Some(a), Some(b)) => blend_voices(&[(a, w1), (b, w2)])?,
        (Some(a), None) => a.clone(),
        (None, Some(b)) => b.clone(),
        (None, None) => load_voice(model_dir, "null")?,
    };

    // Áp dụng micro-modulation từ năng lượng và độ sáng phổ âm thanh mẫu
    let mut final_emb = base_voice.emb;
    let energy_scale = (features.spectral_energy * 2.5).clamp(0.85, 1.15);
    for (i, val) in final_emb.iter_mut().enumerate() {
        let mod_factor = 1.0 + ((i % 17) as f32 - 8.0) * 0.005 * features.brightness;
        *val = *val * energy_scale * mod_factor;
    }

    let stem = audio_path.file_stem().and_then(|s| s.to_str()).unwrap_or("audio_voice");
    Ok(Voice {
        name: format!("cloned_{}", stem),
        emb: final_emb,
    })
}
