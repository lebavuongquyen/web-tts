use wasm_bindgen::prelude::*;
use regex::Regex;

#[wasm_bindgen]
pub fn rtts_version() -> String {
    "rust_tts_runtime WASM v0.1.0".to_string()
}

/// Chuẩn hóa số nguyên thành từ tiếng Việt thuần túy
fn int_to_vietnamese_words(num: i64) -> String {
    if num == 0 {
        return "không".to_string();
    }
    let units = ["", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"];
    let mut n = num;
    let mut groups = Vec::new();
    while n > 0 {
        groups.push(n % 1000);
        n /= 1000;
    }
    let scales = ["", "nghìn", "triệu", "tỷ", "nghìn tỷ"];
    let mut result = Vec::new();

    for (i, &g) in groups.iter().enumerate().rev() {
        if g == 0 {
            continue;
        }
        let h = (g / 100) as usize;
        let t = ((g % 100) / 10) as usize;
        let u = (g % 10) as usize;

        let mut g_str = Vec::new();
        if h > 0 {
            g_str.push(format!("{} trăm", units[h]));
        }
        if t > 1 {
            g_str.push(format!("{} mươi", units[t]));
            if u == 1 {
                g_str.push("mốt".to_string());
            } else if u == 5 {
                g_str.push("lăm".to_string());
            } else if u > 0 {
                g_str.push(units[u].to_string());
            }
        } else if t == 1 {
            g_str.push("mười".to_string());
            if u == 5 {
                g_str.push("lăm".to_string());
            } else if u > 0 {
                g_str.push(units[u].to_string());
            }
        } else if t == 0 && u > 0 {
            if h > 0 {
                g_str.push("lẻ".to_string());
            }
            g_str.push(units[u].to_string());
        }

        if !scales[i].is_empty() {
            g_str.push(scales[i].to_string());
        }
        result.push(g_str.join(" "));
    }

    result.join(" ")
}

/// 1. Chuẩn hóa văn bản tiếng Việt sang âm đọc thuần túy (WASM)
#[wasm_bindgen]
pub fn normalize_vietnamese(text: &str) -> String {
    let mut res = text.to_string();

    // Chuẩn hóa ngày tháng: dd/mm/yyyy -> ngày ... tháng ... năm ...
    let date_re = Regex::new(r"(\d{1,2})/(\d{1,2})/(\d{4})").unwrap();
    res = date_re.replace_all(&res, |caps: &regex::Captures| {
        let d = int_to_vietnamese_words(caps[1].parse().unwrap_or(1));
        let m = int_to_vietnamese_words(caps[2].parse().unwrap_or(1));
        let y = int_to_vietnamese_words(caps[3].parse().unwrap_or(2026));
        format!("ngày {} tháng {} năm {}", d, m, y)
    }).to_string();

    // Chuẩn hóa phần trăm: 100% -> một trăm phần trăm
    let pct_re = Regex::new(r"(\d+)%").unwrap();
    res = pct_re.replace_all(&res, |caps: &regex::Captures| {
        let n = int_to_vietnamese_words(caps[1].parse().unwrap_or(0));
        format!("{} phần trăm", n)
    }).to_string();

    // Chuẩn hóa số nguyên
    let num_re = Regex::new(r"\b(\d+)\b").unwrap();
    res = num_re.replace_all(&res, |caps: &regex::Captures| {
        int_to_vietnamese_words(caps[1].parse().unwrap_or(0))
    }).to_string();

    res
}

/// 2. Tách câu thông minh (Smart Chunking: 10 - 25 từ/câu)
#[wasm_bindgen]
pub fn split_sentences_json(text: &str, max_words: usize) -> String {
    let limit = if max_words == 0 { 22 } else { max_words };
    let mut chunks = Vec::new();

    let p_split = Regex::new(r"(?<=[.!?])\s+").unwrap();
    let sentences: Vec<&str> = p_split.split(text).collect();

    for sent in sentences {
        let trimmed = sent.trim();
        if trimmed.is_empty() {
            continue;
        }
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() <= limit {
            chunks.push(trimmed.to_string());
        } else {
            let mut current = Vec::new();
            for w in words {
                current.push(w);
                if current.len() >= limit {
                    chunks.push(current.join(" "));
                    current.clear();
                }
            }
            if !current.is_empty() {
                chunks.push(current.join(" "));
            }
        }
    }

    serde_json::to_string(&chunks).unwrap_or_else(|_| "[]".to_string())
}

/// 3. Pha trộn 2 vector giọng nói nhị phân (Float32Array) trong WASM
#[wasm_bindgen]
pub fn blend_voice_vectors(v1: &[f32], v2: &[f32], w1: f32, w2: f32) -> Vec<f32> {
    let len = v1.len().min(v2.len());
    let total_w = w1 + w2;
    let norm_w1 = if total_w > 0.0 { w1 / total_w } else { 0.5 };
    let norm_w2 = if total_w > 0.0 { w2 / total_w } else { 0.5 };

    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        out.push(v1[i] * norm_w1 + v2[i] * norm_w2);
    }
    out
}

/// 4. Thuật toán DSP: Trích xuất Pitch F0, RMS, ZCR và sinh ma trận 30KB Float32 (10x768)
#[wasm_bindgen]
pub fn extract_dsp_voice_wasm(pcm_samples: &[f32], sample_rate: u32) -> Vec<f32> {
    if pcm_samples.is_empty() {
        return vec![0.0f32; 7680];
    }

    let mut energy_sum = 0.0f32;
    let mut zcr_count = 0usize;
    for i in 0..pcm_samples.len() {
        energy_sum += pcm_samples[i] * pcm_samples[i];
        if i > 0 && ((pcm_samples[i] >= 0.0 && pcm_samples[i - 1] < 0.0) || (pcm_samples[i] < 0.0 && pcm_samples[i - 1] >= 0.0)) {
            zcr_count += 1;
        }
    }
    let rms = (energy_sum / pcm_samples.len().max(1) as f32).sqrt();
    let zcr = zcr_count as f32 / pcm_samples.len().max(1) as f32;

    // Autocorrelation Pitch F0
    let min_lag = (sample_rate as f32 / 350.0) as usize;
    let max_lag = (sample_rate as f32 / 70.0) as usize;
    let mut best_lag = min_lag;
    let mut max_corr = -1.0f32;

    let eval_len = (pcm_samples.len() - max_lag).min(8000);
    for lag in (min_lag..max_lag).step_by(2) {
        let mut corr = 0.0f32;
        for i in (0..eval_len).step_by(4) {
            corr += pcm_samples[i] * pcm_samples[i + lag];
        }
        if corr > max_corr {
            max_corr = corr;
            best_lag = lag;
        }
    }

    let f0 = sample_rate as f32 / best_lag as f32;
    let pitch_bias = (f0 - 150.0) / 100.0;

    // Chiếu vào 7,680 Float32 (10x768 = 30,720 bytes)
    let mut result = Vec::with_capacity(7680);
    for t in 0..10 {
        for d in 0..768 {
            let base = ((t as f32 * 0.7 + d as f32 * 0.13).sin()) * 0.25;
            let overtone = ((d as f32 * 0.05 + pitch_bias).cos()) * (0.15 + rms * 0.2);
            let brightness = (d as f32 * 0.3).sin() * (zcr * 2.0);
            result.push(base + overtone + brightness);
        }
    }

    result
}

/// 5. Tạo nội dung phụ đề đồng bộ SRT
#[wasm_bindgen]
pub fn generate_srt_wasm(sentences: Vec<String>, durations: &[f32]) -> String {
    let mut srt = String::new();
    let mut current_sec = 0.0f32;

    for (idx, sent) in sentences.iter().enumerate() {
        let dur = if idx < durations.len() { durations[idx] } else { 2.5 };
        let start_str = format_srt_time(current_sec);
        let end_str = format_srt_time(current_sec + dur);

        srt.push_str(&format!("{}\n{} --> {}\n{}\n\n", idx + 1, start_str, end_str, sent));
        current_sec += dur + 0.35; // 0.35s pause
    }

    srt
}

fn format_srt_time(sec: f32) -> String {
    let m = (sec / 60.0) as u32;
    let s = (sec % 60.0) as u32;
    let ms = ((sec % 1.0) * 1000.0) as u32;
    format!("{:02}:{:02}:{:02},{:03}", m / 60, m % 60, s, ms)
}
