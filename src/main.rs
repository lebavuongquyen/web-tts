use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use ort::session::Session;
use ort::value::Value;
use rand::Rng;
use tokenizers::Tokenizer;

use zerotts_rust::audio::save_audio;
use zerotts_rust::chunking::split_sentences;
use zerotts_rust::srt::{write_srt_file, SubtitleItem};
use zerotts_rust::text_norm::normalize_vietnamese_text;

#[derive(Parser, Debug)]
#[command(name = "zerotts-rust")]
#[command(about = "ZeroTTS High-Performance Standalone Rust Runner (MP3, SRT, Long-text Chapter & Number Normalization)", long_about = None)]
struct Args {
    /// Văn bản tiếng Việt cần đọc (hoặc '-' để đọc từ stdin)
    #[arg(index = 1)]
    text: Option<String>,

    /// Đọc văn bản từ file (hỗ trợ file chapter truyện dài .txt)
    #[arg(short, long)]
    file: Option<PathBuf>,

    /// Tên giọng đọc (mặc định: huuduc)
    #[arg(short, long, default_value = "huuduc")]
    voice: String,

    /// Đường dẫn file âm thanh xuất ra (.mp3 hoặc .wav, mặc định: output.mp3)
    #[arg(short, long, default_value = "output.mp3")]
    output: PathBuf,

    /// Bitrate khi xuất file MP3 (mặc định: 192k)
    #[arg(short, long, default_value = "192k")]
    bitrate: String,

    /// Tự động sinh file phụ đề .srt tương ứng với từng câu
    #[arg(long, default_value_t = true)]
    generate_srt: bool,

    /// Đường dẫn file phụ đề .srt tùy chọn (mặc định cùng tên với file output)
    #[arg(long)]
    srt: Option<PathBuf>,

    /// Khoảng lặng giữa các câu (giây, mặc định: 0.35s)
    #[arg(long, default_value_t = 0.35)]
    pause_dur: f64,

    /// Bỏ qua chuẩn hóa số tiếng Việt
    #[arg(long, default_value_t = false)]
    no_norm: bool,

    /// Thư mục chứa model weights (hỗ trợ cả FP32 và INT8 Quantized)
    #[arg(short, long, default_value = "/home/quyen/08_TTS/ZeroTTS/model_weights")]
    model_dir: PathBuf,

    /// Bật CUDA Execution Provider (nếu có GPU NVIDIA)
    #[arg(long, default_value_t = true)]
    cuda: bool,

    /// Giới hạn số frame sinh tối đa cho mỗi câu
    #[arg(long, default_value_t = 1500)]
    max_frames: usize,
}

fn load_f32_bin(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| e.to_string())?;
    let mut f32_vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let val = f32::from_le_bytes(chunk.try_into().map_err(|e: std::array::TryFromSliceError| e.to_string())?);
        f32_vec.push(val);
    }
    Ok(f32_vec)
}

fn synthesize_sentence(
    text: &str,
    sess_text: &mut Session,
    sess_prefix: &mut Session,
    sess_decode: &mut Session,
    sess_codec: &mut Session,
    tokenizer: &Tokenizer,
    voice_emb: &[f32],
    max_frames: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // 1. Tokenize văn bản tiếng Việt
    let cleaned_text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let enc = tokenizer.encode(cleaned_text.as_str(), true).map_err(|e| format!("{:?}", e))?;
    let raw_ids = enc.get_ids();
    let mut ids: Vec<i64> = Vec::with_capacity(raw_ids.len() + 2);
    ids.push(1); // <bos>
    for &id in raw_ids {
        ids.push(id as i64);
    }
    ids.push(2); // <eot>
    let l_text = ids.len();

    // 2. Text Encoder Pass
    let val_text_ids = Value::from_array(([1, l_text], ids)).map_err(|e| format!("{:?}", e))?;
    let val_txt_lengths = Value::from_array(([1], vec![l_text as i64])).map_err(|e| format!("{:?}", e))?;

    let out_text = sess_text.run(ort::inputs![
        "text_ids" => val_text_ids,
        "txt_lengths" => val_txt_lengths,
    ]).map_err(|e| format!("{:?}", e))?;

    let (text_valid_shape, text_valid_slice) = out_text["text_valid"].try_extract_tensor::<bool>().map_err(|e| format!("{:?}", e))?;
    let text_valid_shape = text_valid_shape.to_vec();
    let text_valid_vec: Vec<bool> = text_valid_slice.to_vec();

    let (_, soa_slice) = out_text["soa_embed"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
    let soa_vec: Vec<f32> = soa_slice.to_vec();

    let (cross_kv_shape, cross_kv_slice) = out_text["cross_kv"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
    let cross_kv_shape = cross_kv_shape.to_vec();
    let cross_kv_vec: Vec<f32> = cross_kv_slice.to_vec();
    drop(out_text);

    // 3. Prefix Step Init (Cold Start)
    let v_len = 10;
    let d_model = 768;
    let t_init = v_len + 1;

    let mut external_embed = Vec::with_capacity(t_init * d_model);
    external_embed.extend_from_slice(voice_emb);
    external_embed.extend_from_slice(&soa_vec);

    let val_ext = Value::from_array(([1, t_init, d_model], external_embed)).map_err(|e| format!("{:?}", e))?;
    let val_use_ext = Value::from_array(([1, t_init], vec![true; t_init])).map_err(|e| format!("{:?}", e))?;
    let val_frame_codes = Value::from_array(([1, t_init, 16], vec![0i64; t_init * 16])).map_err(|e| format!("{:?}", e))?;
    let val_new_pos = Value::from_array(([1, t_init], (0..t_init as i64).collect::<Vec<_>>())).map_err(|e| format!("{:?}", e))?;
    let val_new_valid = Value::from_array(([1, t_init], vec![true; t_init])).map_err(|e| format!("{:?}", e))?;

    let mut new_bidir = vec![true; v_len];
    new_bidir.push(false);
    let val_new_bidir = Value::from_array(([1, t_init], new_bidir)).map_err(|e| format!("{:?}", e))?;

    let val_packed_kv = Value::from_array(([9, 2, 1, 12, 0, 64], Vec::<f32>::new())).map_err(|e| format!("{:?}", e))?;
    let val_past_valid = Value::from_array(([1, 0], Vec::<bool>::new())).map_err(|e| format!("{:?}", e))?;
    let val_cross_kv = Value::from_array((cross_kv_shape.clone(), cross_kv_vec.clone())).map_err(|e| format!("{:?}", e))?;
    let val_text_valid = Value::from_array((text_valid_shape.clone(), text_valid_vec.clone())).map_err(|e| format!("{:?}", e))?;

    let out_prefix_init = sess_prefix.run(ort::inputs![
        "external_embed" => val_ext,
        "use_external_embed" => val_use_ext,
        "frame_codes" => val_frame_codes,
        "new_pos" => val_new_pos,
        "new_valid" => val_new_valid,
        "packed_kv" => val_packed_kv,
        "new_bidirectional" => val_new_bidir,
        "past_valid" => val_past_valid,
        "cross_kv" => val_cross_kv,
        "text_valid" => val_text_valid,
    ]).map_err(|e| format!("{:?}", e))?;

    let (hidden_shape, hidden_slice) = out_prefix_init["hidden"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
    let last_frame_start = (hidden_shape[1] - 1) as usize * d_model;
    let mut current_h = hidden_slice[last_frame_start..last_frame_start + d_model].to_vec();

    let (pk_shape, pk_slice) = out_prefix_init["new_packed_kv"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
    let mut current_packed_kv_shape = pk_shape.to_vec();
    let mut current_packed_kv = pk_slice.to_vec();

    let (fv_shape, fv_slice) = out_prefix_init["full_valid"].try_extract_tensor::<bool>().map_err(|e| format!("{:?}", e))?;
    let mut current_full_valid_shape = fv_shape.to_vec();
    let mut current_full_valid = fv_slice.to_vec();
    drop(out_prefix_init);

    // 4. Vòng lặp Autoregressive (AR Loop)
    let mut rng = rand::thread_rng();
    let mut seen_mask = vec![false; 1 * 16 * 1024];
    let mut generated_codes: Vec<Vec<i32>> = Vec::new();

    let min_frames = 4;
    let eoa_extra_frames = 1;
    let mut tail_left: Option<usize> = None;

    for frame_idx in 0..max_frames {
        let forbid_eoa = frame_idx < min_frames || tail_left.is_some();

        let ctrl_u: f32 = rng.gen();
        let mut audio_u = vec![0.0f32; 16];
        for u in audio_u.iter_mut() {
            *u = rng.gen();
        }

        let val_h = Value::from_array(([1, d_model], current_h.clone())).map_err(|e| format!("{:?}", e))?;
        let val_forbid = Value::from_array(([1], vec![forbid_eoa])).map_err(|e| format!("{:?}", e))?;
        let val_txt_temp = Value::from_array(([1], vec![1.0f32])).map_err(|e| format!("{:?}", e))?;
        let val_txt_topk = Value::from_array(([1], vec![50i64])).map_err(|e| format!("{:?}", e))?;
        let val_aud_temp = Value::from_array(([1], vec![0.8f32])).map_err(|e| format!("{:?}", e))?;
        let val_aud_topk = Value::from_array(([1], vec![25i64])).map_err(|e| format!("{:?}", e))?;
        let val_aud_topp = Value::from_array(([1], vec![0.95f32])).map_err(|e| format!("{:?}", e))?;
        let val_aud_rep = Value::from_array(([1], vec![1.2f32])).map_err(|e| format!("{:?}", e))?;
        let val_seen = Value::from_array(([1, 16, 1024], seen_mask.clone())).map_err(|e| format!("{:?}", e))?;
        let val_ctrl_u = Value::from_array(([1], vec![ctrl_u])).map_err(|e| format!("{:?}", e))?;
        let val_aud_u = Value::from_array(([1, 16], audio_u)).map_err(|e| format!("{:?}", e))?;
        let val_cfg = Value::from_array(([1], vec![1.0f32])).map_err(|e| format!("{:?}", e))?;

        let out_decode = sess_decode.run(ort::inputs![
            "global_hidden" => val_h,
            "forbid_eoa" => val_forbid,
            "text_temperature" => val_txt_temp,
            "text_topk" => val_txt_topk,
            "audio_temperature" => val_aud_temp,
            "audio_topk" => val_aud_topk,
            "audio_topp" => val_aud_topp,
            "audio_repetition_penalty" => val_aud_rep,
            "seen_mask" => val_seen,
            "ctrl_random_u" => val_ctrl_u,
            "audio_random_u" => val_aud_u,
            "cfg_scale" => val_cfg,
        ]).map_err(|e| format!("{:?}", e))?;

        let (_, is_eoa_slice) = out_decode["is_eoa"].try_extract_tensor::<i32>().map_err(|e| format!("{:?}", e))?;
        let is_eoa = is_eoa_slice[0] == 1;

        let (_, codes_slice) = out_decode["codes"].try_extract_tensor::<i32>().map_err(|e| format!("{:?}", e))?;
        let frame_codes_i32 = codes_slice.to_vec();
        drop(out_decode);

        for c in 0..16 {
            let code_val = frame_codes_i32[c] as usize;
            if code_val < 1024 {
                seen_mask[c * 1024 + code_val] = true;
            }
        }

        if tail_left.is_none() && is_eoa {
            tail_left = Some(eoa_extra_frames);
        }

        if let Some(tl) = tail_left {
            if tl == 0 {
                break;
            }
        }

        generated_codes.push(frame_codes_i32.clone());

        if let Some(ref mut tl) = tail_left {
            *tl -= 1;
            if *tl == 0 {
                break;
            }
        }

        // Bước tiếp theo qua prefix_step
        let codes_i64: Vec<i64> = frame_codes_i32.iter().map(|&x| x as i64).collect();
        let pos_val = (v_len + 1 + frame_idx) as i64;

        let val_f_ext = Value::from_array(([1, 1, d_model], vec![0.0f32; d_model])).map_err(|e| format!("{:?}", e))?;
        let val_f_use_ext = Value::from_array(([1, 1], vec![false])).map_err(|e| format!("{:?}", e))?;
        let val_f_codes = Value::from_array(([1, 1, 16], codes_i64)).map_err(|e| format!("{:?}", e))?;
        let val_f_pos = Value::from_array(([1, 1], vec![pos_val])).map_err(|e| format!("{:?}", e))?;
        let val_f_valid = Value::from_array(([1, 1], vec![true])).map_err(|e| format!("{:?}", e))?;
        let val_f_packed_kv = Value::from_array((current_packed_kv_shape.clone(), current_packed_kv)).map_err(|e| format!("{:?}", e))?;
        let val_f_past_valid = Value::from_array((current_full_valid_shape.clone(), current_full_valid)).map_err(|e| format!("{:?}", e))?;
        let val_f_cross = Value::from_array((cross_kv_shape.clone(), cross_kv_vec.clone())).map_err(|e| format!("{:?}", e))?;
        let val_f_bidir = Value::from_array(([1, 1], vec![false])).map_err(|e| format!("{:?}", e))?;
        let val_f_text_valid = Value::from_array((text_valid_shape.clone(), text_valid_vec.clone())).map_err(|e| format!("{:?}", e))?;

        let out_step = sess_prefix.run(ort::inputs![
            "external_embed" => val_f_ext,
            "use_external_embed" => val_f_use_ext,
            "frame_codes" => val_f_codes,
            "new_pos" => val_f_pos,
            "new_valid" => val_f_valid,
            "packed_kv" => val_f_packed_kv,
            "past_valid" => val_f_past_valid,
            "cross_kv" => val_f_cross,
            "new_bidirectional" => val_f_bidir,
            "text_valid" => val_f_text_valid,
        ]).map_err(|e| format!("{:?}", e))?;

        let (_, step_slice) = out_step["hidden"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
        current_h = step_slice.to_vec();

        let (new_pk_shape, new_pk_slice) = out_step["new_packed_kv"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
        current_packed_kv_shape = new_pk_shape.to_vec();
        current_packed_kv = new_pk_slice.to_vec();

        let (new_fv_shape, new_fv_slice) = out_step["full_valid"].try_extract_tensor::<bool>().map_err(|e| format!("{:?}", e))?;
        current_full_valid_shape = new_fv_shape.to_vec();
        current_full_valid = new_fv_slice.to_vec();
        drop(out_step);
    }

    let num_gen_frames = generated_codes.len();

    // 5. Decode Waveform qua MossCodec Decoder
    let mut flat_codes = Vec::with_capacity(num_gen_frames * 16);
    for frame in &generated_codes {
        flat_codes.extend_from_slice(frame);
    }

    let val_codec_codes = Value::from_array(([1, num_gen_frames, 16], flat_codes)).map_err(|e| format!("{:?}", e))?;
    let val_codec_len = Value::from_array(([1], vec![num_gen_frames as i32])).map_err(|e| format!("{:?}", e))?;

    let out_codec = sess_codec.run(ort::inputs![
        "audio_codes" => val_codec_codes,
        "audio_code_lengths" => val_codec_len,
    ]).map_err(|e| format!("{:?}", e))?;

    let (audio_shape, audio_slice) = out_codec["audio"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
    let channels = audio_shape[1] as usize;
    let max_len = audio_shape[2] as usize;
    let n_samples = if let Some(out_len) = out_codec.get("audio_lengths") {
        let (_, l_t) = out_len.try_extract_tensor::<i32>().map_err(|e| format!("{:?}", e))?;
        (l_t[0] as usize).min(max_len)
    } else {
        max_len
    };

    let mut mono_samples = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += audio_slice[c * max_len + i];
        }
        mono_samples.push(sum / (channels as f32));
    }

    Ok(mono_samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 1. Đọc văn bản đầu vào
    let raw_text = if let Some(ref file_path) = args.file {
        println!("Đang đọc văn bản từ file: {:?}", file_path);
        fs::read_to_string(file_path)?
    } else if let Some(ref text) = args.text {
        if text == "-" {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        } else {
            text.clone()
        }
    } else {
        eprintln!("Lỗi: Vui lòng cung cấp văn bản cần đọc hoặc đường dẫn file qua --file <path>");
        std::process::exit(1);
    };

    if raw_text.trim().is_empty() {
        eprintln!("Lỗi: Văn bản đầu vào rỗng.");
        std::process::exit(1);
    }

    println!("============================================================");
    println!("     ZeroTTS High-Performance Standalone Rust Runner");
    println!("============================================================");
    println!("Đầu ra âm thanh: {:?} (Bitrate: {})", args.output, args.bitrate);
    println!("Giọng đọc:       {}", args.voice);
    println!("Thư mục model:   {:?}", args.model_dir);

    // 2. Chuẩn hóa số tiếng Việt
    let processed_text = if !args.no_norm {
        let norm = normalize_vietnamese_text(&raw_text);
        if norm != raw_text {
            println!("\n[Chuẩn hóa tiếng Việt] Đã chuyển đổi các con số/ngày tháng thành chữ.");
        }
        norm
    } else {
        raw_text
    };

    // 3. Tách văn bản thành các câu ngắn (chunking chapter truyện)
    let sentences = split_sentences(&processed_text);
    println!("\n[Phân rã câu] Tổng số câu cần tổng hợp: {} câu", sentences.len());

    let t_start = Instant::now();

    // 4. Khởi tạo môi trường ONNX Runtime trong Rust
    let candidates = [
        std::env::var("ORT_DYLIB_PATH").ok(),
        Some("/home/quyen/08_TTS/ZeroTTS/venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so".to_string()),
        Some("/usr/local/lib/libonnxruntime.so".to_string()),
    ];

    let mut env_opt = None;
    for path_str in candidates.into_iter().flatten() {
        let p = Path::new(&path_str);
        if p.exists() {
            if let Ok(builder) = ort::init_from(p) {
                if let Ok(env) = builder.build() {
                    env_opt = Some(env);
                    break;
                }
            }
        }
    }

    let env = match env_opt {
        Some(e) => e,
        None => ort::init().build().map_err(|e| format!("Lỗi khởi tạo ORT env: {:?}", e))?,
    };

    let create_session = |path: PathBuf| -> Result<Session, Box<dyn std::error::Error>> {
        let builder = Session::builder(&env)
            .map_err(|e| format!("{:?}", e))?
            .with_intra_threads(4)
            .map_err(|e| format!("{:?}", e))?;
        let mut builder = if args.cuda {
            match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("Cảnh báo: Không thể bật CUDA ({:?}), chuyển về CPU", e);
                    Session::builder(&env)
                        .map_err(|e| format!("{:?}", e))?
                        .with_intra_threads(4)
                        .map_err(|e| format!("{:?}", e))?
                }
            }
        } else {
            builder
        };
        let sess = builder.commit_from_file(path).map_err(|e| format!("{:?}", e))?;
        Ok(sess)
    };

    println!("\n[Nạp đồ thị ONNX] Đang tải các mô hình...");
    let t_load0 = Instant::now();
    let onnx_dir = args.model_dir.join("onnx");
    let mut sess_text = create_session(onnx_dir.join("text_encoder.onnx"))?;
    let mut sess_prefix = create_session(onnx_dir.join("prefix_step.onnx"))?;
    let mut sess_decode = create_session(onnx_dir.join("local_frame_decode.onnx"))?;
    let mut sess_codec = create_session(onnx_dir.join("codec").join("moss_audio_tokenizer_decode_full.onnx"))?;

    let tok_path = args.model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| format!("{:?}", e))?;

    // Nạp Voice Embeddings
    let null_voice_path = Path::new("/home/quyen/08_TTS/zerotts-rust/null_voice_emb.bin");
    let huuduc_voice_path = Path::new("/home/quyen/08_TTS/zerotts-rust/huuduc.bin");

    let null_voice = load_f32_bin(null_voice_path)?;
    let voice_emb = if args.voice == "huuduc" && huuduc_voice_path.exists() {
        load_f32_bin(huuduc_voice_path)?
    } else {
        null_voice
    };
    println!("✓ Đã nạp xong model & tokenizer trong: {:.2?}", t_load0.elapsed());

    // 5. Tổng hợp từng câu và ghép nối
    println!("\n[Tổng hợp giọng nói] Đang xử lý...");
    let sample_rate = 48000;
    let pause_samples = (args.pause_dur * sample_rate as f64).round() as usize;
    let pause_audio = vec![0.0f32; pause_samples];

    let mut full_audio: Vec<f32> = Vec::new();
    let mut subtitles: Vec<SubtitleItem> = Vec::new();
    let mut current_time = 0.0;

    let t_synth_all = Instant::now();

    for (idx, sentence) in sentences.iter().enumerate() {
        let sent_num = idx + 1;
        let t0 = Instant::now();
        let audio_piece = synthesize_sentence(
            sentence,
            &mut sess_text,
            &mut sess_prefix,
            &mut sess_decode,
            &mut sess_codec,
            &tokenizer,
            &voice_emb,
            args.max_frames,
        )?;
        let dur = audio_piece.len() as f64 / sample_rate as f64;
        let elapsed = t0.elapsed().as_secs_f64();
        let rtf = elapsed / dur;

        let start_sec = current_time;
        let end_sec = current_time + dur;

        subtitles.push(SubtitleItem {
            id: sent_num,
            start_sec,
            end_sec,
            text: sentence.clone(),
        });

        full_audio.extend_from_slice(&audio_piece);
        full_audio.extend_from_slice(&pause_audio);
        current_time = end_sec + args.pause_dur;

        let preview = if sentence.chars().count() > 45 {
            let truncated: String = sentence.chars().take(45).collect();
            format!("{}…", truncated)
        } else {
            sentence.clone()
        };

        println!(
            "[{:02}/{:02}] {:<48} | {:.2}s audio | RTF: {:.2}x",
            sent_num,
            sentences.len(),
            format!("\"{}\"", preview),
            dur,
            rtf
        );
    }

    let total_synth_elapsed = t_synth_all.elapsed().as_secs_f64();
    let total_audio_dur = full_audio.len() as f64 / sample_rate as f64;

    // 6. Lưu file âm thanh hoàn chỉnh (MP3 hoặc WAV)
    println!("\n[Lưu file âm thanh] Đang ghi file {:?}...", args.output);
    save_audio(&full_audio, sample_rate, &args.output, &args.bitrate)?;

    let file_size_mb = fs::metadata(&args.output)
        .map(|m| m.len() as f64 / (1024.0 * 1024.0))
        .unwrap_or(0.0);

    // 7. Sinh file phụ đề SRT
    let srt_path = if let Some(ref custom_srt) = args.srt {
        Some(custom_srt.clone())
    } else if args.generate_srt {
        Some(args.output.with_extension("srt"))
    } else {
        None
    };

    if let Some(ref p_srt) = srt_path {
        write_srt_file(&subtitles, p_srt)?;
        println!("✓ Đã xuất file phụ đề SRT: {:?}", p_srt);
    }

    let total_elapsed = t_start.elapsed().as_secs_f64();
    let total_rtf = total_synth_elapsed / total_audio_dur;

    println!("\n============================================================");
    println!("             HOÀN TẤT TỔNG HỢP (ZERO-TTS RUST)");
    println!("============================================================");
    println!("✓ File âm thanh:      {:?} ({:.2} MB)", args.output, file_size_mb);
    if let Some(ref p_srt) = srt_path {
        println!("✓ File phụ đề SRT:    {:?}", p_srt);
    }
    println!("✓ Tổng số câu:        {} câu", sentences.len());
    println!("✓ Thời lượng audio:   {:.2} giây (~{:.2} phút)", total_audio_dur, total_audio_dur / 60.0);
    println!("✓ Thời gian render:   {:.2} giây (RTF: {:.2}x - Gấp {:.1}x thời gian thực)", total_synth_elapsed, total_rtf, 1.0 / total_rtf);
    println!("✓ Tổng thời gian:     {:.2} giây (bao gồm nạp model)", total_elapsed);
    println!("============================================================");

    Ok(())
}
