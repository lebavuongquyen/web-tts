use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hound::{WavSpec, WavWriter};
use ort::session::Session;
use ort::value::Value;
use rand::Rng;
use serde_json::json;
use tokenizers::Tokenizer;

use rust_tts_runtime::voice::{list_voices, load_voice, parse_and_blend_spec, Voice};

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[derive(Parser, Debug)]
#[command(name = "rust_tts_runtime")]
#[command(
    author = "Quyen Le",
    version = "0.1.0",
    about = "High-Performance Pure TTS Runtime Engine - Generates Clean PCM WAV",
    long_about = None
)]
struct Args {
    /// Văn bản hoặc chuỗi âm vị cần đọc (1 câu đơn lẻ)
    #[arg(index = 1)]
    text: Option<String>,

    /// Tên giọng đọc (huuduc, kimoanh...), đường dẫn file .bin (ZeroTTS) HOẶC đường dẫn model .onnx (VITS)
    #[arg(short, long, default_value = "huuduc")]
    voice: String,

    /// Pha trộn giọng lai cho ZeroTTS (ví dụ: "huuduc:0.7,kimoanh:0.3")
    #[arg(long)]
    blend: Option<String>,

    /// Danh sách mã âm vị Phoneme IDs (dành cho VITS Engine: ví dụ "1, 23, 45, 2")
    #[arg(long)]
    phoneme_ids: Option<String>,

    /// Tần số lấy mẫu mong muốn khi chạy VITS (mặc định: 22050)
    #[arg(long, default_value_t = 22050)]
    vits_sample_rate: u32,

    /// Đường dẫn file âm thanh xuất ra (.wav)
    #[arg(short, long, default_value = "output.wav")]
    output: PathBuf,

    /// Thư mục chứa model weights của ZeroTTS
    #[arg(short, long, default_value = "/home/quyen/08_TTS/ZeroTTS/model_weights")]
    model_dir: PathBuf,

    /// Bật CUDA Execution Provider (nếu có GPU NVIDIA)
    #[arg(long, default_value_t = true)]
    cuda: bool,

    /// Giới hạn số frame sinh tối đa cho mỗi câu (ZeroTTS)
    #[arg(long, default_value_t = 1500)]
    max_frames: usize,

    /// Liệt kê tất cả các giọng có sẵn trong model rồi thoát
    #[arg(long, default_value_t = false)]
    list_voices: bool,

    /// Xuất log dạng JSON thuần túy (cho Wrapper dễ parse)
    #[arg(long, default_value_t = false)]
    json: bool,
}

/// Ghi mảng f32 [-1.0, 1.0] ra file WAV chuẩn PCM 16-bit mono
fn write_wav_file(samples: &[f32], sample_rate: u32, path: &Path) -> Result<(), Box<dyn std::error::Error>> {
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

fn create_ort_session(
    env: &ort::environment::Environment,
    path: &Path,
    use_cuda: bool,
) -> Result<Session, Box<dyn std::error::Error>> {
    let builder = Session::builder(env)?
        .with_intra_threads(4)
        .map_err(|e| format!("{:?}", e))?;

    let mut builder = if use_cuda {
        match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("Cảnh báo: Không thể bật CUDA ({:?}), chuyển về CPU", e);
                Session::builder(env)?
                    .with_intra_threads(4)
                    .map_err(|e| format!("{:?}", e))?
            }
        }
    } else {
        builder
    };

    let sess = builder.commit_from_file(path).map_err(|e| format!("{:?}", e))?;
    Ok(sess)
}

/// Chạy động cơ VITS (1-step forward pass)
fn run_vits_engine(
    model_path: &Path,
    phoneme_ids: &[i64],
    _sample_rate: u32,
    env: &ort::environment::Environment,
    use_cuda: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut sess = create_ort_session(env, model_path, use_cuda)?;

    let l_phonemes = phoneme_ids.len();
    let val_input = Value::from_array(([1, l_phonemes], phoneme_ids.to_vec())).map_err(|e| format!("{:?}", e))?;
    let val_lengths = Value::from_array(([1], vec![l_phonemes as i64])).map_err(|e| format!("{:?}", e))?;
    let val_scales = Value::from_array(([3], vec![0.667f32, 1.0f32, 0.8f32])).map_err(|e| format!("{:?}", e))?;

    let outputs = sess.run(ort::inputs![
        "input" => val_input,
        "input_lengths" => val_lengths,
        "scales" => val_scales,
    ]).map_err(|e| format!("{:?}", e))?;

    let (_, audio_slice) = outputs["output"].try_extract_tensor::<f32>().map_err(|e| format!("{:?}", e))?;
    Ok(audio_slice.to_vec())
}

/// Chạy động cơ ZeroTTS (Text Encoder + Prefix AR Loop + MossCodec)
fn run_zerotts_engine(
    text: &str,
    voice_emb: &[f32],
    model_dir: &Path,
    max_frames: usize,
    env: &ort::environment::Environment,
    use_cuda: bool,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let onnx_dir = model_dir.join("onnx");
    let mut sess_text = create_ort_session(env, &onnx_dir.join("text_encoder.onnx"), use_cuda)?;
    let mut sess_prefix = create_ort_session(env, &onnx_dir.join("prefix_step.onnx"), use_cuda)?;
    let mut sess_decode = create_ort_session(env, &onnx_dir.join("local_frame_decode.onnx"), use_cuda)?;
    let mut sess_codec = create_ort_session(env, &onnx_dir.join("codec").join("moss_audio_tokenizer_decode_full.onnx"), use_cuda)?;

    let tok_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tok_path).map_err(|e| format!("{:?}", e))?;

    // 1. Tokenize Text
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

    // 2. Cold start prefix step
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

    // 3. Autoregressive Loop
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
    if num_gen_frames == 0 {
        return Ok(Vec::new());
    }

    // 4. Codec Decode
    let mut flat_codes = Vec::with_capacity(num_gen_frames * 16);
    for frame in &generated_codes {
        for &c in frame {
            flat_codes.push(c as i32);
        }
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

    // 1. Liệt kê danh sách giọng (nếu yêu cầu)
    if args.list_voices {
        match list_voices(&args.model_dir) {
            Ok(voices) => {
                if args.json {
                    println!("{}", json!({ "voices": voices }).to_string());
                } else {
                    println!("============================================================");
                    println!("         DANH SÁCH GIỌNG ĐỌC CÓ SẴN (MODEL VOICES)");
                    println!("============================================================");
                    for (idx, v) in voices.iter().enumerate() {
                        println!(
                            "[{:02}] {:<12} | {:<12} | Giới tính: {:<6} | {}",
                            idx + 1,
                            v.name,
                            v.display_name,
                            v.gender,
                            v.description
                        );
                    }
                }
            }
            Err(e) => eprintln!("Lỗi liệt kê giọng: {}", e),
        }
        return Ok(());
    }

    // 2. Kiểm tra văn bản hoặc phoneme input
    let text = if let Some(ref t) = args.text {
        if t == "-" {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        } else {
            t.clone()
        }
    } else if args.phoneme_ids.is_none() {
        eprintln!("Lỗi: Vui lòng cung cấp văn bản cần đọc hoặc --phoneme-ids");
        std::process::exit(1);
    } else {
        String::new()
    };

    let t_start = Instant::now();

    // 3. Khởi tạo môi trường ORT
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

    // 4. Nhận diện động cơ: VITS ONNX hay ZeroTTS
    let is_vits = args.voice.ends_with(".onnx") || Path::new(&args.voice).extension().map_or(false, |ext| ext == "onnx");

    let (samples, sample_rate) = if is_vits {
        // Chạy VITS Engine
        let model_path = Path::new(&args.voice);
        let phonemes: Vec<i64> = if let Some(ref p_str) = args.phoneme_ids {
            p_str
                .split([',', ' '])
                .filter_map(|s| s.trim().parse::<i64>().ok())
                .collect()
        } else {
            eprintln!("Lỗi: Động cơ VITS ONNX yêu cầu truyền --phoneme-ids");
            std::process::exit(1);
        };

        let raw_samples = run_vits_engine(model_path, &phonemes, args.vits_sample_rate, &env, args.cuda)?;
        (raw_samples, args.vits_sample_rate)
    } else {
        // Chạy ZeroTTS Engine (48kHz)
        let selected_voice: Voice = if let Some(ref blend_spec) = args.blend {
            parse_and_blend_spec(&args.model_dir, blend_spec)?
        } else {
            load_voice(&args.model_dir, &args.voice)?
        };

        let raw_samples = run_zerotts_engine(&text, &selected_voice.emb, &args.model_dir, args.max_frames, &env, args.cuda)?;
        (raw_samples, 48000)
    };

    let dur_sec = samples.len() as f64 / sample_rate as f64;
    let elapsed = t_start.elapsed().as_secs_f64();
    let rtf = if dur_sec > 0.0 { elapsed / dur_sec } else { 0.0 };

    // 5. Ghi file WAV PCM thuần túy
    write_wav_file(&samples, sample_rate, &args.output)?;

    if args.json {
        let res = json!({
            "status": "success",
            "output": args.output.to_string_lossy(),
            "samples": samples.len(),
            "sample_rate": sample_rate,
            "duration": dur_sec,
            "elapsed": elapsed,
            "rtf": rtf
        });
        println!("{}", res.to_string());
    } else {
        println!("✓ File âm thanh WAV: {:?}", args.output);
        println!("✓ Thời lượng: {:.2}s | Thời gian: {:.2}s | RTF: {:.2}x", dur_sec, elapsed, rtf);
    }

    Ok(())
}
