use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use serde_json::json;

use rust_tts_runtime::voice::{
    extract_voice_from_audio, list_voices, load_voice, parse_and_blend_spec, save_voice_bin, Voice,
};

#[derive(Parser, Debug)]
#[command(name = "rtts-voice")]
#[command(
    author = "Quyen Le",
    version = "0.1.0",
    about = "Pure Rust Voice Tool: Clone voice from audio (.wav/.mp3), Blend voices, and generate 30KB voice.bin",
    long_about = None
)]
struct Args {
    /// File âm thanh mẫu (.wav hoặc .mp3) để trích xuất ra voice.bin
    #[arg(short, long)]
    input: Option<PathBuf>,

    /// Pha trộn giọng lai độc bản (ví dụ: "huuduc:0.7,kimoanh:0.3")
    #[arg(short, long)]
    blend: Option<String>,

    /// Đường dẫn file .bin xuất ra (chuẩn 30.720 bytes)
    #[arg(short, long, default_value = "custom_voice.bin")]
    output: PathBuf,

    /// Thư mục chứa model weights
    #[arg(short, long, default_value = "/home/quyen/08_TTS/ZeroTTS/model_weights")]
    model_dir: PathBuf,

    /// Liệt kê các giọng có sẵn trong model
    #[arg(long, default_value_t = false)]
    list: bool,

    /// Xuất kết quả dạng JSON
    #[arg(long, default_value_t = false)]
    json: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // 1. Liệt kê giọng
    if args.list {
        let voices = list_voices(&args.model_dir)?;
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
        return Ok(());
    }

    let t0 = Instant::now();

    // 2. Xử lý tạo voice
    let voice: Voice = if let Some(ref blend_spec) = args.blend {
        if !args.json {
            println!("Đang pha trộn giọng lai theo cấu hình: \"{}\"...", blend_spec);
        }
        parse_and_blend_spec(&args.model_dir, blend_spec)?
    } else if let Some(ref in_audio) = args.input {
        if !in_audio.exists() {
            eprintln!("Lỗi: File audio không tồn tại: {:?}", in_audio);
            std::process::exit(1);
        }
        if in_audio.extension().map_or(false, |e| e == "bin") {
            load_voice(&args.model_dir, in_audio.to_str().unwrap_or_default())?
        } else {
            if !args.json {
                println!("Đang phân tích đặc trưng DSP thuần Rust từ file {:?}...", in_audio);
            }
            extract_voice_from_audio(in_audio, &args.model_dir)?
        }
    } else {
        eprintln!("Lỗi: Vui lòng cung cấp file audio mẫu qua --input <file.wav/mp3> HOẶC cấu hình --blend <spec>");
        std::process::exit(1);
    };

    // 3. Lưu file .bin chuẩn 30.720 bytes
    save_voice_bin(&voice, &args.output)?;
    let elapsed = t0.elapsed().as_secs_f64();

    if args.json {
        println!(
            "{}",
            json!({
                "status": "success",
                "voice_name": voice.name,
                "output": args.output.to_string_lossy(),
                "byte_size": voice.emb.len() * 4,
                "elapsed_ms": (elapsed * 1000.0).round()
            })
            .to_string()
        );
    } else {
        println!("============================================================");
        println!("      HOÀN TẤT TẠO FILE VOICE.BIN (PURE RUST 100%)");
        println!("============================================================");
        println!("✓ Tên giọng:          {}", voice.name);
        println!("✓ File xuất ra:       {:?} ({} bytes)", args.output, voice.emb.len() * 4);
        println!("✓ Thời gian xử lý:    {:.2} ms ({:.4} giây)", elapsed * 1000.0, elapsed);
        println!("✓ Mẹo sử dụng:        Truyền trực tiếp vào rust_tts_runtime:");
        println!("  ./target/release/rust_tts_runtime \"Xin chào!\" --voice {:?}", args.output);
        println!("============================================================");
    }

    Ok(())
}
