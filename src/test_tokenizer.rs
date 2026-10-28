use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tok = Tokenizer::from_file("/home/quyen/08_TTS/ZeroTTS/model_weights/tokenizer.json")?;
    let text = "Xin chào, tôi là phiên bản ZeroTTS chạy trực tiếp bằng ngôn ngữ Rust.";
    let cleaned = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let enc = tok.encode(cleaned.as_str(), true)?;
    let mut ids: Vec<i64> = vec![1];
    ids.extend(enc.get_ids().iter().map(|&x| x as i64));
    ids.push(2);
    println!("Tokens: {:?}", enc.get_tokens());
    println!("Rust IDs: {:?}", ids);
    Ok(())
}
