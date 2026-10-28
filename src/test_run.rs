use ort::session::Session;
use ort::value::Value;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let env = ort::init().build()?;
    let mut session = Session::builder(&env)?
        .commit_from_file("/home/quyen/08_TTS/ZeroTTS/model_weights/onnx/text_encoder.onnx")?;
    
    let text_ids_val = Value::from_array(([1, 5], vec![922i64, 9, 316, 9, 23]))?;
    let txt_lengths_val = Value::from_array(([1], vec![5i64]))?;

    let outputs = session.run(ort::inputs![
        "text_ids" => text_ids_val,
        "txt_lengths" => txt_lengths_val,
    ])?;

    println!("Outputs count: {}", outputs.len());
    for (name, _) in outputs.iter() {
        println!("Output: {}", name);
    }
    Ok(())
}
