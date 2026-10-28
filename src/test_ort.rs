use ort::session::{Session, builder::GraphOptimizationLevel};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing ORT crate with CUDA 12 dynamic loading...");
    let so_path = "/home/quyen/08_TTS/ZeroTTS/venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so";
    let env = ort::init_from(so_path)?.build()?;

    let builder = Session::builder(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level2)?
        .with_intra_threads(4)?;

    let mut builder = match builder.with_execution_providers([ort::ep::CUDA::default().build()]) {
        Ok(b) => {
            println!("CUDA Provider attached!");
            b
        }
        Err(e) => {
            println!("Failed to attach CUDA EP: {:?}", e);
            Session::builder(&env)?
                .with_optimization_level(GraphOptimizationLevel::Level2)?
                .with_intra_threads(4)?
        }
    };

    let _session = builder.commit_from_file("/home/quyen/08_TTS/ZeroTTS/model_weights/onnx/text_encoder.onnx")?;
    println!("SUCCESS: Session committed on CUDA 12!");
    Ok(())
}
