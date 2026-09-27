use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;

use magna_middleware::api::public_api::Middleware;
use magna_middleware::utils::errors::{MiddlewareConfig, Precision};

fn inference_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");
    // Ensure accurate benchmarking
    group.measurement_time(Duration::from_secs(10));
    group.warm_up_time(Duration::from_secs(3));
    group.sample_size(50);

    let model_path =
        std::env::var("MAGNA_BENCH_MODEL").unwrap_or_else(|_| "mobilenetv2-7.onnx".to_string());
    let image_path = std::env::var("MAGNA_BENCH_IMAGE")
        .unwrap_or_else(|_| "tests/assets/sample.jpg".to_string());

    // Only run if the model exists to prevent bench failures when setup is missing
    if !std::path::Path::new(&model_path).exists() {
        println!(
            "Skipping bench: {} not found. Set MAGNA_BENCH_MODEL to test.",
            model_path
        );
        return;
    }
    if !std::path::Path::new(&image_path).exists() {
        println!(
            "Skipping bench: {} not found. Set MAGNA_BENCH_IMAGE to test.",
            image_path
        );
        return;
    }

    let mw = Middleware::new();
    let config = MiddlewareConfig {
        fallback_precision: Precision::FP32,
        warmup_runs: 0,
        debug: false,
        labels_path: None,
        model_path: Some(model_path.clone()),
        grpc_address: String::new(),
    };

    mw.initialize(config)
        .expect("Middleware initialization failed");
    let info = mw.load_engine(&model_path).expect("Engine load failed");

    group.bench_function(format!("infer_{}", info.name), |b| {
        b.iter(|| {
            let _ = mw.infer_from_image(&image_path).unwrap();
        });
    });

    group.finish();
}

criterion_group!(benches, inference_benchmark);
criterion_main!(benches);
