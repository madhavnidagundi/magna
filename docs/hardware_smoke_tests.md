# Hardware Smoke Tests Design

**Purpose**: Define minimal smoke tests to validate that compiled binaries can execute basic inference operations on target hardware.

**Scope**: Phase 1.5 - bridge between compilation validation (Phase 1) and full test automation (Phase 2).

---

## 1. Overview

### TI runtime-selection regression tests

The implemented TI tests are in [middleware/tests/ti_backend_tests.rs](../middleware/tests/ti_backend_tests.rs).
Run from the repository root:

- Without a native TI SDK: `cargo test -p magna-middleware --no-default-features --features ti --test ti_backend_tests`
  verifies that an existing dummy model is rejected with `BackendUnavailable` and no engine becomes ready.
- Without a native TI SDK: `cargo test -p magna-middleware --no-default-features --features mock-ti --test ti_backend_tests`
  verifies the explicit `ti-mock` engine, lifecycle, exact shapes, and uniform mock output.

`mock-ti` implies `ti`, but does not override an installed native runtime. These two
host-only tests are cfg-gated to run only when no native runtime was selected.
Production builds never fall back to mock output unless `mock-ti` was enabled.

For **each** native implementation, use a real single-input/single-output FP32 model
with a known output element count and a matching preprocessed FP32 input file:

1. **DLR:** set `MAGNA_TI_FORCE_DLR=1`, `DLR_INCLUDE_DIR`, and `DLR_LIB_DIR` at
    build time. Set `TEST_TI_MODEL_PATH` to the compiled DLR model directory.
2. **TIDL-enabled ONNX Runtime:** unset `MAGNA_TI_FORCE_DLR` (setting it to `0`
    still forces DLR), set `ORT_TIDL_INCLUDE_DIR` and `ORT_TIDL_LIB_DIR` at build
    time, and set `MAGNA_TIDL_ARTIFACTS_DIR` to the model's TIDL artifacts directory.
    Set `TEST_TI_MODEL_PATH` to the corresponding ONNX model.
3. Set `TEST_TI_INPUT_PATH` to the matching raw FP32 input and
    `TEST_TI_OUTPUT_ELEMS` to the independently known output element count.
4. Run `cargo test -p magna-middleware --no-default-features --features ti --test ti_backend_tests ti_native_lifecycle -- --ignored --exact --nocapture`.

The native test is explicitly ignored during ordinary host tests. When requested,
missing assets or an unavailable native runtime cause a failure rather than a
successful skip. The test is compiled even without the native cfg, so a regression
in either build branch cannot silently turn this command into a zero-test success.
It validates loading, allocation, inference, output size/finite values, and release;
it does not establish numerical accuracy or DSP offload. Do not pass dummy models
to the native runtime. The SDK-dependent native test must be run on a configured
runner; host-only tests are not evidence of native inference.

### Goals
- **Primary**: Verify compiled binaries can load models and perform at least one inference
- **Secondary**: Detect runtime issues (model loading failures, memory errors, SDK mismatches)
- **Non-Goal**: Performance benchmarking, accuracy validation, or stress testing

### Current State (Post-Phase 1)
- ✅ Compilation validates code builds correctly
- ✅ Binary linking verified via `ldd`
- ✅ `--help` flag executes without crashes
- ❌ No validation that inference actually works on hardware

### Smoke Test Philosophy
> **Smoke Test**: Minimal sanity check that core functionality works. If smoke test fails, full testing is pointless.

---

## 2. Smoke Test Requirements

### 2.1 Test Scope (Per Backend)

**Minimal viable test**:
1. Binary launches successfully
2. Loads a tiny test model (e.g., MobileNetV2 or similar)
3. Preprocesses a single test image
4. Runs inference once
5. Returns a result (any result - accuracy not checked)
6. Exits cleanly

**Success criteria**:
- ✅ Process exits with code 0
- ✅ No segfaults or crashes
- ✅ Model loads without SDK errors
- ✅ Inference completes in reasonable time (<30 seconds)

**Failure examples**:
- ❌ CUDA out of memory
- ❌ TensorRT version mismatch
- ❌ Missing SDK libraries at runtime
- ❌ Inference hangs or crashes

### 2.2 Test Assets

**Required files** (stored in `tests/assets/`):
- `test_image.jpg` - Simple 224x224 image for inference
- `test_model.onnx` - Minimal model (~5MB, fast to load)
- Expected output format (for verification, not accuracy)

**Model requirements**:
- Small enough to build/load quickly (<10 seconds)
- Runs on all backends (ONNX format)
- No external dependencies beyond SDK

---

## 3. NVIDIA Orin Smoke Test

### 3.1 Test Implementation

**Workflow step** (add to `hardware-validation.yml`):

```yaml
- name: Smoke test (nvidia inference)
  working-directory: middleware
  run: |
    source $HOME/.cargo/env
    
    # Check test assets exist
    test -f ../tests/assets/test_image.jpg || { echo "❌ Test image missing"; exit 1; }
    test -f ../tests/assets/test_model.onnx || { echo "❌ Test model missing"; exit 1; }
    
    # Run minimal inference test
    cargo test --release --features nvidia smoke_test_inference -- --nocapture
    echo "✅ Smoke test passed"
```

**Rust test** (in `middleware/tests/smoke_tests.rs`):

```rust
#[cfg(feature = "nvidia")]
#[test]
fn smoke_test_inference() {
    use std::path::Path;
    
    // Setup paths
    let model_path = Path::new("../tests/assets/test_model.onnx");
    let image_path = Path::new("../tests/assets/test_image.jpg");
    
    // Load model (backend-specific)
    let model = load_model_nvidia(model_path).expect("Failed to load model");
    
    // Load and preprocess image
    let input = preprocess_image(image_path).expect("Failed to preprocess");
    
    // Run inference
    let output = model.infer(&input).expect("Inference failed");
    
    // Verify output shape (not accuracy)
    assert!(!output.is_empty(), "Output tensor is empty");
    
    println!("✅ Inference completed successfully");
    println!("   Output shape: {:?}", output.shape());
}
```

### 3.2 Test Duration

**Expected timing**:
- Model load: 2-5 seconds
- Inference: 0.1-1 second
- Total: <10 seconds

**Timeout**: Set to 30 seconds (3x expected max)

### 3.3 Resource Requirements

**Minimal model specs**:
- Size: ~5MB ONNX
- Input: 224x224x3 (standard ImageNet size)
- Example: MobileNetV2 quantized or EfficientNet-Lite0

**Test image**:
- Format: JPEG
- Size: 224x224
- Content: Any recognizable object (cat, dog, car)
- Stored in git (small enough, ~50KB)

---

## 4. Feature Combination Testing

### 4.1 Build Matrix

**Current Orin workflow** builds 3 configurations:
1. **Default (CPU)**: Pure software fallback
2. **NVIDIA only**: Hardware-accelerated backend
3. **CPU+NVIDIA**: Runtime backend selection

**Smoke test strategy**:
- Test only the **final build** (CPU+NVIDIA)
- Smoke test validates runtime backend selection works
- No need to test each build separately (compilation already verified)

### 4.2 Backend Selection Test

**Test both backends in one run**:

```rust
#[test]
fn smoke_test_backend_switching() {
    let model_path = Path::new("../tests/assets/test_model.onnx");
    let image_path = Path::new("../tests/assets/test_image.jpg");
    let input = preprocess_image(image_path).unwrap();
    
    // Test CPU backend
    let cpu_model = load_model_cpu(model_path).expect("CPU backend failed");
    let cpu_output = cpu_model.infer(&input).expect("CPU inference failed");
    println!("✅ CPU backend works");
    
    // Test NVIDIA backend (if available)
    #[cfg(feature = "nvidia")]
    {
        let gpu_model = load_model_nvidia(model_path).expect("NVIDIA backend failed");
        let gpu_output = gpu_model.infer(&input).expect("NVIDIA inference failed");
        println!("✅ NVIDIA backend works");
        
        // Verify outputs are similar (rough check)
        assert_eq!(cpu_output.len(), gpu_output.len(), "Backend outputs differ in size");
    }
}
```

---

## 5. Test Asset Preparation

### 5.1 Model Selection

**Recommended model**: MobileNetV2 (quantized INT8)
- Available pre-trained: ✅
- Small size: ~5MB
- Fast inference: <100ms
- Works on all backends: ✅

**Alternative**: EfficientNet-Lite0
- Slightly larger (~10MB)
- Better for testing modern architectures

### 5.2 Test Image

**Requirements**:
- Standard ImageNet preprocessing compatible
- Recognizable object (for manual verification)
- Stored in git (not downloaded at runtime)

**Suggested image**: 
- Sample from ImageNet validation set
- Or: MIT-licensed stock photo of common object

### 5.3 Storage Location

**Existing assets** (already in repository):

```
mi-isal/
├── assets/
│   ├── sample.jpg                    # Test image for inference
│   └── imagenet_class_index.json     # ImageNet labels
├── models/
│   ├── mobilenetv4_nvidia_optimized_int8.engine
│   └── efficientnet_v2_s_nvidia_optimized_int8.engine
```

**Smoke tests use**:
- Model: `models/mobilenetv4_nvidia_optimized_int8.engine`
- Image: `assets/sample.jpg`
- Labels: `assets/imagenet_class_index.json`

No additional test assets need to be created - the existing production assets are suitable for smoke testing.

---

## 6. Implementation Plan

### 6.1 Phase 1.5 Rollout

**Week 1: Test Implementation** ✅ **COMPLETED**
- [x] Write `smoke_tests.rs` module
- [x] Implement basic NVIDIA backend tests
- [x] Update workflow to run smoke tests
- [x] Use existing models and images from repository

**Week 2: Validation**
- [ ] Run smoke test 10x on Orin to ensure stability
- [ ] Document failure modes
- [ ] Implement full inference test (currently commented out)

**Week 3: Enhancement**
- [ ] Add Thor runner testing
- [ ] Update monitoring/alerts
- [ ] Document any additional edge cases

### 6.2 Success Metrics

**Acceptance criteria**:
- Smoke test passes consistently (>95% success rate)
- Completes in <30 seconds
- Catches real runtime issues (not just compilation)
- No false positives from network/timing issues

---

## 7. Failure Scenarios & Handling

### 7.1 Expected Failures

**Model loading failures**:
- Symptom: "Failed to load model" error
- Cause: ONNX Runtime not found, wrong TensorRT version
- Action: Check SDK installation, verify `LD_LIBRARY_PATH`

**Inference crashes**:
- Symptom: Segfault or "CUDA error"
- Cause: GPU memory issue, driver problem
- Action: Check `nvidia-smi`, restart runner service

**Timeout**:
- Symptom: Test exceeds 30 seconds
- Cause: Model too large, GPU busy, network issue
- Action: Review model size, check GPU utilization

### 7.2 Debugging Workflow

```bash
# On Orin, manually run smoke test
cd ~/actions-runner/_work/mi-isal/mi-isal/middleware
source ~/.cargo/env
cargo test --release --features nvidia smoke_test_inference -- --nocapture

# Check GPU status
nvidia-smi

# Verify model files
ls -lh ../tests/assets/

# Check library linkage
ldd target/release/deps/magna_middleware-*
```

---

## 8. Future Enhancements (Phase 2)

**Beyond smoke tests** (deferred):
- Full ImageNet accuracy validation
- Performance benchmarking (latency, throughput)
- Multi-threading stress tests
- Memory leak detection
- Power consumption monitoring

**Smoke tests remain simple**: Just prove inference works, nothing more.

---

## 9. Other Boards

### 9.1 TI TDA4 Smoke Test

**Same approach**:
- Load TI-optimized model
- Run inference with TI SDK
- Verify output

**TI-specific considerations**:
- Model format: TIDL binary (not ONNX)
- Preprocessing: May differ from NVIDIA
- Test assets: TI-specific model required

### 9.2 Qualcomm Smoke Test

**Same pattern**:
- QNN-optimized model
- Qualcomm DSP/HTP inference
- Output verification

**Qualcomm-specific**:
- Model format: DLC (Deep Learning Container)
- Preprocessing: Quantization-aware
- Test assets: Qualcomm-specific

---

## 10. Open Questions

1. **Model licensing**: Can we include MobileNetV2 weights in repo? (Apache 2.0 - yes)
2. **Test image**: Use public domain image or create synthetic?
3. **Output verification**: Check output values or just shape?
4. **Failed test retry**: Auto-retry once on timeout, or fail immediately?
5. **Notification**: Alert on smoke test failure differently than build failure?

---

## 11. References

- ONNX Runtime: https://onnxruntime.ai/
- TensorRT Inference: https://docs.nvidia.com/deeplearning/tensorrt/
- MobileNetV2 Model: https://pytorch.org/vision/stable/models.html
