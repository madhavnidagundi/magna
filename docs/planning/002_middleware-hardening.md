# Phase 2 — Middleware Hardening

**Status:** Proposed  
**Audience:** Development Team  
**Prerequisite:** Phase 1 capability and tensor-contract refactoring  
**Source:** [Middleware assessment](000_middleware-assessment.md)
**Primary outcome:** Make prototype inference fail safely and predictably when inputs, model metadata, SDKs, or native runtimes are invalid.

## Student Ownership and Decision Authority

This phase defines safety and correctness outcomes, not one mandatory implementation. Under [the planning roadmap](README.md), the Student Development Team should inspect each trust boundary, compare alternatives and agree on the validation, error and recovery design before implementation.

The proposed structures, numeric limits, error mappings, feature names and sequence below are starting points. The team may replace them with evidence-based alternatives. It must, however, preserve the non-negotiable behavior: untrusted sizes are checked, invalid input cannot reach unsafe operations, backend availability is truthful, failures leave a defined state and required tests cannot pass without exercising their declared environment.

Before committing H1–H7, verify which Phase 1 contracts have landed and inventory available hardware, SDKs, models and experienced reviewers. The areas remain open for volunteers. Hardware/FFI areas should use pairs or small groups, while portable validation and CI work may proceed independently. If a designated real-hardware environment is unavailable, mark the affected area blocked or deferred rather than weakening its acceptance criteria.

## 1. Purpose

This phase hardens correctness boundaries around the Rust/native interface and the gRPC input path. “Hardening” here means reliable experimental software: invalid inputs are rejected, backend availability is truthful, failures are observable and CI does not report unexecuted hardware tests as success.

This phase is not a production-security project. Production-only concerns are listed separately at the end.

## 2. Goals and Non-Goals

### Goals

1. Enforce exact TensorRT input/output byte and precision contracts.
2. Prevent overflow and unreasonable allocations before FFI calls.
3. Remove silent mock or SDK-missing success paths.
4. Validate gRPC tensors before constructing runtime buffers.
5. Validate native metadata and return codes defensively.
6. Make CI distinguish compile checks, mock tests and real hardware tests.
7. Add recovery and negative-path coverage.

### Non-Goals

- Authentication, authorization, TLS, or public-network exposure
- Signed models or supply-chain provenance
- High-availability service operation
- Kubernetes, deployment orchestration, or incident response
- Maximizing throughput; profiling and benchmarking are separate phases

## 3. Hardening Principles

1. **Validate at every trust boundary.** gRPC data is checked at ingress; tensors are checked against model metadata; native wrappers check sizes again.
2. **No implicit conversion.** A precision mismatch is an error until a tested conversion routine exists.
3. **No silent fallback.** Missing SDKs, missing native symbols and stubs must be visible in capabilities and errors.
4. **Checked arithmetic only.** Shape products and byte sizes must never wrap.
5. **Test results must be truthful.** A required hardware test cannot pass by returning early.
6. **Limits are configurable and documented.** Prototype caps prevent accidental resource exhaustion without claiming production-grade isolation.

## 4. Shared Validation Model

Phase 1 introduces central tensor validation. This phase extends it with explicit limits:

```rust
pub struct InferenceLimits {
	 pub max_inputs: usize,
	 pub max_rank: usize,
	 pub max_dimension: usize,
	 pub max_tensor_elements: usize,
	 pub max_tensor_bytes: usize,
	 pub max_total_input_bytes: usize,
}
```

The team should derive conservative defaults suitable for current image models and decide how operators configure them. Do not use multi-gigabyte constants without evidence. Values worth evaluating initially are:

- 8 input tensors;
- rank up to 8;
- each dimension up to 1,000,000;
- 64 million elements per tensor;
- 512 MiB per tensor;
- 1 GiB total request tensor bytes.

The final values must be justified against an inventory of available repository models and, where relevant, hardware experiments. Record for each model its binding count, rank, largest dimension, element count and byte size, then document the safety margin used to select each default. One model is enough to establish a provisional prototype limit, but not to claim broad model coverage. Every multiplication and sum must use `checked_mul` or `checked_add`.

Validation order should be cheap to expensive:

1. input count;
2. datatype;
3. rank and dimensions;
4. checked element count;
5. checked expected byte count;
6. actual byte count;
7. model-spec and backend-capability matching.

## 5. Candidate Work Areas

These areas identify correctness boundaries that require investigation and evidence. They do not prescribe the final technical solution or issue breakdown. Students should review the relevant implementation, discuss alternatives and create concrete work packages only after agreeing on the approach.

### H1 — Exact TensorRT transfer contract

**Files:**

- [nvidia/adapter.rs](../../middleware/src/backends/nvidia/adapter.rs)
- [trt_c_api.cpp](../../middleware/src/backends/nvidia/trt_c_api.cpp)
- [traits.rs](../../middleware/src/inference/traits.rs)

**Current failure mode:** The Rust adapter allows FP32 byte length for lower-precision engines, while the C++ wrapper copies the supplied length into an engine-sized device buffer. The wrapper performs no datatype conversion. Returned output is also labelled FP32 regardless of the engine output precision.

**Required design:**

1. Rust must require input shape, precision and byte length to match `TensorSpec` exactly.
2. Remove the FP32 “auto-conversion” exception.
3. Return output with `output_spec.precision`.
4. C++ must store allocated input/output byte capacities in `TrtContext`.
5. `trt_infer` and `trt_infer_graph` must reject any transfer length different from the stored capacity.
6. Check element-count and byte-size multiplication before `cudaMalloc`.
7. Check every CUDA and TensorRT return value, including tensor-address assignment, dynamic-shape assignment, graph capture, allocation, enqueue and copies.
8. Do not infer FP8 versus INT8 from byte width alone. Query and return the actual TensorRT datatype through the C ABI.
9. Reject unsupported TensorRT datatypes explicitly.

**C ABI change:** Prefer functions returning a stable internal datatype code rather than only element size. Keep all Rust/C++ code mapping in one documented table.

**Tests:**

- Pure Rust tests for exact shape/precision/byte matching
- C++ unit or harness tests for over-sized and under-sized transfer rejection
- Hardware tests for FP32, FP16 and INT8 engines
- FP8 test only on hardware and TensorRT versions that support it
- Test that returned buffer precision equals engine metadata

**Acceptance criteria:**

- No branch accepts mismatched input bytes.
- No host-to-device or device-to-host copy can exceed stored capacity.
- Output precision is preserved end to end.
- Unsupported datatypes fail during engine loading.
- Required tests pass on the designated NVIDIA runner.

### H2 — Checked tensor arithmetic and prototype limits

**Files:**

- [traits.rs](../../middleware/src/inference/traits.rs)
- Phase 1 `inference/validation.rs`
- all backend adapters

**Tasks:**

1. Replace `num_elements() -> usize` usage on untrusted shapes with `checked_num_elements() -> MiddlewareResult<usize>`.
2. Add `checked_byte_size()`.
3. Reject zero concrete dimensions unless scalar/dynamic semantics explicitly allow them.
4. Apply `InferenceLimits` before allocating output or entering FFI.
5. Validate native-reported dimensions, ranks, element sizes and counts before converting signed values to `usize`.
6. Ensure empty outputs and zero-element outputs produce defined errors rather than division by zero or invalid allocations.

**Tests:**

- multiplication overflow;
- total-byte addition overflow;
- zero dimensions;
- excessive rank/dimension/elements/bytes;
- boundary values exactly at each configured limit;
- malformed native metadata represented through test helpers.

**Acceptance criteria:**

- No untrusted shape uses `.product()` directly.
- Every allocation size derives from checked arithmetic.
- Limit errors identify the tensor and violated limit.
- Defaults are documented and configurable by the server.

### H3 — gRPC inference validation

**Files:**

- [magna_server.rs](../../middleware/src/bin/magna_server.rs)
- [magna.proto](../../middleware/proto/magna.proto)
- new focused tests for server request conversion

**Tasks:**

1. Extract request conversion into a testable function returning `Result<Vec<TensorBuffer>, Status>`.
2. Reject empty names where the selected backend requires named bindings.
3. Reject unknown datatype strings instead of defaulting to FP32.
4. Reject non-positive `int64` dimensions before conversion to `usize`.
5. Apply configured rank, dimension, element, tensor-byte and total-byte limits.
6. Require `raw_data.len()` to equal the checked expected size.
7. Validate input names/counts against loaded model metadata.
8. Map client mistakes to `invalid_argument`; unavailable backend to `failed_precondition`; internal/native failures to `internal`.

Changing datatype from string to a protobuf enum is desirable but should be a separate protocol-version PR because existing Python clients use strings.

**Tests:**

- unknown datatype;
- negative, zero, oversized and overflowing shapes;
- byte-size mismatch;
- duplicate and unknown names;
- too many inputs;
- valid FP32, FP16, INT8 and FP8 conversion;
- stable gRPC status-code mapping.

**Acceptance criteria:**

- Invalid requests never reach `Middleware::infer()`.
- No signed-to-unsigned conversion occurs before positivity checks.
- Tests exercise request conversion without starting a network server.
- At least one end-to-end gRPC invalid-request test is included.

### H4 — Isolate mocks and report backend availability truthfully

**Files:**

- [qualcomm/adapter.rs](../../middleware/src/backends/qualcomm/adapter.rs)
- [ti/adapter.rs](../../middleware/src/backends/ti/adapter.rs)
- [build.rs](../../middleware/build.rs)
- [Cargo.toml](../../middleware/Cargo.toml)
- Phase 1 capability types

**Tasks:**

1. Move Qualcomm `FAKE`/`DUMMY` model handling behind `#[cfg(test)]` or a non-default `mock-qualcomm` feature.
2. Move TI uniform-score simulation behind an explicit `mock-ti` feature.
3. Ensure real `qualcomm` or `ti` features without required SDK components report `Unavailable` and fail engine loading.
4. Decide the NVIDIA build behavior:
	- real `nvidia` feature fails the build when required headers/libraries are absent; or
	- introduce `mock-nvidia` for SDK-free compile checks.
5. Never label a backend available based only on a Cargo feature.
6. Include availability and mock/stub state in `GetModelInfo` and health output.

Mock features must never be included in defaults or hardware release commands.

**Tests:**

- tiny fake Qualcomm files are rejected in normal builds;
- mock feature permits deterministic fake inference;
- missing SDK produces an explicit unavailable error;
- TI stub cannot report real readiness;
- health/model metadata distinguishes real, mock, stub and unavailable states.

**Acceptance criteria:**

- Normal builds contain no content-triggered fake model bypass.
- Real-backend readiness implies a native context was created successfully.
- CI has separate jobs for mock contract checks and real hardware execution.

### H5 — Native metadata and FFI defensive checks

**Files:**

- NVIDIA, Qualcomm, TI and system-ORT adapters and native wrappers

**Tasks:**

1. Validate native counts before signed-to-unsigned casts.
2. Validate ranks before slicing fixed arrays.
3. Reject null names or invalid UTF-8 according to a documented fallback policy.
4. Reject zero/unknown element sizes rather than defaulting to FP32.
5. Ensure partial load failures destroy all handles exactly once.
6. Make load/reload commit new handles only after complete validation and buffer allocation.
7. Convert boolean-only native failures to stable error codes where feasible.
8. Audit every `unsafe impl Send/Sync` against vendor thread-safety guarantees; synchronization alone does not make a vendor context thread-safe.
9. Document ownership for every raw pointer and which function destroys it.

**Acceptance criteria:**

- Every FFI return value that affects correctness is checked.
- Negative metadata cannot become a large Rust allocation.
- Failure at each load stage leaves no active partial context.
- Safety comments describe verified invariants rather than assumptions alone.
- A reviewer completes the FFI checklist in Section 9.

### H6 — Failure recovery and state consistency

**Files:**

- [public_api.rs](../../middleware/src/api/public_api.rs)
- [engine_manager.rs](../../middleware/src/inference/engine_manager.rs)
- [state_manager.rs](../../middleware/src/lifecycle/state_manager.rs)

**Tasks:**

1. Define behavior for failed first load, failed reload, inference failure and release failure.
2. Preserve the old ready engine until a replacement is fully loaded, or explicitly release first and transition to a documented non-ready state. Do not mix both semantics.
3. Avoid forcing the entire middleware into `Error` for recoverable input mistakes.
4. Reserve `Error` for corrupted or unavailable runtime state.
5. Add failure injection through test-only backend implementations rather than magic model contents.

**Tests:**

- initialization failure and retry;
- first engine-load failure and retry;
- failed reload behavior;
- inference input error leaves engine ready;
- native execution failure behavior;
- shutdown/release remains idempotent where documented.

**Acceptance criteria:**

- State after every tested failure is specified and asserted.
- Recoverable client errors do not destroy a valid engine.
- No test relies on undocumented internal state.

### H7 — Truthful CI and hardware tests

**Files:**

- [middleware-quality.yml](../../.github/workflows/middleware-quality.yml)
- hardware validation and runner workflows under `.github/workflows`
- [Makefile](../../Makefile)
- integration and smoke tests

**Tasks:**

1. Add `misal/**` and `misal/Cargo.toml` to relevant workflow path filters.
2. Separate jobs into:
	- portable CPU tests;
	- SDK-free mock contract tests;
	- native backend binary link checks;
	- real hardware inference tests.
3. Build complete backend binaries, not only libraries, on SDK-equipped runners.
4. Run each binary's `--help`, inspect dynamic links where useful, load a known model and perform one inference.
5. Replace early-return pseudo-skips with `#[ignore]` for optional tests.
6. When a hardware job declares `TEST_ENGINE_PATH`, missing assets must fail the job.
7. Store backend name, SDK version, model checksum, precision and test result as artifacts.
8. Add a protobuf regeneration check.

**Acceptance criteria:**

- CI cannot report a required hardware inference test as passed without inference.
- Native jobs prove the complete binary links and starts.
- Mock and real backend results are visibly distinct.
- A `misal`-only pull request triggers quality and dependency checks.

## 6. Implementation Sequence

One reasonable implementation order is:

1. **H2:** Checked arithmetic and limits
2. **H3:** gRPC conversion and validation
3. **H1a:** TensorRT Rust-side exact contract
4. **H1b:** TensorRT C++ capacity/datatype contract
5. **H4:** Explicit mock features and availability reporting
6. **H5:** Backend-by-backend FFI audit; one backend per PR
7. **H6:** Failure injection and recovery semantics
8. **H7:** CI and hardware-test restructuring

H1 changes require either NVIDIA hardware validation or an explicit record that hardware validation remains pending.

## 7. Test Matrix

| Layer | CPU | NVIDIA mock | NVIDIA real | Qualcomm mock | Qualcomm real | TI mock | TI real |
|---|---:|---:|---:|---:|---:|---:|---:|
| Formatting/clippy | Required | Required | Required | Required | Required | Required | Required |
| Validation unit tests | Required | Required | Required | Required | Required | Required | Required |
| Binary link/start | Required | Optional | Required | Optional | Required | Optional | Required |
| Known-model inference | Required | Deterministic mock | Required | Deterministic mock | Required | Deterministic mock | Required when implemented |
| Precision contract | FP32 | Contract only | FP32/FP16/INT8 | Contract only | Model precision | Contract only | Model precision |

Real hardware tests must record the SDK/runtime version because engine compatibility depends on it.

## 8. Phase Definition of Done

- Every inference request passes checked validation before FFI.
- TensorRT copies exactly the allocated number of bytes and preserves output precision.
- Qualcomm and TI mocks require explicit mock/test configuration.
- Backend availability reflects a usable implementation, not only a feature flag.
- Native metadata is validated before allocation or casting.
- Recovery behavior is documented and tested.
- Required asset tests fail when assets are missing.
- Real hardware jobs build complete binaries and execute known-model inference.
- Formatting, clippy, tests, rustdoc, audit and deny checks pass.

## 9. FFI Review Checklist

Reviewers must answer each item for every changed native call:

- [ ] Can any signed native value be cast to unsigned before validation?
- [ ] Are pointer nullability and lifetime documented?
- [ ] Is ownership transfer explicit?
- [ ] Is destruction guaranteed exactly once on every error path?
- [ ] Are source and destination capacities checked before copying?
- [ ] Can shape or byte arithmetic overflow?
- [ ] Is the native return code checked and mapped to a useful Rust error?
- [ ] Does the vendor permit this context to move across or be called from threads?
- [ ] Can an unsupported datatype or dynamic shape reach this call?
- [ ] Is there a negative-path test or a documented hardware-only validation step?

## 10. Deferred Production Work

The following are intentionally excluded because the middleware is currently a research prototype:

- gRPC authentication, authorization and TLS;
- atomic/signed model deployment and trust policy;
- public-service rate limiting and tenant isolation;
- sandboxing native model parsers;
- production containers and service orchestration;
- SBOM, container scanning, release signing, SLOs and incident runbooks.

If production use becomes a goal, these items require a separate architecture and threat-model phase. They should not be mixed into the student correctness work above.

## 11. Implementation Notes

Before proposing work packages for a candidate area, read the linked source files and existing tests. Add regression coverage for each behavior change and state whether validation used CPU, a mock backend, or real hardware.

For hardware validation, record the model, precision, board and SDK versions. Do not weaken checks with `#[allow]`, silent fallback, or early-return test skips. Update the relevant capability and error documentation when behavior changes.
