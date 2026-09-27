# Phase 3 — Extend the CPU Backend with OpenVINO

**Status:** Proposed  
**Audience:** Development Team  
**Scope:** MISAL inference middleware only  
**Primary outcome:** Add an OpenVINO-based CPU inference implementation behind the existing `InferenceBackend` abstraction without changing the middleware public API or replacing the existing ONNX Runtime CPU implementations.

## Student Ownership and Decision Authority

This brief defines the desired OpenVINO capability and compatibility outcomes. It does not preselect the binding library, FFI shape, internal state model or exact feature arrangement. Following [the planning roadmap](README.md), the Student Development Team should investigate supported OpenVINO APIs and target packages, prototype the risky integration boundary and discuss its recommendation before committing to an implementation.

The team may revise the proposed files, type names and lifecycle mechanics when the selected design remains isolated behind `InferenceBackend`, reports accurate metadata, fails clearly when unavailable and can be validated with the real runtime. The investigation, rejected alternatives and supported platform assumptions are part of the deliverable.

## Prerequisite and milestone gate

Runtime/package investigation may begin immediately. Adapter implementation should begin only after the Phase 1 backend trait, tensor metadata, validation and lifecycle contracts it uses are accepted; Phase 2 checks that affect the native boundary must be incorporated before the backend is presented as complete.

At kickoff, the team must select the first supported Linux architecture, pinned OpenVINO release, model fixture and milestone depth. O1–O5 remain open work areas for volunteers. A first milestone may deliberately support only a documented subset such as static FP32 models, provided unsupported dynamic shapes, element types or binding patterns fail explicitly and the broader goals remain stretch or deferred work. Do not claim general multi-input, dynamic-shape or architecture support from a single fixture.

## 1. Purpose

MISAL currently provides two CPU inference implementations:

- the default `cpu` implementation, which uses the Rust `ort` crate; and
- `cpu-system-ort`, which uses a direct ONNX Runtime C API wrapper for affected embedded systems.

This task adds OpenVINO as a third, explicitly selected CPU implementation. It must load ONNX models, compile them for the OpenVINO `CPU` device, expose accurate model metadata and execute inference through the existing middleware lifecycle:

```text
Middleware
	|
	v
EngineManager
	|
	v
InferenceBackend
	|
	v
CpuAdapter (cpu-openvino)
	|
	v
OpenVINO Core -> compiled model -> infer request -> CPU plugin
```

OpenVINO must remain private to the backend. Callers continue to use `Middleware`, `EngineInfo`, `TensorSpec` and `TensorBuffer`; no OpenVINO handles or types may enter the public Rust or gRPC interfaces.

## 2. Goals and Non-Goals

### Goals

1. Implement every operation required by `InferenceBackend` for OpenVINO CPU inference.
2. Load ONNX models directly and select the OpenVINO `CPU` device explicitly.
3. Discover all model input and output names, ranks, shapes and supported element types.
4. Support models with multiple inputs and outputs rather than making ImageNet-specific assumptions.
5. Validate tensor count, names, ranks, dimensions, precision and byte lengths before inference.
6. Provide deterministic resource ownership, release and error handling across any FFI boundary.
7. Preserve the existing ONNX Runtime implementations and accelerator backend selection behavior.
8. Provide real OpenVINO tests and reproducible performance measurements on supported Linux CPU targets.

### Non-Goals

- Replacing the default ONNX Runtime CPU backend
- Changing the `InferenceBackend` trait, middleware lifecycle or gRPC protocol unless implementation proves an existing abstraction insufficient
- Supporting OpenVINO GPU, NPU, AUTO or heterogeneous device execution in the first milestone
- Adding implicit image preprocessing, layout conversion, precision casting or output decoding to the backend
- Implementing model quantization or ONNX-to-OpenVINO-IR conversion in the first milestone
- Claiming real-time, safety-critical or accelerator-equivalent performance
- Providing a simulated success path when the OpenVINO SDK or runtime is unavailable

## 3. Existing Integration Points

The implementation should use the current middleware boundaries:

| Existing component | Required integration |
|---|---|
| `middleware/Cargo.toml` | Define the OpenVINO CPU feature and its relationship to `cpu` |
| `middleware/build.rs` | Discover OpenVINO headers/libraries and build or link the selected API boundary |
| `middleware/src/backends/cpu/mod.rs` | Export exactly one `CpuAdapter` implementation |
| `middleware/src/backends/cpu/adapter.rs` | Retain the portable ONNX Runtime implementation |
| `middleware/src/backends/cpu/c_api_adapter.rs` | Retain the system ONNX Runtime implementation |
| `middleware/src/inference/traits.rs` | Reuse `InferenceBackend`, `EngineInfo`, `TensorSpec` and `TensorBuffer` |
| `middleware/src/inference/engine_manager.rs` | Continue selecting CPU after NVIDIA, Qualcomm and TI |
| `middleware/src/utils/errors.rs` | Return typed middleware errors with OpenVINO context |

Possible new files are:

```text
middleware/src/backends/cpu/
	openvino_adapter.rs
	openvino_c_api.cpp       # only if a project-owned C ABI is required
```

The implementation should not create a parallel engine manager or a second public inference path.

## 4. Feature and Backend Selection

One option is a `cpu-openvino` Cargo feature that implies `cpu`:

```text
default                     -> portable ONNX Runtime CpuAdapter
cpu                         -> portable ONNX Runtime CpuAdapter
cpu + cpu-system-ort        -> system ONNX Runtime CpuAdapter
cpu + cpu-openvino          -> OpenVINO CpuAdapter
```

`cpu-system-ort` and `cpu-openvino` must be mutually exclusive. Enabling both must produce a clear compile-time error because both select the implementation exported as `CpuAdapter`.

The top-level backend priority remains:

1. NVIDIA
2. Qualcomm
3. TI
4. CPU

No OpenVINO-specific branch is required in `EngineManager`; CPU sub-features determine which `CpuAdapter` is compiled. `backend_name()` must return `cpu-openvino` so health information, logs and benchmark reports distinguish it from the ONNX Runtime implementations.

## 5. OpenVINO Adapter Design

### 5.1 State and lifecycle

The adapter should own state equivalent to:

```text
OpenVinoState
	core
	model or compiled_model
	infer_request
	engine_info
	buffers_allocated
```

Mutable state must be protected by one appropriate synchronization primitive so the adapter satisfies `Send + Sync`. The first implementation may serialize inference through one infer request. An infer-request pool should be considered only after profiling demonstrates a need and its bounds are defined.

The trait operations must behave as follows:

| Operation | Required behavior |
|---|---|
| `backend_name()` | Return `cpu-openvino` |
| `load_engine()` | Validate the path, create the OpenVINO Core, read the ONNX model, compile it for `CPU`, enumerate all bindings and retain `EngineInfo` |
| `allocate_buffers()` | Create the infer request and any reusable resources; a no-op is acceptable only if allocation is genuinely deferred and readiness remains accurate |
| `infer()` | Validate and bind every named input, run synchronous inference and return every output with its actual runtime metadata |
| `release()` | Drop infer requests, compiled models, models and Core-owned resources in a deterministic order; repeated release must be safe |
| `build_engine()` | Return `NotSupported` for the first milestone |
| `engine_info()` | Return metadata for the currently loaded model |
| `is_ready()` | Return true only when a model is compiled and inference resources are available |

A failed load must not leave stale metadata or a partially ready adapter. Loading a second model must either replace the first model atomically or fail while preserving a documented valid state.

### 5.2 Model metadata

`load_engine()` must enumerate every model input and output. It must not assume:

- one input or one output;
- an input binding named `input` or `data`;
- NCHW layout;
- shape `[1, 3, 224, 224]`;
- classification output `[1, 1000]`; or
- FP32 precision.

Static dimensions are copied into `TensorSpec`. Dynamic dimensions are represented as `0`, matching the current CPU metadata convention. The adapter must preserve rank and binding names. If OpenVINO exposes aliases, the adapter must select and document one canonical, stable name and reject ambiguous request bindings.

The current `EngineInfo::memory_bytes` cannot necessarily represent OpenVINO's complete runtime allocation. It should be set only from a defensible measurement; otherwise it remains `0` and the limitation is documented.

### 5.3 Precision mapping

The first milestone supports only lossless mappings representable by MISAL:

| OpenVINO element type | MISAL precision |
|---|---|
| `f32` | `FP32` |
| `f16` | `FP16` |
| signed `i8` | `INT8` |

Unsupported or ambiguous element types must fail model loading with an error that identifies the tensor and OpenVINO element type. In particular, unsigned integers must not be labeled as signed `INT8`. FP8 must not be advertised until the model format, OpenVINO CPU plugin and MISAL byte representation are verified to use the same format.

If common models require element types that `Precision` cannot represent, extending the shared type system must be proposed and reviewed separately rather than hidden inside this adapter.

### 5.4 Input validation and inference

Before invoking OpenVINO, `infer()` must:

1. Require the adapter to be ready.
2. Require exactly one supplied tensor for every declared model input.
3. Reject empty, duplicate and unknown tensor names.
4. Match inputs by name rather than slice position.
5. Call `TensorBuffer::validate()` to verify byte length.
6. Require matching rank and precision.
7. Require every static dimension to match.
8. Require concrete, positive values for model dimensions marked dynamic.
9. Detect multiplication or byte-size overflow before allocating or binding memory.

The adapter must not silently cast precision, transpose data, add a batch dimension, resize an image or normalize values. Those operations belong to preprocessing and must be explicit before creating `TensorBuffer` values.

All model outputs must be returned. Each output `TensorBuffer` must contain the canonical binding name, concrete runtime shape, mapped precision and an owned byte buffer of the validated size.

## 6. Native API and Build Integration

The team should compare the OpenVINO C API, an established Rust binding and a narrow project-owned C ABI over the OpenVINO C++ API. The selected boundary should minimize unsafe surface area while still exposing:

- runtime creation and destruction;
- model reading and CPU compilation;
- input/output count and metadata queries;
- infer-request creation and destruction;
- named tensor binding;
- synchronous inference; and
- output metadata and data access.

If a C++ wrapper is used, it must:

- expose only opaque handles and C-compatible values;
- catch all C++ exceptions before they cross the ABI;
- return status codes and retrievable diagnostic messages;
- define ownership for every pointer and buffer;
- validate null pointers, indexes and lengths; and
- never unwind into Rust.

Every Rust `unsafe` block and native pointer assumption must have the safety documentation required by the repository quality standard.

Build discovery must prefer explicit `OPENVINO_INCLUDE_DIR` and `OPENVINO_LIB_DIR` values, followed by documented SDK environment variables or package metadata. When `cpu-openvino` is enabled, missing headers or libraries must cause an actionable build failure. The build must not silently compile a stub adapter.

`build.rs` must:

- emit `rerun-if-changed` for native wrapper sources;
- emit `rerun-if-env-changed` for supported OpenVINO environment variables;
- apply OpenVINO include, link and runtime-path settings only when `cpu-openvino` is enabled;
- avoid changing non-OpenVINO builds; and
- support documentation-only builds through an explicit, documented policy without making runtime tests appear successful.

The implementation pull request must pin the supported OpenVINO release range and Linux architectures based on CI-tested packages. Runtime library deployment requirements must be documented separately from compile-time discovery.

## 7. Errors and Observability

OpenVINO failures must map to the most specific existing `MiddlewareError` variant while retaining operation and runtime details. Errors should identify whether failure occurred during SDK discovery, Core creation, model reading, CPU compilation, metadata conversion, tensor binding or inference.

Logs and metrics should include:

- backend identity `cpu-openvino`;
- model name or path according to the repository's logging policy;
- OpenVINO runtime version;
- selected device (`CPU`);
- model load and compilation duration;
- inference duration; and
- input/output counts.

Logs must not dump tensor contents or expose model data unintentionally. OpenVINO errors must not be converted into panics.

## 8. Candidate Work Areas

These areas provide a possible decomposition of the OpenVINO investigation. Students should validate the dependencies between them and create the actual work packages, owners and estimates after selecting the native API and integration design.

### O1 — Confirm runtime and API baseline

1. Select and pin the supported OpenVINO release range.
2. Confirm packages for the required Linux architectures.
3. Choose the direct C API or narrow C++ wrapper approach.
4. Record redistribution and runtime deployment requirements.

**Exit criteria:** A minimal proof loads the repository's ONNX fixture and runs one inference through the selected native API on every required architecture.

### O2 — Add feature and build integration

1. Add `cpu-openvino` and mutual-exclusion checks.
2. Add deterministic SDK discovery and conditional native compilation/linking.
3. Export the OpenVINO adapter as `CpuAdapter` only for the selected feature combination.
4. Verify existing default and accelerator builds are unchanged.

**Exit criteria:** Supported feature combinations build with zero warnings, conflicting CPU implementation features fail clearly and missing OpenVINO dependencies produce actionable errors.

### O3 — Implement lifecycle and metadata

1. Implement state ownership and all trait lifecycle methods.
2. Load and compile ONNX models for `CPU`.
3. Discover all input/output metadata and reject unsupported types.
4. Make partial initialization and repeated release safe.

**Exit criteria:** Static, dynamic and multi-binding fixtures report correct `EngineInfo`; invalid models fail without leaks, stale state or panics.

### O4 — Implement inference

1. Add strict named-input validation.
2. Bind supported tensor types without implicit conversion.
3. Execute inference and copy all outputs into validated `TensorBuffer` values.
4. Add structured diagnostics and timing.

**Exit criteria:** Known fixtures produce expected outputs within documented tolerances and all invalid request classes return typed errors.

### O5 — Validate and document

1. Add unit, integration, feature-matrix and repeated-lifecycle tests.
2. Compare results and performance with the default ONNX Runtime backend.
3. Document installation, build, runtime-library setup, model constraints and known limitations.
4. Record exact runtime, CPU, OS, model checksum and thread settings in benchmark results.

**Exit criteria:** A clean supported system can build, test and run the OpenVINO backend using only documented steps.

## 9. Testing Strategy

### Unit tests without inference assumptions

- Backend identity and initial readiness
- Missing model and invalid-path handling
- Precision mapping, including rejected element types
- Tensor count, name, rank, shape, precision and byte-length validation
- Static and dynamic dimension matching
- Duplicate and unknown input rejection
- State cleanup after partial initialization
- Idempotent release

### OpenVINO integration tests

- Load a small redistributable ONNX fixture
- Verify all reported input and output metadata
- Run deterministic FP32 inference and compare against checked expected values
- Exercise FP16 and signed INT8 only with valid reference fixtures
- Exercise multiple inputs and outputs
- Exercise at least one dynamic dimension if supported by the chosen fixture
- Repeat load, infer and release cycles under leak and sanitizer tooling where available
- Verify errors for malformed models and unsupported element types

Tests requested with `cpu-openvino` must execute the real runtime. They must fail clearly if OpenVINO is unavailable rather than skip or simulate success. CI may omit the feature from generic jobs, but at least one declared CI job must install the pinned OpenVINO runtime and run the complete OpenVINO test suite.

### Regression and feature-matrix tests

The following configurations must remain valid:

- default features;
- `--no-default-features --features cpu`;
- `--no-default-features --features cpu-system-ort` in its supported environment;
- `--no-default-features --features cpu-openvino` in the OpenVINO environment; and
- existing accelerator feature combinations.

The combination `cpu-system-ort,cpu-openvino` must fail at compile time with a targeted explanation.

## 10. Performance Validation

Benchmarking must separate:

1. model read and compilation time;
2. first-inference warm-up time;
3. steady-state inference latency;
4. throughput;
5. middleware tensor-copy overhead; and
6. peak and steady-state memory use where measurable.

Results must record CPU model, core count, OS, OpenVINO version, model checksum, input shapes, precision, thread configuration and power mode. OpenVINO and ONNX Runtime comparisons must use identical model files, tensor bytes and correctness tolerances.

Performance targets should be established only after a baseline is measured. Optimization must not weaken tensor validation, error handling or backend isolation.

## 11. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| SDK discovery differs across Linux distributions | Pin supported packages, prefer explicit paths and test installation in CI |
| Runtime libraries are found at build time but absent during deployment | Document runtime packaging and add a startup diagnostic |
| OpenVINO types cannot be represented by `Precision` | Support only explicit lossless mappings and reject unsupported models |
| Dynamic shapes bypass validation | Preserve rank, mark only dynamic dimensions as `0` and validate every concrete request |
| C/C++ exceptions or invalid pointers cross the FFI boundary | Use opaque handles, status results, exception guards and documented ownership |
| Adapter locking serializes workloads | Start with correctness; profile before introducing a bounded request pool |
| OpenVINO and ONNX Runtime differ numerically | Use fixed fixtures and tolerances and record both runtime versions |
| Existing CPU or accelerator builds regress | Run the complete feature matrix and keep OpenVINO compilation conditional |

## 12. Definition of Done

- `cpu-openvino` selects a complete OpenVINO implementation through the existing `CpuAdapter` boundary.
- Existing public Rust and gRPC interfaces remain unchanged.
- Existing ONNX Runtime CPU implementations remain available and pass regression tests.
- OpenVINO is compiled and linked only when explicitly requested.
- Conflicting CPU implementation features and missing dependencies fail clearly.
- ONNX models are compiled explicitly for the OpenVINO `CPU` device.
- All model inputs and outputs are discovered without ImageNet-specific assumptions.
- Input count, names, ranks, dimensions, precision and byte lengths are validated before inference.
- Only explicitly supported, losslessly mapped element types are accepted.
- All outputs are returned with actual names, concrete shapes, precision and validated byte buffers.
- Partial initialization, repeated load/release and inference errors do not panic or leak resources.
- Real OpenVINO integration tests pass in a declared CI environment with zero warnings.
- Numerical comparison against a trusted reference is within documented tolerances.
- Installation, build, runtime deployment, usage and known limitations are documented.

