# Phase 1 — Middleware Refactoring

**Status:** Proposed  
**Audience:** Development Team  
**Source:** [Middleware assessment](000_middleware-assessment.md)
**Primary outcome:** Make the middleware contracts clear, internally consistent and easier to extend before adding more pipeline features.

## Student Ownership and Decision Authority

This document is a design brief, not a prescribed patch series. The Student Development Team owns the resulting architecture, issue breakdown and implementation sequence under the working agreement in [the planning roadmap](README.md). The types, method names, file locations and migration steps below are proposals to evaluate against the current code.

Before implementation, the team should verify the assessment findings, compare reasonable contract designs and agree on a solution together. An alternative is acceptable when it preserves explicit backend behavior, validation before FFI, transactional lifecycle semantics and discoverable model metadata. Consequential departures should be recorded with their rationale and validation plan.

At kickoff, classify R1–R6 as committed, stretch or deferred using actual capacity. The areas are intentionally unassigned: contributors should propose the concrete issues and volunteer as owners under the roadmap's open-ownership process. Phase 1 does not require all six areas to be attempted at once; however, any interface needed by a later committed phase must be completed or explicitly gated before that dependent work begins.

## 1. Purpose

The middleware has good foundational abstractions, but its public API is more generic than some backend implementations. This phase aligns the architecture with actual behavior. It is a refactoring phase, that keeps existing inference functionality working and doesn't introduce any production deployment features.

This phase addresses:

- unclear ownership between runtime, server, client and protocol code;
- inconsistent input/output behavior across backends;
- unused or misleading configuration and buffer abstractions;
- non-transactional lifecycle initialization;
- clients hard-coding model input details;
- documentation that does not accurately describe backend capabilities.

## 2. Goals and Non-Goals

### Goals

1. Define one explicit tensor and backend contract.
2. Reject unsupported backend behavior instead of silently ignoring tensors.
3. Make middleware initialization atomic from the caller's perspective.
4. Expose model metadata needed by clients.
5. Remove dead or misleading APIs.
6. Keep the CPU backend and existing CLI workflows functional.
7. Leave the code in a state that can support later vision-pipeline, profiling and CPU-backend phases.

### Non-Goals

- Authentication, TLS, service deployment, or other production controls
- Performance optimization before profiling data exists
- Implementing every vendor backend's multi-binding support in this phase
- Changing model formats or vendor SDKs
- Adding OpenVINO; that belongs to the CPU-backend extension brief

## 3. Target Architecture

The target dependency direction is:

```text
shared tensor/protocol types
		  ↑
backend traits ← backend adapters
		  ↑
engine manager
		  ↑
middleware public API
		  ↑
CLI / gRPC server / clients
```

The public API must not imply behavior that a selected backend cannot provide. Backend limitations must be represented as data and enforced before FFI calls.

## 4. Design Decisions

### 4.1 Backend capability contract

Add a serializable `BackendCapabilities` type near `InferenceBackend` in [traits.rs](../../middleware/src/inference/traits.rs). It should report at least:

- backend name;
- runtime availability (`Available`, `Unavailable`, or `Stub`);
- maximum supported input and output binding counts, or explicit multi-binding flags;
- supported input and output precisions;
- whether engine building is supported;
- whether dynamic shapes are supported.

Add the following trait method:

```rust
fn capabilities(&self) -> BackendCapabilities;
```

Capabilities describe implemented behavior, not vendor-platform potential. For example, NVIDIA must report one input and one output until its adapter enumerates and transfers all bindings.

### 4.2 Unsupported bindings fail explicitly

`EngineManager::infer()` must compare the request with `EngineInfo` and `BackendCapabilities` before delegation. A backend that supports one input must reject two inputs with a clear error. It must never ignore additional tensors.

Output metadata returned by a backend must match the loaded `EngineInfo`. If a backend cannot accurately introspect an engine, loading should return an explicit unsupported-metadata error rather than inventing generic shapes.

### 4.3 Fixed and dynamic dimensions

The current `Vec<usize>` shape representation uses `0` for unknown ONNX dimensions. This is ambiguous. Introduce a tensor-dimension representation such as:

```rust
pub enum Dimension {
	Fixed(usize),
	Dynamic,
}
```

Use it in `TensorSpec`, while `TensorBuffer` continues to carry concrete positive dimensions. Matching rules are:

- `Fixed(n)` matches only `n`;
- `Dynamic` matches any positive runtime dimension;
- empty shapes are invalid for inference tensors unless scalar tensors are explicitly supported and tested.

If this change is too large for one pull request, first add helper methods that treat `0` as dynamic, then migrate the serialized type in a separate PR.

### 4.4 Configuration ownership

`MiddlewareConfig` should contain only runtime-library settings that `Middleware::initialize()` actually consumes.

Proposed ownership:

| Setting | Owner | Decision |
|---|---|---|
| `fallback_precision` | Runtime/preprocessing | Keep until metadata-based preprocessing is complete |
| `labels_path` | Runtime postprocessing | Keep |
| `debug` | Binary logging setup | Remove from runtime config |
| `model_path` | CLI/server startup | Remove from runtime config |
| `grpc_address` | gRPC server | Remove from runtime config |
| `warmup_runs` | Profiling/benchmark layer | Remove now; reintroduce during profiling if measured |

The repository is pre-1.0, so a clean API change is preferred over retaining unused compatibility fields.

### 4.5 Buffer manager decision

[buffer_manager.rs](../../middleware/src/inference/buffer_manager.rs) is unused and supports only one input and output. The team should decide whether to remove it, replace it or demonstrate a current owner and use case. Vendor adapters already own device buffers, while host-buffer reuse should be justified by profiling evidence. Preserve useful design intent in the profiling plan rather than maintaining unexplained dead code.

### 4.6 Model metadata discovery

Add a `GetModelInfo` RPC to [magna.proto](../../middleware/proto/magna.proto). The response should include:

- loaded model name;
- backend name and availability;
- every input and output name;
- concrete or dynamic dimensions;
- precision;
- backend binding limits;
- optional preprocessing profile identifier.

Do not expose internal Rust serialization directly through gRPC. Define stable protobuf messages and map them explicitly.

The Rust client must call `GetModelInfo` after connecting and use the returned input name, shape and precision instead of hard-coding `"input"`. If preprocessing metadata is unavailable, the client must require explicit CLI arguments rather than guess silently.

## 5. Candidate Work Areas

The areas below describe problems, likely touchpoints and expected evidence. They are not pre-approved implementation packages. After investigation and team discussion, students should turn the agreed solution into their own work packages, estimates, ownership and sequencing.

### R1 — Introduce backend capabilities

**Files:**

- [traits.rs](../../middleware/src/inference/traits.rs)
- [engine_manager.rs](../../middleware/src/inference/engine_manager.rs)
- all adapters under [backends](../../middleware/src/backends)
- [public_api.rs](../../middleware/src/api/public_api.rs)

**Tasks:**

1. Define `BackendAvailability` and `BackendCapabilities`.
2. Implement `capabilities()` for CPU, NVIDIA, Qualcomm and TI.
3. Expose capabilities through `EngineManager` and `Middleware`.
4. Add capability unit tests for every feature configuration.
5. Document compile-time backend priority without describing CPU as a runtime fallback when another backend is selected.

**Acceptance criteria:**

- Every compiled backend reports capabilities.
- Capability values match implemented behavior.
- Unsupported input/output counts fail before adapter execution.
- No adapter silently drops additional inputs.

### R2 — Centralize tensor/spec matching

**Files:**

- [traits.rs](../../middleware/src/inference/traits.rs)
- new `middleware/src/inference/validation.rs`
- [engine_manager.rs](../../middleware/src/inference/engine_manager.rs)

**Tasks:**

1. Add checked element-count and byte-size methods.
2. Add `TensorBuffer::validate_against(&TensorSpec)`.
3. Add request-level validation for tensor count, unique names, shape, precision and bytes.
4. Define dynamic-dimension matching.
5. Call validation once in `EngineManager` so all backends receive validated tensors.
6. Keep adapter-specific checks as defense in depth.

**Acceptance criteria:**

- No inference path uses unchecked shape multiplication.
- Invalid buffers fail before entering FFI.
- CPU multi-input models remain supported.
- Hardware adapters reject unsupported binding counts explicitly.

### R3 — Make initialization transactional

**Files:**

- [public_api.rs](../../middleware/src/api/public_api.rs)
- [state_manager.rs](../../middleware/src/lifecycle/state_manager.rs)

**Tasks:**

1. Validate the current state without changing it.
2. Construct the engine manager and load labels into temporary values.
3. Commit fields and transition to `Initialized` only after all preparation succeeds.
4. Make `Error` fail every normal `require_at_least()` check.
5. Define retry behavior after initialization and engine-load failures.
6. Ensure reload failure does not expose partially replaced engine metadata.

**Acceptance criteria:**

- Failed initialization leaves state and resources unchanged.
- A caller can correct configuration and retry.
- `Error` never satisfies `Initialized`, `EngineLoaded`, or `Ready` requirements.
- Tests cover label failure, load failure, retry, reload and shutdown.

### R4 — Remove or relocate misleading APIs

**Files:**

- [errors.rs](../../middleware/src/utils/errors.rs)
- [buffer_manager.rs](../../middleware/src/inference/buffer_manager.rs)
- all `MiddlewareConfig` construction sites

**Tasks:**

1. Remove unused configuration fields according to Section 4.4.
2. Remove `BufferManager` and its tests.
3. Search for and update all config literals in binaries and tests.
4. Update examples and rustdoc.

**Acceptance criteria:**

- Every remaining configuration field has a verified reader.
- No dead `BufferManager` module remains.
- Workspace formatting, clippy, tests and rustdoc pass.

### R5 — Add model metadata RPC and update clients

**Files:**

- [magna.proto](../../middleware/proto/magna.proto)
- [grpc.rs](../../middleware/src/api/grpc.rs)
- [magna_server.rs](../../middleware/src/bin/magna_server.rs)
- [client/src/main.rs](../../client/src/main.rs)
- `misal/src/infer.rs`
- generated Python stubs and relevant Python clients

**Tasks:**

1. Define model, tensor-spec, dimension and capability protobuf messages.
2. Implement `GetModelInfo` from the currently loaded `EngineInfo`.
3. Make the Rust client discover its input binding.
4. Add explicit client overrides for shape/layout/preprocessing when metadata is incomplete.
5. Regenerate Python stubs from the same proto.
6. Add a CI check that regenerated stubs produce no diff.

**Acceptance criteria:**

- A client can discover the loaded model's bindings before inference.
- The Rust client no longer hard-codes the input name.
- Rust and Python generated protocol files are synchronized.
- An end-to-end gRPC test verifies metadata mapping.

### R6 — Align documentation with implemented capability

**Files:**

- [README.md](../../README.md)
- module rustdoc under `middleware/src`
- new `docs/backend-capabilities.md`

**Tasks:**

1. Publish a capability matrix for each backend.
2. Label backends as verified, experimental, simulated, or stub.
3. Correct server binary and argument examples.
4. Document binding-count, dynamic-shape, precision and preprocessing limits.
5. Link the capability matrix from the main README.

**Acceptance criteria:**

- Documentation contains no claim contradicted by capability tests.
- All documented commands pass a `--help` or dry-run smoke check.

## 6. Implementation Sequence

One reasonable implementation order is:

1. **R1:** Capability types and implementations
2. **R2a:** Checked tensor arithmetic and validation helpers
3. **R2b:** Engine-manager enforcement and adapter cleanup
4. **R3:** Transactional lifecycle behavior
5. **R4:** Configuration and buffer-manager cleanup
6. **R5a:** Protobuf metadata contract and server implementation
7. **R5b:** Rust/Python client migration and generated-stub check
8. **R6:** Documentation alignment

## 7. Required Tests

- Unit tests for checked shape and byte arithmetic
- Table-driven tensor/spec matching tests
- Capability tests for each backend feature
- Lifecycle rollback and retry tests
- CPU multi-input regression test using a suitable fixture
- Hardware-backend rejection tests for excess bindings
- gRPC `GetModelInfo` integration test
- Client test proving discovered binding names are used
- Documentation command smoke checks where practical

Asset-dependent tests must use `#[ignore]` or fail when a CI job declares assets required. Printing `SKIPPED` and returning from an ordinary test is not acceptable for required CI coverage.

## 8. Phase Definition of Done

- Outcomes for all committed candidate work areas are met, or deferrals are documented and agreed.
- `cargo fmt --all --check` passes.
- `cargo clippy --workspace --all-targets -- -D warnings` passes.
- `cargo test --workspace` passes.
- Backend feature lint/build jobs pass.
- Protocol stubs are reproducibly generated.
- No existing CPU inference workflow regresses.
- The assessment's architecture, lifecycle, binding, configuration and documentation findings are closed or explicitly deferred with rationale.

## 9. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Public type changes create a large diff | Split type introduction and call-site migration into separate PRs |
| Dynamic-shape semantics become too broad | Support only matching/validation first; defer dynamic allocation |
| Capability values drift from implementation | Test capabilities alongside adapter contract tests |
| Proto changes break Python users | Regenerate and test all stubs in the same PR |
| Removing `BufferManager` loses future work | Record allocation measurements during profiling before redesigning it |

## 10. Implementation Notes

Before proposing work packages for a candidate area, read the linked source files and existing tests. Prefer writing the expected behavior as a test first. Avoid vendor-SDK assumptions that are not verified by documentation or hardware and keep mock behavior behind explicit test/mock configuration.

Any behavior change should update the relevant rustdoc, capability documentation and tests.
