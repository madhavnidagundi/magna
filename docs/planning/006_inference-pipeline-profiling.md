# Phase 6 — Automated Inference Middleware Profiling Across Backends

**Status:** Proposed  
**Audience:** Development Team  
**Prerequisites:** The Magna middleware can load the model and complete inference on the backend under test; each hardware backend passes its smoke test; a backend-specific runtime container or supported host environment is available  
**Primary outcome:** Build an automated profiling system that isolates the Magna inference middleware, verifies output correctness, measures stable middleware and backend boundaries, captures optional backend-native profiles and publishes reproducible machine-readable results.

## Student Ownership and Decision Authority

This brief defines the questions that profiling must answer and the evidence needed for credible results. It does not prescribe a specific load generator, telemetry stack, result schema, statistics library or vendor-profiler integration. Following [the planning roadmap](README.md), the Student Development Team should propose the measurement model, review sources of bias together and implement the agreed approach incrementally.

The team may simplify instrumentation and backend coverage for a milestone when the reduced scope is declared before measurement. It should preserve correctness gating, backend identity, separated run phases, explicit metric boundaries and reproducible provenance. Unsupported hardware or counters should be reported honestly rather than simulated or encoded as successful zero values.

## Staged entry and coverage gate

Treat backend coverage as independent lanes, not as a requirement that every board be ready before useful work starts:

1. **portable baseline:** versioned manifest/result schemas, deterministic fixture, correctness gate and one local CPU gRPC run;
2. **measurement baseline:** correlated middleware timing, controlled latency/throughput runs and repeatability evidence on CPU;
3. **hardware lanes:** add one real, smoke-tested backend at a time with its own runner owner, identity checks and native tooling; and
4. **trend publication:** enable comparisons or regression thresholds only after variance is understood.

P1–P6 remain open candidate areas. The team should derive smaller issues and volunteer owners after agreeing on the measurement contract. Hardware lanes may proceed in parallel, but shared schema and correctness changes require cross-lane review. If Phase 3 OpenVINO is complete and smoke-tested when a profiling milestone is scoped, classify it explicitly as required, optional or excluded rather than leaving the matrix stale.

## 1. Purpose

This work profiles the inference middleware itself. It must answer two related but distinct questions:

1. **Middleware profiling:** What latency, throughput, concurrency and resource costs are introduced from receipt of an `Infer` gRPC request through response completion?
2. **Backend profiling:** Where does the selected inference runtime spend time and resources while loading a model, preparing buffers, transferring tensors and executing inference?

The profiling boundary begins at the middleware's gRPC interface and ends when the middleware response is returned. ROS 2 image publication, camera capture, vision-node scheduling, DDS transport and ROS 2 result publication are deliberately excluded. They may be measured by a separate vision-pipeline profiling effort.

The system must not confuse middleware and backend measurements. A backend-native kernel profile does not include protobuf or middleware overhead, while an RPC latency does not identify slow kernels or device transfers.

```text
Profile controller / CI workflow
          |
          +-- resolve immutable test manifest
          +-- select eligible hardware runner
          +-- start backend-specific middleware build/container
          +-- verify backend, model and readiness
          +-- run deterministic tensor correctness gate
          +-- measure cold start and warm-up
          +-- run steady-state RPC workload
          +-- optionally run direct-API control workload
          +-- optionally run backend-native profiler
          +-- collect, validate and publish artifacts
```

The primary benchmark exercises the public gRPC `Infer` contract in `middleware/proto/magna.proto`. A direct Rust API benchmark may be used as a control to isolate transport overhead, but it must be reported separately and must never be substituted for the primary RPC measurement.

## 2. Scope

### Goals

1. Define one versioned profile manifest for the middleware build, backend, model, tensor fixture, runtime settings and measurement policy.
2. Send deterministic input tensors through the existing gRPC `Infer` API without requiring ROS 2, a camera or image-file preprocessing.
3. Measure stable middleware boundaries consistently across all implemented backends.
4. Separate client-observed RPC latency, server request-handling latency and backend invocation latency.
5. Measure model load, buffer allocation, warm-up, steady-state inference and shutdown behavior.
6. Collect middleware process, container and accelerator resource metrics.
7. Invoke backend-native profiling tools in separate diagnostic runs where available.
8. Verify tensor contracts and output correctness before accepting performance data.
9. Store raw and summarized results with enough provenance to reproduce each run.
10. Execute automatically on labeled hardware runners without treating unavailable hardware or simulated adapters as a pass.

### Non-Goals

- Profiling ROS 2 nodes, DDS, image topics, camera capture or result publication
- Measuring image decode, resize, crop or normalization performed outside the gRPC tensor contract
- Ranking unlike hardware solely by one latency number
- Treating performance profiling as full model-accuracy validation
- Running proprietary accelerator stacks on unsupported emulators
- Installing hardware drivers inside ordinary middleware containers
- Making privileged containers the default
- Enabling expensive native profilers during normal middleware execution
- Replacing focused backend microbenchmarks or vendor tools
- Publishing simulated, fallback, thermally unstable or differently configured runs as comparable results

If `Middleware::infer_from_image` is profiled, it is a separate convenience-API experiment. Its image preprocessing and classification postprocessing timings must not be merged into the gRPC tensor benchmark.

## 3. Middleware Boundary and Backend Matrix

### 3.1 Profiled request path

The common RPC path is:

1. client serializes an `InferenceRequest` containing one or more `InferInputTensor` values;
2. the server receives and decodes the request;
3. the gRPC handler maps protobuf tensors to `TensorBuffer` values;
4. `Middleware::infer_generic` enters the public API;
5. lifecycle and readiness checks run;
6. `EngineManager::infer` invokes the compile-time-selected backend;
7. the backend validates bindings, transfers inputs as required, executes inference and returns outputs;
8. optional generic result construction or classification runs;
9. the server maps output buffers to protobuf tensors and encodes the response; and
10. the client receives and decodes the response.

Instrumentation may refine a stage internally, but all backends must preserve these common outer boundaries. Backend-specific stages such as host-to-device transfer or graph execution belong in backend-native evidence unless they can be measured consistently and without changing semantics.

### 3.2 Backend matrix

The initial matrix reflects middleware implementations and build features currently present in the repository:

| Backend variant | Runtime | Required runner | Publication rule |
|---|---|---|---|
| Portable CPU (`cpu`) | ONNX Runtime through the Rust adapter | Supported `x86_64` or `aarch64` Linux host | Eligible when a real model is loaded and executed |
| System CPU (`cpu-system-ort`) | System ONNX Runtime C API | Board with the approved system runtime | Report separately from portable CPU |
| NVIDIA (`nvidia` with `orin` or `thor`) | TensorRT/CUDA | Matching Orin or Thor runner | Eligible only with the expected physical device and runtime |
| Qualcomm (`qualcomm`) | QNN/QAIRT | Matching Qualcomm target | Simulation or placeholder execution is invalid |
| TI (`ti`) | TIDL/DLR integration | Matching TI target | Stub or simulated execution is invalid |

The `rockchip` Cargo feature does not currently have a selectable backend adapter and is **excluded** until an implementation and smoke test exist. OpenVINO is also excluded until it is represented by an implemented middleware backend rather than a prospective runtime. Re-evaluate this row at each profiling milestone kickoff; once Phase 3 passes a real-runtime smoke test, add `cpu-openvino` as its own CPU variant or record why it remains deferred.

Backend selection is compile-time and follows the current priority `nvidia > qualcomm > ti > cpu` when multiple features are enabled. Profiling builds should enable exactly one backend variant wherever possible. Every run must verify the runtime-reported backend instead of inferring it only from the requested Cargo features.

The matrix must classify jobs as:

- **required**, where an unavailable runner or failed profile fails the workflow;
- **optional**, where unavailability is reported without invalidating unrelated jobs; or
- **excluded**, with a recorded reason such as an unimplemented adapter.

No job may silently substitute CPU, a stub or simulation for the requested backend.

## 4. Profile Manifest

Every run must be driven by a checked-in or archived manifest. At minimum, the resolved manifest must identify:

- run ID and schema version;
- source revision and whether the worktree was clean;
- middleware artifact or container image digest;
- enabled Cargo features and build profile;
- backend name, hardware variant and expected runtime provider;
- model name, immutable checksum, format and precision;
- backend-specific model artifact and checksum;
- model profile and checksum, including input and output tensor contracts;
- fixture set and checksum;
- request mode (`grpc` or the separately reported `direct-api-control`);
- client and server placement;
- client concurrency, connection count and pacing policy;
- warm-up count and measured request count or duration;
- request timeout and maximum in-flight requests;
- backend thread, stream, execution-provider and power settings;
- resource limits and CPU affinity, when used;
- requested profiler mode; and
- output directory or artifact destination.

Defaults must be versioned, visible in the resolved manifest and identical across comparable runs. Environment variables may override settings for CI, but resolved values must be recorded. A run must fail before measurement if required files, checksums, devices, runtimes or settings are unavailable.

## 5. Measurement Model

### 5.1 Correlation and clocks

The team should evaluate whether each request needs a stable correlation identifier from the load generator through middleware events and back to the client. The server's current process-local request counter is useful for logs but is not sufficient for client/server correlation because it is not present in the protobuf request or response. If per-request stage reconciliation is selected, use a backward-compatible request ID field, tested gRPC metadata propagation or another justified mechanism.

Structured events or trace spans must correlate:

1. client serialization start and completion;
2. client RPC send;
3. server request receipt;
4. protobuf-to-`TensorBuffer` conversion completion;
5. middleware API entry and readiness-check completion;
6. backend invocation start and completion;
7. result construction completion;
8. response serialization and send; and
9. client response receipt and decode completion.

Use a monotonic clock for durations. Wall-clock timestamps may be retained for logs but must not be subtracted to calculate latency. Client-observed latency is measured entirely in the client's clock domain. Server and backend stage durations are measured in the server's clock domain. Cross-host timestamps may be displayed only when synchronization quality and uncertainty are recorded; they must not be used to derive stage durations by subtraction.

### 5.2 Common latency and throughput metrics

Every backend run must report:

- client serialization latency;
- client-observed RPC latency;
- server request decode and tensor-conversion latency;
- middleware API and lifecycle-check latency;
- backend invocation latency;
- result construction or middleware postprocessing latency;
- response construction and serialization latency;
- model load and buffer-allocation time;
- startup-to-process-live time;
- startup-to-backend-ready time;
- startup-to-first-successful-inference time;
- warm-up behavior;
- offered request rate and achieved inference throughput;
- successful request, failed request and timeout counts;
- active and maximum in-flight requests; and
- queue wait or contention time when a queue or lock boundary can be observed.

The existing `InferenceResponse.inference_time_ms` surrounds `Middleware::infer_generic`; it is not client-observed RPC latency and does not include request conversion or response construction. The existing metrics collector times `Middleware::infer` and reports count, minimum, maximum and average. Both may be retained for compatibility, but neither alone satisfies this measurement contract.

Latency summaries must include sample count, minimum, maximum, mean, median, standard deviation and at least p90, p95 and p99. Retain raw per-request records or a validated histogram with sufficient precision for audit and recomputation. Warm-up and failed requests must not be silently included in successful steady-state latency distributions.

Throughput must be calculated from measured wall duration and successful completions, not as only the reciprocal of average latency. Sequential latency and concurrent throughput are different experiments and must be labeled separately.

### 5.3 Resource metrics

Collect, where supported:

- host and middleware-container CPU usage;
- middleware process and per-thread CPU usage;
- resident, proportional and peak memory;
- container memory limit, throttling and OOM events;
- gRPC network bytes and packets;
- block I/O during model loading;
- file-cache state when cold-load comparisons require it;
- accelerator utilization and memory;
- CPU, GPU or NPU frequency and throttling state;
- system temperature and power mode; and
- profiler and collector overhead.

Unsupported counters must be represented as unavailable, not zero. Sampling interval, collector method and tool versions must be recorded. Model memory reported as zero by an adapter must be treated as unknown unless the adapter explicitly proves that zero is meaningful.

### 5.4 Run phases

Automation must separate these phases:

1. **Preflight:** verify the build features, backend identity, hardware, runtime, model and fixture checksums, power and thermal state, free storage and exclusive-runner status.
2. **Cold load:** start from a documented cache and process state; measure process startup, model loading, buffer allocation and first successful inference.
3. **Warm-up:** execute an explicit number of unreported requests or a versioned stability policy.
4. **Steady-state latency:** execute a controlled low-concurrency workload and collect per-request stage measurements.
5. **Steady-state throughput:** execute one or more declared concurrency levels and collect throughput, contention and saturation behavior.
6. **Direct API control:** optionally repeat the fixture through the Rust API to estimate gRPC and serialization overhead; publish separately.
7. **Native profile:** optionally repeat a bounded workload with the backend profiler enabled.
8. **Recovery:** restart or reload the middleware and verify readiness and resumed inference.
9. **Collection and cleanup:** stop cleanly, validate expected artifacts and remove transient state.

Cold-load, warm-up, steady-state latency, throughput, direct-API and native-profiler measurements must never be combined into one distribution.

## 6. Common Instrumentation

The profiling implementation should build on existing hooks while correcting their limitations:

- `Middleware::infer` already measures aggregate backend-path latency when the `metrics` feature is enabled.
- `Middleware::get_metrics` exposes count, last, average, minimum, maximum, reciprocal-average throughput and model memory.
- `InferenceResponse.inference_time_ms` currently measures the `infer_generic` call in the gRPC handler.
- optional OpenTelemetry support exports inference latency and request counters.
- `middleware/benches/inference_bench.rs` provides a Criterion direct-API benchmark.

Required additions are:

1. a request correlation ID propagated across the RPC boundary;
2. structured per-request stage records or spans with explicit units;
3. success, failure, timeout and cancellation status for each request;
4. concurrency and contention observations that remain correct under simultaneous requests;
5. raw or histogram latency retention sufficient for percentiles;
6. explicit metric availability rather than using zero as an unavailable sentinel; and
7. a documented instrumentation-overhead test with instrumentation enabled and disabled.

Instrumentation must be bounded. It must not retain tensor payloads, output values or unbounded per-request state in the server. Raw request measurements should be streamed or exported by the profiling harness rather than accumulated indefinitely in the middleware process.

## 7. Backend-Native Profiling

Common middleware instrumentation is mandatory. Backend-native tools are optional diagnostic adapters selected by the manifest and run separately from baseline measurements.

| Backend | Candidate tools | Expected evidence |
|---|---|---|
| CPU / ONNX Runtime | Linux `perf`, ONNX Runtime profiling where supported | CPU samples, call stacks, thread activity and runtime events |
| NVIDIA TensorRT | Nsight Systems and TensorRT layer profiling | CPU/GPU timeline, transfers, kernels, layer timing and utilization |
| Qualcomm QNN | Approved QNN/QAIRT profiling tools | graph/node timing, accelerator activity and transfers |
| TI TIDL | Approved TIDL profiling tools | layer/core timing, transfers and accelerator utilization |

Exact tools and versions must be confirmed against the installed SDK. Tool output must be retained in its native format plus a human-readable index or summary. The automation must not fabricate a common per-layer metric when runtimes define layers, fused operations or timing differently.

Profiler images or host collectors may require extra binaries, mounts, capabilities or host settings. These must be isolated in a dedicated profiling image or optional target and justified individually. Production middleware images must not inherit profiler privileges or unnecessary tools. If host policy prevents collection, a requested profile must fail explicitly or be marked unsupported with a reason; it must not produce an empty successful artifact.

## 8. Correctness and Identity Gate

Performance data is valid only after the run proves that the intended middleware and backend executed the intended workload. Before steady-state measurement, automation must verify:

- the middleware reports `Ready` through `HealthCheck`;
- the reported backend matches the requested backend and build features;
- the expected physical device and runtime libraries are active;
- no simulation, stub or unintended fallback path is active;
- the expected model and artifact checksums were loaded;
- input names, shapes, dimensions, precision and byte lengths match the model profile;
- output names, shapes, precision and byte lengths match the expected contract;
- fixture outputs contain no unexpected non-finite values;
- numerical outputs or decoded results are within versioned, task-appropriate tolerances; and
- the number of unexplained inference errors is zero.

Backend identity cannot rely only on a user-supplied label or Cargo feature. Adapters that can operate in simulation mode must expose machine-readable execution mode and device identity before their results can be published.

For classification fixtures, compare expected top results and bounded score tolerances. For other tensor outputs, define model-specific numeric or task-level comparisons. Quantized backends may use different approved tolerances, but these must be versioned in the model profile.

A failed gate invalidates the performance run. Results must be marked failed and must not enter trend charts or backend comparisons.

## 9. Container and Runner Architecture

### 9.1 Middleware image

Each backend profile should run one middleware server artifact or container built for exactly that backend variant. The runtime image must contain only the middleware, required backend runtime libraries and operational dependencies. Models and fixtures should be read-only mounts or immutable image inputs; results must use a separate writable location.

Profiling does not require ROS 2, a camera driver or a visualization process. The load generator should run as a separate process and, for the primary measurement, communicate through the same gRPC interface used in deployment. Whether it runs on the same host or a separate load-generator host must be declared in the manifest.

The deployment definition must identify:

- immutable middleware image or binary digest;
- enabled backend feature and hardware variant;
- devices and runtime hooks;
- read-only model and configuration mounts;
- writable result volume;
- gRPC endpoint and health check;
- resource limits and CPU affinity;
- profiler-specific permissions; and
- provenance labels.

Do not expose the gRPC port beyond the profiling network unless remote load generation requires it.

### 9.2 Hardware runners

Use labeled self-hosted runners for hardware-dependent jobs. Each runner must publish an inventory containing board model, architecture, OS, kernel, driver and runtime versions, available devices, power modes and profiler availability. The job must verify this inventory at runtime rather than trusting the scheduling label alone.

Only one exclusive performance job should use a target at a time. The runner must detect and report competing workloads, thermal throttling, incorrect power mode and insufficient cooling. These conditions should invalidate or retry the run according to a bounded documented policy.

The client load generator must not become the bottleneck. Record its host, CPU usage and placement, and validate that it can sustain a higher request rate than the middleware under test.

## 10. Automation Workflow

An automated workflow should cover the following responsibilities; the team may combine steps when boundaries and failure reporting remain clear:

1. Validate the profile manifest and determine eligible backend jobs.
2. Resolve source, middleware artifact, model and fixture digests.
3. Schedule each job on a compatible labeled runner.
4. Record runner inventory and preflight environmental state.
5. Pull or build the immutable backend-specific middleware artifact.
6. Start the middleware with the selected model and runtime configuration.
7. Wait for process liveness, middleware readiness and successful model loading.
8. Verify backend identity, execution mode and tensor contracts.
9. Execute the deterministic correctness fixture.
10. Run cold-load, warm-up, latency and throughput phases separately.
11. Optionally run the direct-API control benchmark.
12. Optionally run the backend-native profiler in a separate invocation.
13. Execute restart or reload recovery validation.
14. Collect logs, request records, traces, metrics, profiles and resolved configuration.
15. Validate artifact completeness and generate summaries.
16. Tear down the middleware even after failure or cancellation.
17. Publish results and update trends only for valid runs.

Timeout and cancellation handling must always perform bounded cleanup. Concurrent jobs must write to unique result locations and must not share mutable model caches unless cache state is explicitly part of the experiment.

## 11. Result Format and Reporting

Each run must produce a self-contained result bundle such as:

```text
results/<run-id>/<backend>/
    manifest.resolved.yaml
    environment.json
    backend-identity.json
    correctness.json
    summary.json
    requests.csv
    metrics/
    traces/
    native-profile/
    logs/
    checksums.txt
```

The exact schema is a Development Team deliverable. It must be versioned and include units on every metric. `summary.json` must distinguish measured zero from unavailable data and identify whether the run is valid, failed, unsupported or incomplete.

Every request record should include at least correlation ID, phase, client concurrency, start time relative to the run, status and available stage durations. It must not contain input or output tensor payloads.

The generated report must include:

- client RPC, middleware and backend latency distributions;
- overhead deltas between adjacent measured boundaries where valid;
- throughput by declared concurrency;
- failures, timeouts and contention indicators;
- model-load and first-inference behavior;
- middleware process and device resource utilization;
- links to raw and native artifacts;
- correctness and backend-identity status;
- run-to-run variability; and
- changes against an explicitly selected compatible baseline.

Cross-backend charts must prominently display hardware, precision, model artifact and runtime. Comparisons across different devices are descriptive system comparisons, not proof that one runtime is intrinsically faster. Comparisons intended to isolate runtime effects must use the same host, model semantics, precision where possible, workload, client placement and resource policy.

## 12. Candidate Work Areas

These areas identify the responsibilities of a credible profiling system. Students should decide which components to combine or separate, select appropriate tools and then define implementation work packages for the agreed measurement design and milestone scope.

### P1 — Define the middleware profiling contract

1. Define manifest and result schemas.
2. Define the exact gRPC, public-API and backend timing boundaries.
3. Define correlation propagation and monotonic-clock rules.
4. Define deterministic tensor fixtures and backend-specific tolerances.
5. Classify required, optional and excluded backend jobs.

**Acceptance criteria:** One fixture run can be represented completely, schemas reject missing provenance, and the team can explain which middleware and backend metrics are directly comparable.

### P2 — Add common middleware instrumentation

1. Propagate one correlation ID through client, gRPC handler, public API and backend events.
2. Instrument request conversion, lifecycle checks, backend invocation, result construction and response conversion.
3. Export bounded metrics and traces without logging tensor payloads.
4. Add middleware process and container resource collection.
5. Measure instrumentation overhead with collection enabled and disabled.

**Acceptance criteria:** A request can be correlated across the middleware boundary, measured stages reconcile with client-observed latency within documented unmeasured transport overhead, and instrumentation does not create unbounded state.

### P3 — Build the middleware profile runner

1. Add manifest validation, server startup, readiness, execution, collection and cleanup commands.
2. Implement deterministic gRPC tensor fixtures and configurable concurrency.
3. Add backend-specific runtime profiles and hardware preflight checks.
4. Make a local portable-CPU profile the first supported path.
5. Retain the Criterion benchmark as a separately labeled direct-API control.

**Acceptance criteria:** One command or CI job produces a complete, valid CPU result bundle from a clean supported host without ROS 2 or camera dependencies.

### P4 — Add backend-native adapters

1. Integrate each approved profiler in a separate bounded run.
2. Keep native outputs and generate non-lossy references from the common report.
3. Minimize and document profiler permissions.
4. Fail explicitly when a requested profiler cannot collect data.

**Acceptance criteria:** Every supported backend produces either a validated native profile or an explicit unsupported result with a reason; no empty profile is reported as success.

### P5 — Automate the hardware matrix

1. Add labeled runner inventory and exclusive scheduling.
2. Execute required and optional backend jobs according to policy.
3. Verify actual device, runtime and non-simulated execution.
4. Publish immutable artifacts and machine-readable summaries.
5. Add trend reporting and bounded regression rules after baseline stability is demonstrated.

**Acceptance criteria:** The workflow identifies the actual backend device and runtime, isolates runner failures from middleware failures and never substitutes one backend for another silently.

### P6 — Validate reproducibility

1. Repeat runs across time, clean process starts and controlled cache states.
2. Quantify run-to-run variance, instrumentation overhead and native-profiler overhead.
3. Document thermal, power, affinity, concurrency and cache controls.
4. Produce an operations and troubleshooting guide.

**Acceptance criteria:** Repeated valid runs remain within documented variance, and another team member can reproduce a middleware result from its bundle and runbook.

## 13. Required Deliverables

The Development Team must produce:

1. A middleware profiling architecture and measurement specification.
2. Versioned manifest and result schemas with examples.
3. Deterministic tensor fixtures, expected outputs and tolerance policies.
4. Correlated common middleware instrumentation and tests.
5. A gRPC load generator with controlled concurrency and raw result export.
6. Backend-specific middleware runtime or container profiles.
7. An automated local CPU runner and hardware CI matrix.
8. Backend-native profiler adapters and permission documentation.
9. A report generator with raw-artifact links and provenance.
10. Baseline result bundles for every available real backend.
11. A runbook covering execution, failure diagnosis, cleanup and reproduction.

## 14. Acceptance Criteria

- A single versioned manifest selects one middleware backend and reproduces the resolved tensor workload.
- The primary profile uses the existing gRPC inference boundary without ROS 2 or camera dependencies.
- Every result identifies source revision, artifact digest, build features, model checksums, model profile, fixture, backend, hardware, runtime, power mode, client placement and measurement settings.
- Backend and device identity are verified at runtime; fallback, stub and simulation modes invalidate the run.
- Input and output tensor contracts and numerical correctness pass before performance data is accepted.
- Client-observed RPC, server middleware and backend invocation latency are named and reported separately.
- Per-request data supports sample count, minimum, maximum, mean, median, standard deviation, p90, p95 and p99.
- Throughput is measured from successful completions over elapsed time and is reported by concurrency level.
- Cold load, warm-up, steady-state latency, throughput, direct API and native-profiler results are separate.
- Middleware process, container and device resource metrics are collected where supported; unavailable values are not encoded as zero.
- Requested profiling cannot pass by skipping a missing model, fixture, runtime, profiler or device.
- Required hardware jobs fail when their runner is unavailable; optional jobs are visibly marked unavailable.
- Profiling permissions are narrower than blanket privileged mode and are absent from production runtime images.
- Raw artifacts and versioned machine-readable summaries are retained.
- Comparisons disclose hardware, runtime and precision and do not present unlike systems as controlled runtime comparisons.
- Repeated runs have quantified variance and an independently reproducible runbook.

## 15. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Client or network behavior hides middleware cost | Report client-observed, server and backend boundaries separately; include a direct-API control |
| Existing timers are interpreted as full RPC latency | Give every timer a precise boundary name and document what it excludes |
| Profiler overhead changes observed performance | Separate baseline and native-profile runs and measure instrumentation overhead |
| Wrong backend, stub or simulation produces plausible output | Verify backend, device and execution mode and reject fallback before measurement |
| Shared locks serialize concurrent requests unexpectedly | Measure contention and throughput across declared concurrency levels |
| The load generator becomes the bottleneck | Monitor client resources and validate spare load-generation capacity |
| Thermal or power variation dominates results | Record sensors and power mode, use exclusive runners and invalidate throttled runs |
| Cross-host clocks distort stage timing | Measure durations within one monotonic clock domain and avoid cross-host timestamp subtraction |
| Missing hardware is reported as success | Use explicit required/optional policy and fail requested jobs that cannot execute |
| Native tools require excessive permissions | Use dedicated profile images and grant only documented devices and capabilities |
| Result schemas drift across backends | Version and validate common schemas; retain native output separately |
| Correctness regresses while latency improves | Gate publication on tensor-contract and model-appropriate correctness checks |
| Stored profiles expose model or customer data | Use approved fixtures, redact paths and logs, and never capture tensor payloads by default |

## 16. Guidance

Begin with the portable CPU backend and deterministic tensors sent through the gRPC `Infer` API. Establish trustworthy request correlation, correctness, latency boundaries and result schemas before adding vendor tools. Use the existing Criterion benchmark only as a direct-API control; it does not replace the gRPC middleware profile.

Add hardware backends one at a time while preserving the same model semantics and tensor workload. Optimize this system for reproducibility, diagnosis and middleware regression detection rather than for producing the largest possible collection of metrics.