# Phase 4 — ROS 2 Vision-Pipeline Integration

**Status:** Proposed  
**Audience:** Development Team  
**Prerequisites:** Phase 1 middleware refactoring; Phase 2 middleware hardening   
**Primary outcome:** Define and integrate camera, image-loader, inference and visualization nodes as one ROS 2 vision pipeline. Implement a `misal_inference` node that connects image sources to MISAL inference middleware through a documented, testable gRPC interface and expose observable results to Lichtblick through `foxglove_bridge`, without coupling ROS 2 code to vendor-specific inference APIs.

## Student Ownership and Decision Authority

This document describes the desired system behavior and candidate boundaries. The Student Development Team owns the ROS 2 package structure, implementation language, node composition, QoS choices, message selection and detailed workflow under [the planning roadmap](README.md).

The team should begin by examining the existing vision workspace and maintained ROS 2 packages, then propose a coherent pipeline for discussion. The diagram and responsibility table below are a baseline to challenge, not an instruction to create unnecessary components. Alternatives are welcome when they preserve source independence, explicit preprocessing and decoding, bounded behavior, middleware isolation and observable end-to-end results.

## Prerequisite and first-slice gate

Contract, fixture and ROS 2 environment investigation may overlap with Phase 2. Integration against the real server requires stable model metadata and hardened gRPC tensor validation. Before implementation, select exactly one initial task/model, model checksum, deterministic image fixture, output message, ROS 2 distribution and target platform. Record who provides camera hardware and ROS 2 infrastructure.

V1–V7 are open candidate areas, not seven preassigned jobs. The first committed slice should normally be the file-based path through `image_loader`, `misal_inference`, a test double or portable CPU server and one standard result message. Live camera, visualization and comparative accelerator measurements can become later committed slices after that path is reproducible.

## 1. Purpose

This phase defines the vision-inference portion of the system as one unified pipeline. Camera nodes should own live image acquisition, an image-loader node should provide deterministic file-based image input, `misal_inference` should own image adaptation and inference transport and visualization nodes should expose results and diagnostics to Lichtblick. MISAL should own model loading, backend selection, tensor validation, inference execution and inference metadata.

The existing MISAL gRPC service is the baseline process boundary to evaluate:

```text
ROS 2 camera nodes          image_loader node
		  |                         |
		  +---- sensor_msgs/Image --+
		  |    + CameraInfo         |
		v
		misal_inference ROS 2 node
		|
	image preprocessing and
	MISAL gRPC request handling
		|
	MISAL gRPC client (inside `misal_inference`)
		|
		v
	MISAL middleware server
		|
	selected inference backend
		|
		v
     inference outputs / metadata
		|
		v
 ROS 2 results, diagnostics, debug images
		|
		v
 foxglove_bridge -> Lichtblick
```

The gRPC client is an internal component of `misal_inference`, not a second ROS 2 node. The node is the single integration point between ROS 2 topics and the MISAL service.

Camera and image-loader nodes are alternative producers of the same documented image contract. They should not require separate inference implementations. `foxglove_bridge` is a transport bridge to Lichtblick, not the owner of inference semantics or model-specific decoding.

This phase extends the existing vision and inference implementation without replacing its core components. The first milestone should prove the end-to-end path with one file-based image source and one supported model, followed by one live camera, before adding multiple cameras or advanced synchronization.

## 2. Goals and Non-Goals

### Goals

1. Define the interfaces between ROS 2, `misal_inference` and MISAL.
2. Convert ROS 2 image messages into validated MISAL inference requests.
3. Convert MISAL outputs into standard ROS 2 messages that work with the ROS 2 ecosystem and visualization tooling.
4. Preserve timestamps, frame IDs, sequence information and model metadata.
5. Make configuration explicit instead of hard-coding model names, input bindings, dimensions or transport addresses.
6. Provide bounded queues, timeout behavior, reconnect behavior and observable diagnostics.
7. Demonstrate the pipeline with a recorded bag and then with a live camera on a supported target.
8. Keep the integration independent of TensorRT, QNN, TIDL and other vendor APIs.
9. Define reusable camera-node and image-loader contracts so live and file-based images exercise the same inference path.
10. Provide a supported Lichtblick workflow through `foxglove_bridge` for source images, inference results, diagnostics and optional annotated images.

### Non-Goals

- Replacing ROS 2 with gRPC or replacing the MISAL middleware API with ROS 2 APIs
- Implementing a new inference backend
- Supporting every ROS 2 image encoding or inference task representation in the first milestone
- Defining project-specific output messages when an existing standard ROS 2 message accurately represents the result
- Developing a new browser visualization application or forking Lichtblick
- Reimplementing `foxglove_bridge`; the supported upstream ROS 2 package should be configured and integrated
- Building custom camera drivers when a maintained ROS 2 driver already satisfies the documented image contract
- Building a production control or safety-critical system
- Adding authentication, TLS, public-network exposure or deployment orchestration
- Claiming real-time or safety-critical guarantees before measurements and a separate safety review

## 3. Responsibilities and Boundary

| Concern | Camera / image-loader nodes | `misal_inference` node | Visualization path | MISAL middleware |
|---|---|---|---|---|
| Live sensor acquisition | Camera node owns | Does not own | Does not own | Does not own |
| File-based image acquisition | Image-loader owns | Does not own | Does not own | Does not own |
| Source timestamps, frame IDs and camera calibration | Publishes valid source metadata | Preserves and correlates | Displays | Does not own |
| ROS 2 image transport | Publishes `Image` and, when available, `CameraInfo` | Subscribes | Bridge observes selected topics | Does not own |
| Resize, normalization and layout conversion | Does not perform model-specific conversion | Owns explicitly; profile must be documented | Does not own | Provides model metadata and rejects invalid tensors |
| Model loading and backend selection | Does not own | Configures or queries | Displays status only | Owns |
| Tensor shape, datatype and byte validation | Does not own | Performs preflight validation | Does not own | Owns final enforcement |
| gRPC request/response handling | Does not own | Owns | Does not own | Serves |
| Inference execution | Does not own | Requests and consumes result | Does not own | Owns |
| Detection decoding and annotated-image generation | Does not own | Owns according to model contract | Displays published data | Returns raw output tensors and metadata |
| Lichtblick connectivity | Does not own | Publishes observable topics | `foxglove_bridge` owns protocol transport | Does not own |
| Runtime health and diagnostics | Publishes source health | Publishes inference health | Exposes selected diagnostics | Reports backend/model/inference status |

The `misal_inference` node must not infer semantics from a tensor's byte width or silently reinterpret a model output. Preprocessing and decoder settings must be defined by the selected model profile rather than independently selected at runtime.

The Development Team owns the node and package design needed to satisfy these boundaries, including deciding whether target-specific camera adapters share one package or use maintained upstream camera packages. All image sources must converge on the same topic contract so `misal_inference` remains source-independent.

## 4. Integration Contract

### 4.1 Input contract

The first implementation should support one explicitly selected `sensor_msgs/msg/Image` topic and a documented subset of encodings, for example `rgb8` and `bgr8`. The `misal_inference` node must:

1. Validate the message encoding, dimensions, row stride, data length, timestamp and frame ID.
2. Convert the supported encoding to the model's expected channel order.
3. Apply the selected resize, normalization, datatype and layout policy.
4. Query or configure the MISAL model input metadata instead of assuming the binding is named `input`.
5. Construct a gRPC request containing the exact input name, datatype, shape and bytes.
6. Associate the request with the source ROS 2 timestamp and sequence number.

The node must reject unsupported encodings and incomplete preprocessing configuration. It must not silently fall back to a different layout, precision or model input.

### 4.2 Output contract

The first milestone must use a standard ROS 2 output representation. For object detection, the default contract is `vision_msgs/msg/Detection2DArray`. Each `Detection2D` must contain the source header, a `BoundingBox2D` in source-image pixel coordinates and one or more `vision_msgs/msg/ObjectHypothesisWithPose` results containing class ID and confidence.

Other inference tasks must use the closest semantically correct standard message, for example:

- `vision_msgs` classification messages for classification results;
- `sensor_msgs/msg/Image` for segmentation masks when its encoding and pixel semantics are documented; and
- `sensor_msgs/msg/PointCloud2`, `geometry_msgs` or other established standard messages for spatial results when their semantics match.

A custom message is permitted only when no maintained standard ROS 2 message can represent the required semantics without ambiguity. Such an exception requires a documented gap analysis and design review. Raw middleware tensors are debugging data, not the primary ROS 2 result contract; if exposed, they must use a separately documented debug topic and must not be required by normal consumers.

The decoder must document:

- output tensor names and expected shapes;
- mapping from model coordinates to the standard message fields;
- coordinate space, units and normalization, with detection boxes published in source-image pixel coordinates;
- class-label source and version;
- confidence and non-maximum-suppression policy;
- header propagation and result ordering;
- behavior for malformed or unsupported outputs.

Published result headers must preserve the source image timestamp and `frame_id` so standard ROS 2 synchronization, bagging, transforms and visualization tools can correlate results with the input. Topic names and QoS profiles must be configurable and documented. Consumers must not need model-specific tensor knowledge to interpret the primary result topic.

### 4.3 Control and diagnostics contract

The team should define a configuration surface that covers concerns such as:

- MISAL server address;
- input topic and output topic;
- model profile or configuration file, containing the model identifier and its preprocessing and decoder contract;
- maximum in-flight requests;
- request timeout;
- queue/drop policy;
- reconnect policy;
- debug output and metrics settings.

The node should publish connection state, last successful inference time, request failures, dropped frames, queue depth and model/backend identity through ROS 2 diagnostics or an equivalent project-standard mechanism.

### 4.4 Camera-node contract

At least one live camera node must publish:

- `sensor_msgs/msg/Image` on a configurable topic;
- `sensor_msgs/msg/CameraInfo` when calibration is available;
- a stable `frame_id` consistent with the associated transform tree;
- capture timestamps from the best available source clock; and
- diagnostics for connection state, capture failures, frame rate and dropped frames.

The selected camera node may wrap a maintained ROS 2 camera driver or be implemented by the project when no suitable driver exists. Hardware-specific controls, pixel formats and SDK objects must remain inside that node. The camera node must not preprocess images for a particular inference model. Any unavoidable sensor-side conversion must be documented as part of the camera output contract.

The first live-camera milestone supports one camera and one configured output mode. Device discovery, reconnect behavior, calibration-file handling and failure when the requested device or mode is unavailable must be explicit and testable.

### 4.5 Image-loader-node contract

An `image_loader` node must provide a hardware-independent source for development, tests and demonstrations. It should accept a file or ordered directory through parameters and publish the same `sensor_msgs/msg/Image` contract consumed from a camera node. It must define:

- supported file formats and color-decoding behavior;
- publication mode: one-shot, fixed-rate sequence or externally triggered;
- timestamp policy: current ROS time, source metadata or deterministic configured values;
- configurable `frame_id`, topic, rate and repeat behavior;
- deterministic lexical or manifest-defined ordering for directories; and
- behavior for missing, corrupt or unsupported files.

The loader should optionally publish configured `CameraInfo`, but must not invent calibration. It must not perform model-specific resize, normalization or layout conversion. Tests should be able to use the loader without a physical camera and receive identical message content across repeated runs when deterministic timestamps are selected.

Recorded ROS 2 bags remain useful integration fixtures, but they do not replace the image-loader node's simple file-based workflow.

### 4.6 Lichtblick and `foxglove_bridge` contract

The visualization path should use the maintained `foxglove_bridge` ROS 2 package and a compatible Lichtblick client. The bridge configuration must expose only the topics needed for the workflow, initially:

- source image and optional `CameraInfo`;
- decoded inference results;
- optional annotated debug image;
- inference and image-source diagnostics; and
- relevant transforms when spatial visualization requires them.

The Development Team must provide a version-controlled Lichtblick layout that identifies expected topic names, panels and message schemas. The minimum layout should show the source or annotated image, decoded results, inference latency, backend/model identity, connection state and dropped-frame counters.

Visualization is observational: losing Lichtblick or `foxglove_bridge` must not stop image acquisition or inference. The bridge must use bounded subscriptions and must not force expensive debug-image generation unless that output is enabled. Network bind address, port, allowed topics and trusted-network assumptions must be explicit; public exposure, authentication and TLS remain outside this phase unless separately approved.

## 5. Candidate Work Areas

These areas describe capabilities that the complete pipeline may need, not a mandatory node or issue decomposition. Students should first confirm what existing components already provide, agree on the architecture and then define the concrete work packages needed for the selected solution.

### V1 — Confirm the system contract

**Tasks:**

1. Identify the ROS 2 distribution, language, supported target platforms, camera message types and existing launch/configuration conventions.
2. Identify the first model and whether it produces classification, detection, segmentation or another output.
3. Select the first input topic, standard `vision_msgs` output type, supported image encodings and model profile.
4. Decide whether `misal_inference` runs as a standalone ROS 2 process or a composable node within the vision pipeline.
5. Record the expected latency, frame-rate, queue and failure behavior as targets rather than guarantees.
6. Select the first live camera driver or define the required adapter, including image and calibration topics.
7. Define the image-loader modes, timestamp policy and supported file formats.
8. Define the `foxglove_bridge` topic allowlist and the version-controlled Lichtblick layout.

**Acceptance criteria:**

- A short interface specification names every input, output, parameter, message and failure state.
- The primary inference result uses a standard ROS 2 message, with any custom-message exception justified by a reviewed gap analysis.
- The first model's input and output tensor contract is recorded.
- Ownership of preprocessing, decoding, labels, configuration and diagnostics is unambiguous.
- Open ROS 2 and hardware assumptions are listed instead of hidden in implementation tasks.
- Camera, image-loader and visualization ownership and topic contracts are documented.

### V2 — Build image-source nodes

**Tasks:**

1. Integrate or implement one camera node that satisfies the camera-node contract.
2. Implement the `image_loader` node with one-shot and deterministic fixed-rate sequence modes.
3. Make topic names, frame IDs, calibration files, source paths and publication rates configurable.
4. Add common contract tests that can be run against both source types.
5. Add diagnostics and explicit failure behavior for unavailable devices and invalid files.

**Acceptance criteria:**

- Camera and image-loader outputs can be substituted without changing `misal_inference` source code.
- Both sources publish structurally valid images with documented encoding, timestamp and frame-ID behavior.
- The image loader produces reproducible message bytes and ordering from fixed fixtures.
- Missing camera hardware and malformed image files produce visible failures rather than empty success.

### V3 — Build the `misal_inference` node

**Tasks:**

1. Create the ROS 2 package and `misal_inference` executable or component in the vision-pipeline workspace.
2. Subscribe to one image topic and validate message structure.
3. Implement the MISAL gRPC client inside the node with explicit connection, timeout and reconnect behavior.
4. Implement bounded buffering and a documented frame-drop policy.
5. Add a deterministic test seam so image conversion and gRPC request construction can be tested without a camera, GPU or live middleware server.

**Acceptance criteria:**

- A recorded ROS 2 image stream reaches the `misal_inference` node and a test double with correct timestamps, dimensions, encoding and frame ID.
- Queue growth is bounded.
- Disconnects do not block the ROS 2 executor indefinitely.
- No vendor SDK headers or libraries are required to build the `misal_inference` node.

### V4 — Implement metadata-driven preprocessing

**Tasks:**

1. Use MISAL model metadata when available, including input name, shape, datatype and layout.
2. Implement only the preprocessing operations required by the selected model.
3. Define color order, resize policy, normalization constants, batch dimension and layout in the versioned model profile; do not expose them as unrelated runtime choices.
4. Validate the resulting tensor in `misal_inference` before sending it and preserve the source frame metadata separately.
5. Add golden-vector tests comparing known image inputs with expected tensor bytes or numerically bounded values.

**Acceptance criteria:**

- The `misal_inference` node does not hard-code the input binding name.
- Invalid or incomplete preprocessing configuration fails clearly at startup or request construction.
- Golden-vector tests cover at least one image for every supported input encoding.
- The same preprocessing profile is documented for CPU and hardware-backed runs.

### V5 — Decode and publish inference results

**Tasks:**

1. Implement the selected model output decoder.
2. Validate output names, datatypes, shapes and byte lengths before decoding.
3. Map results in `misal_inference` into the selected standard ROS 2 output message; use `vision_msgs/msg/Detection2DArray` for the first object-detection model.
4. Preserve source timestamp and frame ID; document behavior when output latency is nonzero.
5. Add optional debug publication of raw outputs or annotated images without making it part of the minimum data path.

**Acceptance criteria:**

- A known model and fixture produce expected detections or classifications.
- Standard ROS 2 subscribers can consume the primary result topic without project-specific message packages or tensor-decoding logic.
- Malformed output tensors produce diagnostics and are not published as valid results.
- Class-label files and decoder versions are identifiable in logs or metadata.
- Output coordinate conventions are tested and documented.
- Source headers, standard message fields and QoS behavior are verified in contract tests.

### V6 — Add launch, configuration and visualization integration

**Tasks:**

1. Add ROS 2 parameters, launch files and example configuration for the first supported target.
2. Connect `misal_inference` to the existing vision visualization or debugging workflow.
3. Add runtime status and model/backend information to the web-based debugging surface where appropriate.
4. Define startup ordering and behavior when the MISAL server is unavailable.
5. Keep model upload, deployment or runtime reconfiguration separate from the minimum inference path unless the system already has an approved mechanism.
6. Configure `foxglove_bridge` with the approved topic allowlist and add the version-controlled Lichtblick layout.
7. Ensure visualization disconnects and slow clients do not block the inference path.

**Acceptance criteria:**

- A clean target setup can launch the camera, `misal_inference`, MISAL server and visualization with documented commands.
- Configuration changes do not require source edits.
- Server unavailability is visible and recoverable without crashing the ROS 2 graph.
- The debugging UI does not claim inference readiness before the backend reports a ready state.
- Lichtblick displays the source or annotated image, decoded result, latency and health information using the documented layout.
- Stopping the bridge or disconnecting Lichtblick does not interrupt image publication or inference.

### V7 — Validate performance and failure behavior

**Tasks:**

1. Measure capture-to-publish latency, middleware inference latency, end-to-end latency, throughput, queue depth and dropped frames.
2. Test recorded bags before live hardware.
3. Test server restart, network interruption, malformed frames, unsupported encodings, slow inference and model mismatch.
4. Compare CPU and selected accelerator behavior where hardware is available.
5. Record board, OS, ROS 2 distribution, middleware revision, model checksum, backend, SDK versions, resolution and precision.
6. Test camera disconnect/reconnect, corrupt image files, end-of-sequence behavior, bridge restart and slow visualization clients.

**Acceptance criteria:**

- Measurements identify where latency and frame drops occur.
- Failure tests produce documented diagnostics and bounded recovery behavior.
- Results distinguish functional correctness from performance targets.
- No test reports success when the camera, server, model or required hardware was not actually available.

## 6. Recommended Milestones

1. **Contract:** interface specification for camera, image loader, inference, visualization and the first model profile.
2. **File-based pipeline:** image fixture through `image_loader`, `misal_inference` and a MISAL test double.
3. **Offline visualization:** file-based inference results and diagnostics visible in the version-controlled Lichtblick layout.
4. **Live inference:** one camera topic to `misal_inference`, one MISAL model and one ROS 2 result topic.
5. **Live visualization:** camera image, results and health visible through `foxglove_bridge` without affecting inference behavior.
6. **Validation:** measured live, image-loader and recorded-bag results with failure-path evidence.

These milestones are technical guidance. The Student Development Team should decide the issue breakdown, sequencing, branch strategy and review flow, and may combine milestones when an existing component already satisfies the intended evidence.

## 7. Testing Strategy

### Portable tests

- Camera/image-loader shared message-contract tests
- Image-loader ordering, timing, repeat and corrupt-file tests
- ROS 2 message validation tests using fixtures
- Image encoding, stride, color-order, resize and normalization tests
- Tensor shape, datatype and byte-size tests
- Output decoder golden-vector tests
- Standard `vision_msgs` field-mapping and header-propagation tests
- Configuration and parameter validation tests
- gRPC client tests using a fake or local test server
- Queue, timeout, cancellation, reconnect and frame-drop tests
- `foxglove_bridge` configuration and Lichtblick layout schema checks

### Integration tests

- Image fixture through `image_loader`, `misal_inference` and a middleware test double
- Recorded ROS 2 bag through `misal_inference` and a CPU MISAL backend
- Real MISAL gRPC server with the selected model
- Model metadata discovery followed by inference
- Server restart and reconnect
- Malformed request and malformed output handling
- Camera or image-loader source through inference to topics observed via `foxglove_bridge`
- Bridge and Lichtblick disconnect without interruption of source or inference nodes

### Hardware validation

When an accelerator is used, record the exact board, operating system, SDK/runtime, model artifact checksum, precision, camera resolution and ROS 2 distribution. Hardware tests must be separate from portable tests and must fail when required assets or devices are absent.

## 8. Performance and Real-Time Considerations

The first implementation should prefer correctness and bounded behavior over zero-copy optimization. Potential optimizations such as shared memory, intra-process communication, compressed image transport, batching, pinned memory or a native in-process middleware API should be considered only after measurements identify a bottleneck.

The `misal_inference` node must define:

- whether frames are processed in order;
- whether old frames are dropped when inference is busy;
- the maximum number of in-flight requests;
- timeout and cancellation semantics;
- whether results may be published after a newer frame has been received.

The `misal_inference` node should not call a blocking gRPC operation on a ROS 2 callback thread in a way that can stall unrelated vision callbacks. Use an executor-compatible design and test it under load.

## 9. Intellectual-Property and Project Governance

This technical plan does not determine ownership, assignment, licensing, confidentiality, publication or disclosure obligations. Those matters must be handled through the applicable university and company representatives and any signed agreements governing participation in the project.

Before new code, model artifacts, datasets, interface specifications or implementation ideas are incorporated into the system, the responsible university and company contacts should confirm:

1. which project materials may be shared;
2. whether pre-existing university, student, open-source or third-party components are excluded from assignment;
3. whether ROS 2 packages, `misal_inference` code, generated files, models, logs and test data are covered;
4. how contributors and universities should be credited;
5. where confidential material may be stored and who may access it;
6. which repository, license, publication and release rules apply;
7. how inventions and disclosures should be reported;
8. whether an independent legal or university technology-transfer review is required.

Contributors should not be asked to decide ownership or disclosure obligations from this planning document. Participation, signing, publication and repository-access questions must be handled through the applicable university and company representatives. Development should use approved repositories and sanitized fixtures until the applicable governance requirements are confirmed.

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ROS 2 and middleware use incompatible tensor assumptions | Make preprocessing and model metadata explicit; add golden-vector tests |
| gRPC latency causes stale results or callback starvation | Use bounded asynchronous queues, timeouts, cancellation and a defined drop policy |
| Output decoder is wrong but appears plausible | Validate names/shapes and test against labeled fixtures |
| A custom result schema prevents standard tooling interoperability | Require standard ROS 2 messages by default and review any exception through a documented gap analysis |
| Camera encoding differs across boards | Support a small documented encoding set first and reject the rest |
| Camera node embeds model-specific processing | Keep acquisition and unavoidable sensor conversion separate from configured inference preprocessing |
| File loader behaves differently from a live source | Enforce one shared ROS 2 image contract and run common contract tests against both |
| Source timestamps or frame IDs are inconsistent | Define policies per source and verify correlation in end-to-end tests |
| Server restart loses the pipeline | Add health state, reconnect behavior and restart integration tests |
| Visualization traffic adds latency or backpressure | Limit bridged topics, make debug images optional and verify bridge disconnection under load |
| Bridge is exposed beyond the trusted development network | Configure bind address and topic allowlist; document that public-network security is out of scope |
| Hardware-specific behavior leaks into ROS 2 code | Keep vendor APIs behind MISAL; use gRPC contract tests in portable CI |
| Project governance restricts artifact use | Obtain written guidance before incorporating non-public artifacts; use sanitized fixtures |
| Performance goals are asserted without evidence | Measure each pipeline segment and record environment details |

## 11. Definition of Done

- The ROS 2 input, `misal_inference` request/response behavior, MISAL request, MISAL response and ROS 2 output contracts are documented.
- The primary inference output uses the appropriate standard ROS 2 message; object detections use `vision_msgs/msg/Detection2DArray`.
- One live camera node and the `image_loader` node satisfy the same documented image-source contract.
- The image loader supports reproducible one-shot and sequence workflows without camera hardware.
- A `misal_inference` ROS 2 node can be launched with documented parameters and connects to the MISAL gRPC service.
- One supported camera topic completes inference through the gRPC bridge.
- The `misal_inference` node uses explicit model/preprocessing metadata and does not silently guess.
- Results preserve source timestamp and frame ID.
- Queue, timeout, reconnect and frame-drop behavior are bounded and tested.
- Portable tests run without accelerator SDKs or physical cameras.
- A recorded-bag test and one live hardware demonstration are reproducible.
- A version-controlled Lichtblick layout displays images, results, latency and health through a restricted `foxglove_bridge` configuration.
- Visualization disconnection does not stop source publication or inference.
- Performance and failure results include enough environment metadata to interpret them.
- Ownership, access, disclosure and licensing questions are confirmed through the authorized university/company contacts before non-public artifacts are incorporated.
- Documentation includes setup, launch, configuration, model profile, diagnostics and known limitations.

## 12. Implementation Notes

Start with one model, the deterministic image loader, one output type and one target platform. Add the live camera only after the file-based inference path is stable, then add the Lichtblick workflow without placing visualization in the critical path. Keep both the gRPC and visualization bridges narrow until the end-to-end contract is stable. Do not make production, safety-critical or zero-copy claims from a functional demo alone.
