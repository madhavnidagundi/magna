# Phase 5 — Vision-Pipeline Containerization Proposal

**Status:** Proposed  
**Audience:** Student Development Team  
**Prerequisite:** The ROS 2 vision pipeline and MISAL gRPC boundary are functionally validated outside containers  
**Primary outcome:** Produce and validate an evidence-based containerization proposal for deploying the vision pipeline. The proposal must compare a single application container with a split deployment in which ROS 2 applications and MISAL inference run in separate containers.

## Student Ownership and Decision Authority

This phase is intentionally an investigation rather than a predetermined container architecture. Under [the planning roadmap](README.md), the Student Development Team owns the alternatives, weighting, prototypes and final recommendation. The two topologies below establish a common comparison baseline; they do not imply that either must become the final deployment.

The team may vary process placement, base images, build tooling, network mode and operational mechanisms when both alternatives remain comparable and the trade-offs are demonstrated. The depth of each prototype should be agreed at phase kickoff: a milestone may require a focused feasibility prototype, while a final architecture study may retain the complete measurement and operations criteria below.

## Entry and scope gate

Do not start the comparative implementation until the Phase 4 deterministic file-based path works outside containers. A short survey of existing Dockerfiles, target runtimes and redistribution constraints may happen earlier.

At kickoff, choose one of two explicit depths:

- **feasibility milestone:** minimal, comparable A/B prototypes using `image_loader`, a portable backend and a focused set of build, readiness, connectivity and latency evidence; or
- **deployment study:** the complete measurement, security, operations and hardware criteria in this brief.

The feasibility milestone must not be presented as a production deployment decision. Prototype and evidence areas remain open for volunteers, but each alternative needs a named owner and a shared validation owner so the comparison uses one protocol and does not favor either topology.

## 1. Purpose

The vision pipeline contains several processes with different dependencies and hardware responsibilities:

- a camera node or file-based `image_loader` node;
- the `misal_inference` ROS 2 node;
- `foxglove_bridge` for Lichtblick connectivity; and
- the MISAL middleware server with a selected CPU or accelerator backend.

The existing gRPC boundary makes both a combined and a split deployment possible. This task does not prescribe which topology is correct. The Development Team must investigate the alternatives, document advantages and disadvantages, build enough of each alternative to test its assumptions and recommend a strategy for the first supported target.

Containerization must package the application; it must not change ROS 2 topic contracts, standard message types, model semantics or the MISAL gRPC API.

## 2. Required Alternatives

The proposal should begin with the following two alternatives and may refine them during team review.

### Alternative A — Single application container

```text
Host
	|
	+-- application container
				|
				+-- camera or image_loader ROS 2 node
				+-- misal_inference ROS 2 node
				+-- foxglove_bridge
				+-- MISAL middleware server
				+-- selected inference runtime
```

The processes may be started by a launch system or a minimal supervisor, but each process must remain independently observable. A container entrypoint must not hide child-process failures.

The proposal must assess these likely characteristics rather than assume they apply:

**Potential advantages**

- Simple deployment and startup ordering
- No container-to-container gRPC or DDS networking configuration
- One versioned application artifact
- Straightforward file sharing for models and configuration
- Potentially simpler access to camera and accelerator devices

**Potential disadvantages**

- ROS 2, native inference runtimes and hardware SDKs create a larger image
- ROS 2 and inference components cannot be upgraded or restarted independently
- One process failure or image vulnerability has a larger blast radius
- Hardware-specific runtime layers may need to be duplicated across target images
- Scaling or replacing only the inference service is difficult
- Combined logs and lifecycle management may be harder to diagnose

### Alternative B — ROS 2 and inference containers

```text
Host
	|
	+-- ROS 2 application container
	|     +-- camera or image_loader node
	|     +-- misal_inference node
	|     +-- foxglove_bridge
	|
	+-- MISAL inference container
				+-- middleware gRPC server
				+-- selected inference runtime
```

The ROS 2 application container communicates with the MISAL container through the existing gRPC interface. Raw images should remain on ROS 2 topics inside the ROS 2 side unless measurements justify another boundary.

The proposal must assess these likely characteristics rather than assume they apply:

**Potential advantages**

- Clear separation between ROS 2 integration and hardware-specific inference dependencies
- Independent builds, updates, restarts and vulnerability scanning
- The inference image can vary by CPU or accelerator while the ROS 2 image remains stable
- The gRPC boundary can be tested and versioned independently
- Failures and resource limits can be isolated per service
- The inference server can potentially run on another host in a future deployment

**Potential disadvantages**

- Additional networking, name resolution, health checking and startup coordination
- gRPC traffic crosses a container network boundary
- More deployment artifacts and compatibility combinations
- Model/configuration ownership and volume permissions become more complex
- Device assignment and vendor runtime configuration remain platform-specific
- Network policy and exposed-port mistakes can widen the attack surface

### Optional alternatives

The team may evaluate additional topologies, such as placing `foxglove_bridge` in its own container or keeping a hardware camera driver on the host. These may be recommended only when supported by a concrete constraint or measurement. Optional alternatives do not replace the required A/B comparison.

## 3. Questions the Proposal Must Answer

### 3.1 Deployment scope

1. What is the first supported board, CPU architecture, operating system and container engine?
2. Which inference backend and vendor runtime must be present?
3. Is the deployment intended for development, demonstration, CI or field operation?
4. Which components are target-specific and which images can be reused across targets?
5. Is offline installation required?

### 3.2 Process and image boundaries

1. Which process belongs in each container, and why?
2. Is the camera node containerized, or does a hardware/driver constraint require host execution?
3. Does `foxglove_bridge` share the ROS 2 container or warrant an independent lifecycle?
4. Who owns model files, model profiles, calibration files and Lichtblick layouts?
5. Which components must be independently restartable or upgradeable?
6. How are incompatible ROS 2, gRPC, model and backend versions prevented from being combined?

### 3.3 Networking

1. How does ROS 2 DDS discovery operate in the selected network mode?
2. Is host networking necessary, or can a bridged network satisfy discovery and multicast requirements?
3. Which gRPC address and port are used between services?
4. Which port is exposed for `foxglove_bridge`, and to which interface?
5. Which ports must remain internal?
6. How are DNS/service names, ROS domain ID and DDS implementation configured?
7. What changes when Lichtblick runs on a different machine?

Host networking must not be selected only because it is convenient. If it is selected, the proposal must explain the portability, isolation and security trade-offs. If bridge networking is selected, ROS 2 discovery and cross-host behavior must be demonstrated.

### 3.4 Hardware access

The proposal must identify the minimum required host resources for each target:

- camera device nodes and permissions;
- GPU, NPU or other accelerator devices;
- vendor driver libraries and container runtime hooks;
- shared memory requirements;
- USB or media-controller devices;
- CPU affinity, real-time scheduling or memory-lock capabilities, if measured as necessary; and
- architecture-specific runtime libraries.

Do not use blanket `--privileged` mode as the default solution. Every mounted device, host path, group, Linux capability and security-profile exception must be justified. Host kernel drivers must remain compatible with the user-space runtime inside the image.

### 3.5 Data and configuration

The team must classify each item as image content, read-only bind mount, named volume, runtime parameter or secret:

- model artifacts and checksums;
- model profiles;
- camera calibration;
- image fixtures or recorded bags;
- ROS 2 parameter files and launch configuration;
- Lichtblick layouts;
- middleware configuration;
- logs and metrics; and
- generated caches or optimized engines.

Images must not contain credentials, private keys or unrestricted proprietary datasets. Writable application state must not be stored implicitly in the container layer. The proposal must define ownership, permissions, retention and upgrade behavior for every writable volume.

### 3.6 Lifecycle and failure behavior

The proposal must define:

- startup dependencies and readiness conditions;
- behavior while the inference server is unavailable;
- health checks that distinguish process liveness from model/backend readiness;
- restart policies and retry/backoff behavior;
- graceful shutdown and request draining;
- log collection and correlation across ROS 2 and MISAL;
- behavior when the camera disconnects;
- behavior when `foxglove_bridge` or Lichtblick disconnects; and
- rollback to a previously known-good image set.

Container startup order alone is not readiness. The ROS 2 application must tolerate the MISAL server starting later or restarting, as required by the vision-pipeline design.

### 3.7 Security and supply chain

The proposal must address:

- pinned base images, preferably by digest for releases;
- reproducible dependency installation and lock files;
- multi-stage builds and removal of build tools from runtime images;
- non-root runtime users where hardware access permits;
- read-only root filesystems and dropped capabilities where practical;
- image vulnerability and license scanning;
- software bill of materials and image provenance;
- secrets handling;
- gRPC and `foxglove_bridge` network exposure; and
- update and patch ownership.

TLS, authentication and public-network deployment remain separate design decisions, but the proposal must state the trusted-network boundary and must not expose services unintentionally.

## 4. Required Trade-Off Analysis

The proposal must include a decision matrix. At minimum, both required alternatives must be scored and explained against:

| Criterion | Evidence expected |
|---|---|
| Build complexity | Dockerfile stages, native SDK installation and build duration |
| Runtime complexity | Services, startup dependencies, networks and volumes |
| Image size | Compressed and unpacked image measurements |
| Startup and readiness | Time until the first successful inference |
| Inference performance | Warm latency, throughput and jitter |
| End-to-end performance | Image publication to ROS 2 result publication |
| Copy/network overhead | gRPC latency and CPU/memory effects across the chosen boundary |
| Hardware portability | Differences across CPU and accelerator targets |
| Development workflow | Rebuild scope and local debugging experience |
| Upgradeability | Ability to update and roll back ROS 2 and inference independently |
| Fault isolation | Observed behavior after terminating each process/container |
| Observability | Log, health, metrics and trace correlation |
| Security | Privileges, devices, mounts, ports and attack surface |
| Reproducibility | Clean-host build and deployment success |
| Operational ownership | Number of artifacts and compatibility combinations maintained |

Scores without evidence are insufficient. Weighting must reflect the stated deployment scope, and the team must explain how changing the weights could change the recommendation.

## 5. Prototype and Measurement Requirements

Before recommending a topology, the team should agree on the smallest fair prototype needed to test both alternatives using the deterministic `image_loader`. A physical camera and accelerator may be added after the portable comparison.

Both prototypes must use:

- the same model and checksum;
- the same model profile and input fixtures;
- the same ROS 2 message and QoS contracts;
- the same middleware revision and inference backend;
- equivalent resource limits; and
- the same host and measurement procedure.

Measure at least:

- clean and cached image build time;
- image sizes;
- container startup to service readiness;
- startup to first successful inference;
- steady-state inference latency and throughput;
- end-to-end image-to-result latency;
- CPU and memory use per process/container;
- dropped images and queue depth under load;
- shutdown time; and
- recovery after terminating and restarting the MISAL server.

Run enough warm iterations to report median and tail latency, including at least the 95th percentile. Record host hardware, OS, kernel, container-engine version, image digests, middleware revision, model checksum and backend/runtime versions. Do not compare results collected from materially different environments.

## 6. Deliverables

The Development Team must produce:

1. **Architecture proposal** — scope, assumptions, diagrams and process/image boundaries.
2. **Alternatives analysis** — explicit pros, cons, risks and the weighted decision matrix.
3. **Prototype artifacts** — multi-stage Dockerfiles and a declarative deployment definition for both required alternatives.
4. **Configuration contract** — networks, ports, devices, mounts, volumes, environment variables, health checks and resource limits.
5. **Measurement report** — reproducible procedure, raw results and interpretation.
6. **Security review** — privileges, capabilities, device access, network exposure, base-image policy, scanning and SBOM strategy.
7. **Operations runbook** — build, start, inspect, stop, update, roll back and troubleshoot procedures.
8. **Architecture decision record** — selected strategy, rejected alternatives, evidence, limitations and conditions that trigger reconsideration.

The final recommendation must state whether it applies only to the first target or is intended as the project-wide deployment pattern.

## 7. Acceptance Criteria

- Both the single-container and split-container alternatives are described with concrete advantages, disadvantages and risks.
- Minimal prototypes of both alternatives complete deterministic file-based inference through the existing ROS 2 and gRPC contracts.
- The comparison uses equivalent inputs, model, backend, host and resource assumptions.
- The decision matrix is supported by measurements and documented operational requirements.
- ROS 2 DDS discovery, gRPC connectivity and Lichtblick access work in the proposed network mode.
- The deployment definition identifies all ports, devices, mounts, volumes, health checks, restart policies and resource limits.
- Camera and accelerator access use the minimum justified privileges; blanket privileged mode is not the default.
- The MISAL container reports readiness only after the backend and model are ready.
- The ROS 2 side remains responsive while MISAL is unavailable and reconnects according to the documented policy.
- Stopping or disconnecting visualization does not interrupt image publication or inference.
- Runtime images contain only required runtime dependencies and run as non-root where target device permissions allow.
- Base images and application images are versioned; release inputs are pinned and traceable.
- A clean supported host can reproduce the build and deployment using only the runbook.
- The architecture decision record recommends one topology and explains what evidence would justify changing it.

## 8. Non-Goals

- Selecting a topology before the alternatives are prototyped and measured
- Replacing the existing MISAL gRPC or ROS 2 message contracts
- Moving raw camera images across gRPC when the existing ROS 2-side preprocessing boundary is sufficient
- Building a custom container orchestrator
- Requiring Kubernetes for a single-device prototype
- Solving public-cloud deployment, fleet management or over-the-air updates in the first proposal
- Embedding development toolchains, test data or credentials in production runtime images
- Treating an image that merely starts its processes as proof that inference is ready

## 9. Guidance

Prefer the smallest deployable design that meets the measured requirements, but do not optimize only for the number of containers. A single container may be appropriate for a tightly coupled demonstration; a split design may be appropriate when hardware-specific inference dependencies, independent updates or fault isolation dominate. The recommendation must follow from the target constraints and evidence rather than from a general rule that more or fewer containers are inherently better.

Reuse the repository's existing hardware-specific container work where it is suitable, but distinguish CI/optimizer images from minimal deployment images. Existing Dockerfiles are evidence and reusable input, not automatically the final runtime architecture.
