.PHONY: all help clean
.PHONY: security audit deny
.PHONY: quality fmt-check clippy test docs
.PHONY: build build-cpu build-lib-backends build-lib-nvidia build-lib-qualcomm build-lib-ti
.PHONY: clippy-cpu clippy-nvidia clippy-qualcomm clippy-ti clippy-cpu-metrics
.PHONY: clippy-cpu-opentelemetry clippy-nvidia-opentelemetry clippy-qualcomm-opentelemetry clippy-ti-opentelemetry
.PHONY: test-cpu test-cpu-metrics test-cpu-opentelemetry
.PHONY: test-nvidia-opentelemetry test-qualcomm-opentelemetry test-ti-opentelemetry test-hardware
.PHONY: build-cpu-opentelemetry build-nvidia-opentelemetry build-qualcomm-opentelemetry build-ti-opentelemetry
.PHONY: hardware nvidia-build radxa-build qualcomm-build ti-build cpu-verify

# Default target - runs all quality checks that can be performed locally
all: security quality build-lib-backends docs
	@echo "✓ All quality checks passed!"

# Help target
help:
	@echo "Makefile for mi-isal project"
	@echo ""
	@echo "Main targets:"
	@echo "  all                  - Run all security and quality checks (default)"
	@echo "  security             - Run all security audits"
	@echo "  quality              - Run all code quality checks"
	@echo "  build                - Build all components"
	@echo "  docs                 - Generate and check documentation"
	@echo "  clean                - Clean build artifacts"
	@echo ""
	@echo "Security targets:"
	@echo "  audit                - Run cargo audit (vulnerability scan)"
	@echo "  deny                 - Run cargo deny (policy enforcement)"
	@echo ""
	@echo "Quality targets:"
	@echo "  fmt-check            - Check code formatting"
	@echo "  clippy               - Run clippy on all feature combinations"
	@echo "  clippy-*-opentelemetry - Run clippy with OpenTelemetry for specific backend"
	@echo "  test                 - Run tests for all testable features"
	@echo "  test-*-opentelemetry - Run tests with OpenTelemetry for specific backend"
	@echo ""
	@echo "Build targets:"
	@echo "  build-cpu            - Build and test CPU backend"
	@echo "  build-lib-nvidia     - Build NVIDIA backend (library only)"
	@echo "  build-lib-qualcomm   - Build Qualcomm backend (library only)"
	@echo "  build-lib-ti         - Build TI backend (library only)"
	@echo ""
	@echo "Hardware targets (require specific SDKs/hardware):"
	@echo "  nvidia-build         - Build NVIDIA TensorRT engine (requires Docker)"
	@echo "  radxa-build          - Build RADXA RKNN model (requires Docker)"
	@echo "  qualcomm-build       - Build Qualcomm QNN model (requires SDK)"
	@echo "  ti-build             - Build TI TIDL artifacts (requires SDK)"
	@echo "  cpu-verify           - Verify CPU optimization"

# ── Security Targets ─────────────────────────────────────────────────────────

security: audit deny

audit:
	@echo "Running cargo audit (vulnerability scan)..."
	cargo audit

deny:
	@echo "Running cargo deny (policy enforcement)..."
	cargo deny check

# ── Quality Targets ──────────────────────────────────────────────────────────

quality: fmt-check clippy test

fmt-check:
	@echo "Checking code formatting..."
	cargo fmt --all --check

clippy: clippy-cpu clippy-nvidia clippy-qualcomm clippy-ti clippy-cpu-metrics clippy-cpu-opentelemetry clippy-nvidia-opentelemetry clippy-qualcomm-opentelemetry clippy-ti-opentelemetry
	@echo "✓ All clippy checks passed!"

clippy-cpu:
	@echo "Running clippy for CPU backend..."
	cargo clippy --features cpu --all-targets -- -D warnings

clippy-nvidia:
	@echo "Running clippy for NVIDIA backend..."
	cargo clippy --features nvidia --all-targets -- -D warnings

clippy-qualcomm:
	@echo "Running clippy for Qualcomm backend..."
	cargo clippy --features qualcomm --all-targets -- -D warnings

clippy-ti:
	@echo "Running clippy for TI backend..."
	cargo clippy --features ti --all-targets -- -D warnings

clippy-cpu-metrics:
	@echo "Running clippy for CPU backend with metrics..."
	cargo clippy --features cpu,metrics --all-targets -- -D warnings

clippy-cpu-opentelemetry:
	@echo "Running clippy for CPU backend with OpenTelemetry..."
	cargo clippy --features cpu,opentelemetry --all-targets -- -D warnings

clippy-nvidia-opentelemetry:
	@echo "Running clippy for NVIDIA backend with OpenTelemetry..."
	cargo clippy --features nvidia,opentelemetry --all-targets -- -D warnings

clippy-qualcomm-opentelemetry:
	@echo "Running clippy for Qualcomm backend with OpenTelemetry..."
	cargo clippy --features qualcomm,opentelemetry --all-targets -- -D warnings

clippy-ti-opentelemetry:
	@echo "Running clippy for TI backend with OpenTelemetry..."
	cargo clippy --features ti,opentelemetry --all-targets -- -D warnings

test: test-cpu test-cpu-metrics test-cpu-opentelemetry

test-hardware: test-nvidia-opentelemetry test-qualcomm-opentelemetry test-ti-opentelemetry

test-cpu:
	@echo "Running tests for CPU backend..."
	cargo test --features cpu

test-cpu-metrics:
	@echo "Running tests for CPU backend with metrics..."
	cargo test --features cpu,metrics

test-cpu-opentelemetry:
	@echo "Running tests for CPU backend with OpenTelemetry..."
	cargo test --features cpu,opentelemetry

test-nvidia-opentelemetry:
	@echo "Running tests for NVIDIA backend with OpenTelemetry..."
	cargo test --lib --features nvidia,opentelemetry

test-qualcomm-opentelemetry:
	@echo "Running tests for Qualcomm backend with OpenTelemetry..."
	cargo test --lib --features qualcomm,opentelemetry

test-ti-opentelemetry:
	@echo "Running tests for TI backend with OpenTelemetry..."
	cargo test --lib --features ti,opentelemetry

# ── Build Targets ────────────────────────────────────────────────────────────

build: build-cpu build-lib-backends

build-cpu:
	@echo "Building and testing CPU backend..."
	cargo build --features cpu
	cargo test --features cpu

build-lib-backends: build-lib-nvidia build-lib-qualcomm build-lib-ti
	@echo "✓ All backend libraries built successfully!"

build-lib-nvidia:
	@echo "Building NVIDIA backend library..."
	cargo build --lib --features nvidia

build-lib-qualcomm:
	@echo "Building Qualcomm backend library..."
	cargo build --lib --features qualcomm

build-lib-ti:
	@echo "Building TI backend library..."
	cargo build --lib --features ti

build-nvidia-opentelemetry:
	@echo "Building NVIDIA backend with OpenTelemetry..."
	cargo build --lib --features nvidia,opentelemetry

build-qualcomm-opentelemetry:
	@echo "Building Qualcomm backend with OpenTelemetry..."
	cargo build --lib --features qualcomm,opentelemetry

build-ti-opentelemetry:
	@echo "Building TI backend with OpenTelemetry..."
	cargo build --lib --features ti,opentelemetry

build-cpu-opentelemetry:
	@echo "Building CPU backend with OpenTelemetry..."
	cargo build --features cpu,opentelemetry

# ── Documentation ────────────────────────────────────────────────────────────

docs:
	@echo "Generating and checking documentation..."
	MAGNA_DOCS_BUILD=1 RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features

# ── Hardware-Specific Targets ────────────────────────────────────────────────
# These require specific SDKs, Docker images, or actual hardware

hardware: nvidia-build radxa-build qualcomm-build ti-build cpu-verify

nvidia-build:
	@echo "Building NVIDIA TensorRT engine (requires Docker and GPU)..."
	@echo "Note: This requires the misal-tensorrt Docker image"
	docker run --gpus all --rm \
		-v $(PWD):/workspace \
		misal-tensorrt:latest \
		optimize --hw ORIN \
		--model /workspace/models/mobilenetv4.onnx \
		--precision int8 \
		--output /workspace/models/mobilenetv4.engine

radxa-build:
	@echo "Building RADXA RKNN model (requires Docker)..."
	docker build -t misal-radxa-optimizer:latest -f docker/Dockerfile.radxa-optimizer .
	docker run --rm \
		-v $(PWD):/workspace \
		misal-radxa-optimizer:latest \
		optimize --hw RADXA-Q6A \
		--model /workspace/models/mobilenetv4.onnx \
		--precision int8 \
		--calibration-data /workspace/scripts/data_prep/imagenet_val_subset \
		--output /workspace/models/mobilenetv4.rknn

qualcomm-build:
	@echo "Building Qualcomm QNN model (requires QNN SDK)..."
	@if [ -z "$(QNN_SDK_ROOT)" ]; then \
		echo "Error: QNN_SDK_ROOT environment variable not set"; \
		exit 1; \
	fi
	cargo run --bin misal -- optimize --hw SNAPDRAGON \
		--model models/mobilenetv4.onnx \
		--precision int8 \
		--output /tmp/mobilenetv4.bin \
		--dry-run

ti-build:
	@echo "Building TI TIDL artifacts (requires TIDL tools)..."
	@if [ -z "$(TIDL_TOOLS_PATH)" ]; then \
		echo "Error: TIDL_TOOLS_PATH environment variable not set"; \
		exit 1; \
	fi
	cargo run --bin misal -- optimize --hw TDA4 \
		--model models/mobilenetv4.onnx \
		--precision int8 \
		--output /tmp/mobilenetv4_tidl \
		--dry-run

cpu-verify:
	@echo "Verifying CPU optimization..."
	cargo run --bin misal -- optimize --hw CPU \
		--model models/mobilenetv4.onnx \
		--output /tmp/mobilenetv4_cpu.onnx

# ── Clean ────────────────────────────────────────────────────────────────────

clean:
	@echo "Cleaning build artifacts..."
	cargo clean
	rm -rf target/
	rm -f models/*.engine models/*.rknn
	@echo "✓ Clean complete!"

# ── Development Dependencies ─────────────────────────────────────────────────

.PHONY: deps-install deps-check

deps-check:
	@echo "Checking required dependencies..."
	@command -v cargo >/dev/null 2>&1 || { echo "cargo not found!"; exit 1; }
	@command -v protoc >/dev/null 2>&1 || { echo "protobuf-compiler not found!"; exit 1; }
	@command -v nasm >/dev/null 2>&1 || { echo "nasm not found!"; exit 1; }
	@echo "✓ All required dependencies are installed"

deps-install:
	@echo "Installing required dependencies (Ubuntu/Debian)..."
	sudo apt-get update
	sudo apt-get install -y protobuf-compiler nasm
	@echo "✓ Dependencies installed!"
