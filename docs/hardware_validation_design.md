# Hardware Validation Design: Self-Hosted GitHub Runners

**Purpose**: Validate that student code changes compile and link correctly on target hardware platforms (NVIDIA Orin/Thor, TI, Qualcomm) after merge to main.

**Scope**: This document covers the compilation validation infrastructure. Test automation (benchmarks, accuracy tests) is deferred to a second phase.

---

## 1. Overview

### Goals
- **Primary**: Verify code compiles on actual hardware platforms immediately after merge to main
- **Secondary**: Catch platform-specific build issues (missing SDK symbols, architecture mismatches)
- **Non-Goal (Phase 1)**: Automated inference testing, benchmarking, or accuracy validation

### Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│  GitHub Enterprise (MAGNA-Global/mi-isal)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  Workflow: hardware-validation.yml                                   │  │
│  │  Trigger: push to main                                               │  │
│  └──────────┬──────────────┬──────────────┬──────────────┬──────────────┘  │
│             │              │              │              │                 │
└─────────────┼──────────────┼──────────────┼──────────────┼─────────────────┘
              │              │              │              │
              ▼              ▼              ▼              ▼
       ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
       │   Orin   │   │   Thor   │   │  TI TDA4 │   │ Qualcomm │
       │(aarch64) │   │(aarch64) │   │(aarch64) │   │(aarch64) │
       │          │   │          │   │          │   │          │
       │ Runner:  │   │ Runner:  │   │ Runner:  │   │ Runner:  │
       │  "orin"  │   │  "thor"  │   │   "ti"   │   │"qualcomm"│
       └──────────┘   └──────────┘   └──────────┘   └──────────┘
            ▲              ▲              ▲              ▲
            │              │              │              │
            └──────────────┴──────────────┴──────────────┘
                        Devel LAN Network
                   (x86 PC w/ GPU for model preparation)
```

---

## 2. Hardware Runner Configuration

### 2.1 Board Setup

Each hardware board will be configured as a GitHub Actions self-hosted runner:

| Board | Label | Architecture | SDK Installed | Purpose |
|-------|-------|--------------|---------------|---------|
| NVIDIA Orin | `self-hosted, orin, nvidia` | aarch64-linux-gnu | JetPack 6.x (CUDA 12.2, TensorRT 8.6.2.3) | NVIDIA backend validation |
| NVIDIA Thor | `self-hosted, thor, nvidia` | aarch64-linux-gnu | JetPack 7.x (CUDA 13.0, TensorRT 10.13.3.9) | NVIDIA backend validation |
| TI TDA4 | `self-hosted, ti` | aarch64-linux-gnu | TI Processor SDK | TI backend validation |
| Qualcomm | `self-hosted, qualcomm` | aarch64-linux-gnu | QNN/SNPE SDK | Qualcomm backend validation |

### 2.2 Runner Installation

**Prerequisites** (install before runner setup):

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y protobuf-compiler nasm cmake libssl-dev pkg-config libclang-dev

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Verify installations
rustc --version
cargo --version
protoc --version
nasm --version
```

**Runner Installation**:

On each board, install GitHub Actions runner:

```bash
# On target board (as user with sudo access, e.g., 'alexsch6')
mkdir ~/actions-runner && cd ~/actions-runner
curl -o actions-runner-linux-arm64-2.314.1.tar.gz -L \
  https://github.com/actions/runner/releases/download/v2.314.1/actions-runner-linux-arm64-2.314.1.tar.gz
tar xzf ./actions-runner-linux-arm64-*.tar.gz

# Configure runner (requires GitHub repo admin token)
# Get token from: https://github.com/MAGNA-Global/mi-isal/settings/actions/runners/new
./config.sh --url https://github.com/MAGNA-Global/mi-isal \
            --token <REGISTRATION_TOKEN> \
            --labels self-hosted,orin,nvidia \
            --name orin-runner-hostname \
            --work _work

# Note: During configuration, press Enter for:
# - Runner group: [Default]
# - Runner name: [hostname] or custom name
# - Additional labels: orin,nvidia (comma-separated)
# - Work folder: [_work]

# Create environment file for SDK paths
cat > .env << 'EOF'
CUDA_ROOT=/usr/local/cuda
TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu
TENSORRT_LIB=/usr/lib/aarch64-linux-gnu
PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
HTTP_PROXY=http://ecsproxy2.van.magna.global:3128
HTTPS_PROXY=http://ecsproxy2.van.magna.global:3128
http_proxy=http://ecsproxyv.van.magna.global:3128
https_proxy=http://ecsproxyv.van.magna.global:3128
no_proxy=.van.magna.global,.magna.global
EOF

# Install as systemd service for auto-restart
sudo ./svc.sh install
sudo ./svc.sh start

# Verify service is running
sudo ./svc.sh status
```

### 2.3 Runner Environment

Each runner must have:
- **Rust toolchain**: Installed via `rustup` (stable channel)
- **System dependencies**: `protobuf-compiler`, `libssl-dev`, `pkg-config`, `cmake`, `nasm`, `libclang-dev`
- **Backend SDK**: Platform-specific (CUDA/TensorRT for NVIDIA, Processor SDK for TI, QNN for Qualcomm)
- **Build cache**: Local Cargo cache to speed up incremental builds (~/.cargo)
- **Disk space**: Minimum 50GB free for builds and models

**Environment variables** (set in runner service):
```bash
# NVIDIA boards
export CUDA_ROOT=/usr/local/cuda
export TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu
export TENSORRT_LIB=/usr/lib/aarch64-linux-gnu

# TI boards
export TI_SDK_ROOT=/opt/ti-processor-sdk

# Qualcomm boards
export QNN_SDK_ROOT=/opt/qnn-sdk
```

---

## 3. GitHub Workflow Design

### 3.1 Workflow File Structure

Create `.github/workflows/hardware-validation.yml`:

```yaml
name: Hardware Validation

on:
  push:
    branches:
      - main
  workflow_dispatch:  # Allow manual trigger
    inputs:
      board:
        description: 'Target board (all/orin/thor/ti/qualcomm)'
        required: false
        default: 'all'

jobs:
  # Gate: Only run if quality checks passed
  check-quality:
    runs-on: ubuntu-latest
    steps:
      - name: Verify quality workflow passed
        run: echo "Quality gates passed, proceeding to hardware validation"

  # NVIDIA Orin validation
  validate-orin:
    needs: check-quality
    if: github.event.inputs.board == 'all' || github.event.inputs.board == 'orin' || github.event.inputs.board == ''
    runs-on: [self-hosted, orin]
    steps:
      - uses: actions/checkout@v4
      - name: Build middleware (nvidia)
        working-directory: middleware
        run: cargo build --release --features nvidia
      - name: Build client
        working-directory: client
        run: cargo build --release
      - name: Verify binaries
        run: |
          ldd middleware/target/release/magna_server
          ./middleware/target/release/magna_server --help

  # NVIDIA Thor validation (similar to Orin)
  validate-thor:
    needs: check-quality
    if: github.event.inputs.board == 'all' || github.event.inputs.board == 'thor' || github.event.inputs.board == ''
    runs-on: [self-hosted, thor]
    steps:
      - uses: actions/checkout@v4
      - name: Build middleware (nvidia)
        working-directory: middleware
        run: cargo build --release --features nvidia
      - name: Build client
        working-directory: client
        run: cargo build --release
      - name: Verify binaries
        run: |
          ldd middleware/target/release/magna_server
          ./middleware/target/release/magna_server --help

  # TI TDA4 validation
  validate-ti:
    needs: check-quality
    if: github.event.inputs.board == 'all' || github.event.inputs.board == 'ti' || github.event.inputs.board == ''
    runs-on: [self-hosted, ti]
    steps:
      - uses: actions/checkout@v4
      - name: Build middleware (ti)
        working-directory: middleware
        run: cargo build --release --features ti
      - name: Build client
        working-directory: client
        run: cargo build --release
      - name: Verify binaries
        run: |
          ldd middleware/target/release/magna_server
          ./middleware/target/release/magna_server --help

  # Qualcomm validation (when board arrives)
  validate-qualcomm:
    needs: check-quality
    if: github.event.inputs.board == 'all' || github.event.inputs.board == 'qualcomm' || github.event.inputs.board == ''
    runs-on: [self-hosted, qualcomm]
    steps:
      - uses: actions/checkout@v4
      - name: Build middleware (qualcomm)
        working-directory: middleware
        run: cargo build --release --features qualcomm
      - name: Build client
        working-directory: client
        run: cargo build --release
      - name: Verify binaries
        run: |
          ldd middleware/target/release/magna_server
          ./middleware/target/release/magna_server --help
```

### 3.2 Workflow Behavior

**Trigger conditions**:
- **Automatic**: On every push to `main` branch
- **Manual**: Via workflow_dispatch with optional board selection

**Success criteria**:
- ✅ Code compiles with target backend feature
- ✅ Binary links successfully against SDK libraries
- ✅ `--help` executes without crashes (basic smoke test)

**Failure handling**:
- ❌ Compilation fails → GitHub check fails, notification sent
- ❌ Linking fails → Indicates missing SDK dependencies
- ❌ Binary crashes → Runtime issue detected early

**Duration estimate**: 5-10 minutes per board (parallel execution)

---

## 4. Security Considerations

### 4.1 Network Isolation

- Boards are on **devel network only**
- Runners pull code from GitHub (outbound only)
- No inbound ports exposed to public internet

### 4.2 Runner Security

**Best practices**:
- Dedicated user account (`magna-ci`) with minimal privileges
- Runner workspace isolated (`~/actions-runner/_work`)
- No sensitive credentials stored in repository
- Use GitHub Secrets for any API tokens (future)

**Risk mitigation**:
- Self-hosted runners only execute on `main` branch (after merge)
- No execution of untrusted PR code (PRs run on GitHub cloud runners)
- Regular security updates on boards via `apt update && apt upgrade`

### 4.3 Repository Settings

Configure repository to:
- Disable self-hosted runner usage for forks
- Require approval for first-time contributors (already standard)
- Restrict workflow modifications to maintainers

---

## 5. x86 GPU Server Role

### 5.1 Model Optimization Pipeline

**Location**: x86 terminal PC with GPU (not a GitHub runner)

**Purpose**:
- Convert PyTorch models to ONNX
- Optimize models for target hardware (INT8 quantization, pruning)
- Build TensorRT engines for NVIDIA targets
- Store optimized models in `models/` directory

**Workflow (manual for Phase 1)**:
```bash
# On x86 PC
cd /path/to/mi-isal/scripts/quantization
python build_int8_engine.py --model mobilenetv4 --target orin

# Commit optimized model to repo
cd ../..
git add models/mobilenetv4_orin_int8.engine
git commit -m "Add optimized model for Orin"
git push origin main
```

### 5.2 Future Automation

**Phase 2 potential**:
- x86 PC as GitHub runner with label `x86-gpu-optimization`
- Pre-merge job: optimize models for all targets
- Post-merge job: hardware runners pull latest models for testing

---

## 6. Operational Procedures

### 6.1 Adding a New Board

1. Install runner software on board
2. Register runner with GitHub (repo settings → Actions → Runners)
3. Configure SDK environment variables
4. Add validation job to `hardware-validation.yml`
5. Test with manual workflow dispatch

### 6.2 Maintenance

**Weekly**:
- Check runner disk space: `df -h`
- Clean old build artifacts: `cargo clean` in runner workspace

**Monthly**:
- Update runner version: `./svc.sh stop && ./config.sh remove && ./config.sh --url ... && ./svc.sh install`
- Update system packages: `sudo apt update && sudo apt upgrade`

**Troubleshooting**:
- Runner offline: `sudo systemctl status actions.runner.*`
- Build failures: Check SDK paths in environment
- Disk full: Clear `~/.cargo/registry/cache`

### 6.3 Monitoring

**Metrics to track** (manual for now):
- Build success rate per board
- Average build duration
- Frequency of SDK-related failures

**Notifications**:
- GitHub sends email/notification on workflow failure
- Consider Slack integration for real-time alerts (Phase 2)

---

## 7. Rollout Plan

### Phase 1: Compilation Validation (Current)

**Setup** ✅ **COMPLETED (May 6, 2026)**
- [x] Install runner on NVIDIA Orin (vanlindv60004)
- [x] Configure JetPack SDK environment (CUDA 12.2, TensorRT 8.6.2.3)
- [x] Test with manual workflow dispatch (test-orin-runner.yml successful)
- [x] Install runner on NVIDIA Thor (VANDVL00018) - **May 8, 2026**
- [x] Configure JetPack SDK environment for Thor (CUDA 13.0, TensorRT 10.13.3.9)
- [x] Test with manual workflow dispatch (test-thor-runner.yml)

**Initial Deployment** 🔄 **IN PROGRESS**
- [x] Create `hardware-validation.yml` workflow
- [x] Add Thor validation job to workflow
- [ ] Enable automatic trigger on push to main
- [ ] Monitor for 2 weeks, collect feedback

**Expand Coverage**
- [x] Add Thor runner (completed May 8, 2026)
- [ ] Add TI TDA4 runner
- [ ] Document SDK setup procedures

**Qualcomm Support**
- [ ] Add Qualcomm runner when board arrives
- [ ] Validate all four backends working

### Phase 2: Test Automation (Future)

**Deferred to separate design document**:
- Inference smoke tests (model loads, single inference succeeds)
- Benchmark suite (latency, throughput)
- Accuracy validation (ImageNet subset)
- Performance regression detection
- Power consumption monitoring

## 8. Implementation Notes (May 6, 2026)

### 8.1 Orin Runner Setup (vanlindv60004)

**Hardware Configuration**:
- Board: NVIDIA Jetson Orin (aarch64)
- Hostname: vanlindv60004
- OS: Ubuntu 22.04.5 LTS (Linux 5.15.136-tegra)
- Disk: 1.8TB total, 1.5TB available

**Software Installed**:
- **CUDA**: 12.2 (installed at /usr/local/cuda-12.6)
- **TensorRT**: 8.6.2.3-1+cuda12.2 (development libraries installed)
- **Rust**: 1.95.0 (installed via rustup)
- **System dependencies**: protobuf-compiler 3.12.4, nasm 2.15.05, cmake 3.22.1, libssl-dev, pkg-config, libclang-14

**Runner Configuration**:
- **Service name**: `actions.runner.MAGNA-Global-mi-isal.vanlindv60004`
- **Runner name**: vanlindv60004
- **Labels**: `self-hosted`, `Linux`, `ARM64`, `orin`
- **Version**: 2.334.0 (auto-updated from 2.314.1)
- **Work directory**: `~/actions-runner/_work`
- **User**: alexsch6 (uid: 1006)

**Environment Variables** (configured in `~/actions-runner/.env`):
```bash
CUDA_ROOT=/usr/local/cuda
TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu
TENSORRT_LIB=/usr/lib/aarch64-linux-gnu
PATH=/usr/local/cuda/bin:/home/alexsch6/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
```

**Systemd Service**:
- Auto-start enabled via systemd
- Status: Active and listening for jobs
- Connected to: https://github.com/MAGNA-Global/mi-isal

**Initial Testing**:
- Test workflow (`test-orin-runner.yml`) executed successfully
- Verified system info, Rust toolchain, NVIDIA environment, and repository checkout
- Runner responds to manual workflow_dispatch triggers

### 10.2 Corporate Proxy Configuration

**Issue**: Initial workflow runs failed with cargo timeout errors when accessing crates.io index due to corporate proxy requirements.

**Solution**: Added proxy environment variables to the runner's `.env` file:
```bash
HTTP_PROXY=http://ecsproxy2.van.magna.global:3128
HTTPS_PROXY=http://ecsproxy2.van.magna.global:3128
http_proxy=http://ecsproxyv.van.magna.global:3128
https_proxy=http://ecsproxyv.van.magna.global:3128
no_proxy=.van.magna.global,.magna.global
```

**Note**: The runner systemd service does not inherit shell proxy environment variables by default. Proxy settings must be explicitly added to `~/actions-runner/.env` for cargo and other tools to access external resources through the corporate network.

**Verification**: After adding proxy configuration and restarting the service (`sudo ./svc.sh stop && sudo ./svc.sh start`), cargo successfully downloads dependencies from crates.io.

### 10.3 Thor Runner Setup (VANDVL00018) - May 8, 2026

**Hardware Configuration**:
- Board: NVIDIA Jetson Thor (aarch64)
- Hostname: VANDVL00018
- OS: Ubuntu 24.04.4 LTS (Linux 6.8.12-tegra)
- Disk: 936GB total, 864GB available
- Memory: 122GB RAM

**Software Installed**:
- **CUDA**: 13.0 V13.0.48 (installed at /usr/local/cuda-13.0)
- **TensorRT**: 10.13.3.9-1+cuda13.0 (development libraries installed)
- **Rust**: 1.95.0 (installed via rustup)
- **System dependencies**: protobuf-compiler 3.21.12, nasm 2.16.01, cmake 3.28.3, libssl-dev, pkg-config, libclang-18-dev

**Runner Configuration**:
- **Service name**: `actions.runner.MAGNA-Global-mi-isal.VANDVL00018`
- **Runner name**: VANDVL00018
- **Labels**: `self-hosted`, `Linux`, `ARM64`, `thor`
- **Version**: 2.334.0 (auto-updated during initial start)
- **Work directory**: `~/actions-runner/_work`
- **User**: alexsch6 (uid: 1000)

**Environment Variables** (configured in `~/actions-runner/.env`):
```bash
CUDA_ROOT=/usr/local/cuda
TENSORRT_INCLUDE=/usr/include/aarch64-linux-gnu
TENSORRT_LIB=/usr/lib/aarch64-linux-gnu
PATH=/usr/local/cuda/bin:/home/alexsch6/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu
HTTP_PROXY=http://ecsproxy2.van.magna.global:3128
HTTPS_PROXY=http://ecsproxy2.van.magna.global:3128
http_proxy=http://ecsproxyv.van.magna.global:3128
https_proxy=http://ecsproxyv.van.magna.global:3128
no_proxy=.van.magna.global,.magna.global
```

**Systemd Service**:
- Auto-start enabled via systemd
- Status: Active and listening for jobs
- Connected to: https://github.com/MAGNA-Global/mi-isal

**Initial Testing**:
- Test workflow (`test-thor-runner.yml`) created for manual validation
- Runner responds to manual workflow_dispatch triggers
- **Corporate proxy configured from initial setup**: Based on lessons learned from Orin runner (see section 10.2), proxy environment variables were added to `.env` file during initial configuration, preventing cargo timeout errors from the start. No service restart required.

**Notable Differences from Orin**:
- Newer CUDA version (13.0 vs 12.2)
- Newer TensorRT version (10.13.3.9 vs 8.6.2.3)
- Ubuntu 24.04 LTS vs 22.04 LTS
- Significantly more RAM (122GB vs ~32GB typical for Orin)
- Larger storage capacity (936GB vs 1.8TB on Orin)

### 10.4 Known Limitations

1. **nvcc not in PATH**: CUDA compiler is installed but not automatically in PATH. Build scripts may need to source `/usr/local/cuda/bin` explicitly or rely on environment variables.
2. **Limited redundancy**: Currently two NVIDIA boards configured (Orin and Thor). No redundancy for hardware failures on TI or Qualcomm platforms yet.
3. **Manual trigger only**: Hardware validation workflow requires manual trigger initially; auto-trigger on push to main will be enabled after verification period.
4. **Version differences**: Orin runs CUDA 12.2/TensorRT 8.6.2.3, while Thor runs CUDA 13.0/TensorRT 10.13.3.9. This may expose version-specific compatibility issues.

### 10.5 Next Steps

1. ✅ Test full hardware validation workflow with actual middleware build on Orin
2. ✅ Verify cargo build succeeds with `--features nvidia` on Orin
3. ✅ Resolve corporate proxy configuration for crates.io access
4. ✅ Add Thor runner (completed May 8, 2026)
5. ⏳ Test Thor runner with manual workflow dispatch
6. ⏳ Monitor build times and resource usage on both NVIDIA boards
7. Enable automatic triggering on push to main after 1-week monitoring period
8. Add TI TDA4 runner when available
9. Add Qualcomm runner when hardware arrives

