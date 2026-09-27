# Qualcomm QNN / Radxa Dragon Q6A Deployment Guide

This is the single reference for deploying and running ONNX models on Qualcomm
Snapdragon NPUs through this project — specifically validated against the
**Radxa Dragon Q6A** (Qualcomm QCS6490, Hexagon v68 HTP DSP), via the
**Qualcomm AI Runtime (QAIRT/QNN)**.

It supersedes `QUALCOMM_PIPELINE.md`, `RADXA_BOARD_DETAILS.txt`,
`qcboradsetup.txt`, `RADXA_NEXT_STEPS.md`, and `issues.txt`, which have been
consolidated here (see **"What was removed and why"** at the bottom).

---

## Architecture

Qualcomm deployment requires a strict separation between **optimization**
(portable, can run anywhere) and **native context generation** (must run on
the target board, because it depends on the exact DSP firmware and VTCM
layout of that device). The pipeline has five stages:

```
Host (x86 + Docker)                          Radxa board
────────────────────                         ───────────
 .onnx
   │  qairt-converter
   ▼
 FP32 .dlc
   │  qairt-quantizer
   ▼
 INT8 .dlc  ───── deploy (gRPC/SCP) ────────►  INT8 .dlc
                                                  │  qnn-context-binary-generator
                                                  ▼
                                                Native .bin (hardware-specific)
                                                  │  qnn-net-run / magna_server
                                                  ▼
                                                HTP inference
```

1. **ONNX** — given as input to this pipeline; not produced by it. Exporting
   an ONNX model from a training framework (e.g. PyTorch) is an external,
   upstream step and out of scope for this document.
2. **Dockerized FP32 conversion** (host) — `.onnx` → intermediate `_fp.dlc`.
3. **Dockerized INT8 quantization** (host) — → portable `_int8.dlc`.
4. **Deployment** (host → board) — transfer the `.dlc`, via gRPC or SCP.
5. **Native context generation** (on board) — `.dlc` → hardware-specific
   `.bin`, then execute on the Hexagon HTP.

---

## 1. Prerequisites

The Qualcomm SDK is proprietary and the target is physical hardware, so these
steps can't be automated by Docker — do them manually first:

1. **Download the QAIRT SDK** (requires a Qualcomm account). This pipeline
   has been validated against **QAIRT 2.47.0.260601**. Create a workspace and
   unzip the SDK into it:

   ```bash
   mkdir -p /opt/qairt_workspace/qairt
   unzip v2.47.0.260601.zip -d /opt/qairt_workspace/qairt/
   # → /opt/qairt_workspace/qairt/2.47.0.260601
   ```
2. **Prepare calibration data** — INT8 quantization needs raw calibration
   images (or `.raw` tensors matching your model's expected input encoding):

   ```bash
   mkdir -p /opt/qairt_workspace/<your_project>/calib/
   ```
3. **Provision the target board** — the Radxa Dragon Q6A must be powered on
   and reachable via SSH. See [§4](#4-board-one-time-workspace-setup) for the
   exact on-board layout and environment this pipeline expects. There is no
   fixed provisioning path baked into the tooling — set it up once per board
   and keep the environment variables in `~/.bashrc` (§4).
4. **Generating large evaluation datasets (optional)** — for full-dataset
   accuracy runs (e.g. the ImageNet validation set), host-side disk
   constraints can be bypassed by running `scripts/eval/colab_generator.py`
   in Google Colab, then SCPing the resulting archive directly to the board.

---

## 2. Host: Build the optimizer Docker image

```bash
cd ~/mi-isal
docker build -t qualcomm-optimizer -f docker/Dockerfile.qualcomm-optimizer .
```

This builds a lightweight, CPU-only container (no CUDA bloat) whose
entrypoint runs `qairt-converter` followed by `qairt-quantizer`
(`scripts/quantization/build_qnn_int8.py`).

## 3. Host: Convert ONNX → INT8 `.dlc`

### Option A — `misal optimize` (recommended)

```bash
cd ~/mi-isal
export QAIRT_SDK_ROOT=/opt/qairt_workspace/qairt/2.47.0.260601   # or QNN_SDK_ROOT

cargo run --release --bin misal -- optimize \
    --model models/your_model.onnx \
    --hw radxa-q6a \
    --precision int8 \
    --input-dim input:1,3,224,224 \
    --calibration-data /opt/qairt_workspace/<your_project>/calib/ \
    --num-images 100 \
    --output models/your_model.dlc
```

- `--input-dim NAME:DIMS` is **required** — there is no default input shape;
  state the exact dimensions for your model. Repeat the flag for models with
  multiple inputs.
- `--num-images` bounds how many calibration images are used (optional).
- The container runs as your host UID/GID, so output files aren't
  root-owned.
- The Dockerized quantizer currently only supports `--precision int8` (it
  always runs the converter + quantizer together — there's no FP-only mode).

### Option B — manual Docker invocation

```bash
docker run --rm \
    -v $(pwd):/workspace \
    -v /opt/qairt_workspace/qairt/2.47.0.260601:/opt/qairt \
    -w /workspace \
    --user "$(id -u):$(id -g)" \
    qualcomm-optimizer \
    --onnx /workspace/models/your_model.onnx \
    --dataset /workspace/<calib_dir> \
    --output_dlc /workspace/models/your_model_int8.dlc \
    --input_dim input 1,3,224,224
```

---

## 4. Board: One-time workspace setup

**There is no single fixed workspace path** — this pipeline has moved
locations more than once as boards were reprovisioned (`~/qc_magna_final` →
`~/qc_magna_final_v247` → `~/middleware`), and `magna_server` resolves
`./bin/`, `./lib/`, `./models/`, `./generated_ctx/` **relative to its own
working directory**, wherever that is. Pick one directory, be consistent, and
run everything from it. The layout itself is fixed:

```text
<workspace>/
├── bin/
│   ├── qnn-net-run
│   └── qnn-context-binary-generator
├── lib/
│   ├── libQnnHtp.so
│   ├── libQnnHtpPrepare.so
│   ├── libQnnModelDlc.so
│   ├── libQnnHtpV68Stub.so
│   ├── libQnnSystem.so
│   └── libQnnCpu.so            # CPU backend, for isolating quant-math bugs — see §11
├── dsp/
│   └── libQnnHtpV68Skel.so     # Hexagon v68 DSP skeleton
├── models/                     # deployed .dlc files
├── generated_ctx/              # on-device-generated .bin context binaries
└── eval_data/                  # raw tensors + labels for accuracy evaluation
```

Copy the executables/libraries from the SDK:

```bash
cd <workspace>
mkdir -p bin lib dsp models generated_ctx

SDK=<path-to-qairt-sdk-on-board>/2.47.0.260601
LIBDIR=$SDK/lib/aarch64-oe-linux-gcc11.2
BINDIR=$SDK/bin/aarch64-oe-linux-gcc11.2

cp $BINDIR/qnn-net-run $BINDIR/qnn-context-binary-generator bin/
cp $LIBDIR/libQnnCpu.so $LIBDIR/libQnnHtp.so $LIBDIR/libQnnHtpPrepare.so \
   $LIBDIR/libQnnModelDlc.so $LIBDIR/libQnnHtpV68Stub.so $LIBDIR/libQnnSystem.so lib/
cp $SDK/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so dsp/

chmod +x bin/qnn-net-run bin/qnn-context-binary-generator
```

`libQnnSystem.so` and `libQnnModelDlc.so` are both **required**, not
optional — the FFI code `dlopen()`s `libQnnHtp.so` and `libQnnSystem.so` by
bare name, resolved via `LD_LIBRARY_PATH`.

### Required environment variables

```bash
export QAIRT_SDK_ROOT=$SDK                       # or QNN_SDK_ROOT
export LD_LIBRARY_PATH=<workspace>/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="<workspace>/dsp;/usr/lib/rfsa/adsp;/usr/lib/rfsa/cdsp"
export CDSP_LIBRARY_PATH=/usr/lib/rfsa/cdsp
export QNN_ENABLE_DSP=1
export QNN_HTP_FORCE_PD_SESSION=3
```

`ADSP_LIBRARY_PATH`/`CDSP_LIBRARY_PATH`/`QNN_HTP_FORCE_PD_SESSION` are
**confirmed required, not optional** — without them,
`qnn-context-binary-generator` fails with `DspTransport.openSession qnn_open failed` / `Failed to load skel, error: 1002`, even when basic
DSP/driver sanity checks (`fastrpc_test -a v68`) pass fine. `LD_LIBRARY_PATH`
only helps the ARM-side linker find `libQnnHtp.so`; the Hexagon skeleton
(`libQnnHtpV68Skel.so`) is located via FastRPC's own separate search
mechanism, which is what `ADSP_LIBRARY_PATH` controls.

**Make these permanent** — a dropped SSH session loses anything only
`export`ed inline for that shell. Append to `~/.bashrc` on the board instead
of re-exporting every session.

---

## 5. Board: Generate the native context binary

```bash
cd <workspace>
./bin/qnn-context-binary-generator \
    --model ./lib/libQnnModelDlc.so \
    --backend ./lib/libQnnHtp.so \
    --dlc_path ./models/your_model.dlc \
    --binary_file your_model_native \
    --output_dir ./generated_ctx
```

Produces `generated_ctx/your_model_native.bin`. This binary is **hardware/
runtime-specific** — it must always be generated on the actual target board,
never copied from another device or SDK version.

**Note:** `.dlc` files cannot be loaded directly by `magna_server` — the
native FFI (`qnn_load_context`) only calls `QnnContext_createFromBinary` on
an already-compiled context binary; there is no `libQnnModelDlc.so`
graph-composition path in the FFI. `magna_server` calls `load_engine()`
synchronously at startup and **exits if it fails**, so a raw `.dlc` passed as
`--model` will make the server fail to start. The context binary must exist
*before* `magna_server` is started for the first time — the gRPC
`PrepareContext` RPC (§7) can only refresh/add models once a server is
already up and holding a working engine.

---

## 6. Board: Run `magna_server`

```bash
cd <workspace>
nohup ./magna_server \
    --model generated_ctx/your_model_native.bin \
    --precision int8 \
    --address 0.0.0.0:50051 \
    > magna_server.log 2>&1 &
disown
tail -20 magna_server.log   # expect "Listening for gRPC connections"
```

Run it **detached** (`nohup ... &` + `disown`, or `tmux`/`screen`) — running
in the foreground means a dropped SSH connection kills the server.

There is **no hot-swap RPC**: `infer` always uses whatever engine was loaded
at startup, so switching to a different model/context binary means
restarting the server pointed at the new one:

```bash
pkill -f magna_server   # graceful SIGTERM — see the wedged-DSP note in §12, don't use -9
# ...then start it again with the new --model path
```

Verify the server is reachable via `HealthCheck()` — expect `ready: true`,
`backend: "qualcomm"`, and the loaded model's name.

---

## 7. Remote deploy without SSH

Once `magna_server` is running, you can deploy new models and generate their
context binaries entirely over gRPC — no SCP/SSH needed for these two steps
(only for starting/restarting the server itself, §6):

```bash
# Upload a .dlc to the board (auto-routes to models/ by extension)
./target/release/misal deploy models/your_model.dlc \
    --server http://<board-host>:50051

# Trigger on-device context binary generation
./target/release/misal prepare-context your_model.dlc \
    --server http://<board-host>:50051
```

`prepare-context` produces `<workspace>/generated_ctx/your_model_native.bin`
on the board. Restart `magna_server` (§6) against it to make it active.

---

## 8. Running inference

### Via gRPC (`misal infer`)

```bash
./target/release/misal infer <input.raw> \
    --server http://<board-host>:50051 \
    --shape 1,3,224,224 \
    --quant-scale <scale> --quant-offset <offset> \
    --output /tmp/output.raw
```

`--shape` has no default — state your model's exact input dimensions.
`--quant-scale`/`--quant-offset` are optional and only needed if your input
file is float32 and the model expects quantized uint8 input (see §9 for how
to find these values for your model). Omit both to send the file's bytes
as-is at `--precision` (default `FP32`).

### In-process, with accuracy + OpenTelemetry metrics (`eval_50k`)

For batch accuracy evaluation against a labeled dataset, `eval_50k` calls the
middleware directly in-process (no gRPC hop) and reports running top-1/top-5
accuracy as Prometheus metrics on the same `:9090/metrics` endpoint
`magna_server` uses:

```bash
./eval_50k --model generated_ctx/your_model_native.bin \
    --raw-dir eval_data/raw --labels eval_data/labels.txt \
    --shape 1,3,224,224 --quant-scale <scale> --quant-offset <offset>
```

`--shape` falls back to the shape reported by the loaded engine if omitted;
`--quant-scale`/`--quant-offset` behave the same as `misal infer`. Nothing in
this tool is hardcoded to a specific model or dataset size — it works with
any labeled single-label classification set laid out as
`eval_<index>.raw` + `labels.txt`.

**Only one process can hold a DSP session at a time** — stop `magna_server`
before running `eval_50k` against the same model, and vice versa.

View live metrics from a second session: `curl -s localhost:9090/metrics | grep eval_`. To view from the host instead of curling on-board, set
`OTEL_METRICS_HOST=0.0.0.0` before starting `eval_50k` (default `127.0.0.1`
only binds loopback).

---

## 9. Finding your model's quantization parameters

INT8 `.dlc` models encode inputs/outputs as `q = round(x / scale) - offset`,
clamped to `[0, 255]`. **These values are per-model, per-calibration-run —
never assume or reuse a value from a different model.** Query them directly
from your `.dlc`:

```bash
$QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qairt-dlc-info --input_dlc your_model.dlc
```

Look at the input/output tensor encoding table. For example, one validated
run against a MobileNetV4 INT8 `.dlc` reported:

```text
Input Name  | Dimensions   | Type    | Encoding
input       | 1,3,224,224  | uFxp_8  | scale 0.018658448011, offset -114.000000000000

Output Name | Dimensions  | Type    | Encoding
output      | 1,1000      | uFxp_8  | scale 0.096628904343, offset -192.000000000000
```

That table is specific to that one model/calibration run — re-query it for
every new `.dlc` you deploy.

Two things worth knowing about this encoding:

- **No dequantization is needed to rank outputs.** Since dequantize is
  `x = (q + offset) * scale` with a positive `scale` applied uniformly
  across every class, `argmax`/`argsort` on the raw quantized bytes gives
  the identical ranking as on dequantized values. Both `misal infer` and
  `eval_50k` rely on this.
- **Check the `datatype` field against the byte length.** Current code
  reads the real output encoding from the loaded context
  (`qnn_get_output_precision` → `QualcommAdapter` reports `INT8` for an
  8-bit fixed-point output, `FP32` for a float output) and sizes the
  response buffer to match. A `magna_server` built **before** that support
  (QNN output-precision detection landed 2026-08-27) mislabels every output
  as `FP32` and returns a `numel × 4`-byte buffer holding `numel` uint8
  values plus zero padding — so if `datatype` says `FP32` but the byte
  length is `4 × numel` with a zeroed tail, read the first `numel` bytes as
  uint8 and rebuild the server. `qnn_execute_graph` now also rejects a
  client buffer whose size disagrees with the context tensor, so a
  precision misread fails loudly instead of silently returning garbage.

---

## 10. Profiling / latency

```bash
./bin/qnn-net-run \
    --backend ./lib/libQnnHtp.so \
    --retrieve_context ./generated_ctx/your_model_native.bin \
    --input_list ./eval_data/input_list.txt \
    --profiling_level detailed
```

This produces `qnn-profiling-data.log` under `output/`.

- Quick on-board check: `grep -i "execute" output/qnn-profiling-data.log`
- Detailed microsecond-level table: fetch the log to the host and run the
  SDK's viewer (path must match your SDK version):

  ```bash
  scp <board>:<workspace>/output/qnn-profiling-data_0.log ./
  $QAIRT_SDK_ROOT/bin/x86_64-linux-clang/qnn-profile-viewer \
      --input_log qnn-profiling-data_0.log
  ```

  Look at **Execute Stats (Average) → Backend (Accelerator execute time)**
  for the absolute hardware-level NPU latency.

---

## 11. `.dlc` vs. context binary — and isolating quantization bugs from NPU bugs

- **`.dlc` (portable)** — used with `--model libQnnModelDlc.so --dlc_path model.dlc`. Compiles the graph at runtime on every launch (slower startup,
  fully portable across devices of the same SoC family).
- **Context binary (hardware-specific, `.bin`)** — used with
  `--retrieve_context model.bin`. Pre-compiled, serialized DSP graph; much
  faster startup/execution, but tied to the exact hardware/runtime it was
  generated on (§5). This is the production deployment format.

If accuracy looks wrong, isolate **quantization math** from **NPU hardware
behavior** by running the same `.dlc` on the pure-software CPU backend
(`libQnnCpu.so`) instead of the HTP:

```bash
./bin/qnn-net-run \
    --backend ./lib/libQnnCpu.so \
    --model ./lib/libQnnModelDlc.so \
    --dlc_path ./models/your_model.dlc \
    --input_list ./eval_data/input_list.txt \
    --output_dir ./output_cpu \
    --use_native_output_files
```

If CPU-backend results also look wrong, the bug is in quantization/input
encoding (see §9), not the DSP/HTP path.

---

## 12. Known failure modes

**Wedged DSP session after an abrupt kill.** If `magna_server` (or
`qnn-context-binary-generator`) is killed abruptly (e.g. a dropped SSH pipe)
instead of shutting down gracefully, the *next* run can fail with
`DspTransport.openSession qnn_open failed` / `Failed to load skel, error: 1002` — even with every environment variable correct and no ARM-side process
still holding anything (a clean `ps`). This is a stuck Hexagon DSP-side
process-domain (PD) session, not visible or fixable from userspace.

**Fix: reboot the board.** No other recovery was found necessary (no kernel
module reload, no dmesg access even available as non-root). Prefer Ctrl-C /
SIGTERM over closing the terminal or letting the SSH connection drop, to
avoid triggering this in the first place.

---

## 13. Scope: single input / single output only

The native QNN wrapper (`middleware/src/backends/qualcomm/qnn_c_api.cpp`)
currently supports context binaries with **exactly one input tensor and one
output tensor per graph**. This is enforced, not merely assumed:

- `qnn_load_context()` rejects any binary whose `numGraphInputs != 1` or
  `numGraphOutputs != 1` and fails the load. `load_engine()` on a
  multi-input/output `.bin` therefore returns
  `MiddlewareError::EngineLoadFailed` — it does **not** load and then produce
  wrong results.
- `QualcommAdapter::infer()` additionally rejects any call with more than one
  input `TensorBuffer` (`MiddlewareError::InferenceFailed`), so a caller that
  passes extra tensors gets a clear error instead of having `inputs[1..]`
  silently dropped.

**Practical consequence:** the Qualcomm backend cannot currently serve
multi-input fusion models or multi-output detection models. Export such a
model as single-input/single-output, or split it, before deploying it here.

Lifting this restriction means reworking the C ABI (arrays of buffer
pointers/sizes instead of the single-pointer form) and the Rust
`EngineInfo`/`TensorBuffer` plumbing in `adapter.rs`, then validating against
a real multi-tensor model on hardware. Tracked in issue #54.

Regression coverage includes request-count validation in
`middleware/src/inference/validation.rs` and the
`info.inputs.len() == 1` / `info.outputs.len() == 1` assertions in
`smoke_test_qualcomm_native_load_infer_release`
(`middleware/tests/smoke_tests.rs`, real hardware).

---

## 14. Runtime contract and metadata discovery

The Qualcomm gRPC path validates request tensors before they reach the QNN FFI
boundary. For the current single-input/single-output QNN wrapper, the server
rejects:

- missing or duplicate input tensors;
- empty input names;
- unknown datatypes;
- non-positive dimensions;
- rank above the wrapper limit;
- shape, precision, name, or byte-size mismatches against the loaded context
  metadata.

Clients should call `GetModelInfo` after connecting and use the returned input
binding name, shape, and precision. The Rust client and `misal infer` follow
that flow, so Qualcomm requests no longer need to assume the binding is named
`input`.

`GetModelInfo` also reports Qualcomm backend capabilities. These are the
implemented middleware limits, not Qualcomm's theoretical QNN feature set:
currently one input, one output, concrete context-binary shapes, and no
middleware-side engine building.

---

## 15. Qualcomm regression checks

### CI runner configuration

The Radxa workflow runs on the machine registered with the `radxa` label. It
does not install the proprietary SDK or connect to a separate board over SSH.
The SDK, runtime libraries, DSP access, and model must be available to that
runner's service account. A successful SDK-less build is not hardware validation.

In the repository running the workflow, open **Settings → Secrets and variables
→ Actions → Variables**, and configure:

| Variable | Required value |
|---|---|
| `QAIRT_SDK_ROOT` (or legacy `QNN_SDK_ROOT`) | Absolute installed SDK directory containing `include/QNN` |
| `TEST_QNN_CONTEXT_PATH` | Absolute path to a real, board-compatible context `.bin` generated as described in §5 |

Use actual paths on the runner, not the illustrative paths in this guide. These
variables select existing installations; they do not download an SDK or model.
Repository SDK configuration takes precedence over runner-service SDK settings,
with `QAIRT_SDK_ROOT` preferred over `QNN_SDK_ROOT` within each source. Without
repository variables, the workflow retains the service environment. Both SDK
variable names are normalized to the selected root for subsequent build/runtime
steps. The model variable similarly falls back to the service environment.

Alternatively, configure these values in the Actions runner's service environment
(for example, its runner `.env` file) and restart the service after changes. Exports
in an interactive terminal or `~/.bashrc` alone do not configure a running service.
Keep the required `LD_LIBRARY_PATH` and `ADSP_LIBRARY_PATH` from §4 in the runner
service environment as well, using absolute paths accessible to its account.

If validation stops at **Verify QNN SDK headers are present**, inspect the SDK
variables printed by **Setup environment**. Empty values mean configuration is
missing; a nonexistent `include/QNN` means the selected installation is wrong or
inaccessible. Fix provisioning and rerun the workflow; do not remove the native
header gate. Passing that gate only allows native compilation to start, and does
not by itself prove runtime or inference success.

### Local and hardware checks

On a host without the SDK, run:

```bash
cargo test -p magna-middleware --features qualcomm --lib
cargo clippy --features qualcomm --all-targets -- -D warnings
```

These validate the no-SDK error path and Rust request validation. They do not
prove native compilation or inference. The header-availability smoke test is
intentionally a failure without the SDK and is reserved for configured runners.

On a configured Radxa runner, set `QAIRT_SDK_ROOT` or `QNN_SDK_ROOT` and the
runtime library paths described above. Set `TEST_QNN_CONTEXT_PATH` to a real
context binary for this board, then run:

```bash
cargo test -p magna-middleware --release --features qualcomm,opentelemetry --test smoke_tests smoke_test_qualcomm -- --nocapture --test-threads=1
```

Run the hardware tests serially and stop other DSP clients first. The native
configuration test checks the compiled `have_qnn_headers` cfg; do not search
normal Cargo console output for build-script directives. The Radxa workflow
uses `cargo run` for the server startup check so workspace and custom Cargo
target directories are handled consistently.

`smoke_test_qualcomm_native_failed_reload_preserves_engine` starts with a loaded,
allocated engine and successful inference, attempts three replacements with an
existing invalid context file, and verifies retained metadata/readiness and
successful inference without reallocating the original engine's buffers. This
tests transactional failed replacement rather than only load/release cycles.
The model path is required; native tests do not silently skip missing assets.

These are lifecycle checks, not numerical reference validation. Issue #54's
comparison against trusted outputs for multiple distinct inputs still requires
model-specific fixtures and hardware results.

---

## What was removed and why

The following files documented the same pipeline but had drifted out of
date; their content has been merged in above and they've been removed to
avoid future contributors following stale instructions:

- **`RADXA_BOARD_DETAILS.txt`** and **`qcboradsetup.txt`** — both documented
  on-board workspace layouts (`~/qc_magna_final` and `~/qc_magna_final_v247`
  respectively) that **no longer exist**; the board's SD card was reset and
  neither survived. The durable parts (DLC-vs-context-binary explanation,
  CPU-backend quant-isolation trick, env var list) are folded into §4/§11
  above, generalized away from the dead paths.
- **`RADXA_NEXT_STEPS.md`** — a chronological debugging log for one specific
  validation session. Its durable findings (required env vars, the wedged-
  DSP failure mode, the quantization-encoding investigation, the
  `datatype`-field caveat) are folded into §4, §9, and §12 above. Its
  session narrative and now-resolved bug writeups (a Docker image tag
  mismatch and an entrypoint-argument mismatch in `misal optimize --hw radxa-q6a`) are **not carried forward — both are already fixed** in
  the current `misal/src/optimize.rs` and `docker/Dockerfile.qualcomm- optimizer` (confirmed by reading the current code: the image is tagged
  `qualcomm-optimizer` consistently, and `build_qualcomm` calls the image's
  Python entrypoint directly with `--onnx`/`--dataset`/`--output_dlc`, not
  the old bash-wrapper invocation the bug report described).
- **`issues.txt`** — a tracked-issues list. **Every Qualcomm-specific item on
  it is already fixed** in the current code, confirmed by direct inspection:
  `qnn_execute_graph()` now calls the real `QnnGraph_execute` (it was a
  no-op); the QAIRT SDK path is resolved dynamically via
  `QAIRT_SDK_ROOT`/`QNN_SDK_ROOT` (it was hardcoded to `2.46.0.260424`); the
  path-to-`CString` conversion uses proper error handling (it used to
  `.unwrap()` and could panic); tensor shapes are extracted dynamically from
  the loaded engine (they were hardcoded to `[1,3,224,224]`/`[1,1000]`); and
  the Qualcomm module is properly `#[cfg(feature = "qualcomm")]`-gated. None
  of it reflects the current codebase, so it was not merged forward.
- **`QUALCOMM_PIPELINE.md`** — superseded by this file (same scope, now
  consolidated with the board-setup and troubleshooting material above).

**Not touched:** `middleware/requirements/qualcomm.txt` — this is also
stale (it lists SNPE, Hexagon SDK, `onnxruntime-qnn`, and Android NDK, none
of which the actual Docker+QAIRT pipeline above uses), but it's one of four
sibling per-backend requirement files (`generic.txt`, `nvidia.txt`,
`qualcomm.txt`, `ti.txt`), so I left it as-is rather than editing it
unilaterally — flagging it here for a decision on whether to update or
remove it.
