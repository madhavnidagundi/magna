# Contributing

Magna middleware follows a zero-warning policy for Rust changes. Every pull request is expected to pass formatting, clippy, and rustdoc checks before review or merge.

## Required Local Checks

Run these commands from `middleware/`:

```bash
cargo fmt --all --check
cargo clippy --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features
```

Validate the backend you are working on as well:

```bash
cargo clippy --features cpu --all-targets -- -D warnings
cargo clippy --features nvidia --all-targets -- -D warnings
cargo clippy --features qualcomm --all-targets -- -D warnings
cargo clippy --features ti --all-targets -- -D warnings
```

## Quick Fixes

If formatting fails, run:

```bash
cargo fmt --all
```

## Standards

- Remove unused imports, dead code, and redundant clones instead of silencing warnings by default.
- Prefer idiomatic Rust patterns that satisfy clippy without adding unnecessary complexity.
- Treat `#[allow(...)]` as a last resort, and include a clear reason when it is truly necessary.
- Keep backend-specific code feature-gated so the full lint matrix stays green.
- Do not commit generated logs, local benchmark outputs, or middleware build artifacts.

## CI Enforcement

Pull requests automatically run:

- `cargo fmt --all --check`
- `cargo clippy --all-features --all-targets -- -D warnings`
- `cargo clippy --features <backend> --all-targets -- -D warnings` for `cpu`, `nvidia`, `qualcomm`, and `ti`
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --all-features`
- `cargo audit` to detect dependency vulnerabilities
- `cargo deny check` to detect license, source, and bloat issues

## Unsafe Code

The middleware uses `unsafe` Rust to interface with C/C++ hardware SDKs (TensorRT, QNN, SNPE, TIDL). This is necessary but must be handled with discipline.

### When to Use `unsafe`

Only use `unsafe` when required by an FFI boundary, raw pointer operation, or a trait implementation (`Send`/`Sync`) where you can manually verify correctness. Never use `unsafe` to work around a borrow checker error.

### Required: `// SAFETY:` Comments

**Every single use of `unsafe` must be preceded by a `// SAFETY:` comment** explaining why the code is safe. This applies to:

- `unsafe impl Send` / `unsafe impl Sync`
- `unsafe { ... }` blocks
- `unsafe fn` definitions

Each comment must address:
1. **Invariants** — What preconditions does the unsafe code rely on?
2. **Justification** — Why are those preconditions guaranteed to hold here?
3. **Assumptions** — Any assumptions about the behaviour of external C libraries.

```rust
// SAFETY: `ctx.ptr` is a valid initialized TensorRT engine context.
// The `input.data` and `output_data` slices are valid memory allocations,
// passed with their exact lengths. The `RwLock` read guard on `state` ensures
// that `ctx` is not modified or dropped during this inference call.
let result = unsafe { trt_infer(ctx.ptr, input.data.as_ptr(), ...) };
```

A safety comment must explain *why* the code is safe, not just *what* it does.

### Common FFI Safety Considerations

When documenting FFI calls, address:

- **Pointer validity**: Is the pointer non-null and pointing to valid, initialized memory?
- **Lifetime/ownership**: Who owns the memory? When is it freed?
- **Thread safety**: Can the C function be called concurrently?
- **Preconditions**: What must hold before the call? (e.g. "C string is null-terminated")
- **Error handling**: Is the return value checked before use?

### Further Reading

- [Rustonomicon](https://doc.rust-lang.org/nomicon/) — The Dark Arts of Unsafe Rust
- [Rust API Guidelines: C-UNSAFE-DOC](https://rust-lang.github.io/api-guidelines/documentation.html#c-unsafe-doc)

---

## Security Vulnerabilities and Policy Violations

We enforce supply chain security using `cargo-audit` and `cargo-deny`. If the CI fails due to a security scan, you must resolve it before merging:

1. **Vulnerabilities (`cargo audit`)**: Update the vulnerable dependency to a safe version. If a direct update is not possible, try using `cargo update -p <crate_name>` to pull a patched semver-compatible version.
2. **Policy Violations (`cargo deny check`)**: 
   - **Licenses**: If a new dependency uses a disallowed license (e.g., GPL-3.0), find an alternative crate. 
   - **Multiple Versions**: If there are multiple versions of the same crate in the tree, attempt to unify them by updating your `Cargo.toml`.
   - **Exceptions**: In rare cases where a vulnerability is proven to be a false positive or completely inapplicable to our use case, it may be added to `[advisories.ignore]` in `middleware/deny.toml` with a detailed comment explaining the accepted risk.
