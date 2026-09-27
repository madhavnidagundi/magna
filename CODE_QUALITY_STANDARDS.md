# Code Quality Standards

This document outlines the strict code quality standards expected in the Magna Middleware codebase. All contributions must adhere to these policies.

## Unsafe Code Documentation Policy

The middleware interfaces directly with C/C++ hardware SDKs (TensorRT, QNN, SNPE, TIDL) which inherently requires `unsafe` Rust. We maintain strict control over unsafe code to guarantee memory safety, thread safety, and maintainability.

### 1. Enforcement

- `#![deny(unsafe_op_in_unsafe_fn)]`: All `unsafe fn` bodies must explicitly use `unsafe {}` blocks for unsafe operations.
- `#![warn(clippy::undocumented_unsafe_blocks)]`: Every `unsafe` block or `unsafe impl` must have a corresponding `// SAFETY:` comment immediately preceding it. This is enforced as an error on CI via `-D warnings`.

### 2. Comment Requirements

Every `// SAFETY:` comment must explicitly detail:
1. **The Invariants**: What preconditions are required by the unsafe code?
2. **The Justification**: Why are those preconditions guaranteed to be met in this exact context?
3. **Assumptions**: Any assumptions about external C libraries or state.

A valid safety comment explains *why* the code is safe, not just *what* it does.

Example:
```rust
// SAFETY: `ctx.ptr` is a valid initialized TensorRT engine context.
// The `input.data` and `output_data` slices are valid memory allocations, passed with their
// exact lengths. The `RwLock` read guard on `state` ensures that `ctx` is not modified or
// dropped during this inference call.
let result = unsafe {
    trt_infer(
        ctx.ptr,
        input.data.as_ptr(),
        input.data.len(),
        output_data.as_mut_ptr(),
        output_bytes,
    )
};
```
