# CLAUDE.md

Rust native kernels for GATK (a replacement for Intel GKL), exposed to Java through JNI. See README.md for the layout and build commands.

## Verification

Run all of these before considering work done: `cargo ci-fmt`, `cargo ci-lint`, `cargo ci-test`, `cargo ci-doc`, `cargo ci-deny` (aliases in `.cargo/config.toml`; they need `cargo-nextest` and `cargo-deny`), then `./gradlew build`. The Rust tests compare every SIMD backend against the scalar reference ports (`reference.rs`, `pd_reference.rs`, the aligner's `reference.rs`); the Java tests compare the JNI path against Java ports of the same. On Apple silicon the x86 backends can be tested under Rosetta with `cargo test --target x86_64-apple-darwin --release` (scalar and, where Rosetta exposes them, AVX paths); AVX-512 needs real hardware.

## Design constraints

- The public Java contract is `gatk-native-bindings`; keep implementing those interfaces so GATK needs only a dependency swap.
- No threading inside the library: every call computes on the calling thread. Parallelism belongs to the caller.
- Numerics must stay equivalent to GATK's Java implementation: double precision within 1e-9 in log10 space, single precision within 1e-4, with the same underflow policy as GKL (recompute below 1e-28).
- Kernel code that touches vector types must be `#[inline(always)]` so it compiles inside the `#[target_feature]` entry points (`kernel.rs`, `pd_kernel.rs`, the aligner's `diag.rs`); the x86 backends are only sound when instantiated there.
- Generate test data programmatically (`synthetic.rs`, the Java `Region` helper); do not commit data files.
