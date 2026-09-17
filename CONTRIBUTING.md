# Contributing to fgkl

## Prerequisites

| Tool | Version | Install |
|---|---|---|
| Rust | the release pinned in `rust-toolchain.toml` (currently 1.98.1) | [rustup](https://rustup.rs): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh`. rustup installs the pinned toolchain automatically the first time `cargo` runs in the checkout. |
| JDK | 17 | Any distribution, e.g. [Temurin](https://adoptium.net) or `brew install --cask temurin@17`. Gradle compiles against the JDK 17 API through its toolchain support whatever JDK runs it. |
| Gradle | wrapper | `./gradlew`; nothing to install. |
| cargo-nextest, cargo-deny | latest | `cargo install cargo-nextest cargo-deny --locked`. Used by the `ci-test` and `ci-deny` gates. |

The pinned toolchain is always a current stable release and is bumped as releases come out; `Cargo.toml` additionally declares 1.89 as the oldest compiler the code is kept building with, which CI checks.

For the multi-platform build and testing (see below): [Docker](https://docs.docker.com/get-docker/) with `linux/amd64` emulation enabled (Docker Desktop has it on by default), [zig](https://ziglang.org) via `brew install zig`, and `cargo install cargo-zigbuild --locked`.

## Building and testing

```
cargo build --release          # kernels and developer tools
cargo ci-test                  # Rust tests on every backend this CPU supports
./gradlew build                # native library for the host, the JAR, and the Java tests
```

`./gradlew build` runs cargo itself (the `buildNative` task), so no separate `cargo build` is needed. It puts the host's native library in `build/native/<os>-<arch>/` and packages it at `native/<os>-<arch>/` inside the JAR. Set the system property `fgkl.library.path` to a directory to load a library from there instead of the JAR, which is how a rebuilt kernel is tried in GATK without repackaging.

The Rust tests compare every SIMD backend against scalar ports of GATK's Java implementations; the Java tests compare the JNI path against Java ports of the same. Each machine only exercises the backends its CPU has, so the full matrix is:

| backend | where it runs |
|---|---|
| scalar | everywhere |
| NEON | Apple silicon, Linux aarch64 (CI) |
| AVX2 | Linux x86_64 (CI), or a `linux/amd64` container on an Apple silicon machine (see below) |
| AVX-512 | Linux x86_64 with AVX-512 (current GitHub runners have it); not available under Docker emulation |

### Testing the Linux libraries on macOS

`scripts/test-linux.sh` cross-compiles the JNI library for Linux x86_64 and aarch64 and runs the Java test suite against each in a Docker container, which is also what the release script does before publishing:

```
scripts/test-linux.sh            # both architectures
scripts/test-linux.sh amd64      # one of them
```

The x86_64 container runs under emulation and exposes AVX2 but not AVX-512.

## Before opening a pull request

Run the gate set, which CI runs unchanged:

```
cargo ci-fmt && cargo ci-lint && cargo ci-test && cargo ci-doc && cargo ci-deny && ./gradlew build
```

Conventions: `rustfmt.toml` is authoritative for formatting; clippy runs with warnings denied; every public item has a doc comment; comments explain why, not what; tests are named after the behaviour they assert and generate their data in code. Numerics must stay equivalent to GATK's Java implementation (double within 1e-9 in log10 space, single within 1e-4, recomputing below 1e-28 in double as GKL does); `docs/design.md` explains the kernels.

Kernel code that touches vector types must be `#[inline(always)]`, because it is only sound when compiled inside the `#[target_feature]` entry points that instantiate it after runtime feature detection.

## Releasing

Releases must be made with user-supplied Sonatype credentials (`SONATYPE_USER`, `SONATYPE_PASS`) and, for non-snapshot versions, a PGP signing key (`PGP_SECRET`, `PGP_PASSPHRASE`) in the environment. `publish.sh` cross-compiles the native library for linux-x86_64, linux-aarch64, osx-x86_64, osx-aarch64 and windows-x86_64, assembles the JAR, runs the Java tests against the Linux libraries in Docker, and publishes to Maven Central. The Windows library is built but not tested by the script; the manually triggered `windows` workflow on GitHub runs the Java tests on a Windows runner when needed.

The version lives in `Cargo.toml` and is bumped and tagged by `cargo release`; Gradle reads the same version from the tag, and commits between tags publish as the next patch version with `-SNAPSHOT`.

```
cargo release patch --execute   # bump, commit, tag vX.Y.Z (nothing goes to crates.io)
git push && git push --tags
./publish.sh                    # build, test, publish: a release when HEAD is tagged, else a snapshot
./publish.sh --dry-run          # everything except the upload
```
