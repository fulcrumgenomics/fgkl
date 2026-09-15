# fgkl

Native compute kernels for GATK, written in Rust, shipped as one JAR with prebuilt libraries for Linux and macOS on x86_64 and aarch64. A replacement for the archived Intel Genomics Kernel Library (GKL).

The project name is a placeholder.

## Kernels

- **PairHMM** (`com.fulcrumgenomics.fgkl.pairhmm.FgklPairHmm`): implements `org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding`, so GATK can use it in place of `IntelPairHmm` / `IntelPairHmmOMP`. Results match GATK's Java `LoglessPairHMM` (double precision) and Intel GKL (single precision with double-precision recomputation of underflowing pairs). SIMD lanes hold different reads, the DP is swept row by row, and haplotypes sorted lexicographically share DP columns for their common prefix. Backends: scalar, NEON, AVX2, AVX-512, selected at runtime. Every call runs on the calling thread; run independent calls concurrently for parallelism.

- **Partially determined PairHMM** (`com.fulcrumgenomics.fgkl.pdhmm.FgklPdHmm`): implements `org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeBinding` in place of `IntelPDHMM` for HaplotypeCaller's `--dragen-378-concordance-mode`, where every likelihood goes through GATK's `LoglessPDPairHMM`. Haplotypes carry per-base flags for undetermined SNP alleles and deletions; the kernel resolves the flags once per haplotype into runs of columns with a fixed deletion state and dispatches each run to a specialised inner loop, so the per-cell cost matches the plain kernel. Same lane layout, backends and precision policy as the PairHMM; haplotype columns are not shared because the flags make each haplotype's DP its own.

- **Smith-Waterman** (`com.fulcrumgenomics.fgkl.smithwaterman.FgklSmithWaterman`): implements `SWAlignerNativeBinding` in place of `IntelSmithWaterman`, reproducing GATK's `SmithWatermanJavaAligner` exactly (gap model, tie-breaking, the four overhang strategies, CIGARs and offsets). The matrix is filled along anti-diagonals in SIMD lanes (NEON, AVX2, AVX-512) with one byte of traceback per cell: 16-bit lanes first, holding scores relative to the anti-diagonal so that every step is non-positive and saturation can be detected at the alignment end, and 32-bit lanes for the rare pair that saturates. Set `FGKL_SW_STATS` to get per-parameter-set call, cell, duplicate and fallback counts on exit.

## Layout

- `crates/pairhmm`: the PairHMM and partially determined PairHMM kernels (`fgkl-pairhmm`), scalar reference ports of GATK's algorithms, synthetic data generation, the `pairhmm-bench` and `pdhmm-bench` throughput benchmarks, and `pairhmm-replay`, which replays a GATK `--pair-hmm-results-file` dump.
- `crates/smithwaterman`: the Smith-Waterman aligner (`fgkl-smithwaterman`) with its reference port and `sw-bench`.
- `crates/jni`: the JNI cdylib (`libfgkl`).
- `src/main/java`: the Java API and native loader; `src/test/java`: tests against a Java port of the reference.
- `docs/literature-survey.md`: survey of published PairHMM and Smith-Waterman acceleration work.

## Building

Requires a Rust toolchain (pinned in `rust-toolchain.toml`) and JDK 17.

```
cargo test                    # kernel tests
cargo run --release --bin pairhmm-bench -- --reads 1000 --haps 32
./gradlew build               # builds the native library with cargo, then the JAR and Java tests
```

`./gradlew build` places the native library under `src/main/resources/native/<os>-<arch>/`. Set `FGKL_PLATFORM` (for example `linux-aarch64`) to cross-compile for another platform with the corresponding Rust target installed. Set the system property `fgkl.library.path` to load a library from a directory instead of the JAR.

Before submitting changes run `cargo fmt --all`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test` and `./gradlew build`.
