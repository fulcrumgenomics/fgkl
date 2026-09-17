[![CI](https://github.com/fulcrumgenomics/fgkl/actions/workflows/ci.yml/badge.svg)](https://github.com/fulcrumgenomics/fgkl/actions/workflows/ci.yml)
[![License](http://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/fulcrumgenomics/fgkl/blob/main/LICENSE)
[![Maven Central](https://img.shields.io/maven-central/v/com.fulcrumgenomics/fgkl)](https://central.sonatype.com/artifact/com.fulcrumgenomics/fgkl)
[![Language](https://img.shields.io/badge/language-rust-orange.svg)](https://github.com/fulcrumgenomics/fgkl)
[![Language](https://img.shields.io/badge/language-java-brightgreen.svg)](https://github.com/fulcrumgenomics/fgkl)
[![Javadoc](https://javadoc.io/badge/com.fulcrumgenomics/fgkl.svg)](https://javadoc.io/doc/com.fulcrumgenomics/fgkl)

# fgkl

Native compute kernels for GATK, written in Rust, packaged as one JAR for Linux, macOS and Windows on x86_64 and aarch64. A replacement for the archived [Intel Genomics Kernel Library (GKL)](https://github.com/Intel-HLS/GKL). MIT licensed; the reference implementations are ports of GATK code (Apache-2.0), see `NOTICE`.

<p>
<a href="https://fulcrumgenomics.com">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset=".github/logos/fulcrumgenomics-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset=".github/logos/fulcrumgenomics-light.svg">
  <img alt="Fulcrum Genomics" src=".github/logos/fulcrumgenomics-light.svg" height="100">
</picture>
</a>
</p>

[Fulcrum Genomics](https://www.fulcrumgenomics.com) - supporting the bioinformatics and computational biology community.

<a href="mailto:contact@fulcrumgenomics.com?subject=[GitHub inquiry]"><img src="https://img.shields.io/badge/Email_us-%2338b44a.svg?&style=for-the-badge&logo=gmail&logoColor=white"/></a>
<a href="https://www.fulcrumgenomics.com"><img src="https://img.shields.io/badge/Visit_Us-%2326a8e0.svg?&style=for-the-badge&logo=wordpress&logoColor=white"/></a>

## Performance

HaplotypeCaller on a representative 5.8 Mb shard of a 30x human WGS sample (HG00123 from NYGC 1KG), one JVM, wall / CPU seconds. fgkl always runs single-threaded on the calling thread; GKL's PairHMM used its default 4 OpenMP threads.

| mode, hardware | GKL | fgkl |
|---|---|---|
| `--dragen-mode`, Xeon 6975P-C (AVX-512), 4 vCPU | 97 / 250 | 90 / 123 |
| `--dragen-378-concordance-mode`, same | 104 / 315 | 77 / 105 |
| `--dragen-378-concordance-mode`, Apple M4 Max (NEON) | n/a (Java fallback: 282 / 296) | 49 / 66 |

Output is byte-identical to GATK's Java implementations in double precision, and to GKL's in DRAGEN 3.7.8 mode; in the default single precision a handful of records per shard differ from GKL in genotype-quality fields only.

## Kernels

- **PairHMM** (`com.fulcrumgenomics.fgkl.pairhmm.FgklPairHmm`): implements `org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding`, so GATK can use it in place of `IntelPairHmm` / `IntelPairHmmOMP`. Results match GATK's Java `LoglessPairHMM` (double precision) and Intel GKL (single precision with double-precision recomputation of underflowing pairs). SIMD lanes hold different reads, the DP is swept row by row, and haplotypes sorted lexicographically share DP columns for their common prefix. Backends: scalar, NEON, AVX2, AVX-512, selected at runtime. Every call runs on the calling thread; run independent calls concurrently for parallelism.

- **Partially determined PairHMM** (`com.fulcrumgenomics.fgkl.pdhmm.FgklPdHmm`): implements `org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeBinding` in place of `IntelPDHMM` for HaplotypeCaller's `--dragen-378-concordance-mode`, where every likelihood goes through GATK's `LoglessPDPairHMM`. Haplotypes carry per-base flags for undetermined SNP alleles and deletions; the kernel resolves the flags once per haplotype into runs of columns with a fixed deletion state and dispatches each run to a specialised inner loop, so the per-cell cost matches the plain kernel. Same lane layout, backends, precision policy and prefix sharing as the PairHMM, with sharing restricted to haplotypes that agree on bases, flags and row-end deletion state.

- **Smith-Waterman** (`com.fulcrumgenomics.fgkl.smithwaterman.FgklSmithWaterman`): implements `SWAlignerNativeBinding` in place of `IntelSmithWaterman`, reproducing GATK's `SmithWatermanJavaAligner` exactly (gap model, tie-breaking, the four overhang strategies, CIGARs and offsets). The matrix is filled along anti-diagonals in SIMD lanes (NEON, AVX2, AVX-512) with one byte of traceback per cell: 16-bit lanes first, holding scores relative to the anti-diagonal so that every step is non-positive and saturation can be detected at the alignment end, and 32-bit lanes for the rare pair that saturates. Setting the environment variable `FGKL_SW_STATS` (to any value) in the process that loads the library makes it print, at process exit, one line per scoring-parameter set with the number of calls, cells, distinct sequence pairs and 16-bit saturation fallbacks.

The backend is chosen at runtime from what the CPU reports: AVX-512 (F, plus BW for the aligner), then AVX2 with FMA, then NEON on aarch64, and otherwise the scalar kernels, which are the same algorithms without vector instructions and run on any x86_64 or aarch64 CPU. Every backend produces results within the documented tolerances of the others.

## Layout

- `crates/`: the Rust workspace.
  - `pairhmm`: the PairHMM and partially determined PairHMM kernels (`fgkl-pairhmm`), scalar reference ports of GATK's algorithms, synthetic data generation, the `pairhmm-bench` and `pdhmm-bench` throughput benchmarks, and `pairhmm-replay` / `pdhmm-replay`, which replay GATK `--pair-hmm-results-file` / `--pdhmm-results-file` dumps. The Rust type is `PdPairHmm` (after GATK's `LoglessPDPairHMM`), the Java class `FgklPdHmm` (after GKL's `IntelPDHMM` and GATK's `pdhmm` package), and the tools `pdhmm-*` (after GATK's flag).
  - `smithwaterman`: the Smith-Waterman aligner (`fgkl-smithwaterman`) with its reference port and `sw-bench`.
  - `jni`: the JNI cdylib (`libfgkl`), the only crate with a dependency.
- `src/main/java`: the Java API and native loader; `src/test/java`: tests against Java ports of the references.
- `docs/design.md`: how the kernels work and why.

## Building

Requires the Rust toolchain pinned in `rust-toolchain.toml` (rustup installs it on first use) and JDK 17.

```
cargo test                    # kernel tests on every backend this CPU supports
cargo run --release --bin pairhmm-bench -- --reads 1000 --haps 32
./gradlew build               # builds the native library with cargo, then the JAR and the Java tests
```

A local `./gradlew build` produces a JAR holding the host platform's library only. [CONTRIBUTING.md](CONTRIBUTING.md) covers the full gate set, testing the Linux libraries on this machine, and how the multi-platform JAR is built and released.
