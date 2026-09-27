# Changelog

Notable changes to fgkl, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-27

First release: Rust kernels for GATK behind the `org.broadinstitute:gatk-native-bindings:1.1.0` interfaces, so GATK can replace Intel GKL with a dependency swap.

### Added

- `FgklPairHmm` (`PairHMMNativeBinding`): the PairHMM in single precision, recomputing underflowing pairs in double as GKL does, or in double precision; haplotypes share DP columns for their common prefix.
- `FgklPdHmm` (`PDHMMNativeBinding`): the partially determined PairHMM used by HaplotypeCaller's `--dragen-378-concordance-mode`.
- `FgklSmithWaterman` (`SWAlignerNativeBinding`): an exact port of GATK's Java aligner, filling anti-diagonals in 16-bit lanes with a 32-bit fallback.
- Scalar, NEON, AVX2 and AVX-512 backends chosen at runtime; every call runs on the calling thread.
- One JAR with native libraries for Linux (x86_64 and aarch64, glibc 2.17), macOS (x86_64 and aarch64) and Windows (x86_64, built but not yet tested in CI).

[Unreleased]: https://github.com/fulcrumgenomics/fgkl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/fulcrumgenomics/fgkl/releases/tag/v0.1.0
