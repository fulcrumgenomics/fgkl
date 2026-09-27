# Changelog

Notable changes to fgkl, newest first. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-27

### Changed

- PairHMM: haplotypes that share a suffix now share work through a backward pass computed once per shared suffix, roughly halving PairHMM kernel time on HaplotypeCaller's assembly regions (1.8-2.4x across NEON, AVX2 and AVX-512). Results stay within tolerance but are not bit-identical to 0.1.0: single-precision likelihoods move by at most 1.7e-6 log10, double-precision ones by at most 6e-14, and no pair enters or leaves the double-precision recomputation.
- PairHMM: the kernel is a further 20-30% faster with bit-identical results. Per-row and per-batch overhead is gone (haplotype bases are validated once per sweep, rows are no longer split at shared-prefix columns, the suffix join vectorises), a column's match, insertion and deletion values are stored together, and AVX-512 runs 16-lane batches, with a region's last few reads on AVX2 lanes, instead of 32-lane ones. Replaying HaplotypeCaller's regions, kernel time drops 21-25% on NEON, 21-32% on AVX2 and 22-29% on AVX-512, except 6-13% on centromeric regions with AVX-512. The AVX-512 backend now also requires AVX2 and FMA.
- Smith-Waterman: the anti-diagonal fill is 1.3-2.2x faster (2.2x on NEON, 1.6x on AVX2, 1.3-1.6x on AVX-512), checking bounds once per anti-diagonal and no longer clearing the traceback buffer on every call. Alignments are unchanged.

## [0.1.0] - 2026-09-27

First release: Rust kernels for GATK behind the `org.broadinstitute:gatk-native-bindings:1.1.0` interfaces, so GATK can replace Intel GKL with a dependency swap.

### Added

- `FgklPairHmm` (`PairHMMNativeBinding`): the PairHMM in single precision, recomputing underflowing pairs in double as GKL does, or in double precision; haplotypes share DP columns for their common prefix.
- `FgklPdHmm` (`PDHMMNativeBinding`): the partially determined PairHMM used by HaplotypeCaller's `--dragen-378-concordance-mode`.
- `FgklSmithWaterman` (`SWAlignerNativeBinding`): an exact port of GATK's Java aligner, filling anti-diagonals in 16-bit lanes with a 32-bit fallback.
- Scalar, NEON, AVX2 and AVX-512 backends chosen at runtime; every call runs on the calling thread.
- One JAR with native libraries for Linux (x86_64 and aarch64, glibc 2.17), macOS (x86_64 and aarch64) and Windows (x86_64, built but not yet tested in CI).

[Unreleased]: https://github.com/fulcrumgenomics/fgkl/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/fulcrumgenomics/fgkl/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/fulcrumgenomics/fgkl/releases/tag/v0.1.0
