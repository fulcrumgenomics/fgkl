# Using fgkl from GATK

GATK talks to native PairHMM implementations through `org.broadinstitute:gatk-native-bindings`, which fgkl implements, so wiring it in is a dependency change plus one constructor.

`gatk-integration.patch` (against GATK master at 0cde69eed, September 2026) does four things:

- adds the fgkl jar to `build.gradle` as a file dependency (a Maven coordinate once fgkl is published),
- makes `VectorLoglessPairHMM` construct `com.fulcrumgenomics.fgkl.pairhmm.FgklPairHmm` for both the `AVX` and `OMP` implementations instead of `IntelPairHmm` / `IntelPairHmmOMP`, logging the selected backend, and
- makes `VectorLoglessPairPDHMM` construct `com.fulcrumgenomics.fgkl.pdhmm.FgklPdHmm` instead of `IntelPDHMM`, so `--dragen-378-concordance-mode` gets a native partially determined PairHMM on every platform (GATK otherwise falls back to Java wherever GKL's PD kernel does not load, which is all of arm64 and macOS), and
- makes `SmithWatermanIntelAligner` construct `com.fulcrumgenomics.fgkl.smithwaterman.FgklSmithWaterman` instead of `IntelSmithWaterman`.

With that patch, `--pair-hmm-implementation FASTEST_AVAILABLE` (the default) uses fgkl on every platform fgkl ships a native library for, and `--native-pair-hmm-use-double-precision` selects the double-precision kernel, whose output is byte-identical to GATK's Java `LOGLESS_CACHING` implementation. `--native-pair-hmm-threads` and the `--native-dragen-pd-pair-hmm-*` arguments are accepted and ignored: each likelihood call runs on the calling thread. The PD kernel has no precision argument in GATK; `FgklPdHmm` runs single precision with double recomputation of underflowing pairs unless `setDoublePrecision(true)` is called.

Apply with `git apply docs/gatk-integration.patch` in a GATK checkout, build the jar with `./gradlew localJar`, and run HaplotypeCaller as usual. A cleaner long-term change for GATK is to discover the binding through `java.util.ServiceLoader` so no vendor class is named in GATK at all.
