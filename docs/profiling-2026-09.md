# Profiling findings and options (September 2026)

Measurements of the v6 kernel (the one benchmarked against GKL) on an AWS m8i.xlarge (Granite Rapids, AVX-512) and an M4 Max (NEON), what they say about where time goes, and what each candidate next step is worth. Raw outputs: `hc-perf/aws/results-x86.md`, `hc-perf/aws/perf-x86/`, `hc-perf/aws/perf-mac/`, `hc-perf/aws/replay-out/`.

## 1. Where HaplotypeCaller time goes on x86 (v6 kernel, serial JVMs)

| shard | config | wall s | PairHMM s | JNI setup s | native SW s | PairHMM share of wall |
|---|---|---|---|---|---|---|
| 0228 median | gkl4 | 97 | 13.0 | 1.4 | 2.9 | 13% |
| 0228 | fgkl32 | 94 | 12.4 | 1.4 | 2.8 | 13% |
| 0228 | fgkl64 | 96 | 15.9 | 1.5 | 2.8 | 17% |
| 0093 segdup | gkl4 | 159 | 48.1 | 1.7 | 9.4 | 30% |
| 0093 | fgkl32 | 178 | 70.5 | 1.8 | 9.5 | 40% |
| 0093 | fgkl64 | 161 | 53.9 | 1.8 | 9.4 | 34% |
| 0122 pericentromere | gkl4 | 623 | 296 | 5.0 | 15.5 | 48% |
| 0122 | fgkl32 | 689 | 365 | 5.2 | 16.6 | 53% |
| 0122 | fgkl64 | 667 | 349 | 5.1 | 15.4 | 52% |

Two things follow. The JNI marshalling (GATK's holder construction plus our flattening) is 1-2% of PairHMM time and under 1.5% of wall, and native Smith-Waterman is 2-3% of wall, so no change to the Java/native contract can buy more than a few percent. And the pathological shards are pathological in one window each: on 0122, 475 of the 689 wall seconds are spent in chr4:49.0-49.5 Mb and another 124 s in the next 500 kb, with every other 500 kb window taking about 10 s; on 0093, chr3:75.5-76.0 Mb takes 97 of 178 s. Those windows are also where the non-PairHMM Java work concentrates, which is the other session's territory (read caps, region-level parallelism).

## 2. Kernel profiles

Both ISAs were profiled on `pairhmm-replay` over real GATK pair dumps and on the synthetic bench (`perf record` with source-line attribution on x86, `samply` on the Mac).

- **Zero-filling scratch memory was the largest single cost on ordinary regions.** `Workspace::prepare` cleared every snapshot column for every batch, about 9 MB per call, whether or not the column would be used. `memset` was 32% of samples on x86 and 15% on the Mac for the norm8 replay, and 5-6% on the hotspot and synthetic runs. Fixed: `AlignedVec::resize_no_fill` sizes the row buffers and snapshots without clearing them (they are always written before being read). Effect: Mac norm8 replay 3.4 to 4.1 Gcells/s float (+20%) and 2.15 to 2.49 double (+16%); x86 synthetic bench 5.95 to 6.38 Gcells/s AVX-512 float (+7%), 3.0 to 3.3 double (+10%). The same clearing was paid once per fallback batch (14 MB for an 8-lane double batch), which is most of why the fallback path was so slow; see section 4.
- **The inner loop is instruction-bound, and a quarter to a third of its instructions are slice bookkeeping.** IPC is 2.5 on norm8, 3.1 on the synthetic bench and 3.7 on the hotspot replay, so memory is not the limit. Source-line attribution puts 25-35% of kernel samples on `core::slice::index`, range iteration and `usize` comparisons: every `S::load(&prev.m[off..])` and `store` in `dp_segment` builds a subslice with a bounds check, seven per vector cell, and `Wide<S, K>` slices again per lane group. The DP arithmetic itself (`kernel.rs:585-587`) is only about 4% of samples. Removing that bookkeeping (one bounds check per row segment, then unchecked or iterator-based access) is the largest remaining kernel win on every region type; I would expect 15-30% on the DP-bound part, to be measured.
- **Smith-Waterman is fine.** The SIMD fill is 50% of `sw-bench` samples (the rest is the scalar reference used for validation and page faults), IPC 4.4, 1.9 Gcells/s on AVX-512; with SW at 2-3% of HC wall there is nothing to chase there now.

## 3. What real regions look like (corrected replay)

The replay tool's parser was splitting regions at duplicate reads (GATK dumps duplicates back to back with identical text), so the earlier "normal regions average 5.6 reads" was an artifact. With read boundaries detected by the haplotype list starting over, the dumps reconstruct completely:

| dump | regions | reads per region | haplotypes per region | pairs | prefix-shared cells | pairs underflowing in f32 |
|---|---|---|---|---|---|---|
| norm8 (1 Mb, chr8 normal) | 1047 | 46 | 3.4 | 211k | 54% | 0.42% |
| sd3_5kb (chr3 segdup hotspot) | 28 | 203 | 61 | 410k | 68% | 0% |
| sd3_10k (chr3:75.69-75.70 Mb) | 45 | 210 | 69 | 718k | 70% | 0.37% |
| cen4_1k (chr4:49.107-49.108 Mb centromere) | 5 | 4230 | 98 | 2.37M | 65% | 4.2% |

So lane utilisation is not the small-batch problem the earlier notes assumed: a 46-read region fills two 32-lane AVX-512 batches at 72% and six 8-lane NEON batches at 96%. Underflow (the double-precision fallback) is rare on normal sequence and reaches 4% in the centromere, where reads from other satellite copies mismatch the haplotypes heavily. It is a property of read-versus-haplotype divergence, not of depth.

Replay throughput with the no-fill fix (v7), x86 AVX-512 unless noted. "float-nofb" is single precision with the double fallback disabled, i.e. the ceiling for a kernel that never underflows:

| dump | float, no fallback | float | double |
|---|---|---|---|
| norm8 | 5.97 Gcells/s (654 ms) | 5.29 (737 ms) | 3.53 (1105 ms) |
| sd3_5kb | 10.6 | 10.7 | 5.7 |
| sd3_10k | 11.4 | 10.8 | 6.1 |
| cen4_1k | 10.1 (9.7 s) | 7.6 (12.9 s) | 5.0 (19.4 s) |
| norm8, Mac NEON | 5.02 | 4.69 | 2.64 |
| sd3_5kb, Mac NEON | 7.61 | 7.58 | 3.83 |

Kernel results agree with GATK's recorded values to 5e-5 (the dump's `%e` precision) in every mode.

## 4. Precision policy: what each option is worth

The cost of the fallback path is the point. A fallback pair costs 33 µs on the centromere and 93 µs on norm8 against 3-4 µs for a pair in the main pass, because the recomputation runs one haplotype at a time with mostly empty lanes and rebuilds the per-haplotype tables (and, in v6, cleared 14 MB per batch). That is why 0.4% of pairs cost 11% of norm8 time and 4% of pairs cost 33% of centromere time, and why in HaplotypeCaller double beat float on 0093 and 0122 with v6: not because float is slow there, but because the fallback path was. The HaplotypeCaller-level A/B of the no-fill build (v7) on the same instance confirms it:

| shard | v6 float | v7 float | v6 double | v7 double | GKL 4 threads |
|---|---|---|---|---|---|
| 0228 median, PairHMM s | 12.4 | 8.6 | 15.9 | 11.8 | 13.0 |
| 0093 segdup | 70.5 | 43.0 | 53.9 | 48.2 | 48.1 |
| 0122 pericentromere | 365 | 243 | 349 | 326 | 296 |
| 0122 wall s | 689 | 573 | 667 | 658 | 623 |

Single-threaded v7 float now beats GKL's 4-thread OpenMP kernel on every shard, at 40-50% of its CPU seconds, and float beats double everywhere (1.34x on the centromere shard, matching the replay).

| option | expected PairHMM time, centromere | normal sequence | effort | notes |
|---|---|---|---|---|
| current v7 (float, recompute underflows in double) | 1.0 | 1.0 | done | float beats double 1.5x on the centromere in the replay |
| predict double from region properties | at best equal to double: 1.5x slower | worse | small | double is 2x float per cell, so switching whole regions to double can only lose; depth does not predict underflow anyway |
| cheaper fallback path (batch by read, full lanes, no table rebuild) | ~0.85 | ~0.95 | small | bounded by the 8-30x per-pair penalty shrinking to maybe 3x |
| per-row power-of-two rescaling in f32, no fallback at all | ~0.75-0.8 | ~0.9 | medium | ceiling is float-nofb (1.33x on centromere, 1.13x on norm8) minus the rescale overhead; removes the double pass, the fallback bookkeeping, and the precision policy entirely |

Rescaling mechanics: track the row maximum (or row sum) per lane in the inner loop, and when a lane's row drops below 2^-40 multiply that lane's row by 2^80 and add 80 to an integer exponent lane; the log10 result is `log10(sum) + exponent * log10(2)`. Precision is unchanged (f32 relative error does not depend on magnitude), so results agree with double to the same 1e-4 as today; byte-identical gVCFs stay a double-mode property as they are now. The overhead question is one extra vector op per cell for the running maximum, roughly 7-12% of the arithmetic, which is why the bounds-check work should land first: it makes room. I recommend rescaling over the cheaper-fallback option because it also deletes code paths, but the cheaper fallback is a one-hour change if we want a stopgap.

## 5. API changes: not for speed

With marshalling at 1-2% of PairHMM time and SW at 2-3% of wall, a flat-array PairHMM contract or a batched SW contract would each be worth under 2% of HC wall, and both need a new `gatk-native-bindings` release plus GATK call-site changes. The one API item with leverage is adoption, not speed: `ServiceLoader` discovery so GATK names no vendor class. I suggest raising that with Broad and leaving the contract otherwise alone until the kernel work is done.

## 6. Proposed order

1. Remove per-cell slice bookkeeping from `dp_segment` (and `Wide`), measure on both ISAs. Broadest gain, no numerics change.
2. Choose the AVX-512 narrow instantiation per 32-read chunk rather than per call, so a 46-read region runs as 32 wide + 14 narrow (48 lane slots instead of 64). Small change, x86 only, worth up to ~20% on normal regions.
3. Per-row rescaling in the f32 kernel; delete the fallback path; keep double as an explicit mode.
4. Re-run the x86 shards and the Mac five-unit set; update `docs/pairhmm-design.md`, whose lane-utilisation section rests on the wrong 5.6-read figure.
5. Then PDHMM, naming, packaging, and the Broad conversation (ServiceLoader).

## 7. What was done after this analysis (same day)

1. Bounds checks: `dp_segment` checks bounds once per segment and uses pointer loads and stores. NEON 1.8x on both region types; AVX-512 unchanged (that loop was already at its FP-port roofline, the checks ran on spare integer ports). Bit-identical.
2. Narrow/wide per chunk on AVX-512: about 2% on normal regions. Bit-identical.
3. Rescaling: built and measured three variants (per-row exponents with prefix-mass tracking, then source-column-based decisions with a scalar slow path, then a fully vectorised decision with transition-based scaling). All were slower on NEON (10-60%), and the correct variant cannot shrink the double pass: rows before a lane's first rescale flush at `2^-126` as today, and a real norm8 pair lost 0.7% there under flush-to-zero. Rejected; the step-2 kernel with GKL's threshold was kept. Details in `docs/pairhmm-design.md`.
4. Fallback grouping: reads that fail against a third or more of the haplotypes are recomputed with one prefix-shared double sweep. Centromere dump: fallback cost 33% to 16% of float time (10.3 s to 9.0 s on the Mac); chr3 hotspot and norm8 also improve. The above-threshold checksums are unchanged; recomputed pairs move at the last bit.
5. HaplotypeCaller on norm8 (Mac): double mode byte-identical to master (0 of 122,440 records); float mode differs in the same 2 records as before.

Also fixed on the way: the replay tool's region parser (duplicate reads), `Float::exponent` (unused now), and the replay tool gained result checksums, `--only-region`, and `--dump-region`.

## 8. In-GATK round with the final build (x86 and Graviton, same day)

Full tables: `hc-perf/aws/results-x86.md` and `hc-perf/aws/results-graviton.md`.

**x86 (m8i.xlarge, one JVM at a time, 8 GB heap, DRAGEN mode, gVCF).** Single-threaded fgkl float beats GKL's 4-thread OpenMP PairHMM on wall time on every shard: 90 vs 97 s (0228), 144 vs 160 s (0093 segdup), 540 vs 624 s (0122 pericentromere), at 37-49% of GKL's CPU seconds; PairHMM time 8.2 / 35.6 / 205 s against GKL's 13.0 / 48.1 / 296 s with four threads. Double mode 11.7 / 48.0 / 326 s. gVCF differences against GKL are unchanged from the earlier builds (15 / 32 / 143 records for float, 29 / 44 / 226 for double, no variant added or removed); double mode stays byte-identical to GATK's Java PairHMM.

**Where x86 time goes now** (async-profiler, share of CPU samples, fgkl32 vs GKL 4 threads): PairHMM 6.6 vs 9.1% (0228), 19.6 vs 23.8% (0093), 34.6 vs 41.7% (0122); the double recomputation is 4.7% of samples on 0122, 14% of PairHMM time. Native Smith-Waterman is 3-7% of wall (3.1 / 10.4 / 16.7 s), about twice what section 5 assumed, a third of PairHMM on the segdup shard. Everything else is GATK's Java: assembly-graph hashing 23-31%, DRAGEN genotyping 10-13%, and on the median shard JIT warm-up (26%) and reference-confidence emission (21%).

**Graviton (m8g.xlarge, 4 Graviton4 cores).** The v17 build compiles and passes its tests unchanged on aarch64 Linux; GATK loads the NEON PairHMM and Smith-Waterman. Kernel throughput is 0.6-0.7x the AVX-512 Xeon per core (norm8 replay 951 vs 692 ms, cen4 19.8 vs 11.7 s) and 0.45-0.5x the M4 Max. HaplotypeCaller float mode runs at 1.35-1.55x the x86 wall time (128 vs 90 s on 0228, 105 vs 78 s on 0250, 224 vs 144 s on 0093, 826 vs 540 s on 0122), PairHMM at 1.25-1.65x; double mode costs 1.6x the float PairHMM time on both platforms. GKL has no arm64 natives, so an unmodified GATK on Graviton runs the Java PairHMM (hours on the pericentromere shard); that baseline was not rerun. Two arm64 observations: the 4-lane NEON Smith-Waterman is 11% of samples on the segdup shard (5.6% with 16 lanes on AVX-512), so it is the one native kernel with headroom there; and GATK falls back to `java.util.zip` for BAM and gVCF compression on arm64 because `CommandLineProgram` installs GKL's deflater factories instead of leaving htsjdk's libdeflate default in place (about 1% of samples; a GATK-side fix).

**Cross-platform numerics.** Above GKL's threshold, float results are bit-identical between Linux x86 and Linux arm64 on three of the four dumps and differ on the centromere dump in exactly 149 of 2.27 million pairs, all within two log10 units of the threshold, by at most 7e-8 log10: x86 runs flush-to-zero as GKL does and NEON keeps subnormals. Double results are byte-identical between the two Linux platforms on every pair of every dump; against macOS only libm's last bit differs. `pairhmm-replay` now prints a checksum of results rounded to 1e-6 and can write per-pair results (`--write-results DIR`) for such comparisons (`hc-perf/aws/compare_results.py`). Details in `docs/pairhmm-design.md`, Numerics.

## 9. What the Smith-Waterman is asked to do (HaplotypeCaller on the Mac, `FGKL_SW_STATS=1`)

Per call the aligner records the parameter set, overhang strategy, lengths, cells and distinct sequence pairs (`hc-perf/aws/sw-stats-mac.txt`). Shard 0228 (60 s wall, 4.8 s native Smith-Waterman) and 0093 (109 s, 18.0 s):

| parameter set | calls 0228 / 0093 | cells 0228 / 0093 | share of cells | distinct pairs |
|---|---|---|---|---|
| haplotype to reference, 200/-150/-260/-11, softclip | 31,940 / 103,946 | 3.96 G / 16.98 G | 85% / 95% | 100% |
| read to best haplotype, 10/-15/-30/-5, softclip | 31,768 / 36,328 | 0.70 G / 0.93 G | 15% / 5% | 99.7% / 98.7% |
| dangling ends, 25/-50/-110/-6, leading indel | 3,809 / 5,448 | 0.013 G / 0.018 G | under 0.5% | 96% / 94% |

The native aligner never sees an exact match: GATK's `SWNativeAlignerWrapper` answers those with a substring search before calling it, so the kernel's own shortcut is idle under GATK. Haplotype-to-reference alignments dominate the cells, each about 500 x 500 after GATK pads both sequences with ten Ns, and every one of them is a distinct sequence pair, so there is nothing to cache. A 16-bit kernel chosen by a score bound would cover only the read and dangling-end cases (`fits_i16`), 5-15% of the cells: the 200-scale parameters exceed i16 on the best score alone and cannot be rescaled (gcd 1). The design that covers them is a single kernel over 16-bit lanes with saturating arithmetic and match-relative scores, now implemented in `crates/smithwaterman/src/diag.rs`: a cell holds `s * H - (s * match / 2) * (i + j)` plus a headroom constant near the top of the i16 range (`s` = 2 for odd match values), so a match costs 0, a mismatch `s * (mismatch - match)`, and every gap step its penalty minus half a match; every step is non-positive, so a saturated value can only breed saturated values and every stored value above the floor is exact. The end is chosen on absolute scores decoded from the last row and column, where a saturated candidate is only known to lie below a bound; if that bound reaches the best decoded candidate the pair is redone in 32-bit lanes (the same trait, the same fill, `i32` lanes). On the Mac: `sw-bench` 1.04 to 2.00 Gcells/s (1.93x) on both alignment shapes; HaplotypeCaller native Smith-Waterman 4.81 to 2.71 s on 0228 and 18.04 to 9.77 s on 0093 (1.8x), wall 109 to 99 s on 0093, no fallback in 213,239 alignments, gVCFs byte-identical to the 32-bit runs. Verified against the scalar reference on every backend in both lane widths (NEON natively, AVX2 in an amd64 container; the AVX-512 16-bit path compiles but has not run on hardware). The Graviton tables above predate this change; its Smith-Waterman share there (11-14% of wall on 0093) should roughly halve.

## 10. The partially determined PairHMM (DRAGEN 3.7.8 mode)

Measured before writing the kernel, on the Mac with the v17 jar: shard 0228 in `--dragen-378-concordance-mode` took 282 s wall / 296 s CPU, of which GATK's Java `LoglessPDPairHMM` was 240 s (GKL's `IntelPDHMM` does not load on macOS or arm64, so GATK falls back to Java there). The same shard's plain-mode likelihoods take about 5 s in fgkl's PairHMM. Design and results of the kernel are in `pairhmm-design.md`; this section records the measurements.

Synthetic throughput (`pdhmm-bench`, 4000 reads x 1 haplotype so neither kernel shares prefixes): the PD kernel is within 3% of the plain kernel per cell on NEON (4.67 vs 4.76 Gcells/s float, 2.39 vs 2.47 double) and AVX-512 (3.38 vs 3.42 float, 1.79 vs 1.82 double), and about 9% slower on AVX2 (2.43 vs 2.68 float). On a synthetic region of 32 haplotypes the plain kernel is 2.5-3x faster because of prefix sharing, which the PD kernel does not do.

Real data (`pdhmm-replay` on a `--pdhmm-results-file` dump of chr7:125588609-126400000, the first 0.8 Mb of shard 0228, from the Java PD implementation): 1,189 regions, 244,956 pairs, 5.2 Gcells, 3.6 haplotypes and 43 reads per region on average; 59% of the PD haplotypes carry at least one flag but only 1.6% of columns are flagged. NEON float 4.0 Gcells/s (1.30 s for the dump, so roughly 9 s for the whole shard), double 2.1 Gcells/s; 256 pairs (0.1%) underflow single precision; every pair agrees with GATK's recorded likelihood to the dump's seven significant digits. 52% of the cells lie in prefixes that two haplotypes of the same region share (identical bases and flags, same row-end state), so prefix sharing would roughly halve the PD kernel's time; it is not implemented.

In HaplotypeCaller (Mac, shard 0228, VCF output): 282 s wall / 296 s CPU with the Java PD implementation, 53.5 s / 70.2 s with fgkl's kernel; the VCF body is byte-identical (9,980 records). On the x86 box (m8i.xlarge): GKL's `IntelPDHMM` with its default 8 OpenMP threads 104 s wall / 315 s CPU, with 1 thread 107 s / 137 s, fgkl 77 s / 105 s; on the segdup shard 0093, 246 s / 636 s, 301 s / 365 s and 178 s / 208 s. VCF bodies are identical to GKL's on both shards. GATK's `PDPairHMM` timer reports 0.0 for both native classes (the vector class never updates it), so kernel-only times in HC are inferred from the wall difference.
