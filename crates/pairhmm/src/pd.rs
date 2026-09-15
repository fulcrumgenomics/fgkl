//! Partially determined PairHMM likelihoods for GATK's DRAGEN 3.7.8 concordance mode, numerically
//! equivalent to GATK's Java `LoglessPDPairHMM` and to Intel GKL's `IntelPDHMM`.
//!
//! The batching mirrors [`PairHmm`](crate::PairHmm): reads fill SIMD lanes, single precision is
//! the default with double-precision recomputation of any pair that underflows, and every call
//! runs on the calling thread. Haplotype columns are not shared, so haplotypes are computed in
//! the caller's order.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::pd_kernel::{PdBatchRunner, PdHaps, PdRunner};
use crate::{
    Backend, Config, Error, NARROW_BATCH, Precision, ReadRef, RunnerKey, has_narrow_instantiation,
    simd,
};

/// A partially determined haplotype: its bases and, per base, the flags of [`crate::pdhmm`]
/// (`SNP` with `ALT_*` bits, `DEL_START`, `DEL_END`). Both slices have the same length.
#[derive(Clone, Copy, Debug)]
pub struct PdHaplotype<'a> {
    pub bases: &'a [u8],
    pub flags: &'a [u8],
}

/// A configured partially determined PairHMM. Cheap to call repeatedly and safe to share between
/// threads; each call computes on the calling thread.
pub struct PdPairHmm {
    precision: Precision,
    backend: Backend,
    double_fallback: bool,
    /// Pairs whose single-precision result underflowed, summed over every call.
    fallback_pairs: AtomicU64,
}

thread_local! {
    /// PD kernel workspaces, reused across calls on the same thread like the plain kernel's.
    static PD_RUNNERS: RefCell<Vec<(RunnerKey, Box<dyn PdBatchRunner>)>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with the cached PD runner for `key`, creating it on first use.
fn with_pd_runner<R>(key: RunnerKey, f: impl FnOnce(&mut dyn PdBatchRunner) -> R) -> R {
    PD_RUNNERS.with(|cell| {
        let mut runners = cell.borrow_mut();
        let index = match runners.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                runners.push((key, make_pd_runner(key)));
                runners.len() - 1
            }
        };
        f(runners[index].1.as_mut())
    })
}

fn make_pd_runner(key: RunnerKey) -> Box<dyn PdBatchRunner> {
    use simd::{ScalarF32, ScalarF64};
    #[cfg(target_arch = "x86_64")]
    if key.backend == Backend::Avx512 && !key.wide {
        return match key.precision {
            Precision::Float => Box::new(PdRunner::<simd::x86::Avx512F32Narrow>::new()),
            Precision::Double => Box::new(PdRunner::<simd::x86::Avx512F64Narrow>::new()),
        };
    }
    match (key.backend, key.precision) {
        (Backend::Scalar, Precision::Float) => Box::new(PdRunner::<ScalarF32>::new()),
        (Backend::Scalar, Precision::Double) => Box::new(PdRunner::<ScalarF64>::new()),
        #[cfg(target_arch = "aarch64")]
        (Backend::Neon, Precision::Float) => Box::new(PdRunner::<simd::neon::NeonF32>::new()),
        #[cfg(target_arch = "aarch64")]
        (Backend::Neon, Precision::Double) => Box::new(PdRunner::<simd::neon::NeonF64>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx2, Precision::Float) => Box::new(PdRunner::<simd::x86::Avx2F32>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx2, Precision::Double) => Box::new(PdRunner::<simd::x86::Avx2F64>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx512, Precision::Float) => Box::new(PdRunner::<simd::x86::Avx512F32>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx512, Precision::Double) => Box::new(PdRunner::<simd::x86::Avx512F64>::new()),
    }
}

impl PdPairHmm {
    pub fn new(config: &Config) -> Result<Self, Error> {
        let backend = config.backend.unwrap_or_else(Backend::detect);
        if !backend.is_available() {
            return Err(Error::BackendUnavailable(backend));
        }
        Ok(PdPairHmm {
            precision: config.precision,
            backend,
            double_fallback: config.double_fallback,
            fallback_pairs: AtomicU64::new(0),
        })
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// How many pairs so far underflowed in single precision and were (or, with
    /// `double_fallback` off, would have been) recomputed in double precision.
    pub fn fallback_pairs(&self) -> u64 {
        self.fallback_pairs.load(Ordering::Relaxed)
    }

    /// Computes every read against every haplotype, writing the log10 likelihood of read `r`
    /// given haplotype `h` to `out[r * haplotypes.len() + h]`.
    pub fn compute_log10_likelihoods(
        &self,
        reads: &[ReadRef<'_>],
        haplotypes: &[PdHaplotype<'_>],
        out: &mut [f64],
    ) -> Result<(), Error> {
        for (i, read) in reads.iter().enumerate() {
            read.validate(i)?;
        }
        for (i, hap) in haplotypes.iter().enumerate() {
            if hap.bases.is_empty() {
                return Err(Error::EmptyHaplotype(i));
            }
            if hap.flags.len() != hap.bases.len() {
                return Err(Error::MismatchedHaplotypeArrays(i));
            }
        }
        let expected = reads.len() * haplotypes.len();
        if out.len() != expected {
            return Err(Error::OutputLength { expected, actual: out.len() });
        }
        if expected == 0 {
            return Ok(());
        }

        let pairs: Vec<(&[u8], &[u8])> = haplotypes.iter().map(|h| (h.bases, h.flags)).collect();
        let haps = PdHaps::new(&pairs);
        let n_haps = haps.len();
        // Longest reads first so that the lanes of a batch have similar lengths.
        let mut read_order: Vec<usize> = (0..reads.len()).collect();
        read_order.sort_by_key(|&i| std::cmp::Reverse(reads[i].len()));
        let sorted_reads: Vec<ReadRef<'_>> = read_order.iter().map(|&i| reads[i]).collect();

        let mut tmp = vec![0.0f64; expected];
        let fallback = self.run_pass(self.precision, &sorted_reads, &haps, &mut tmp);
        if !fallback.is_empty() {
            self.fallback_pairs.fetch_add(fallback.len() as u64, Ordering::Relaxed);
            if self.double_fallback {
                self.run_fallback(&fallback, &sorted_reads, &haps, &mut tmp);
            }
        }
        for (pos, &read) in read_order.iter().enumerate() {
            out[read * n_haps..(read + 1) * n_haps]
                .copy_from_slice(&tmp[pos * n_haps..(pos + 1) * n_haps]);
        }
        Ok(())
    }

    /// Runs every batch of reads against all haplotypes, returning the `(read, haplotype)` pairs
    /// whose result must be recomputed in double precision.
    fn run_pass(
        &self,
        precision: Precision,
        reads: &[ReadRef<'_>],
        haps: &PdHaps<'_>,
        tmp: &mut [f64],
    ) -> Vec<(usize, usize)> {
        let wide = RunnerKey { backend: self.backend, precision, wide: true };
        let narrow = RunnerKey { backend: self.backend, precision, wide: false };
        let n = reads.len();
        let split = if has_narrow_instantiation(self.backend) {
            let wide_lanes = with_pd_runner(wide, |runner| runner.lanes());
            let rest = n % wide_lanes;
            if rest <= NARROW_BATCH { n - rest } else { n }
        } else {
            n
        };
        let mut fallback = Vec::new();
        let (head, tail) = tmp.split_at_mut(split * haps.len());
        Self::run_batches(wide, &reads[..split], haps, head, 0, &mut fallback);
        Self::run_batches(narrow, &reads[split..], haps, tail, split, &mut fallback);
        fallback
    }

    /// Runs `reads` in batches of the runner's lane count, recording fallback pairs with their
    /// read index offset by `first_read`.
    fn run_batches(
        key: RunnerKey,
        reads: &[ReadRef<'_>],
        haps: &PdHaps<'_>,
        tmp: &mut [f64],
        first_read: usize,
        fallback: &mut Vec<(usize, usize)>,
    ) {
        if reads.is_empty() {
            return;
        }
        with_pd_runner(key, |runner| {
            let lanes = runner.lanes();
            let n_haps = haps.len();
            let mut batch = Vec::new();
            for (b, out) in tmp.chunks_mut(lanes * n_haps).enumerate() {
                let lo = b * lanes;
                let hi = (lo + lanes).min(reads.len());
                batch.clear();
                runner.run(&reads[lo..hi], haps, out, &mut batch);
                fallback.extend(batch.iter().map(|&(r, h)| (first_read + lo + r, h)));
            }
        })
    }

    /// Recomputes the given `(read, haplotype)` pairs in double precision, one haplotype at a
    /// time with its reads packed into lanes.
    fn run_fallback(
        &self,
        pairs: &[(usize, usize)],
        reads: &[ReadRef<'_>],
        haps: &PdHaps<'_>,
        tmp: &mut [f64],
    ) {
        let n_haps = haps.len();
        let mut by_hap: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &(r, h) in pairs {
            by_hap.entry(h).or_default().push(r);
        }
        let key = RunnerKey { backend: self.backend, precision: Precision::Double, wide: false };
        with_pd_runner(key, |runner| {
            let lanes = runner.lanes();
            let mut out = vec![0.0f64; lanes];
            let mut none = Vec::new();
            for (h, read_positions) in by_hap {
                let single = PdHaps::new(&[(haps.bases[h], haps.flags[h])]);
                for chunk in read_positions.chunks(lanes) {
                    let batch: Vec<ReadRef<'_>> = chunk.iter().map(|&p| reads[p]).collect();
                    runner.run(&batch, &single, &mut out[..batch.len()], &mut none);
                    for (i, &p) in chunk.iter().enumerate() {
                        tmp[p * n_haps + h] = out[i];
                    }
                }
            }
            debug_assert!(none.is_empty(), "double precision never falls back");
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PairHmm;
    use crate::pdhmm::{self, ALT_C, ALT_G, ALT_T, DEL_END, DEL_START, SNP};
    use crate::synthetic::{self, PdHap, PdRegion, Read};

    fn reference_all(reads: &[ReadRef<'_>], haps: &[PdHaplotype<'_>]) -> Vec<f64> {
        let mut out = Vec::with_capacity(reads.len() * haps.len());
        for read in reads {
            for hap in haps {
                out.push(pdhmm::reference_log10_likelihood(hap.bases, hap.flags, read));
            }
        }
        out
    }

    fn compute(config: &Config, reads: &[ReadRef<'_>], haps: &[PdHaplotype<'_>]) -> Vec<f64> {
        let (out, _) = compute_counting(config, reads, haps);
        out
    }

    fn compute_counting(
        config: &Config,
        reads: &[ReadRef<'_>],
        haps: &[PdHaplotype<'_>],
    ) -> (Vec<f64>, u64) {
        let hmm = PdPairHmm::new(config).unwrap();
        let mut out = vec![0.0; reads.len() * haps.len()];
        hmm.compute_log10_likelihoods(reads, haps, &mut out).unwrap();
        (out, hmm.fallback_pairs())
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let d = (a - e).abs();
            assert!(d <= tol, "{what}: pair {i}: got {a}, expected {e} (diff {d})");
        }
    }

    fn config(backend: Backend, precision: Precision) -> Config {
        Config { precision, backend: Some(backend), double_fallback: true }
    }

    fn region() -> PdRegion {
        PdRegion::generate(31, 70, 120, 24, 300)
    }

    #[test]
    fn double_precision_matches_the_pd_reference_on_every_backend() {
        let region = region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let out = compute(&config(backend, Precision::Double), &reads, &haps);
            assert_close(&out, &expected, 1e-9, backend.name());
        }
    }

    #[test]
    fn single_precision_matches_the_pd_reference_within_float_tolerance() {
        let region = region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let out = compute(&config(backend, Precision::Float), &reads, &haps);
            assert_close(&out, &expected, 1e-4, backend.name());
        }
    }

    #[test]
    fn flagged_alternates_and_deletions_change_the_result() {
        // The generator's flags must actually be exercised: a read drawn from a haplotype's
        // realized allele scores far better against that haplotype's flags than without them,
        // and flags only ever add alignment paths, so they never lower a likelihood.
        let region = region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let plain: Vec<Vec<u8>> = haps.iter().map(|h| vec![0u8; h.bases.len()]).collect();
        let unflagged: Vec<PdHaplotype<'_>> = haps
            .iter()
            .zip(&plain)
            .map(|(h, f)| PdHaplotype { bases: h.bases, flags: f })
            .collect();
        let cfg = Config { precision: Precision::Double, ..Config::default() };
        let with = compute(&cfg, &reads, &haps);
        let without = compute(&cfg, &reads, &unflagged);
        let changed = with.iter().zip(&without).filter(|(a, b)| (*a - *b).abs() > 1.0).count();
        assert!(changed >= 20, "{changed} of {} pairs changed", with.len());
        assert!(with.iter().zip(&without).all(|(a, b)| a >= b), "flags never lower a likelihood");
    }

    #[test]
    fn haplotypes_without_flags_agree_with_the_plain_kernel() {
        let region = synthetic::Region::generate(5, 40, 100, 8, 200);
        let reads = region.read_refs();
        let plain = region.haplotype_refs();
        let flags: Vec<Vec<u8>> = plain.iter().map(|h| vec![0u8; h.len()]).collect();
        let pd: Vec<PdHaplotype<'_>> =
            plain.iter().zip(&flags).map(|(&h, f)| PdHaplotype { bases: h, flags: f }).collect();
        for backend in Backend::available() {
            for (precision, tol) in [(Precision::Double, 1e-9), (Precision::Float, 1e-4)] {
                let cfg = config(backend, precision);
                let pd_out = compute(&cfg, &reads, &pd);
                let mut plain_out = vec![0.0; reads.len() * plain.len()];
                PairHmm::new(&cfg)
                    .unwrap()
                    .compute_log10_likelihoods(&reads, &plain, &mut plain_out)
                    .unwrap();
                assert_close(&pd_out, &plain_out, tol, &format!("{backend} {precision:?}"));
            }
        }
    }

    fn read(bases: &[u8]) -> Read {
        Read {
            bases: bases.to_vec(),
            quals: vec![30; bases.len()],
            ins_gop: vec![45; bases.len()],
            del_gop: vec![45; bases.len()],
            gcp: vec![10; bases.len()],
        }
    }

    #[test]
    fn flags_on_the_last_columns_carry_their_state_into_the_next_row() {
        let hap = b"ACGTTGCAAGGCTTAGGCTTACG";
        let n = hap.len();
        // A deletion ending on the last column leaves every later row starting after a
        // deletion; one starting on the last column leaves them starting inside one; a SNP on
        // the last column changes only the prior.
        let mut ends_del = vec![0u8; n];
        ends_del[n - 3] = DEL_START;
        ends_del[n - 1] = DEL_END;
        let mut starts_del = vec![0u8; n];
        starts_del[n - 1] = DEL_START;
        let mut single_del = vec![0u8; n];
        single_del[n - 1] = DEL_START | DEL_END;
        let mut first_del = vec![0u8; n];
        first_del[0] = DEL_START | DEL_END;
        first_del[1] = SNP | ALT_G;
        let mut snp = vec![0u8; n];
        snp[n - 1] = SNP | ALT_C | ALT_T;
        let owned = [ends_del, starts_del, single_del, first_del, snp];
        let haps: Vec<PdHaplotype<'_>> =
            owned.iter().map(|f| PdHaplotype { bases: hap, flags: f }).collect();
        let reads_owned = [
            read(b"GTTGCAAGGCTTAG"),
            read(b"GGCTTAGGCTTACG"),
            read(b"GGCTTAGGCTTC"),
            read(b"GGCTTAGGCTTA"),
            read(b"CGTTGCAAGG"),
            read(hap),
        ];
        let reads: Vec<ReadRef<'_>> = reads_owned.iter().map(Read::as_ref).collect();
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let out = compute(&config(backend, Precision::Double), &reads, &haps);
            assert_close(&out, &expected, 1e-9, backend.name());
            let out = compute(&config(backend, Precision::Float), &reads, &haps);
            assert_close(&out, &expected, 1e-4, backend.name());
        }
    }

    #[test]
    fn adjacent_and_single_base_deletions_match_the_reference() {
        let hap = b"ACGTTGCAAGGCTTAGGCTTACGATCGATTACA";
        let n = hap.len();
        let mut back_to_back = vec![0u8; n];
        back_to_back[5] = DEL_START;
        back_to_back[7] = DEL_END;
        back_to_back[8] = DEL_START | DEL_END;
        back_to_back[9] = DEL_START;
        back_to_back[12] = DEL_END;
        let mut singles = vec![0u8; n];
        for j in (3..n).step_by(4) {
            singles[j] = DEL_START | DEL_END;
        }
        let mut snp_in_del = vec![0u8; n];
        snp_in_del[10] = DEL_START;
        snp_in_del[11] = SNP | ALT_T;
        snp_in_del[13] = DEL_END | SNP | ALT_G;
        let owned = [back_to_back, singles, snp_in_del];
        let haps: Vec<PdHaplotype<'_>> =
            owned.iter().map(|f| PdHaplotype { bases: hap, flags: f }).collect();
        let region = PdRegion::generate(9, 30, 25, 1, 33);
        let reads = region.read_refs();
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let out = compute(&config(backend, Precision::Double), &reads, &haps);
            assert_close(&out, &expected, 1e-9, backend.name());
        }
    }

    #[test]
    fn float_underflow_falls_back_to_double() {
        let r = Read {
            bases: vec![b'A'; 80],
            quals: vec![40; 80],
            ins_gop: vec![45; 80],
            del_gop: vec![45; 80],
            gcp: vec![10; 80],
        };
        let reads = vec![r.as_ref()];
        let bases = vec![b'C'; 80];
        let mut flags = vec![0u8; 80];
        flags[10] = DEL_START;
        flags[20] = DEL_END;
        let haps = vec![PdHaplotype { bases: &bases, flags: &flags }];
        let expected = reference_all(&reads, &haps);
        assert!(expected[0].is_finite() && expected[0] < -50.0);
        for backend in Backend::available() {
            let (out, fallbacks) =
                compute_counting(&config(backend, Precision::Float), &reads, &haps);
            assert_close(&out, &expected, 1e-9, backend.name());
            assert_eq!(fallbacks, 1, "{backend}");
        }
    }

    #[test]
    fn a_read_gets_the_same_bits_whatever_shares_its_batch() {
        let region = region();
        let haps = region.haplotype_refs();
        let alone: Vec<ReadRef<'_>> = region.read_refs().into_iter().take(3).collect();
        let mut owned: Vec<Read> = region.reads.iter().take(3).cloned().collect();
        let mut rng = synthetic::Rng::new(77);
        for _ in 0..29 {
            owned.push(Read {
                bases: (0..120).map(|_| rng.base()).collect(),
                quals: vec![30; 120],
                ins_gop: vec![45; 120],
                del_gop: vec![45; 120],
                gcp: vec![10; 120],
            });
        }
        let batched: Vec<ReadRef<'_>> = owned.iter().map(Read::as_ref).collect();
        for precision in [Precision::Float, Precision::Double] {
            let cfg = Config { precision, ..Config::default() };
            let solo = compute(&cfg, &alone, &haps);
            let (with_others, fallbacks) = compute_counting(&cfg, &batched, &haps);
            assert_eq!(solo[..], with_others[..solo.len()], "{precision:?}");
            if precision == Precision::Float {
                assert!(fallbacks > 0, "the random reads must underflow");
            }
        }
    }

    #[test]
    fn more_reads_than_lanes_and_single_read_both_work() {
        for (seed, n_reads) in [(41, 1), (42, 37), (43, 65)] {
            let region = PdRegion::generate(seed, n_reads, 50, 3, 60);
            let (reads, haps) = (region.read_refs(), region.haplotype_refs());
            let expected = reference_all(&reads, &haps);
            let out = compute(&Config::default(), &reads, &haps);
            assert_close(&out, &expected, 1e-4, &format!("{n_reads} reads"));
        }
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let hmm = PdPairHmm::new(&Config::default()).unwrap();
        let good = read(b"ACGT");
        let short_quals = Read { quals: vec![30; 3], ..good.clone() };
        let mut out = vec![0.0; 1];
        let hap = PdHaplotype { bases: b"ACGT", flags: &[0; 4] };
        assert_eq!(
            hmm.compute_log10_likelihoods(&[short_quals.as_ref()], &[hap], &mut out),
            Err(Error::MismatchedReadArrays(0))
        );
        let empty = PdHaplotype { bases: b"", flags: &[] };
        assert_eq!(
            hmm.compute_log10_likelihoods(&[good.as_ref()], &[empty], &mut out),
            Err(Error::EmptyHaplotype(0))
        );
        let short_flags = PdHaplotype { bases: b"ACGT", flags: &[0; 3] };
        assert_eq!(
            hmm.compute_log10_likelihoods(&[good.as_ref()], &[hap, short_flags], &mut out),
            Err(Error::MismatchedHaplotypeArrays(1))
        );
        assert_eq!(
            hmm.compute_log10_likelihoods(&[good.as_ref()], &[hap, hap], &mut out),
            Err(Error::OutputLength { expected: 2, actual: 1 })
        );
        assert!(hmm.compute_log10_likelihoods(&[], &[hap], &mut []).is_ok());
    }

    #[test]
    fn synthetic_region_reads_are_valid_and_haplotypes_carry_flags() {
        let region = region();
        assert!(region.reads.iter().all(|r| !r.bases.is_empty()));
        assert!(region.haplotypes.iter().any(|h: &PdHap| h.flags.iter().any(|&f| f != 0)));
        assert!(region.cells() > 0);
    }
}
