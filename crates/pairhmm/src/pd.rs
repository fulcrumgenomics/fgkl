//! Partially determined PairHMM likelihoods for GATK's DRAGEN 3.7.8 concordance mode, numerically
//! equivalent to GATK's Java `LoglessPDPairHMM` and to Intel GKL's `IntelPDHMM`.
//!
//! [`PdPairHmm`] drives the kernel in `pd_kernel.rs` through the same [`Batcher`] as
//! [`PairHmm`](crate::PairHmm): reads fill SIMD lanes, single precision is the default with
//! double-precision recomputation of any pair that underflows, every call runs on the calling
//! thread, and haplotypes sharing a prefix of bases and flags share its DP columns.

use crate::pd_kernel::SortedPdHaps;
use crate::{Backend, Precision};
use crate::{Batcher, Config, Error, ReadRef};

/// Flag bits of a partially determined haplotype base (GATK's `PartiallyDeterminedHaplotype`).
/// A `SNP` column's `ALT_*` bits name the alternate bases a read may match there.
pub const SNP: u8 = 1;
/// First base a read may skip.
pub const DEL_START: u8 = 2;
/// Last base a read may skip; a single-base deletion carries both flags.
pub const DEL_END: u8 = 4;
/// Alternate base `A` at a `SNP` column.
pub const ALT_A: u8 = 8;
/// Alternate base `C` at a `SNP` column.
pub const ALT_C: u8 = 16;
/// Alternate base `G` at a `SNP` column.
pub const ALT_G: u8 = 32;
/// Alternate base `T` at a `SNP` column.
pub const ALT_T: u8 = 64;

/// A partially determined haplotype: its bases and, per base, the flags above. Both slices have
/// the same length.
#[derive(Clone, Copy, Debug)]
pub struct PdHaplotype<'a> {
    /// The determined bases.
    pub bases: &'a [u8],
    /// Per base, the `SNP`/`ALT_*`/`DEL_START`/`DEL_END` bits; zero for a plain base.
    pub flags: &'a [u8],
}

/// A configured partially determined PairHMM. Cheap to call repeatedly and safe to share between
/// threads; each call computes on the calling thread.
pub struct PdPairHmm {
    inner: Batcher,
}

impl PdPairHmm {
    /// Builds a PD PairHMM for `config`, failing if the requested backend is unavailable.
    pub fn new(config: &Config) -> Result<Self, Error> {
        Ok(PdPairHmm { inner: Batcher::new(config)? })
    }

    /// The vector instruction set in use.
    pub fn backend(&self) -> Backend {
        self.inner.backend
    }

    /// The arithmetic precision in use.
    pub fn precision(&self) -> Precision {
        self.inner.precision
    }

    /// How many pairs so far underflowed in single precision and were (or, with
    /// `double_fallback` off, would have been) recomputed in double precision.
    pub fn fallback_pairs(&self) -> u64 {
        self.inner.fallback_pairs()
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
        self.inner.compute(reads, &SortedPdHaps::new(haplotypes), out);
        Ok(())
    }
}

/// Whether a read base matches the alternate SNP alleles encoded in a haplotype flag byte.
pub fn base_matches_pd(read_base: u8, flags: u8) -> bool {
    if flags & SNP == 0 {
        return false;
    }
    match read_base {
        b'A' | b'a' => flags & ALT_A != 0,
        b'C' | b'c' => flags & ALT_C != 0,
        b'G' | b'g' => flags & ALT_G != 0,
        b'T' | b't' => flags & ALT_T != 0,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PairHmm;
    use crate::pd_reference;
    use crate::synthetic::{self, PdRegion, Read};

    fn reference_all(reads: &[ReadRef<'_>], haps: &[PdHaplotype<'_>]) -> Vec<f64> {
        let mut out = Vec::with_capacity(reads.len() * haps.len());
        for read in reads {
            for hap in haps {
                out.push(pd_reference::log10_likelihood(hap.bases, hap.flags, read));
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
    fn prefix_sharing_handles_nested_prefixes_lengths_and_flag_splits() {
        // Sorted by (row-end state, bases+flags): haplotypes sharing bases diverge where their
        // flags differ, a deletion ending on the last column moves a haplotype to another end
        // state, and lengths differ so snapshots are rescaled.
        let base = b"ACGTTGCAAGGCTTAGGCTTACGATCGATTACAGGT";
        let n = base.len();
        let plain = vec![0u8; n];
        let mut snp_early = plain.clone();
        snp_early[5] = SNP | ALT_T;
        let mut snp_late = plain.clone();
        snp_late[30] = SNP | ALT_C;
        let mut del_mid = plain.clone();
        del_mid[12] = DEL_START;
        del_mid[15] = DEL_END;
        let mut del_mid_and_snp = del_mid.clone();
        del_mid_and_snp[25] = SNP | ALT_G;
        let mut del_last = plain.clone();
        del_last[n - 2] = DEL_START;
        del_last[n - 1] = DEL_END;
        let short: &[u8] = &base[..20];
        let mut short_del = vec![0u8; 20];
        short_del[12] = DEL_START;
        short_del[15] = DEL_END;
        let longer: Vec<u8> = [&base[..], b"ACGTAC"].concat();
        let mut longer_flags = vec![0u8; longer.len()];
        longer_flags[12] = DEL_START;
        longer_flags[15] = DEL_END;
        let haps: Vec<PdHaplotype<'_>> = vec![
            PdHaplotype { bases: base, flags: &del_last },
            PdHaplotype { bases: base, flags: &snp_late },
            PdHaplotype { bases: short, flags: &short_del },
            PdHaplotype { bases: base, flags: &plain },
            PdHaplotype { bases: base, flags: &del_mid_and_snp },
            PdHaplotype { bases: &longer, flags: &longer_flags },
            PdHaplotype { bases: base, flags: &del_mid },
            PdHaplotype { bases: base, flags: &snp_early },
            PdHaplotype { bases: base, flags: &plain },
        ];
        let region = PdRegion::generate(13, 40, 30, 1, 36);
        let reads = region.read_refs();
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let out = compute(&config(backend, Precision::Double), &reads, &haps);
            assert_close(&out, &expected, 1e-9, backend.name());
            let out = compute(&config(backend, Precision::Float), &reads, &haps);
            assert_close(&out, &expected, 1e-4, backend.name());
        }
    }

    #[test]
    fn haplotype_order_does_not_change_results() {
        let region = region();
        let reads = region.read_refs();
        let mut haps = region.haplotype_refs();
        let cfg = Config { precision: Precision::Double, ..Config::default() };
        let forward = compute(&cfg, &reads, &haps);
        haps.reverse();
        let reversed = compute(&cfg, &reads, &haps);
        let n = haps.len();
        for r in 0..reads.len() {
            for h in 0..n {
                let (a, b) = (forward[r * n + h], reversed[r * n + (n - 1 - h)]);
                assert!((a - b).abs() < 1e-11, "read {r} hap {h}: {a} vs {b}");
            }
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
}
