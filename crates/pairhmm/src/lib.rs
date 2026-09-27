//! PairHMM read-versus-haplotype log10 likelihoods, numerically equivalent to GATK's Java
//! `LoglessPairHMM` and to Intel GKL's AVX kernels.
//!
//! The kernel vectorizes across reads (each SIMD lane holds a different read), sweeps the DP
//! matrix row by row, and shares DP columns between haplotypes with a common prefix. Single
//! precision is used by default, with any pair whose scaled probability underflows recomputed in
//! double precision, exactly as GKL does. Every call runs on the calling thread; callers that want
//! parallelism run independent calls concurrently.

mod kernel;
mod model;
mod pd;
mod pd_kernel;
pub mod pd_reference;
mod plan;
pub mod reference;
mod simd;
pub mod synthetic;

pub use pd::{ALT_A, ALT_C, ALT_G, ALT_T, DEL_END, DEL_START, PdHaplotype, PdPairHmm, SNP};
pub use plan::SharedColumns;

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use kernel::SortedHaps;

/// A region's reads left over after its full wide batches use the narrow AVX-512 instantiation
/// when there are at most this many of them.
const NARROW_BATCH: usize = 16;

/// One read's bases and per-base penalties, all of the same length.
#[derive(Clone, Copy, Debug)]
pub struct ReadRef<'a> {
    /// Read bases; `N` matches every haplotype base.
    pub bases: &'a [u8],
    /// Phred base qualities.
    pub quals: &'a [u8],
    /// Phred insertion gap-open penalties.
    pub ins_gop: &'a [u8],
    /// Phred deletion gap-open penalties.
    pub del_gop: &'a [u8],
    /// Phred gap-continuation penalties.
    pub gcp: &'a [u8],
}

impl ReadRef<'_> {
    /// Number of bases.
    pub fn len(&self) -> usize {
        self.bases.len()
    }

    /// Whether the read has no bases.
    pub fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    fn validate(&self, index: usize) -> Result<(), Error> {
        if self.bases.is_empty() {
            return Err(Error::EmptyRead(index));
        }
        let n = self.bases.len();
        if [self.quals.len(), self.ins_gop.len(), self.del_gop.len(), self.gcp.len()]
            .iter()
            .any(|&l| l != n)
        {
            return Err(Error::MismatchedReadArrays(index));
        }
        Ok(())
    }
}

/// Arithmetic precision of the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    /// `f32`, recomputing in `f64` any pair whose scaled probability drops below `1e-28`.
    Float,
    /// `f64` throughout.
    Double,
}

impl Precision {
    /// `"float"` or `"double"`, the spelling `FromStr` accepts.
    pub fn name(self) -> &'static str {
        match self {
            Precision::Float => "float",
            Precision::Double => "double",
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Precision {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "float" => Ok(Precision::Float),
            "double" => Ok(Precision::Double),
            other => Err(format!("unknown precision '{other}' (expected float or double)")),
        }
    }
}

/// The vector instruction set a [`PairHmm`] runs on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    /// Four scalar lanes, for CPUs without a supported vector unit and for checking the others.
    Scalar,
    /// Two 128-bit NEON lane groups.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// One 256-bit AVX2 lane group with FMA.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// Two 512-bit AVX-512 lane groups, with a single-group instantiation for small batches.
    #[cfg(target_arch = "x86_64")]
    Avx512,
}

impl Backend {
    /// The fastest backend this CPU supports.
    pub fn detect() -> Backend {
        Self::all().iter().rev().copied().find(|b| b.is_available()).unwrap_or(Backend::Scalar)
    }

    /// Every backend compiled into this build, slowest first.
    pub fn all() -> &'static [Backend] {
        &[
            Backend::Scalar,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2,
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512,
        ]
    }

    /// The backends this CPU supports, slowest first.
    pub fn available() -> Vec<Backend> {
        Self::all().iter().copied().filter(|b| b.is_available()).collect()
    }

    /// Whether this CPU can run the backend.
    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => is_x86_feature_detected!("avx512f"),
        }
    }

    /// Whether the kernels have a separate narrow (single lane group) instantiation for this
    /// backend, used for the reads a region has left over after its full wide batches.
    pub(crate) fn has_narrow_instantiation(self) -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            self == Backend::Avx512
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Lower-case name, the spelling `FromStr` accepts.
    pub fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => "neon",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => "avx2",
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => "avx512",
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::all()
            .iter()
            .copied()
            .find(|b| b.name() == s)
            .ok_or_else(|| format!("unknown backend '{s}'"))
    }
}

/// How to build a [`PairHmm`].
#[derive(Clone, Debug)]
pub struct Config {
    /// Arithmetic precision; `Float` is GKL's default.
    pub precision: Precision,
    /// `None` selects the fastest available backend.
    pub backend: Option<Backend>,
    /// Recompute in double precision every pair whose single-precision result underflowed
    /// (GKL's policy), and without suffix sharing every pair whose joined result was too small
    /// for the backward values. Disable only to measure what that recomputation costs: the
    /// affected results are then `NaN`.
    pub double_fallback: bool,
    /// Share work between haplotypes with a common suffix as well as a common prefix. Results stay
    /// within tolerance of the prefix-only kernel but differ from it in the last bits wherever a
    /// haplotype joins backward values; disable to reproduce those bits. Ignored by [`PdPairHmm`].
    pub share_suffixes: bool,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            precision: Precision::Float,
            backend: None,
            double_fallback: true,
            share_suffixes: true,
        }
    }
}

/// A configured PairHMM. Cheap to call repeatedly and safe to share between threads; each call
/// computes on the calling thread.
pub struct PairHmm {
    inner: Batcher,
    share_suffixes: bool,
}

impl PairHmm {
    /// Builds a PairHMM for `config`, failing if the requested backend is unavailable on this CPU.
    pub fn new(config: &Config) -> Result<Self, Error> {
        Ok(PairHmm { inner: Batcher::new(config)?, share_suffixes: config.share_suffixes })
    }

    /// The vector instruction set in use.
    pub fn backend(&self) -> Backend {
        self.inner.backend
    }

    /// The arithmetic precision in use.
    pub fn precision(&self) -> Precision {
        self.inner.precision
    }

    /// How many pairs so far lost precision and were (or, with `double_fallback` off, would have
    /// been) recomputed: pairs that underflowed in single precision, and pairs whose result
    /// joined from backward values was too small for them.
    pub fn fallback_pairs(&self) -> u64 {
        self.inner.fallback_pairs()
    }

    /// Computes every read against every haplotype, writing the log10 likelihood of read `r`
    /// given haplotype `h` to `out[r * haplotypes.len() + h]`.
    pub fn compute_log10_likelihoods(
        &self,
        reads: &[ReadRef<'_>],
        haplotypes: &[&[u8]],
        out: &mut [f64],
    ) -> Result<(), Error> {
        for (i, read) in reads.iter().enumerate() {
            read.validate(i)?;
        }
        if let Some(i) = haplotypes.iter().position(|h| h.is_empty()) {
            return Err(Error::EmptyHaplotype(i));
        }
        let expected = reads.len() * haplotypes.len();
        if out.len() != expected {
            return Err(Error::OutputLength { expected, actual: out.len() });
        }
        if expected == 0 {
            return Ok(());
        }
        self.inner.compute(reads, &SortedHaps::new(haplotypes, self.share_suffixes), out);
        Ok(())
    }
}

/// The haplotype columns per read base the PairHMM computes for `haplotypes`, with prefix sharing
/// alone and with suffix sharing too: a diagnostic of how much work sharing saves.
pub fn shared_columns(haplotypes: &[&[u8]]) -> SharedColumns {
    let sorted = SortedHaps::new(haplotypes, true);
    sorted.plan.columns(&sorted.bases, &sorted.lcp, &sorted.dup)
}

/// Why a likelihood computation was refused; indices are into the caller's arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The read has no bases.
    EmptyRead(usize),
    /// The read's qualities or penalties differ in length from its bases.
    MismatchedReadArrays(usize),
    /// The haplotype has no bases.
    EmptyHaplotype(usize),
    /// A partially determined haplotype's flags differ in length from its bases.
    MismatchedHaplotypeArrays(usize),
    /// The output slice does not hold exactly reads x haplotypes elements.
    OutputLength {
        /// Reads x haplotypes.
        expected: usize,
        /// The slice's length.
        actual: usize,
    },
    /// The requested backend is not supported by this CPU.
    BackendUnavailable(Backend),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyRead(i) => write!(f, "read {i} has no bases"),
            Error::MismatchedReadArrays(i) => {
                write!(f, "read {i}: bases, qualities and penalties differ in length")
            }
            Error::EmptyHaplotype(i) => write!(f, "haplotype {i} has no bases"),
            Error::MismatchedHaplotypeArrays(i) => {
                write!(f, "haplotype {i}: bases and flags differ in length")
            }
            Error::OutputLength { expected, actual } => {
                write!(f, "output has {actual} elements but reads x haplotypes is {expected}")
            }
            Error::BackendUnavailable(b) => write!(f, "backend {b} is not supported by this CPU"),
        }
    }
}

impl std::error::Error for Error {}

/// The batching driver shared by [`PairHmm`] and [`PdPairHmm`]: fills the kernel's lanes with
/// reads, recomputes underflowing pairs in double precision, and maps results back to the
/// caller's haplotype order. What differs between the two kernels lives behind [`HapSet`].
pub(crate) struct Batcher {
    precision: Precision,
    backend: Backend,
    double_fallback: bool,
    /// Pairs whose single-precision result underflowed, summed over every call.
    fallback_pairs: AtomicU64,
}

impl Batcher {
    pub fn new(config: &Config) -> Result<Self, Error> {
        let backend = config.backend.unwrap_or_else(Backend::detect);
        if !backend.is_available() {
            return Err(Error::BackendUnavailable(backend));
        }
        Ok(Batcher {
            precision: config.precision,
            backend,
            double_fallback: config.double_fallback,
            fallback_pairs: AtomicU64::new(0),
        })
    }

    pub fn fallback_pairs(&self) -> u64 {
        self.fallback_pairs.load(Ordering::Relaxed)
    }

    /// Computes every read against every haplotype of `haps` (validated and non-empty), writing
    /// `out[read * haps.len() + hap]` in the caller's haplotype order.
    pub fn compute<H: HapSet>(&self, reads: &[ReadRef<'_>], haps: &H, out: &mut [f64]) {
        let n_haps = haps.len();
        // Longest reads first so that the lanes of a batch have similar lengths.
        let mut read_order: Vec<usize> = (0..reads.len()).collect();
        read_order.sort_by_key(|&i| std::cmp::Reverse(reads[i].len()));
        let sorted_reads: Vec<ReadRef<'_>> = read_order.iter().map(|&i| reads[i]).collect();

        let mut tmp = vec![0.0f64; reads.len() * n_haps];
        let fallback = self.run_pass(self.precision, &sorted_reads, haps, &mut tmp);
        if !fallback.is_empty() {
            self.fallback_pairs.fetch_add(fallback.len() as u64, Ordering::Relaxed);
            if self.double_fallback {
                self.run_fallback(self.precision, &fallback, &sorted_reads, haps, &mut tmp);
            }
        }
        for (pos, &read) in read_order.iter().enumerate() {
            for (k, &hap) in haps.order().iter().enumerate() {
                out[read * n_haps + hap] = tmp[pos * n_haps + k];
            }
        }
    }

    /// Runs every batch of reads against all haplotypes, returning the `(read, sorted
    /// haplotype)` pairs whose result must be recomputed in double precision.
    fn run_pass<H: HapSet>(
        &self,
        precision: Precision,
        reads: &[ReadRef<'_>],
        haps: &H,
        tmp: &mut [f64],
    ) -> Vec<(usize, usize)> {
        let wide = RunnerKey { backend: self.backend, precision, wide: true };
        let narrow = RunnerKey { backend: self.backend, precision, wide: false };
        // Full batches run on the wide instantiation; a remainder the narrow one can hold runs
        // there rather than leaving half the wide lanes empty. Only AVX-512 has both.
        let n = reads.len();
        let split = if self.backend.has_narrow_instantiation() {
            let rest = n % H::lanes(wide);
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
    fn run_batches<H: HapSet>(
        key: RunnerKey,
        reads: &[ReadRef<'_>],
        haps: &H,
        tmp: &mut [f64],
        first_read: usize,
        fallback: &mut Vec<(usize, usize)>,
    ) {
        if reads.is_empty() {
            return;
        }
        let lanes = H::lanes(key);
        let n_haps = haps.len();
        let mut batch = Vec::new();
        for (b, out) in tmp.chunks_mut(lanes * n_haps).enumerate() {
            let lo = b * lanes;
            let hi = (lo + lanes).min(reads.len());
            batch.clear();
            haps.run_batch(key, &reads[lo..hi], out, &mut batch);
            fallback.extend(batch.iter().map(|&(r, h)| (first_read + lo + r, h)));
        }
    }

    /// Recomputes the `(read, sorted haplotype)` pairs a pass in precision `first` lost. After
    /// a single-precision pass they are recomputed in double precision; any whose joined double
    /// result is still too small for the backward values, and any lost by a double-precision
    /// pass in the first place, are recomputed once more without suffix sharing, which cannot
    /// lose a pair.
    fn run_fallback<H: HapSet>(
        &self,
        first: Precision,
        pairs: &[(usize, usize)],
        reads: &[ReadRef<'_>],
        haps: &H,
        tmp: &mut [f64],
    ) {
        let lost = match first {
            Precision::Float => self.recompute_in_double(pairs, reads, haps, tmp),
            Precision::Double => pairs.to_vec(),
        };
        if !lost.is_empty() {
            let unshared = haps.without_suffix_sharing();
            let none = self.recompute_in_double(&lost, reads, &unshared, tmp);
            debug_assert!(
                none.is_empty(),
                "double precision without suffix sharing never falls back"
            );
        }
    }

    /// Recomputes the given pairs in double precision, returning those whose joined result lies
    /// below `Float::MIN_ACCEPTED_JOINED`. A read lost against a third or more of the haplotypes is
    /// run against all of them in one shared sweep, which is cheaper than an unshared sweep per
    /// haplotype; the other pairs are grouped by haplotype and run one haplotype at a time.
    fn recompute_in_double<H: HapSet>(
        &self,
        pairs: &[(usize, usize)],
        reads: &[ReadRef<'_>],
        haps: &H,
        tmp: &mut [f64],
    ) -> Vec<(usize, usize)> {
        let n_haps = haps.len();
        let mut lost = Vec::new();
        let mut per_read: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for &(r, h) in pairs {
            per_read.entry(r).or_default().push(h);
        }
        let dense: Vec<usize> =
            per_read.iter().filter(|(_, hs)| hs.len() * 3 >= n_haps).map(|(&r, _)| r).collect();
        if !dense.is_empty() {
            let batch: Vec<ReadRef<'_>> = dense.iter().map(|&r| reads[r]).collect();
            let mut out = vec![0.0f64; batch.len() * n_haps];
            let dense_lost = self.run_pass(Precision::Double, &batch, haps, &mut out);
            lost.extend(dense_lost.into_iter().map(|(i, h)| (dense[i], h)));
            for (i, &r) in dense.iter().enumerate() {
                for &h in &per_read[&r] {
                    tmp[r * n_haps + h] = out[i * n_haps + h];
                }
            }
            for r in &dense {
                per_read.remove(r);
            }
        }
        let mut by_hap: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (&r, hs) in &per_read {
            for &h in hs {
                by_hap.entry(h).or_default().push(r);
            }
        }
        let key = RunnerKey { backend: self.backend, precision: Precision::Double, wide: false };
        let lanes = H::lanes(key);
        let mut out = vec![0.0f64; lanes];
        let mut none = Vec::new();
        for (h, read_positions) in by_hap {
            let single = haps.single(h);
            for chunk in read_positions.chunks(lanes) {
                let batch: Vec<ReadRef<'_>> = chunk.iter().map(|&p| reads[p]).collect();
                single.run_batch(key, &batch, &mut out[..batch.len()], &mut none);
                for (i, &p) in chunk.iter().enumerate() {
                    tmp[p * n_haps + h] = out[i];
                }
            }
        }
        debug_assert!(none.is_empty(), "a single haplotype joins nothing, so it cannot be lost");
        lost
    }
}

/// A sorted haplotype set a kernel can run, with its own per-thread runner cache.
pub(crate) trait HapSet: Sized {
    fn len(&self) -> usize;
    /// `order()[k]` is the caller's index of sorted haplotype `k`.
    fn order(&self) -> &[usize];
    /// The set holding only sorted haplotype `k`, for per-haplotype double recomputation.
    fn single(&self, k: usize) -> Self;
    /// The same set, in the same sorted order, with no haplotype joining backward values: the
    /// last resort for pairs whose joined result lies below `Float::MIN_ACCEPTED_JOINED`.
    fn without_suffix_sharing(&self) -> Self;
    /// Lane count of the kernel instantiation `key` selects.
    fn lanes(key: RunnerKey) -> usize;
    /// Runs at most `lanes(key)` reads against every haplotype on this thread's cached runner;
    /// see `BatchRunner::run` for the output layout.
    fn run_batch(
        &self,
        key: RunnerKey,
        reads: &[ReadRef<'_>],
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    );
}

/// Which kernel instantiation to use: the widest lane group only pays off when a batch can fill it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct RunnerKey {
    pub backend: Backend,
    pub precision: Precision,
    pub wide: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use synthetic::Region;

    fn reference_all(reads: &[ReadRef<'_>], haps: &[&[u8]]) -> Vec<f64> {
        let mut out = Vec::with_capacity(reads.len() * haps.len());
        for read in reads {
            for hap in haps {
                out.push(reference::log10_likelihood(hap, read));
            }
        }
        out
    }

    fn compute(config: &Config, reads: &[ReadRef<'_>], haps: &[&[u8]]) -> Vec<f64> {
        let hmm = PairHmm::new(config).unwrap();
        let mut out = vec![0.0; reads.len() * haps.len()];
        hmm.compute_log10_likelihoods(reads, haps, &mut out).unwrap();
        out
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64, what: &str) {
        assert_eq!(actual.len(), expected.len());
        let mut worst = 0.0f64;
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let d = (a - e).abs();
            assert!(d <= tol, "{what}: pair {i}: got {a}, expected {e} (diff {d})");
            worst = worst.max(d);
        }
        assert!(worst.is_finite());
    }

    fn region() -> Region {
        Region::generate(11, 70, 120, 24, 300)
    }

    #[test]
    fn double_precision_matches_reference_on_every_backend() {
        let region = region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let config = Config {
                precision: Precision::Double,
                backend: Some(backend),
                ..Config::default()
            };
            assert_close(&compute(&config, &reads, &haps), &expected, 1e-9, backend.name());
        }
    }

    #[test]
    fn single_precision_matches_reference_within_float_tolerance() {
        let region = region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let config =
                Config { precision: Precision::Float, backend: Some(backend), ..Config::default() };
            assert_close(&compute(&config, &reads, &haps), &expected, 1e-4, backend.name());
        }
    }

    #[test]
    fn haplotype_order_does_not_change_results() {
        let region = region();
        let reads = region.read_refs();
        let mut haps = region.haplotype_refs();
        let config = Config { precision: Precision::Double, ..Config::default() };
        let forward = compute(&config, &reads, &haps);
        haps.reverse();
        let reversed = compute(&config, &reads, &haps);
        let n = haps.len();
        for r in 0..reads.len() {
            for h in 0..n {
                let (a, b) = (forward[r * n + h], reversed[r * n + (n - 1 - h)]);
                assert!((a - b).abs() < 1e-11, "read {r} hap {h}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn prefix_sharing_handles_nested_and_repeated_prefixes() {
        // Sorted order AAAB, AAAC, AAAC, ABBB, ABBC, ABBCC, AC: snapshots at columns 1, 3 and 4
        // must survive being needed after later haplotypes have moved past them.
        let haps: Vec<&[u8]> = vec![b"ABBC", b"AAAC", b"AC", b"AAAB", b"ABBB", b"AAAC", b"ABBCC"];
        let region = Region::generate(3, 20, 6, 1, 8);
        let reads = region.read_refs();
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let config = Config {
                precision: Precision::Double,
                backend: Some(backend),
                ..Config::default()
            };
            assert_close(&compute(&config, &reads, &haps), &expected, 1e-9, backend.name());
        }
    }

    #[test]
    fn haplotypes_of_different_lengths_share_prefixes_correctly() {
        let haps: Vec<&[u8]> = vec![b"ACGTACGTAC", b"ACGTACGT", b"ACGTACGTACGTACGT", b"ACGT", b"A"];
        let region = Region::generate(9, 40, 12, 1, 16);
        let reads = region.read_refs();
        let expected = reference_all(&reads, &haps);
        let config = Config { precision: Precision::Double, ..Config::default() };
        assert_close(&compute(&config, &reads, &haps), &expected, 1e-9, "lengths");
    }

    #[test]
    fn non_acgtn_bases_match_only_themselves() {
        let read = synthetic::Read {
            bases: b"ACRTN".to_vec(),
            quals: vec![30; 5],
            ins_gop: vec![45; 5],
            del_gop: vec![45; 5],
            gcp: vec![10; 5],
        };
        let reads = vec![read.as_ref()];
        let haps: Vec<&[u8]> = vec![b"ACRTG", b"ACGTG", b"ACNTG", b"RRRRR"];
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let config = Config {
                precision: Precision::Double,
                backend: Some(backend),
                ..Config::default()
            };
            assert_close(&compute(&config, &reads, &haps), &expected, 1e-9, backend.name());
        }
        // R matches R better than G, and N is at least as good as any base.
        assert!(expected[0] > expected[1]);
        assert!(expected[2] >= expected[0]);
    }

    #[test]
    fn float_underflow_falls_back_to_double() {
        let read = synthetic::Read {
            bases: vec![b'A'; 80],
            quals: vec![40; 80],
            ins_gop: vec![45; 80],
            del_gop: vec![45; 80],
            gcp: vec![10; 80],
        };
        let reads = vec![read.as_ref()];
        let hap = vec![b'C'; 80];
        let haps: Vec<&[u8]> = vec![&hap];
        let expected = reference_all(&reads, &haps);
        // The cheapest path inserts the whole read: about 10^-84, far below f32's range once
        // scaled by 2^120, so the single-precision pass must hand this pair to the f64 kernel.
        assert!(expected[0].is_finite() && expected[0] < -50.0);
        for backend in Backend::available() {
            let config =
                Config { precision: Precision::Float, backend: Some(backend), ..Config::default() };
            assert_close(&compute(&config, &reads, &haps), &expected, 1e-9, backend.name());
        }
    }

    fn config(precision: Precision, backend: Backend, share_suffixes: bool) -> Config {
        Config { precision, backend: Some(backend), double_fallback: true, share_suffixes }
    }

    /// A region whose haplotypes share long prefixes and suffixes, with reads of many lengths so
    /// the lanes of a batch end on different rows.
    fn suffix_sharing_region() -> Region {
        Region::generate(21, 90, 150, 40, 260)
    }

    #[test]
    fn suffix_sharing_matches_reference_in_double_on_every_backend() {
        let region = suffix_sharing_region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        assert!(SortedHaps::new(&haps, true).plan.cuts.iter().filter(|c| c.is_some()).count() > 20);
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let actual = compute(&config(Precision::Double, backend, true), &reads, &haps);
            assert_close(&actual, &expected, 1e-9, backend.name());
        }
    }

    #[test]
    fn suffix_sharing_matches_reference_in_float_within_tolerance() {
        let region = suffix_sharing_region();
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let actual = compute(&config(Precision::Float, backend, true), &reads, &haps);
            assert_close(&actual, &expected, 1e-4, backend.name());
        }
    }

    #[test]
    fn random_small_haplotype_sets_match_reference_with_suffix_sharing() {
        // Two-letter haplotypes derived from one another give deep, nested and overlapping
        // shared prefixes and suffixes, and duplicates.
        let mut rng = synthetic::Rng::new(31);
        let letter = |rng: &mut synthetic::Rng| if rng.chance(0.5) { b'A' } else { b'C' };
        for _ in 0..60 {
            let mut haps: Vec<Vec<u8>> =
                vec![(0..6 + rng.below(30)).map(|_| letter(&mut rng)).collect()];
            for _ in 0..rng.below(12) {
                let from = rng.below(haps.len());
                let hap = synthetic::edited(&mut rng, b"AC", &haps[from]);
                haps.push(hap);
            }
            let reads: Vec<synthetic::Read> = (0..1 + rng.below(20))
                .map(|_| {
                    let len = 1 + rng.below(25);
                    synthetic::Read {
                        bases: (0..len).map(|_| letter(&mut rng)).collect(),
                        quals: (0..len).map(|_| 10 + rng.below(31) as u8).collect(),
                        ins_gop: (0..len).map(|_| 20 + rng.below(26) as u8).collect(),
                        del_gop: (0..len).map(|_| 20 + rng.below(26) as u8).collect(),
                        gcp: (0..len).map(|_| 1 + rng.below(10) as u8).collect(),
                    }
                })
                .collect();
            let reads: Vec<ReadRef<'_>> = reads.iter().map(synthetic::Read::as_ref).collect();
            let haps: Vec<&[u8]> = haps.iter().map(Vec::as_slice).collect();
            let expected = reference_all(&reads, &haps);
            for backend in Backend::available() {
                let actual = compute(&config(Precision::Double, backend, true), &reads, &haps);
                assert_close(&actual, &expected, 1e-9, backend.name());
            }
        }
    }

    #[test]
    fn haplotypes_without_a_cut_keep_their_prefix_only_bits() {
        let region = Region::generate(23, 20, 90, 24, 160);
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let sorted = SortedHaps::new(&haps, true);
        let uncut: Vec<usize> = (0..haps.len())
            .filter(|&k| sorted.plan.cuts[k].is_none() && !sorted.dup[k])
            .map(|k| sorted.order[k])
            .collect();
        assert!(!uncut.is_empty());
        for backend in Backend::available() {
            for precision in [Precision::Float, Precision::Double] {
                let shared = compute(&config(precision, backend, true), &reads, &haps);
                let prefix_only = compute(&config(precision, backend, false), &reads, &haps);
                for r in 0..reads.len() {
                    for &h in &uncut {
                        let i = r * haps.len() + h;
                        assert_eq!(
                            shared[i].to_bits(),
                            prefix_only[i].to_bits(),
                            "{backend} {precision}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn deep_pairs_fall_back_to_double_with_suffix_sharing() {
        // Haplotypes of As sharing prefixes and suffixes, against reads of Cs: every pair lies far
        // below single precision's range, whether its haplotype joins backward values or not.
        let base = vec![b'A'; 90];
        let mut haps: Vec<Vec<u8>> = vec![base.clone()];
        for pos in [20, 45, 70] {
            let mut hap = base.clone();
            hap[pos] = b'G';
            haps.push(hap);
        }
        let haps: Vec<&[u8]> = haps.iter().map(Vec::as_slice).collect();
        assert!(SortedHaps::new(&haps, true).plan.cuts.iter().any(Option::is_some));
        let read = synthetic::Read {
            bases: vec![b'C'; 70],
            quals: vec![40; 70],
            ins_gop: vec![45; 70],
            del_gop: vec![45; 70],
            gcp: vec![10; 70],
        };
        let reads = vec![read.as_ref()];
        let expected = reference_all(&reads, &haps);
        assert!(expected.iter().all(|&e| e < -50.0));
        for backend in Backend::available() {
            let hmm = PairHmm::new(&config(Precision::Float, backend, true)).unwrap();
            let mut out = vec![0.0; haps.len()];
            hmm.compute_log10_likelihoods(&reads, &haps, &mut out).unwrap();
            assert_eq!(hmm.fallback_pairs(), haps.len() as u64, "{backend}");
            assert_close(&out, &expected, 1e-9, backend.name());
        }
    }

    #[test]
    fn likelihoods_below_the_backward_range_are_recomputed_without_suffix_sharing() {
        // Long reads unrelated to haplotypes that share suffixes: likelihoods far below 1e-308,
        // where double-precision backward values flush, so the joined results must be replaced.
        let mut rng = synthetic::Rng::new(99);
        let reference: Vec<u8> = (0..700).map(|_| rng.base()).collect();
        let mut haps = vec![reference.clone()];
        for pos in [20, 40, 60] {
            let mut hap = reference.clone();
            hap[pos] = if hap[pos] == b'A' { b'C' } else { b'A' };
            haps.push(hap);
        }
        let haps: Vec<&[u8]> = haps.iter().map(Vec::as_slice).collect();
        assert!(SortedHaps::new(&haps, true).plan.cuts.iter().any(Option::is_some));
        let reads: Vec<synthetic::Read> = [290, 360, 500]
            .iter()
            .map(|&len| synthetic::Read {
                bases: (0..len).map(|_| rng.base()).collect(),
                quals: vec![40; len],
                ins_gop: vec![45; len],
                del_gop: vec![45; len],
                gcp: vec![10; len],
            })
            .collect();
        let reads: Vec<ReadRef<'_>> = reads.iter().map(synthetic::Read::as_ref).collect();
        let expected = reference_all(&reads, &haps);
        assert!(expected.iter().any(|&e| e < -400.0));
        for backend in Backend::available() {
            for precision in [Precision::Double, Precision::Float] {
                let actual = compute(&config(precision, backend, true), &reads, &haps);
                assert_close(&actual, &expected, 1e-9, &format!("{backend} {precision}"));
            }
        }
    }

    fn foreign_reads(seed: u64, n: usize, len: usize) -> Vec<synthetic::Read> {
        // Reads unrelated to any haplotype of the region under test: a random read mismatches
        // three bases in four, so its likelihood is far below single precision's range.
        let mut rng = synthetic::Rng::new(seed);
        (0..n)
            .map(|_| synthetic::Read {
                bases: (0..len).map(|_| rng.base()).collect(),
                quals: (0..len).map(|_| 20 + rng.below(21) as u8).collect(),
                ins_gop: vec![45; len],
                del_gop: vec![45; len],
                gcp: vec![10; len],
            })
            .collect()
    }

    fn compute_counting(config: &Config, reads: &[ReadRef<'_>], haps: &[&[u8]]) -> (Vec<f64>, u64) {
        let hmm = PairHmm::new(config).unwrap();
        let mut out = vec![0.0; reads.len() * haps.len()];
        hmm.compute_log10_likelihoods(reads, haps, &mut out).unwrap();
        (out, hmm.fallback_pairs())
    }

    #[test]
    fn reads_unrelated_to_every_haplotype_are_recomputed_in_double() {
        // Every pair underflows single precision, so every pair goes through the fallback.
        let region = region();
        let haps = region.haplotype_refs();
        let foreign = foreign_reads(5, 40, 120);
        let reads: Vec<ReadRef<'_>> = foreign.iter().map(synthetic::Read::as_ref).collect();
        let expected = reference_all(&reads, &haps);
        assert!(expected.iter().all(|&e| e < -64.0), "test reads must underflow f32");
        for backend in Backend::available() {
            let config =
                Config { precision: Precision::Float, backend: Some(backend), ..Config::default() };
            let (out, fallbacks) = compute_counting(&config, &reads, &haps);
            assert_close(&out, &expected, 1e-9, backend.name());
            assert_eq!(fallbacks, expected.len() as u64, "{}", backend.name());
        }
    }

    #[test]
    fn matching_and_mismatching_reads_of_varied_lengths_share_a_batch() {
        let region = region();
        let haps = region.haplotype_refs();
        let mut owned: Vec<synthetic::Read> = region.reads.iter().take(20).cloned().collect();
        owned.extend(foreign_reads(6, 20, 120));
        // Every lane ends on a different row.
        for (i, read) in owned.iter_mut().enumerate() {
            let len = 30 + (i * 7) % 90;
            read.bases.truncate(len);
            read.quals.truncate(len);
            read.ins_gop.truncate(len);
            read.del_gop.truncate(len);
            read.gcp.truncate(len);
        }
        let reads: Vec<ReadRef<'_>> = owned.iter().map(synthetic::Read::as_ref).collect();
        let expected = reference_all(&reads, &haps);
        for backend in Backend::available() {
            let config =
                Config { precision: Precision::Float, backend: Some(backend), ..Config::default() };
            let (out, fallbacks) = compute_counting(&config, &reads, &haps);
            assert_close(&out, &expected, 1e-4, backend.name());
            assert!(fallbacks > 0, "{}", backend.name());
        }
    }

    #[test]
    fn a_read_gets_the_same_bits_whatever_shares_its_batch() {
        // Lanes are independent, so a read's result must not change with its batch mates, even
        // when they underflow and are recomputed.
        let region = region();
        let haps = region.haplotype_refs();
        let alone: Vec<ReadRef<'_>> = region.read_refs().into_iter().take(3).collect();
        let mut owned: Vec<synthetic::Read> = region.reads.iter().take(3).cloned().collect();
        owned.extend(foreign_reads(7, 29, 120));
        let batched: Vec<ReadRef<'_>> = owned.iter().map(synthetic::Read::as_ref).collect();
        for precision in [Precision::Float, Precision::Double] {
            let config = Config { precision, ..Config::default() };
            let solo = compute(&config, &alone, &haps);
            let with_others = compute(&config, &batched, &haps);
            assert_eq!(solo[..], with_others[..solo.len()], "{precision:?}");
        }
    }

    #[test]
    fn a_read_matching_only_the_previous_haplotype_is_recomputed_in_double() {
        // Reads drawn from H2's random tail match H2 but nothing else. In H2's sweep such a read
        // keeps a high scale, so the shared column's "insert the rest of the read" states, which
        // H1 and H3 need, underflow single precision there; those pairs must be recomputed,
        // while reads from the reference match every haplotype normally.
        let mut rng = synthetic::Rng::new(8);
        let reference: Vec<u8> = (0..300).map(|_| rng.base()).collect();
        let tail: Vec<u8> = (0..150).map(|_| rng.base()).collect();
        let h1 = reference.clone();
        let mut h2 = reference[..150].to_vec();
        h2.extend_from_slice(&tail);
        let mut h3 = reference.clone();
        h3[200] = if h3[200] == b'A' { b'C' } else { b'A' };
        let haps: Vec<&[u8]> = vec![&h1, &h2, &h3];
        let mut owned = Vec::new();
        for k in 0..12 {
            let source: &[u8] = if k % 2 == 0 { &reference } else { &h2 };
            let start = 100 + (k * 13) % 80;
            owned.push(synthetic::Read {
                bases: source[start..start + 100].to_vec(),
                quals: vec![35; 100],
                ins_gop: vec![45; 100],
                del_gop: vec![45; 100],
                gcp: vec![10; 100],
            });
        }
        let reads: Vec<ReadRef<'_>> = owned.iter().map(synthetic::Read::as_ref).collect();
        let expected = reference_all(&reads, &haps);
        assert!(expected.iter().any(|&e| e < -64.0) && expected.iter().any(|&e| e > -10.0));
        for backend in Backend::available() {
            let config =
                Config { precision: Precision::Float, backend: Some(backend), ..Config::default() };
            let (out, fallbacks) = compute_counting(&config, &reads, &haps);
            assert_close(&out, &expected, 1e-4, backend.name());
            assert!(fallbacks > 0 && fallbacks <= 24, "{}: {fallbacks}", backend.name());
        }
    }

    #[test]
    fn more_reads_than_lanes_and_single_read_both_work() {
        let region = Region::generate(21, 1, 50, 3, 60);
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        assert_close(&compute(&Config::default(), &reads, &haps), &expected, 1e-4, "single");
        let region = Region::generate(22, 37, 50, 3, 60);
        let (reads, haps) = (region.read_refs(), region.haplotype_refs());
        let expected = reference_all(&reads, &haps);
        assert_close(&compute(&Config::default(), &reads, &haps), &expected, 1e-4, "many");
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let hmm = PairHmm::new(&Config::default()).unwrap();
        let good = synthetic::Read {
            bases: b"ACGT".to_vec(),
            quals: vec![30; 4],
            ins_gop: vec![45; 4],
            del_gop: vec![45; 4],
            gcp: vec![10; 4],
        };
        let short_quals = synthetic::Read { quals: vec![30; 3], ..good.clone() };
        let mut out = vec![0.0; 1];
        let hap: &[u8] = b"ACGT";
        assert_eq!(
            hmm.compute_log10_likelihoods(&[short_quals.as_ref()], &[hap], &mut out),
            Err(Error::MismatchedReadArrays(0))
        );
        assert_eq!(
            hmm.compute_log10_likelihoods(&[good.as_ref()], &[b""], &mut out),
            Err(Error::EmptyHaplotype(0))
        );
        assert_eq!(
            hmm.compute_log10_likelihoods(&[good.as_ref()], &[hap, hap], &mut out),
            Err(Error::OutputLength { expected: 2, actual: 1 })
        );
        assert!(hmm.compute_log10_likelihoods(&[], &[hap], &mut []).is_ok());
    }
}
