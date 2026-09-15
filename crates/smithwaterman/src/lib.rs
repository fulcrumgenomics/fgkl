//! Smith-Waterman alignment exactly as GATK's `SmithWatermanJavaAligner` computes it: affine gaps
//! tracked per column and per row, ties resolved in favour of the diagonal, then the insertion,
//! then the deletion, and GATK's four overhang strategies for choosing where the alignment ends
//! and how unaligned sequence is reported.
//!
//! [`reference`] is a line-by-line port kept as the oracle; [`Aligner`] produces identical
//! alignments with two score rows, reusable buffers, and one byte of traceback per cell. The SIMD
//! backends fill the matrix anti-diagonal by anti-diagonal in 16-bit lanes first and redo the rare
//! pair whose scores saturate them in 32-bit lanes (see `diag`).

mod diag;
pub mod reference;
mod simd;

use std::fmt;

use diag::{DiagFill, DiagWorkspace, narrow_lanes_apply};

/// GATK's `MATRIX_MIN_CUTOFF`: no cell score drops below this.
const MATRIX_MIN_CUTOFF: i32 = -100_000_000;
/// GATK's `lowInitValue` for gap scores before any gap has been opened.
const LOW_INIT_VALUE: i32 = i32::MIN / 2;

/// Traceback flags, one byte per cell.
const DIR_MASK: u8 = 0b11;
const DIR_DIAG: u8 = 0;
const DIR_INSERTION: u8 = 1;
const DIR_DELETION: u8 = 2;
/// The best vertical gap ending here extends the one ending in the cell above.
const DELETION_EXTENDS: u8 = 0b100;
/// The best horizontal gap ending here extends the one ending in the cell to the left.
const INSERTION_EXTENDS: u8 = 0b1000;

/// Scoring parameters as GATK's `SWParameters`: a positive match value and negative penalties.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwParameters {
    pub match_value: i32,
    pub mismatch_penalty: i32,
    pub gap_open_penalty: i32,
    pub gap_extend_penalty: i32,
}

impl SwParameters {
    pub const fn new(
        match_value: i32,
        mismatch_penalty: i32,
        gap_open_penalty: i32,
        gap_extend_penalty: i32,
    ) -> Self {
        SwParameters { match_value, mismatch_penalty, gap_open_penalty, gap_extend_penalty }
    }
}

/// How overhanging sequence at either end is treated; GATK's `SWOverhangStrategy`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverhangStrategy {
    /// Overhangs of the alternate become soft clips; the alignment may end anywhere.
    SoftClip,
    /// Overhangs at both ends are indels; the alignment runs corner to corner.
    Indel,
    /// A leading overhang is an indel; the alignment ends at the last alternate base.
    LeadingIndel,
    /// Overhangs are folded into the flanking match segment and the offset.
    Ignore,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CigarOp {
    M,
    I,
    D,
    S,
}

impl CigarOp {
    pub fn letter(self) -> char {
        match self {
            CigarOp::M => 'M',
            CigarOp::I => 'I',
            CigarOp::D => 'D',
            CigarOp::S => 'S',
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CigarElement {
    pub len: u32,
    pub op: CigarOp,
}

/// An alignment of an alternate sequence to a reference: the CIGAR in alternate order and the
/// reference offset of its first aligned base (GATK's `alignmentOffset`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Alignment {
    pub cigar: Vec<CigarElement>,
    pub offset: i32,
}

impl Alignment {
    pub fn cigar_string(&self) -> String {
        let mut s = String::with_capacity(self.cigar.len() * 4);
        for e in &self.cigar {
            s.push_str(&e.len.to_string());
            s.push(e.op.letter());
        }
        s
    }
}

impl fmt::Display for Alignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.cigar_string(), self.offset)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    EmptySequence,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptySequence => f.write_str("sequences must be non-empty"),
        }
    }
}

impl std::error::Error for Error {}

/// The vector instruction set an [`Aligner`] fills the matrix with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    /// Row by row without vector instructions.
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "x86_64")]
    Avx2,
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

    pub fn available() -> Vec<Backend> {
        Self::all().iter().copied().filter(|b| b.is_available()).collect()
    }

    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => is_x86_feature_detected!("avx2"),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => {
                is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw")
            }
        }
    }

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

    /// The 16-bit and 32-bit lane workspaces of this backend; none for the scalar backend.
    #[allow(clippy::type_complexity)]
    fn make_fills(self) -> (Option<Box<dyn DiagFill>>, Option<Box<dyn DiagFill>>) {
        match self {
            Backend::Scalar => (None, None),
            #[cfg(target_arch = "aarch64")]
            Backend::Neon => (
                Some(Box::new(DiagWorkspace::<simd::neon::I16x8>::new())),
                Some(Box::new(DiagWorkspace::<simd::neon::I32x4>::new())),
            ),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx2 => (
                Some(Box::new(DiagWorkspace::<simd::x86::I16x16>::new())),
                Some(Box::new(DiagWorkspace::<simd::x86::I32x8>::new())),
            ),
            #[cfg(target_arch = "x86_64")]
            Backend::Avx512 => (
                Some(Box::new(DiagWorkspace::<simd::x86::I16x32>::new())),
                Some(Box::new(DiagWorkspace::<simd::x86::I32x16>::new())),
            ),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// A reusable aligner: keeps its scratch buffers between calls, so one per thread is the
/// intended use.
pub struct Aligner {
    backend: Backend,
    /// The 16-bit-lane fill, tried first; absent on the scalar backend or when disabled.
    narrow: Option<Box<dyn DiagFill>>,
    /// The 32-bit-lane fill, used when the narrow one saturates.
    wide: Option<Box<dyn DiagFill>>,
    narrow_fallbacks: u64,
    h_prev: Vec<i32>,
    h_cur: Vec<i32>,
    /// Best vertical-gap score ending in each column of the current row.
    f: Vec<i32>,
    /// Score of the last column in each row, for choosing the alignment end.
    last_col: Vec<i32>,
    trace: Vec<u8>,
}

impl Default for Aligner {
    fn default() -> Self {
        Aligner::new()
    }
}

impl Aligner {
    /// An aligner on the fastest backend this CPU supports.
    pub fn new() -> Self {
        Aligner::with_backend(Backend::detect())
    }

    /// An aligner on a specific backend, which must be available on this CPU.
    pub fn with_backend(backend: Backend) -> Self {
        assert!(backend.is_available(), "backend {backend} is not supported by this CPU");
        let (narrow, wide) = backend.make_fills();
        Aligner {
            backend,
            narrow,
            wide,
            narrow_fallbacks: 0,
            h_prev: Vec::new(),
            h_cur: Vec::new(),
            f: Vec::new(),
            last_col: Vec::new(),
            trace: Vec::new(),
        }
    }

    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The same aligner using only 32-bit lanes, for measurements and tests.
    pub fn without_narrow_lanes(mut self) -> Self {
        self.narrow = None;
        self
    }

    /// How many alignments so far saturated the 16-bit lanes and were redone in 32-bit lanes.
    pub fn narrow_fallbacks(&self) -> u64 {
        self.narrow_fallbacks
    }

    /// Aligns `alternate` to `reference`. Both must be non-empty.
    pub fn align(
        &mut self,
        reference: &[u8],
        alternate: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) -> Result<Alignment, Error> {
        if reference.is_empty() || alternate.is_empty() {
            return Err(Error::EmptySequence);
        }
        if matches!(strategy, OverhangStrategy::SoftClip | OverhangStrategy::Ignore)
            && let Some(offset) = last_index_of(reference, alternate)
        {
            stats::record(reference, alternate, params, strategy, true);
            return Ok(Alignment {
                cigar: vec![CigarElement { len: alternate.len() as u32, op: CigarOp::M }],
                offset: offset as i32,
            });
        }
        stats::record(reference, alternate, params, strategy, false);
        if narrow_lanes_apply(params)
            && let Some(narrow) = self.narrow.as_mut()
        {
            if let Some((end_row, end_col, trailing)) =
                narrow.fill(reference, alternate, params, strategy)
            {
                let narrow: &dyn DiagFill = &**narrow;
                return Ok(traceback(
                    |i, j| narrow.trace_at(i, j),
                    alternate.len(),
                    end_row,
                    end_col,
                    trailing,
                    strategy,
                ));
            }
            self.narrow_fallbacks += 1;
            stats::record_fallback(params, strategy);
        }
        if let Some(wide) = self.wide.as_mut()
            && let Some((end_row, end_col, trailing)) =
                wide.fill(reference, alternate, params, strategy)
        {
            let wide: &dyn DiagFill = &**wide;
            return Ok(traceback(
                |i, j| wide.trace_at(i, j),
                alternate.len(),
                end_row,
                end_col,
                trailing,
                strategy,
            ));
        }
        let (end_row, end_col, trailing) = self.fill(reference, alternate, params, strategy);
        let cols = alternate.len() + 1;
        let trace = &self.trace;
        Ok(traceback(
            |i, j| trace[i * cols + j],
            alternate.len(),
            end_row,
            end_col,
            trailing,
            strategy,
        ))
    }

    /// Fills the traceback and returns the cell the alignment ends in plus the number of
    /// trailing alternate bases left unaligned by the bottom-row scan.
    fn fill(
        &mut self,
        reference: &[u8],
        alternate: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) -> (usize, usize, usize) {
        let rows = reference.len() + 1;
        let cols = alternate.len() + 1;
        let (w_match, w_mismatch, w_open, w_extend) = (
            params.match_value,
            params.mismatch_penalty,
            params.gap_open_penalty,
            params.gap_extend_penalty,
        );
        self.h_prev.clear();
        self.h_prev.resize(cols, 0);
        self.h_cur.clear();
        self.h_cur.resize(cols, 0);
        self.f.clear();
        self.f.resize(cols, LOW_INIT_VALUE);
        self.last_col.clear();
        self.last_col.resize(rows, 0);
        self.trace.clear();
        self.trace.resize(rows * cols, 0);

        let global_ends =
            matches!(strategy, OverhangStrategy::Indel | OverhangStrategy::LeadingIndel);
        if global_ends {
            let mut v = w_open;
            for j in 1..cols {
                self.h_prev[j] = v;
                v += w_extend;
            }
        }
        let first_col =
            |i: usize| -> i32 { if global_ends { w_open + (i as i32 - 1) * w_extend } else { 0 } };

        for i in 1..rows {
            let a = reference[i - 1];
            self.h_cur[0] = first_col(i);
            let mut e = LOW_INIT_VALUE;
            let row_trace = &mut self.trace[i * cols..(i + 1) * cols];
            for j in 1..cols {
                let b = alternate[j - 1];
                let diag = self.h_prev[j - 1] + if a == b { w_match } else { w_mismatch };
                let mut flags = 0u8;
                let open_down = self.h_prev[j] + w_open;
                self.f[j] += w_extend;
                if open_down > self.f[j] {
                    self.f[j] = open_down;
                } else {
                    flags |= DELETION_EXTENDS;
                }
                let down = self.f[j];
                let open_right = self.h_cur[j - 1] + w_open;
                e += w_extend;
                if open_right > e {
                    e = open_right;
                } else {
                    flags |= INSERTION_EXTENDS;
                }
                let right = e;
                let score = if diag >= down && diag >= right {
                    flags |= DIR_DIAG;
                    diag
                } else if right >= down {
                    flags |= DIR_INSERTION;
                    right
                } else {
                    flags |= DIR_DELETION;
                    down
                };
                self.h_cur[j] = score.max(MATRIX_MIN_CUTOFF);
                row_trace[j] = flags;
            }
            self.last_col[i] = self.h_cur[cols - 1];
            std::mem::swap(&mut self.h_prev, &mut self.h_cur);
        }
        // After the swap the bottom row is in h_prev.
        select_end(strategy, rows - 1, cols - 1, &self.last_col, &self.h_prev)
    }
}

/// Chooses the cell the alignment ends in from the last column and bottom row scores, as GATK's
/// `calculateCigar` does, returning `(row, column, trailing alternate bases)`.
fn select_end(
    strategy: OverhangStrategy,
    ref_len: usize,
    alt_len: usize,
    last_col: &[i32],
    bottom: &[i32],
) -> (usize, usize, usize) {
    let mut p1 = 0usize;
    let mut p2 = alt_len;
    let mut trailing = 0usize;
    if strategy == OverhangStrategy::Indel {
        p1 = ref_len;
    } else {
        let mut max_score = i32::MIN;
        for (i, &score) in last_col.iter().enumerate().take(ref_len + 1).skip(1) {
            if score >= max_score {
                p1 = i;
                max_score = score;
            }
        }
        if strategy != OverhangStrategy::LeadingIndel {
            for (j, &cur) in bottom.iter().enumerate().take(alt_len + 1).skip(1) {
                let closer = (ref_len as i32 - j as i32).abs() < (p1 as i32 - p2 as i32).abs();
                if cur > max_score || (cur == max_score && closer) {
                    p1 = ref_len;
                    p2 = j;
                    max_score = cur;
                    trailing = alt_len - j;
                }
            }
        }
    }
    (p1, p2, trailing)
}

/// Walks the traceback from the end cell `(p1, p2)` and builds the CIGAR exactly as GATK does,
/// reading each cell's flags through `trace`.
fn traceback(
    trace: impl Fn(usize, usize) -> u8,
    alt_len: usize,
    mut p1: usize,
    mut p2: usize,
    trailing: usize,
    strategy: OverhangStrategy,
) -> Alignment {
    let _ = alt_len;
    let mut elements: Vec<CigarElement> = Vec::with_capacity(8);
    let mut segment = trailing;
    if segment > 0 && strategy == OverhangStrategy::SoftClip {
        elements.push(CigarElement { len: segment as u32, op: CigarOp::S });
        segment = 0;
    }
    let mut state = CigarOp::M;
    loop {
        let flags = trace(p1, p2);
        let (new_state, step) = match flags & DIR_MASK {
            DIR_DELETION => {
                let mut len = 1;
                let mut r = p1;
                while trace(r, p2) & DELETION_EXTENDS != 0 {
                    len += 1;
                    r -= 1;
                }
                (CigarOp::D, len)
            }
            DIR_INSERTION => {
                let mut len = 1;
                let mut c = p2;
                while trace(p1, c) & INSERTION_EXTENDS != 0 {
                    len += 1;
                    c -= 1;
                }
                (CigarOp::I, len)
            }
            _ => (CigarOp::M, 1),
        };
        match new_state {
            CigarOp::M => {
                p1 -= 1;
                p2 -= 1;
            }
            CigarOp::I => p2 -= step,
            CigarOp::D => p1 -= step,
            CigarOp::S => unreachable!(),
        }
        if new_state == state {
            segment += step;
        } else {
            if segment > 0 {
                elements.push(CigarElement { len: segment as u32, op: state });
            }
            segment = step;
            state = new_state;
        }
        if p1 == 0 || p2 == 0 {
            break;
        }
    }
    let offset = match strategy {
        OverhangStrategy::SoftClip => {
            elements.push(CigarElement { len: segment as u32, op: state });
            if p2 > 0 {
                elements.push(CigarElement { len: p2 as u32, op: CigarOp::S });
            }
            p1 as i32
        }
        OverhangStrategy::Ignore => {
            elements.push(CigarElement { len: (segment + p2) as u32, op: state });
            p1 as i32 - p2 as i32
        }
        OverhangStrategy::Indel | OverhangStrategy::LeadingIndel => {
            elements.push(CigarElement { len: segment as u32, op: state });
            if p1 > 0 {
                elements.push(CigarElement { len: p1 as u32, op: CigarOp::D });
            } else if p2 > 0 {
                elements.push(CigarElement { len: p2 as u32, op: CigarOp::I });
            }
            0
        }
    };
    elements.reverse();
    Alignment { cigar: elements, offset }
}

/// GATK's `Utils.lastIndexOf`: the start of the last occurrence of `query` in `reference`.
pub fn last_index_of(reference: &[u8], query: &[u8]) -> Option<usize> {
    if query.len() > reference.len() {
        return None;
    }
    (0..=reference.len() - query.len()).rev().find(|&r| &reference[r..r + query.len()] == query)
}

/// Per-parameter-set counters of what the aligner is asked to do, kept only while the
/// environment variable `FGKL_SW_STATS` is set and written to stderr when the process exits.
/// They say how many calls the exact-match shortcut answers, how many cells the kernel fills, how
/// many sequence pairs repeat, and how many alignments fell back from 16-bit to 32-bit lanes.
mod stats {
    use std::collections::{BTreeMap, HashSet};
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::{Mutex, Once, OnceLock};

    use super::{OverhangStrategy, SwParameters};

    #[derive(Default)]
    struct Counters {
        calls: u64,
        exact: u64,
        cells: u64,
        narrow_fallbacks: u64,
        max_reference: usize,
        max_alternate: usize,
        /// Hashes of the (reference, alternate) pairs and of the alternates alone that were
        /// actually aligned, to see how many alignments repeat an earlier one.
        distinct_pairs: HashSet<u64>,
        distinct_alternates: HashSet<u64>,
    }

    fn key(params: &SwParameters, strategy: OverhangStrategy) -> Key {
        let strategy_name = match strategy {
            OverhangStrategy::SoftClip => "softclip",
            OverhangStrategy::Indel => "indel",
            OverhangStrategy::LeadingIndel => "leading_indel",
            OverhangStrategy::Ignore => "ignore",
        };
        (
            params.match_value,
            params.mismatch_penalty,
            params.gap_open_penalty,
            params.gap_extend_penalty,
            strategy_name,
        )
    }

    fn hash_bytes(parts: &[&[u8]]) -> u64 {
        let mut hasher = DefaultHasher::new();
        for part in parts {
            part.hash(&mut hasher);
        }
        hasher.finish()
    }

    type Key = (i32, i32, i32, i32, &'static str);

    static ENABLED: OnceLock<bool> = OnceLock::new();
    static COUNTERS: Mutex<BTreeMap<Key, Counters>> = Mutex::new(BTreeMap::new());
    static REGISTER_EXIT_HOOK: Once = Once::new();

    unsafe extern "C" {
        fn atexit(callback: extern "C" fn()) -> i32;
    }

    extern "C" fn print_at_exit() {
        print();
    }

    pub(super) fn record(
        reference: &[u8],
        alternate: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
        exact: bool,
    ) {
        let (reference_len, alternate_len) = (reference.len(), alternate.len());
        if !*ENABLED.get_or_init(|| std::env::var_os("FGKL_SW_STATS").is_some()) {
            return;
        }
        REGISTER_EXIT_HOOK.call_once(|| {
            // SAFETY: registering a plain `extern "C" fn()` with the C runtime's exit hook.
            unsafe {
                atexit(print_at_exit);
            }
        });
        let mut counters = COUNTERS.lock().unwrap();
        let c = counters.entry(key(params, strategy)).or_default();
        c.calls += 1;
        c.max_reference = c.max_reference.max(reference_len);
        c.max_alternate = c.max_alternate.max(alternate_len);
        if exact {
            c.exact += 1;
            return;
        }
        let cells = reference_len as u64 * alternate_len as u64;
        c.cells += cells;
        c.distinct_pairs.insert(hash_bytes(&[reference, alternate]));
        c.distinct_alternates.insert(hash_bytes(&[alternate]));
    }

    pub(super) fn record_fallback(params: &SwParameters, strategy: OverhangStrategy) {
        if !*ENABLED.get_or_init(|| std::env::var_os("FGKL_SW_STATS").is_some()) {
            return;
        }
        let mut counters = COUNTERS.lock().unwrap();
        counters.entry(key(params, strategy)).or_default().narrow_fallbacks += 1;
    }

    /// Writes one line per parameter set and overhang strategy to stderr.
    pub fn print() {
        let counters = COUNTERS.lock().unwrap();
        for (&(m, x, o, e, st), c) in counters.iter() {
            eprintln!(
                "fgkl sw stats  params={m}/{x}/{o}/{e} strategy={st} calls={} exact_match={} aligned={} distinct_pairs={} distinct_alternates={} cells={} narrow_fallbacks={} max_reference={} max_alternate={}",
                c.calls,
                c.exact,
                c.calls - c.exact,
                c.distinct_pairs.len(),
                c.distinct_alternates.len(),
                c.cells,
                c.narrow_fallbacks,
                c.max_reference,
                c.max_alternate
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HAP_TO_REF: SwParameters = SwParameters::new(200, -150, -260, -11);
    const READ_TO_HAP: SwParameters = SwParameters::new(10, -15, -30, -5);
    const ORIGINAL: SwParameters = SwParameters::new(3, -1, -4, -3);
    const STRATEGIES: [OverhangStrategy; 4] = [
        OverhangStrategy::SoftClip,
        OverhangStrategy::Indel,
        OverhangStrategy::LeadingIndel,
        OverhangStrategy::Ignore,
    ];

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
        fn base(&mut self) -> u8 {
            b"ACGT"[self.below(4)]
        }
        fn chance(&mut self, p: f64) -> bool {
            ((self.next() >> 11) as f64 / (1u64 << 53) as f64) < p
        }
    }

    /// A reference and an edited copy of a window of it, like a haplotype or a read.
    fn related_pair(rng: &mut Rng, ref_len: usize, alt_len: usize) -> (Vec<u8>, Vec<u8>) {
        let reference: Vec<u8> = (0..ref_len).map(|_| rng.base()).collect();
        let start = rng.below(ref_len.saturating_sub(alt_len) + 1);
        let mut alt: Vec<u8> = reference[start..(start + alt_len).min(ref_len)].to_vec();
        for _ in 0..rng.below(4) {
            if alt.is_empty() {
                break;
            }
            let pos = rng.below(alt.len());
            match rng.below(3) {
                0 => alt[pos] = rng.base(),
                1 => {
                    let ins: Vec<u8> = (0..1 + rng.below(6)).map(|_| rng.base()).collect();
                    alt.splice(pos..pos, ins);
                }
                _ => {
                    let end = (pos + 1 + rng.below(6)).min(alt.len());
                    if alt.len() > end - pos {
                        alt.drain(pos..end);
                    }
                }
            }
        }
        if rng.chance(0.3) {
            let extra: Vec<u8> = (0..1 + rng.below(20)).map(|_| rng.base()).collect();
            if rng.chance(0.5) {
                alt.splice(0..0, extra);
            } else {
                alt.extend(extra);
            }
        }
        if alt.is_empty() {
            alt.push(b'A');
        }
        (reference, alt)
    }

    fn assert_same(
        reference: &[u8],
        alt: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) {
        let expected = reference::align(reference, alt, params, strategy);
        for backend in Backend::available() {
            for narrow in [true, false] {
                let mut aligner = Aligner::with_backend(backend);
                if !narrow {
                    aligner = aligner.without_narrow_lanes();
                }
                let actual = aligner.align(reference, alt, params, strategy).unwrap();
                assert_eq!(
                    actual,
                    expected,
                    "{backend} narrow={narrow} {strategy:?} {params:?}\nref={}\nalt={}",
                    String::from_utf8_lossy(reference),
                    String::from_utf8_lossy(alt)
                );
            }
        }
    }

    /// Runs one pair on every SIMD backend with 16-bit lanes and returns how many fell back.
    fn narrow_fallbacks_on_simd_backends(
        reference: &[u8],
        alt: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) -> Vec<(Backend, u64)> {
        Backend::available()
            .into_iter()
            .filter(|&b| b != Backend::Scalar)
            .map(|backend| {
                let mut aligner = Aligner::with_backend(backend);
                let actual = aligner.align(reference, alt, params, strategy).unwrap();
                assert_eq!(actual, reference::align(reference, alt, params, strategy));
                (backend, aligner.narrow_fallbacks())
            })
            .collect()
    }

    #[test]
    fn weak_alignments_saturate_the_narrow_lanes_and_fall_back_exactly() {
        // Two unrelated 700-mers under the haplotype-to-reference scores: the best local alignment
        // is short, so relative to the anti-diagonal every end candidate is deep in saturation.
        let mut rng = Rng(23);
        let r: Vec<u8> = (0..700).map(|_| rng.base()).collect();
        let a: Vec<u8> = (0..700).map(|_| rng.base()).collect();
        for (backend, fallbacks) in
            narrow_fallbacks_on_simd_backends(&r, &a, &HAP_TO_REF, OverhangStrategy::SoftClip)
        {
            assert_eq!(fallbacks, 1, "{backend} should have fallen back once");
        }
    }

    #[test]
    fn a_haplotype_matching_the_far_end_of_the_reference_stays_in_narrow_lanes() {
        // Many last-column candidates saturate (short alternate prefixes against a long reference
        // prefix), but their bounds stay below the real end, so no fallback is needed.
        let mut rng = Rng(29);
        let r: Vec<u8> = (0..600).map(|_| rng.base()).collect();
        let mut a = r[380..].to_vec();
        a[100] = if a[100] == b'A' { b'C' } else { b'A' };
        for (backend, fallbacks) in
            narrow_fallbacks_on_simd_backends(&r, &a, &HAP_TO_REF, OverhangStrategy::SoftClip)
        {
            assert_eq!(fallbacks, 0, "{backend} should not have fallen back");
        }
    }

    #[test]
    fn odd_match_values_are_scaled_into_the_relative_encoding() {
        let mut rng = Rng(31);
        let dangling = SwParameters::new(25, -50, -110, -6);
        for _ in 0..50 {
            let (ref_len, alt_len) = (40 + rng.below(200), 20 + rng.below(150));
            let (r, a) = related_pair(&mut rng, ref_len, alt_len);
            for strategy in STRATEGIES {
                assert_same(&r, &a, &dangling, strategy);
            }
        }
    }

    #[test]
    fn matches_reference_on_random_related_sequences() {
        let mut rng = Rng(7);
        for case in 0..400 {
            let (ref_len, alt_len) = if case % 2 == 0 {
                (40 + rng.below(300), 20 + rng.below(200))
            } else {
                (5 + rng.below(30), 3 + rng.below(30))
            };
            let (r, a) = related_pair(&mut rng, ref_len, alt_len);
            for params in [&HAP_TO_REF, &READ_TO_HAP, &ORIGINAL] {
                for strategy in STRATEGIES {
                    assert_same(&r, &a, params, strategy);
                }
            }
        }
    }

    #[test]
    fn matches_reference_on_unrelated_sequences() {
        let mut rng = Rng(11);
        for _ in 0..100 {
            let r: Vec<u8> = (0..5 + rng.below(80)).map(|_| rng.base()).collect();
            let a: Vec<u8> = (0..5 + rng.below(80)).map(|_| rng.base()).collect();
            for strategy in STRATEGIES {
                assert_same(&r, &a, &HAP_TO_REF, strategy);
                assert_same(&r, &a, &READ_TO_HAP, strategy);
            }
        }
    }

    #[test]
    fn exact_substring_short_circuits_for_softclip_and_ignore() {
        let mut aligner = Aligner::new();
        let a = aligner
            .align(b"AAACGTACGTAAA", b"ACGTACGT", &HAP_TO_REF, OverhangStrategy::SoftClip)
            .unwrap();
        assert_eq!(a.cigar_string(), "8M");
        assert_eq!(a.offset, 2);
        let a =
            aligner.align(b"ACGTACGTACGT", b"ACGT", &HAP_TO_REF, OverhangStrategy::Ignore).unwrap();
        assert_eq!(a.offset, 8, "last occurrence wins");
    }

    #[test]
    fn simple_indels_are_reported_with_gatk_conventions() {
        let mut aligner = Aligner::new();
        let reference = b"NNNNNNNNNNACGTTTGCAAGGCTTAGGCTNNNNNNNNNN";
        let deletion = b"NNNNNNNNNNACGTTTGCGGCTTAGGCTNNNNNNNNNN";
        let a = aligner.align(reference, deletion, &HAP_TO_REF, OverhangStrategy::Indel).unwrap();
        assert_eq!(a.cigar_string(), "18M2D20M");
        assert_eq!(a.offset, 0);
        let insertion = b"NNNNNNNNNNACGTTTGCAAGGCTTTTAGGCTNNNNNNNNNN";
        let a = aligner.align(reference, insertion, &HAP_TO_REF, OverhangStrategy::Indel).unwrap();
        // The extra TT lands at the left of the run of Ts because ties prefer the diagonal.
        assert_eq!(a.cigar_string(), "23M2I17M");
        assert_eq!(a, reference::align(reference, insertion, &HAP_TO_REF, OverhangStrategy::Indel));
    }

    #[test]
    fn soft_clips_read_bases_beyond_the_reference_ends() {
        let mut aligner = Aligner::new();
        let hap = b"ACGTACGTAGGCCTTAGCA";
        let read = b"GGGGACGTACGTAGGCCTTAGCAGGGG";
        let a = aligner.align(hap, read, &READ_TO_HAP, OverhangStrategy::SoftClip).unwrap();
        assert_eq!(a.cigar_string(), "4S19M4S");
        assert_eq!(a.offset, 0);
        // Inside the reference span, overhanging bases are gaps or mismatches, never clips.
        let padded = b"TTTTTTTTTTACGTACGTAGGCCTTAGCATTTTTTTTTT";
        let a = aligner.align(padded, read, &READ_TO_HAP, OverhangStrategy::SoftClip).unwrap();
        assert_eq!(a.cigar_string(), "4I19M4I");
        assert_eq!(a, reference::align(padded, read, &READ_TO_HAP, OverhangStrategy::SoftClip));
    }

    #[test]
    fn empty_sequences_are_rejected() {
        let mut aligner = Aligner::new();
        assert_eq!(
            aligner.align(b"", b"A", &HAP_TO_REF, OverhangStrategy::SoftClip),
            Err(Error::EmptySequence)
        );
        assert_eq!(
            aligner.align(b"A", b"", &HAP_TO_REF, OverhangStrategy::Indel),
            Err(Error::EmptySequence)
        );
    }

    #[test]
    fn buffers_are_reused_across_calls_without_leaking_state() {
        let mut aligner = Aligner::new();
        let mut rng = Rng(3);
        let (r1, a1) = related_pair(&mut rng, 300, 150);
        let (r2, a2) = related_pair(&mut rng, 30, 20);
        let first = aligner.align(&r1, &a1, &HAP_TO_REF, OverhangStrategy::Indel).unwrap();
        aligner.align(&r2, &a2, &READ_TO_HAP, OverhangStrategy::SoftClip).unwrap();
        let again = aligner.align(&r1, &a1, &HAP_TO_REF, OverhangStrategy::Indel).unwrap();
        assert_eq!(first, again);
    }
}
