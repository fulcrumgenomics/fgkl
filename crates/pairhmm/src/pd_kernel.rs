//! The batched partially determined PairHMM kernel.
//!
//! As in `kernel.rs`, SIMD lanes hold different reads and one haplotype's DP matrix is swept row by
//! row. GATK's `LoglessPDPairHMM` adds three branch matrices that remember the state to the left
//! of a flagged deletion, and a per-column deletion state (`Normal`, `InsideDel`, `AfterDel`)
//! that is carried from the last column of one row into the first column of the next. The state
//! depends only on the flags, never on the DP values, so it is identical in every lane and is
//! resolved once per haplotype into runs of columns (`Segment`s) that the row loop dispatches to
//! a specialised inner loop. Because the state at the end of a row is fixed by the flags too, all
//! rows after the first share one run list.
//!
//! The branch matrices are never materialised: within a row they are three running vectors, and
//! only the next row's `DEL_END` columns (whose insertion value, and the `AfterDel` match value
//! one column further right, read the branch values above them) need them stored.
//!
//! Columns are shared between haplotypes as in the plain kernel, with one extra condition. A
//! column's values depend on the flags before it and on the state its row started in, which is
//! the state carried out of the previous row and hence a function of the whole flag string. Two
//! haplotypes with identical bases and flags up to a column and the same row-end state therefore
//! compute identical values there, and haplotypes are sorted by (row-end state, interleaved bases
//! and flags) so that such prefixes are adjacent. The recurrence, `max` included, is homogeneous
//! in the initial deletion value, so a snapshot is rescaled by the haplotype-length ratio exactly
//! as in the plain kernel. Snapshots also hold the branch values of their column.

use std::cell::RefCell;

use crate::kernel::{CellState, Transitions, finish_lane};
use crate::model::{NUM_TRANSITIONS, TABLES};
use crate::pd::{
    ALT_A, ALT_C, ALT_G, ALT_T, DEL_END, DEL_START, PdHaplotype, SNP, base_matches_pd,
};
use crate::simd::{self, AlignedVec, Float, Simd, with_flush_to_zero};
use crate::{Backend, HapSet, Precision, ReadRef, RunnerKey};

const NORMAL: u8 = 0;
const INSIDE_DEL: u8 = 1;
const AFTER_DEL: u8 = 2;

/// Haplotypes in kernel order with their flags resolved into per-column prior codes, `DEL_END`
/// columns and deletion-state runs, plus the bookkeeping for prefix sharing.
pub(crate) struct SortedPdHaps<'a> {
    /// `order[k]` is the caller's index of the k-th sorted haplotype.
    pub order: Vec<usize>,
    pub bases: Vec<&'a [u8]>,
    pub flags: Vec<&'a [u8]>,
    /// Per haplotype, the prior-table code of each column.
    codes: Vec<Vec<u8>>,
    /// Per haplotype, the 1-based columns carrying `DEL_END`.
    del_end_cols: Vec<Vec<usize>>,
    /// Per haplotype, the column runs of the first row and of every later row.
    first_row: Vec<Vec<Segment>>,
    later_rows: Vec<Vec<Segment>>,
    /// `lcp[k]` is the number of leading columns sorted haplotypes `k-1` and `k` share (same
    /// base and flag, same row-end state); `lcp[0]` is 0.
    pub lcp: Vec<usize>,
    match_sets: Vec<MatchSet>,
    pub max_len: usize,
}

impl<'a> SortedPdHaps<'a> {
    pub fn new(haplotypes: &[PdHaplotype<'a>]) -> Self {
        // The first row's runs also give the state carried out of every row, which sorts the
        // haplotypes and starts the runs of the later rows.
        let first_rows: Vec<(Vec<Segment>, u8)> =
            haplotypes.iter().map(|h| segments(h.flags, NORMAL)).collect();
        let end_states: Vec<u8> = first_rows.iter().map(|(_, e)| *e).collect();
        let key = |i: usize| {
            let h = haplotypes[i];
            (end_states[i], h.bases.iter().zip(h.flags))
        };
        let mut order: Vec<usize> = (0..haplotypes.len()).collect();
        order.sort_by(|&a, &b| {
            let (ea, ka) = key(a);
            let (eb, kb) = key(b);
            ea.cmp(&eb).then_with(|| ka.cmp(kb))
        });
        let mut match_sets: Vec<MatchSet> = Vec::new();
        let mut codes = Vec::with_capacity(haplotypes.len());
        let mut del_end_cols = Vec::with_capacity(haplotypes.len());
        let mut first_row = Vec::with_capacity(haplotypes.len());
        let mut later_rows = Vec::with_capacity(haplotypes.len());
        for &i in &order {
            let PdHaplotype { bases, flags } = haplotypes[i];
            assert_eq!(bases.len(), flags.len(), "flags must be as long as the haplotype");
            let hap_codes = bases
                .iter()
                .zip(flags)
                .map(|(&b, &f)| {
                    let set = MatchSet::of(b, f);
                    let code = match match_sets.iter().position(|s| *s == set) {
                        Some(p) => p,
                        None => {
                            match_sets.push(set);
                            match_sets.len() - 1
                        }
                    };
                    u8::try_from(code).expect("at most 256 distinct match sets")
                })
                .collect();
            codes.push(hap_codes);
            del_end_cols.push(
                flags
                    .iter()
                    .enumerate()
                    .filter(|&(_, &f)| f & DEL_END != 0)
                    .map(|(j, _)| j + 1)
                    .collect(),
            );
            let (first, end_state) = &first_rows[i];
            let (later, later_end) = segments(flags, *end_state);
            debug_assert_eq!(*end_state, later_end, "row end state is fixed by the flags");
            first_row.push(first.clone());
            later_rows.push(later);
        }
        let mut lcp = vec![0; order.len()];
        for k in 1..order.len() {
            let (a, b) = (order[k - 1], order[k]);
            if end_states[a] == end_states[b] {
                let (ka, kb) = (key(a).1, key(b).1);
                lcp[k] = ka.zip(kb).take_while(|(x, y)| x == y).count();
            }
        }
        let max_len = haplotypes.iter().map(|h| h.bases.len()).max().unwrap_or(0);
        SortedPdHaps {
            bases: order.iter().map(|&i| haplotypes[i].bases).collect(),
            flags: order.iter().map(|&i| haplotypes[i].flags).collect(),
            order,
            codes,
            del_end_cols,
            first_row,
            later_rows,
            lcp,
            match_sets,
            max_len,
        }
    }

    pub fn len(&self) -> usize {
        self.bases.len()
    }
}

impl HapSet for SortedPdHaps<'_> {
    fn len(&self) -> usize {
        self.bases.len()
    }

    fn order(&self) -> &[usize] {
        &self.order
    }

    fn single(&self, k: usize) -> Self {
        SortedPdHaps::new(&[PdHaplotype { bases: self.bases[k], flags: self.flags[k] }])
    }

    fn lanes(key: RunnerKey) -> usize {
        with_pd_runner(key, |runner| runner.lanes())
    }

    fn run_batch(
        &self,
        key: RunnerKey,
        reads: &[ReadRef<'_>],
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    ) {
        with_pd_runner(key, |runner| runner.run(reads, self, out, fallback))
    }
}

/// Columns `lo+1..=hi`, all computed in the same deletion state and all with (or all without)
/// the `DEL_END` flag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Segment {
    lo: usize,
    hi: usize,
    state: u8,
    del_end: bool,
}

/// The read bases a haplotype column accepts as a match: its own base plus, for a `SNP` column,
/// the flagged alternates. Each distinct set gets one prior table per read row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct MatchSet {
    base: u8,
    /// `ALT_*` bits; zero when the column has no `SNP` flag.
    alts: u8,
}

impl MatchSet {
    fn of(base: u8, flag: u8) -> Self {
        let alts = if flag & SNP != 0 { flag & (ALT_A | ALT_C | ALT_G | ALT_T) } else { 0 };
        MatchSet { base, alts }
    }

    fn accepts(&self, read_base: u8) -> bool {
        read_base == self.base
            || read_base == b'N'
            || self.base == b'N'
            || (self.alts != 0 && base_matches_pd(read_base, SNP | self.alts))
    }
}

thread_local! {
    /// PD kernel workspaces, reused across calls on the same thread like the plain kernel's.
    static PD_RUNNERS: RefCell<Vec<(RunnerKey, Box<dyn PdBatchRunner>)>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with this thread's cached PD runner for `key`, creating it on first use.
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

/// A PD kernel instantiated for one backend and precision, with its scratch memory.
pub(crate) trait PdBatchRunner: Send {
    fn lanes(&self) -> usize;

    /// Computes at most `lanes()` reads against every haplotype. Results land in
    /// `out[read * haps.len() + hap]`; pairs whose result lost precision are appended to
    /// `fallback` as `(read, hap)` with `NaN` written in their place.
    fn run(
        &mut self,
        reads: &[ReadRef<'_>],
        haps: &SortedPdHaps<'_>,
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    );
}

pub(crate) struct PdRunner<S: Simd> {
    ws: Workspace<S>,
}

impl<S: Simd> PdRunner<S> {
    pub fn new() -> Self {
        PdRunner { ws: Workspace::new() }
    }

    #[inline(always)]
    fn run_impl(
        &mut self,
        reads: &[ReadRef<'_>],
        haps: &SortedPdHaps<'_>,
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    ) {
        let ws = &mut self.ws;
        ws.prepare(reads, haps);
        let n_haps = haps.len();
        for k in 0..n_haps {
            let start = haps.lcp[k];
            if k > 0 {
                ws.lcp_counts[start] -= 1;
            }
            let hap_len = haps.bases[k].len();
            ws.snap_cols.clear();
            for q in (start + 1)..=hap_len {
                if ws.lcp_counts[q] > 0 {
                    ws.snap_cols.push(q);
                }
            }
            ws.run_hap(haps, k, start);
            for lane in 0..reads.len() {
                out[lane * n_haps + k] = finish_lane::<S::Elem>(ws.acc[lane], lane, k, fallback);
            }
        }
    }
}

macro_rules! pd_runner_impl {
    ($ty:ty) => {
        impl PdBatchRunner for PdRunner<$ty> {
            fn lanes(&self) -> usize {
                <$ty as Simd>::LANES
            }
            fn run(
                &mut self,
                reads: &[ReadRef<'_>],
                haps: &SortedPdHaps<'_>,
                out: &mut [f64],
                fallback: &mut Vec<(usize, usize)>,
            ) {
                with_flush_to_zero(|| self.run_impl(reads, haps, out, fallback))
            }
        }
    };
    ($ty:ty, $features:literal) => {
        impl PdBatchRunner for PdRunner<$ty> {
            fn lanes(&self) -> usize {
                <$ty as Simd>::LANES
            }
            fn run(
                &mut self,
                reads: &[ReadRef<'_>],
                haps: &SortedPdHaps<'_>,
                out: &mut [f64],
                fallback: &mut Vec<(usize, usize)>,
            ) {
                #[target_feature(enable = $features)]
                unsafe fn go(
                    runner: &mut PdRunner<$ty>,
                    reads: &[ReadRef<'_>],
                    haps: &SortedPdHaps<'_>,
                    out: &mut [f64],
                    fallback: &mut Vec<(usize, usize)>,
                ) {
                    runner.run_impl(reads, haps, out, fallback)
                }
                // SAFETY: runners of this type are only built after `Backend::is_available`
                // confirmed the CPU supports the enabled features.
                with_flush_to_zero(|| unsafe { go(self, reads, haps, out, fallback) })
            }
        }
    };
}

pd_runner_impl!(crate::simd::ScalarF32);
pd_runner_impl!(crate::simd::ScalarF64);
#[cfg(target_arch = "aarch64")]
pd_runner_impl!(crate::simd::neon::NeonF32);
#[cfg(target_arch = "aarch64")]
pd_runner_impl!(crate::simd::neon::NeonF64);
#[cfg(target_arch = "x86_64")]
pd_runner_impl!(crate::simd::x86::Avx2F32, "avx2,fma");
#[cfg(target_arch = "x86_64")]
pd_runner_impl!(crate::simd::x86::Avx2F64, "avx2,fma");
#[cfg(target_arch = "x86_64")]
pd_runner_impl!(crate::simd::x86::Avx512F32, "avx512f");
#[cfg(target_arch = "x86_64")]
pd_runner_impl!(crate::simd::x86::Avx512F64, "avx512f");
#[cfg(target_arch = "x86_64")]
pd_runner_impl!(crate::simd::x86::Avx512F32Narrow, "avx512f");
#[cfg(target_arch = "x86_64")]
pd_runner_impl!(crate::simd::x86::Avx512F64Narrow, "avx512f");

/// One DP row for every column, lane-interleaved: match, insertion and deletion values plus
/// the three branch values, which are only written at `DEL_END` columns.
struct RowBuf<E> {
    m: AlignedVec<E>,
    x: AlignedVec<E>,
    y: AlignedVec<E>,
    bm: AlignedVec<E>,
    bx: AlignedVec<E>,
    by: AlignedVec<E>,
}

impl<E: Float> RowBuf<E> {
    fn new() -> Self {
        RowBuf {
            m: AlignedVec::new(),
            x: AlignedVec::new(),
            y: AlignedVec::new(),
            bm: AlignedVec::new(),
            bx: AlignedVec::new(),
            by: AlignedVec::new(),
        }
    }

    /// Sizes the row without clearing it: `run_hap` initialises every column it reads.
    fn resize(&mut self, len: usize) {
        self.m.resize_no_fill(len);
        self.x.resize_no_fill(len);
        self.y.resize_no_fill(len);
        self.bm.resize_no_fill(len);
        self.bx.resize_no_fill(len);
        self.by.resize_no_fill(len);
    }
}

/// Scratch memory for one batch of reads, laid out lane-interleaved so that element `i` of every
/// lane is one contiguous vector.
struct Workspace<S: Simd> {
    /// Rows of the DP excluding row zero, i.e. the longest read in the batch.
    rows: usize,
    num_codes: usize,
    /// `[row][transition][lane]`.
    trans: AlignedVec<S::Elem>,
    /// `[row][match-set code][lane]`.
    prior: AlignedVec<S::Elem>,
    /// `[row][lane]`: 1 where the lane's read ends at this row, else 0.
    end_mul: AlignedVec<S::Elem>,
    /// Whether any lane's read ends at this row.
    end_rows: Vec<bool>,
    /// Per lane, the raw (scaled) probability after the last haplotype run.
    acc: AlignedVec<S::Elem>,
    row_prev: RowBuf<S::Elem>,
    row_cur: RowBuf<S::Elem>,
    /// Indexed by column; slot 0 is the virtual column before the haplotype.
    snapshots: Vec<Snapshot<S::Elem>>,
    /// Per column, how many haplotypes still to come start from that column.
    lcp_counts: Vec<u32>,
    /// Columns of the current haplotype whose state must be snapshotted.
    snap_cols: Vec<usize>,
    /// The current haplotype's state runs after the shared prefix, cut at the snapshot columns,
    /// for the first row and for every later row; the flag marks a run ending at a snapshot.
    first_runs: Vec<(Segment, bool)>,
    later_runs: Vec<(Segment, bool)>,
}

impl<S: Simd> Workspace<S> {
    fn new() -> Self {
        Workspace {
            rows: 0,
            num_codes: 0,
            trans: AlignedVec::new(),
            prior: AlignedVec::new(),
            end_mul: AlignedVec::new(),
            end_rows: Vec::new(),
            acc: AlignedVec::new(),
            row_prev: RowBuf::new(),
            row_cur: RowBuf::new(),
            snapshots: Vec::new(),
            lcp_counts: Vec::new(),
            snap_cols: Vec::new(),
            first_runs: Vec::new(),
            later_runs: Vec::new(),
        }
    }

    /// Fills the per-read tables for a batch and sizes every buffer for the haplotype set.
    fn prepare(&mut self, reads: &[ReadRef<'_>], haps: &SortedPdHaps<'_>) {
        let lanes = S::LANES;
        assert!(reads.len() <= lanes, "batch larger than the lane count");
        let rows = reads.iter().map(|r| r.len()).max().unwrap_or(0);
        let num_codes = haps.match_sets.len();
        self.rows = rows;
        self.num_codes = num_codes;
        self.trans.resize(rows * NUM_TRANSITIONS * lanes, S::Elem::ZERO);
        self.prior.resize(rows * num_codes * lanes, S::Elem::ZERO);
        self.end_mul.resize(rows * lanes, S::Elem::ZERO);
        self.end_rows.clear();
        self.end_rows.resize(rows, false);
        for (lane, read) in reads.iter().enumerate() {
            for r in 0..read.len() {
                let t = TABLES.transitions(read.ins_gop[r], read.del_gop[r], read.gcp[r]);
                for (k, &prob) in t.iter().enumerate() {
                    self.trans[(r * NUM_TRANSITIONS + k) * lanes + lane] = S::Elem::from_f64(prob);
                }
                let (p_match, p_mismatch) = TABLES.priors(read.quals[r]);
                let read_base = read.bases[r];
                for (code, set) in haps.match_sets.iter().enumerate() {
                    let p = if set.accepts(read_base) { p_match } else { p_mismatch };
                    self.prior[(r * num_codes + code) * lanes + lane] = S::Elem::from_f64(p);
                }
            }
            self.end_mul[(read.len() - 1) * lanes + lane] = S::Elem::ONE;
            self.end_rows[read.len() - 1] = true;
        }
        let cols = haps.max_len + 1;
        self.row_prev.resize(cols * lanes);
        self.row_cur.resize(cols * lanes);
        self.acc.resize(lanes, S::Elem::ZERO);
        if self.snapshots.len() < cols {
            self.snapshots.resize_with(cols, Snapshot::new);
        }
        for snap in &mut self.snapshots[..cols] {
            snap.resize((rows + 1) * lanes, lanes);
        }
        // The virtual column before the haplotype: nothing but the initial deletion probability
        // in row zero, zero branch values (GATK never writes branch column zero), recorded for a
        // haplotype of length one so the length rescaling applies.
        let origin = &mut self.snapshots[0];
        origin.m.fill(S::Elem::ZERO);
        origin.x.fill(S::Elem::ZERO);
        origin.y.fill(S::Elem::ZERO);
        origin.bm.fill(S::Elem::ZERO);
        origin.bx.fill(S::Elem::ZERO);
        origin.by.fill(S::Elem::ZERO);
        origin.y[..lanes].fill(S::Elem::INITIAL_CONSTANT);
        origin.acc.fill(S::Elem::ZERO);
        origin.hap_len = 1;
        self.lcp_counts.clear();
        self.lcp_counts.resize(cols, 0);
        for &p in &haps.lcp[1..] {
            self.lcp_counts[p] += 1;
        }
    }

    /// Runs haplotype `k` from column `start`, whose state comes from the snapshot at that
    /// column, leaving the per-lane raw result in `acc` and recording the snapshots listed in
    /// `snap_cols`.
    #[inline(always)]
    fn run_hap(&mut self, haps: &SortedPdHaps<'_>, k: usize, start: usize) {
        let lanes = S::LANES;
        let rows = self.rows;
        let num_codes = self.num_codes;
        let hap_len = haps.bases[k].len();
        let codes: &[u8] = &haps.codes[k];
        let Workspace {
            trans,
            prior,
            end_mul,
            end_rows,
            acc,
            row_prev,
            row_cur,
            snapshots,
            snap_cols,
            first_runs,
            later_runs,
            ..
        } = self;
        cut_runs(&haps.first_row[k], start, snap_cols, first_runs);
        cut_runs(&haps.later_rows[k], start, snap_cols, later_runs);
        let (before, after) = snapshots.split_at_mut(start + 1);
        let src = &before[start];
        let trans: &[S::Elem] = trans;
        let prior: &[S::Elem] = prior;
        let end_mul: &[S::Elem] = end_mul;
        let zero = S::zero();
        let init = S::splat(S::Elem::from_f64(S::Elem::INITIAL_CONSTANT.to_f64() / hap_len as f64));
        let scale = S::splat(S::Elem::from_f64(src.hap_len as f64 / hap_len as f64));

        // Row zero: free deletions along the haplotype, and zero branch values.
        for j in start..=hap_len {
            zero.store(&mut row_prev.m[j * lanes..]);
            zero.store(&mut row_prev.x[j * lanes..]);
            init.store(&mut row_prev.y[j * lanes..]);
        }
        for &j in &haps.del_end_cols[k] {
            zero.store(&mut row_prev.bm[j * lanes..]);
            zero.store(&mut row_prev.bx[j * lanes..]);
            zero.store(&mut row_prev.by[j * lanes..]);
        }
        let mut acc_v = S::load(&src.acc).mul(scale);
        for &q in snap_cols.iter() {
            let snap = &mut after[q - start - 1];
            snap.hap_len = hap_len;
            zero.store(&mut snap.m[..lanes]);
            zero.store(&mut snap.x[..lanes]);
            init.store(&mut snap.y[..lanes]);
            zero.store(&mut snap.bm[..lanes]);
            zero.store(&mut snap.bx[..lanes]);
            zero.store(&mut snap.by[..lanes]);
            acc_v.store(&mut snap.acc);
        }
        if start == hap_len {
            acc_v.store(acc);
            return;
        }

        for i in 1..=rows {
            let r = i - 1;
            let t = Transitions::<S>::load(trans, r);
            let prior_row = &prior[r * num_codes * lanes..(r + 1) * num_codes * lanes];
            // The shared column's state for this row and the row above, rescaled. The row above's
            // branch values are also placed in the previous row buffer, where an `AfterDel`
            // column right after the shared prefix reads them.
            let mut st = CellState {
                m_diag: S::load(&src.m[(i - 1) * lanes..]).mul(scale),
                x_diag: S::load(&src.x[(i - 1) * lanes..]).mul(scale),
                y_diag: S::load(&src.y[(i - 1) * lanes..]).mul(scale),
                m_left: S::load(&src.m[i * lanes..]).mul(scale),
                x_left: S::load(&src.x[i * lanes..]).mul(scale),
                y_left: S::load(&src.y[i * lanes..]).mul(scale),
            };
            let mut branch = Branch {
                m: S::load(&src.bm[i * lanes..]).mul(scale),
                x: S::load(&src.bx[i * lanes..]).mul(scale),
                y: S::load(&src.by[i * lanes..]).mul(scale),
            };
            S::load(&src.bm[(i - 1) * lanes..]).mul(scale).store(&mut row_prev.bm[start * lanes..]);
            S::load(&src.bx[(i - 1) * lanes..]).mul(scale).store(&mut row_prev.bx[start * lanes..]);
            S::load(&src.by[(i - 1) * lanes..]).mul(scale).store(&mut row_prev.by[start * lanes..]);
            let runs: &[(Segment, bool)] = if i == 1 { first_runs } else { later_runs };
            let ends_here = end_rows[r];
            let mut rowsum = zero;
            for (seg, at_snap) in runs {
                (st, branch) = dispatch::<S>(
                    seg,
                    ends_here,
                    &t,
                    prior_row,
                    codes,
                    row_prev,
                    row_cur,
                    st,
                    branch,
                    &mut rowsum,
                );
                if *at_snap {
                    let snap = &mut after[seg.hi - start - 1];
                    st.m_left.store(&mut snap.m[i * lanes..]);
                    st.x_left.store(&mut snap.x[i * lanes..]);
                    st.y_left.store(&mut snap.y[i * lanes..]);
                    branch.m.store(&mut snap.bm[i * lanes..]);
                    branch.x.store(&mut snap.bx[i * lanes..]);
                    branch.y.store(&mut snap.by[i * lanes..]);
                    if ends_here {
                        let e = S::load(&end_mul[r * lanes..]);
                        e.mul_add(rowsum, S::load(&snap.acc)).store(&mut snap.acc);
                    }
                }
            }
            if ends_here {
                acc_v = S::load(&end_mul[r * lanes..]).mul_add(rowsum, acc_v);
            }
            std::mem::swap(row_prev, row_cur);
        }
        acc_v.store(acc);
    }
}

/// The DP state of one column for every row, kept so a later haplotype can start from it: the
/// match, insertion and deletion values and the branch values after that column.
struct Snapshot<E> {
    m: AlignedVec<E>,
    x: AlignedVec<E>,
    y: AlignedVec<E>,
    bm: AlignedVec<E>,
    bx: AlignedVec<E>,
    by: AlignedVec<E>,
    /// Per lane, the result accumulated over the columns up to this one.
    acc: AlignedVec<E>,
    /// Length of the haplotype this state was computed for; scales the state when reused.
    hap_len: usize,
}

impl<E: Float> Snapshot<E> {
    fn new() -> Self {
        Snapshot {
            m: AlignedVec::new(),
            x: AlignedVec::new(),
            y: AlignedVec::new(),
            bm: AlignedVec::new(),
            bx: AlignedVec::new(),
            by: AlignedVec::new(),
            acc: AlignedVec::new(),
            hap_len: 1,
        }
    }

    /// Sizes the snapshot without clearing it: `run_hap` writes every row of a snapshot before
    /// a later haplotype starts from it, and the origin column is filled by `prepare`.
    fn resize(&mut self, cells: usize, lanes: usize) {
        self.m.resize_no_fill(cells);
        self.x.resize_no_fill(cells);
        self.y.resize_no_fill(cells);
        self.bm.resize_no_fill(cells);
        self.bx.resize_no_fill(cells);
        self.by.resize_no_fill(cells);
        self.acc.resize_no_fill(lanes);
    }
}

/// Clips a row's state runs to the columns after `start` and cuts them at the snapshot columns.
fn cut_runs(segs: &[Segment], start: usize, snap_cols: &[usize], out: &mut Vec<(Segment, bool)>) {
    out.clear();
    let mut next_snap = 0;
    for seg in segs {
        let mut lo = seg.lo.max(start);
        if lo >= seg.hi {
            continue;
        }
        while next_snap < snap_cols.len() && snap_cols[next_snap] <= lo {
            next_snap += 1;
        }
        while next_snap < snap_cols.len() && snap_cols[next_snap] < seg.hi {
            let q = snap_cols[next_snap];
            out.push((Segment { lo, hi: q, ..*seg }, true));
            lo = q;
            next_snap += 1;
        }
        let at_snap = next_snap < snap_cols.len() && snap_cols[next_snap] == seg.hi;
        out.push((Segment { lo, hi: seg.hi, ..*seg }, at_snap));
    }
}

/// Splits a row into runs of columns sharing a deletion state and `DEL_END` flag, starting the
/// row in `state`; returns the runs and the state carried out of the last column.
fn segments(flags: &[u8], mut state: u8) -> (Vec<Segment>, u8) {
    let mut runs: Vec<Segment> = Vec::new();
    for (j, &flag) in flags.iter().enumerate() {
        let col = j + 1;
        let del_end = flag & DEL_END != 0;
        match runs.last_mut() {
            Some(last) if last.state == state && last.del_end == del_end => last.hi = col,
            _ => runs.push(Segment { lo: j, hi: col, state, del_end }),
        }
        if state == AFTER_DEL {
            state = NORMAL;
        }
        if flag & DEL_START != 0 {
            state = INSIDE_DEL;
        }
        if del_end {
            state = AFTER_DEL;
        }
    }
    (runs, state)
}

/// Runs one segment on the inner loop specialised for its state and flags.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dispatch<S: Simd>(
    seg: &Segment,
    ends_here: bool,
    t: &Transitions<S>,
    prior_row: &[S::Elem],
    codes: &[u8],
    prev: &RowBuf<S::Elem>,
    cur: &mut RowBuf<S::Elem>,
    st: CellState<S>,
    branch: Branch<S>,
    rowsum: &mut S,
) -> (CellState<S>, Branch<S>) {
    macro_rules! go {
        ($state:expr, $del_end:expr, $acc:expr) => {
            dp_segment::<S, $state, $del_end, $acc>(
                t, prior_row, codes, seg.lo, seg.hi, prev, cur, st, branch, rowsum,
            )
        };
    }
    match (seg.state, seg.del_end, ends_here) {
        (NORMAL, false, false) => go!(NORMAL, false, false),
        (NORMAL, false, true) => go!(NORMAL, false, true),
        (NORMAL, true, false) => go!(NORMAL, true, false),
        (NORMAL, true, true) => go!(NORMAL, true, true),
        (INSIDE_DEL, false, false) => go!(INSIDE_DEL, false, false),
        (INSIDE_DEL, false, true) => go!(INSIDE_DEL, false, true),
        (INSIDE_DEL, true, false) => go!(INSIDE_DEL, true, false),
        (INSIDE_DEL, true, true) => go!(INSIDE_DEL, true, true),
        (AFTER_DEL, false, false) => go!(AFTER_DEL, false, false),
        (AFTER_DEL, false, true) => go!(AFTER_DEL, false, true),
        (AFTER_DEL, true, false) => go!(AFTER_DEL, true, false),
        (AFTER_DEL, true, true) => go!(AFTER_DEL, true, true),
        _ => unreachable!("unknown deletion state"),
    }
}

/// GATK's three branch values at the column one back in the current row. Kept apart from
/// [`CellState`] because the normal-state inner loop never reads them: there they are simply the
/// values one further column back, and carrying them would cost three more live vectors.
#[derive(Clone, Copy)]
struct Branch<S> {
    m: S,
    x: S,
    y: S,
}

/// Computes columns `lo+1..=hi` of the current row in deletion state `STATE`, every column
/// carrying (`DEL_END`) or lacking the deletion-end flag. With `ACC` set, also sums match plus
/// insertion over the columns into `rowsum`, which the caller applies to lanes whose read ends on
/// this row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dp_segment<S: Simd, const STATE: u8, const DEL_END: bool, const ACC: bool>(
    t: &Transitions<S>,
    prior_row: &[S::Elem],
    codes: &[u8],
    lo: usize,
    hi: usize,
    prev: &RowBuf<S::Elem>,
    cur: &mut RowBuf<S::Elem>,
    mut st: CellState<S>,
    mut branch: Branch<S>,
    rowsum: &mut S,
) -> (CellState<S>, Branch<S>) {
    // Bounds are checked once per segment rather than on every load and store. Column `j`
    // occupies elements `j * lanes..(j + 1) * lanes` of a row, and code `c` occupies
    // `c * lanes..(c + 1) * lanes` of the prior row. The `AfterDel` state reads the branch
    // values one column back, so it also needs `lo * lanes` in range, which `lo < hi` gives.
    let lanes = S::LANES;
    let end = (hi + 1) * lanes;
    assert!(lo < hi);
    assert!(prev.m.len() >= end && prev.x.len() >= end && prev.y.len() >= end);
    assert!(prev.bm.len() >= end && prev.bx.len() >= end && prev.by.len() >= end);
    assert!(cur.m.len() >= end && cur.x.len() >= end && cur.y.len() >= end);
    assert!(cur.bm.len() >= end && cur.bx.len() >= end && cur.by.len() >= end);
    let codes = &codes[lo..hi];
    let num_codes = prior_row.len() / lanes;
    assert!(codes.iter().all(|&c| (c as usize) < num_codes));
    let (prev_m, prev_x, prev_y) = (prev.m.as_ptr(), prev.x.as_ptr(), prev.y.as_ptr());
    let (prev_bm, prev_bx, prev_by) = (prev.bm.as_ptr(), prev.bx.as_ptr(), prev.by.as_ptr());
    let (cur_m, cur_x, cur_y) = (cur.m.as_mut_ptr(), cur.x.as_mut_ptr(), cur.y.as_mut_ptr());
    let (cur_bm, cur_bx, cur_by) = (cur.bm.as_mut_ptr(), cur.bx.as_mut_ptr(), cur.by.as_mut_ptr());
    let priors = prior_row.as_ptr();
    let entry_left = (st.m_left, st.x_left, st.y_left);
    for (k, &code) in codes.iter().enumerate() {
        let off = (lo + 1 + k) * lanes;
        // SAFETY: `off + lanes <= end` and `(code + 1) * lanes <= prior_row.len()` by the checks
        // above, and the previous and current rows are distinct buffers.
        let (prior, m_up, x_up, y_up) = unsafe {
            (
                S::load_ptr(priors.add(code as usize * lanes)),
                S::load_ptr(prev_m.add(off)),
                S::load_ptr(prev_x.add(off)),
                S::load_ptr(prev_y.add(off)),
            )
        };
        // GATK's branch matrices at this column: a copy of the column to the left in the
        // normal state, held through a deletion, and merged with the DP values after one.
        let (bm, bx, by) = match STATE {
            NORMAL => (st.m_left, st.x_left, st.y_left),
            INSIDE_DEL => (branch.m, branch.x, branch.y),
            _ => (branch.m.max(st.m_left), branch.x.max(st.x_left), branch.y.max(st.y_left)),
        };
        let (m, y) = if STATE == AFTER_DEL {
            // The column to the left ended a deletion, so its branch values are stored.
            // SAFETY: `off - lanes >= lo * lanes` is in range by the checks above.
            let (bm_diag, bx_diag, by_diag) = unsafe {
                (
                    S::load_ptr(prev_bm.add(off - lanes)),
                    S::load_ptr(prev_bx.add(off - lanes)),
                    S::load_ptr(prev_by.add(off - lanes)),
                )
            };
            let m_d = bm_diag.max(st.m_diag);
            let x_d = bx_diag.max(st.x_diag);
            let y_d = by_diag.max(st.y_diag);
            let m = prior.mul(y_d.mul_add(t.im, x_d.mul_add(t.im, m_d.mul(t.mm))));
            (m, by.mul_add(t.dd, bm.mul(t.md)))
        } else {
            let m =
                prior.mul(st.y_diag.mul_add(t.im, st.x_diag.mul_add(t.im, st.m_diag.mul(t.mm))));
            (m, st.y_left.mul_add(t.dd, st.m_left.mul(t.md)))
        };
        let x = if DEL_END {
            // SAFETY: as for the loads above; `DEL_END` columns of the previous row were stored.
            let (bm_up, bx_up) =
                unsafe { (S::load_ptr(prev_bm.add(off)), S::load_ptr(prev_bx.add(off))) };
            bx_up.max(x_up).mul_add(t.ii, bm_up.max(m_up).mul(t.mi))
        } else {
            x_up.mul_add(t.ii, m_up.mul(t.mi))
        };
        // SAFETY: as above, for the current row.
        unsafe {
            m.store_ptr(cur_m.add(off));
            x.store_ptr(cur_x.add(off));
            y.store_ptr(cur_y.add(off));
            if DEL_END {
                bm.store_ptr(cur_bm.add(off));
                bx.store_ptr(cur_bx.add(off));
                by.store_ptr(cur_by.add(off));
            }
        }
        if ACC {
            *rowsum = rowsum.add(m.add(x));
        }
        if STATE == AFTER_DEL {
            branch = Branch { m: bm, x: bx, y: by };
        }
        st =
            CellState { m_diag: m_up, x_diag: x_up, y_diag: y_up, m_left: m, x_left: x, y_left: y };
    }
    if STATE == NORMAL {
        // The branch values at the last column are the DP values one column before it, which
        // the loop stored (or, for a single-column segment, received).
        branch = if hi - lo >= 2 {
            let off = (hi - 1) * lanes;
            // SAFETY: `off + lanes <= end` by the checks above, and the column was just written.
            unsafe {
                Branch {
                    m: S::load_ptr(cur_m.add(off)),
                    x: S::load_ptr(cur_x.add(off)),
                    y: S::load_ptr(cur_y.add(off)),
                }
            }
        } else {
            Branch { m: entry_left.0, x: entry_left.1, y: entry_left.2 }
        };
    }
    (st, branch)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(lo: usize, hi: usize, state: u8, del_end: bool) -> Segment {
        Segment { lo, hi, state, del_end }
    }

    #[test]
    fn unflagged_haplotype_is_one_normal_run_in_every_row() {
        let (first, end) = segments(&[0; 5], NORMAL);
        assert_eq!(first, vec![seg(0, 5, NORMAL, false)]);
        assert_eq!(end, NORMAL);
    }

    #[test]
    fn a_deletion_splits_the_row_into_state_runs() {
        // A flag changes the state of the columns after it: columns 1..=3 are normal (column 3
        // starts the deletion), 4..=5 inside it with column 5 carrying DEL_END, column 6 after
        // the deletion, 7..=8 normal again.
        let flags = [0, 0, DEL_START, 0, DEL_END, 0, 0, 0];
        let (first, end) = segments(&flags, NORMAL);
        assert_eq!(
            first,
            vec![
                seg(0, 3, NORMAL, false),
                seg(3, 4, INSIDE_DEL, false),
                seg(4, 5, INSIDE_DEL, true),
                seg(5, 6, AFTER_DEL, false),
                seg(6, 8, NORMAL, false),
            ]
        );
        assert_eq!(end, NORMAL);
    }

    #[test]
    fn a_single_base_deletion_ends_in_the_state_it_started_in() {
        let flags = [0, DEL_START | DEL_END, 0];
        let (first, end) = segments(&flags, NORMAL);
        assert_eq!(
            first,
            vec![seg(0, 1, NORMAL, false), seg(1, 2, NORMAL, true), seg(2, 3, AFTER_DEL, false)]
        );
        assert_eq!(end, NORMAL);
    }

    #[test]
    fn a_deletion_ending_on_the_last_column_carries_after_del_into_later_rows() {
        let flags = [0, DEL_START, DEL_END];
        let (first, end) = segments(&flags, NORMAL);
        assert_eq!(end, AFTER_DEL);
        let (later, later_end) = segments(&flags, end);
        assert_eq!(later[0], seg(0, 1, AFTER_DEL, false));
        assert_eq!(later_end, AFTER_DEL);
        assert_ne!(first[0], later[0]);
    }

    #[test]
    fn match_sets_accept_own_base_n_and_flagged_alternates() {
        let plain = MatchSet::of(b'A', 0);
        assert!(plain.accepts(b'A') && plain.accepts(b'N') && !plain.accepts(b'C'));
        let snp = MatchSet::of(b'A', SNP | ALT_C | ALT_T);
        assert!(snp.accepts(b'C') && snp.accepts(b'T') && !snp.accepts(b'G'));
        assert_eq!(MatchSet::of(b'A', ALT_C), plain, "alternates need the SNP flag");
        assert_eq!(MatchSet::of(b'A', DEL_START | DEL_END), plain);
        assert!(MatchSet::of(b'N', 0).accepts(b'G'));
    }
}
