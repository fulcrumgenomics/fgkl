//! The batched PairHMM kernel.
//!
//! SIMD lanes hold different reads. The dynamic-programming matrix of one haplotype is swept row by
//! row (read position outer, haplotype position inner), which keeps the six per-row transition
//! probabilities in registers and touches only the previous row. Haplotypes are processed in sorted
//! order so each one starts from the DP column its predecessor already computed for their common
//! prefix: whenever a column will be needed as such a starting point later, a snapshot of it is
//! kept. Because the whole recurrence is linear in the initial deletion probability, which GATK sets
//! to `INITIAL / haplotype_length`, a snapshot taken for one haplotype length is rescaled by the
//! ratio of lengths when reused for another.
//!
//! Haplotypes of a region also share suffixes. A backward pass computes, once per shared suffix,
//! the probability mass from each cell to the end of the read; a haplotype that shares a suffix
//! then stops its forward sweep early and joins those backward values at a cut column (see
//! [`SuffixPlan`] and `docs/design.md`).

use std::cell::RefCell;
use std::collections::HashMap;

use crate::model::{
    DELETION_TO_DELETION, INDEL_TO_MATCH, INSERTION_TO_INSERTION, MATCH_TO_DELETION,
    MATCH_TO_INSERTION, MATCH_TO_MATCH, NUM_TRANSITIONS, TABLES,
};
use crate::simd::{self, AlignedVec, Float, Simd, with_flush_to_zero};
use crate::{Backend, HapSet, Precision, ReadRef, RunnerKey};

/// Prior tables always cover these bases; any other byte occurring in a haplotype gets its own.
const STANDARD_BASES: [u8; 5] = *b"ACGTN";

/// Haplotypes in kernel order plus the bookkeeping for prefix sharing.
pub(crate) struct SortedHaps<'a> {
    /// `order[k]` is the caller's index of the k-th sorted haplotype.
    pub order: Vec<usize>,
    pub bases: Vec<&'a [u8]>,
    /// Per haplotype, the prior-table code of each base.
    pub codes: Vec<Vec<u8>>,
    /// `lcp[k]` is the common-prefix length of sorted haplotypes `k-1` and `k`; `lcp[0]` is 0.
    pub lcp: Vec<usize>,
    pub max_len: usize,
    /// Bytes outside ACGTN present in some haplotype, each assigned a prior table.
    pub extra_bases: Vec<u8>,
    /// `dup[k]`: sorted haplotype `k` equals haplotype `k - 1` and takes its results.
    pub dup: Vec<bool>,
    pub plan: SuffixPlan,
}

impl<'a> SortedHaps<'a> {
    /// Sorts `haplotypes` and plans the work they share, including suffixes when
    /// `share_suffixes` is set.
    pub fn new(haplotypes: &[&'a [u8]], share_suffixes: bool) -> Self {
        let mut order: Vec<usize> = (0..haplotypes.len()).collect();
        order.sort_by(|&a, &b| haplotypes[a].cmp(haplotypes[b]));
        let bases: Vec<&[u8]> = order.iter().map(|&i| haplotypes[i]).collect();
        let mut extra_bases: Vec<u8> = Vec::new();
        for hap in &bases {
            for &b in *hap {
                if !STANDARD_BASES.contains(&b) && !extra_bases.contains(&b) {
                    extra_bases.push(b);
                }
            }
        }
        let codes = bases
            .iter()
            .map(|hap| hap.iter().map(|&b| Self::code_of(b, &extra_bases)).collect())
            .collect();
        let mut lcp = vec![0; bases.len()];
        for k in 1..bases.len() {
            lcp[k] = bases[k - 1].iter().zip(bases[k]).take_while(|(x, y)| x == y).count();
        }
        let max_len = bases.iter().map(|h| h.len()).max().unwrap_or(0);
        let dup: Vec<bool> = (0..bases.len()).map(|k| k > 0 && bases[k] == bases[k - 1]).collect();
        let plan = if share_suffixes {
            SuffixPlan::new(&bases, &lcp, &dup)
        } else {
            SuffixPlan::forward_only(bases.len())
        };
        SortedHaps { order, bases, codes, lcp, max_len, extra_bases, dup, plan }
    }

    pub fn len(&self) -> usize {
        self.bases.len()
    }

    pub fn num_codes(&self) -> usize {
        STANDARD_BASES.len() + self.extra_bases.len()
    }

    pub fn byte_of_code(&self, code: usize) -> u8 {
        if code < STANDARD_BASES.len() {
            STANDARD_BASES[code]
        } else {
            self.extra_bases[code - STANDARD_BASES.len()]
        }
    }

    fn code_of(base: u8, extra_bases: &[u8]) -> u8 {
        match STANDARD_BASES.iter().position(|&s| s == base) {
            Some(p) => p as u8,
            None => {
                let p = extra_bases.iter().position(|&e| e == base).expect("extra base registered");
                (STANDARD_BASES.len() + p) as u8
            }
        }
    }
}

/// A backward sweep must skip at least this many columns of a haplotype's forward sweep to pay
/// for its join, which costs about two columns.
const MIN_SAVING: usize = 4;

/// Where a haplotype's forward sweep stops: it computes columns up to `column - 1`, then joins
/// the backward values of `column`, held in backward node `node`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cut {
    pub column: usize,
    pub node: usize,
}

/// One backward sweep along a haplotype's suffix, where depth `d` is the haplotype's column
/// `len - d`. It computes depths `resume + 1..=need` starting from node `start`, which holds depth
/// `resume`, and stores each `(depth, node)` of `writes` (ascending) for later sweeps and cuts.
#[derive(Debug)]
pub(crate) struct BackwardSweep {
    /// Sorted index of the haplotype whose bases the sweep follows.
    pub hap: usize,
    pub resume: usize,
    pub need: usize,
    pub start: usize,
    pub writes: Vec<(usize, usize)>,
}

/// How a sorted haplotype set shares suffixes: which haplotypes stop their forward sweep at a cut,
/// and the backward sweeps that compute the values those cuts join, once per shared suffix.
///
/// Backward values at a column depend only on the haplotype's bases from that column to its end,
/// not on its length or prefix. A haplotype whose longest prefix shared with another distinct
/// haplotype is `p` and longest shared suffix is `s` cuts at column `len - d`, with
/// `d = min(s, len - p - 1)`: the forward sweep then covers exactly its shared prefix and any
/// private middle, and the backward pass only goes as deep as the cuts read. Sweeps visit the
/// haplotypes ordered by reversed bases, each resuming from the deepest depth an earlier sweep
/// computed along its suffix, so each (suffix, depth) is computed once.
#[derive(Debug)]
pub(crate) struct SuffixPlan {
    /// Per sorted haplotype; `None` runs the forward sweep to the end.
    pub cuts: Vec<Option<Cut>>,
    /// In the order they must run.
    pub sweeps: Vec<BackwardSweep>,
    /// Backward nodes the plan uses. Node 0 is depth 0, the last column, whose backward values
    /// depend on the reads alone.
    pub nodes: usize,
}

impl SuffixPlan {
    /// A plan with no cuts: every haplotype runs forward to its end.
    fn forward_only(num_haps: usize) -> Self {
        SuffixPlan { cuts: vec![None; num_haps], sweeps: Vec::new(), nodes: 1 }
    }

    /// Plans suffix sharing for haplotypes in sorted order, given their common-prefix lengths with
    /// their predecessors; `dup[k]` marks a copy of haplotype `k - 1`, which plays no part.
    fn new(bases: &[&[u8]], lcp: &[usize], dup: &[bool]) -> Self {
        let n = bases.len();
        let distinct: Vec<usize> = (0..n).filter(|&k| !dup[k]).collect();
        // `lcp` of a first copy is with the previous distinct haplotype or a copy of it, which
        // has the same bases.
        let shared_prefix =
            |t: usize| lcp[distinct[t]].max(distinct.get(t + 1).map_or(0, |&next| lcp[next]));
        let mut by_suffix = distinct.clone();
        by_suffix.sort_by(|&a, &b| bases[a].iter().rev().cmp(bases[b].iter().rev()));
        let lcs: Vec<usize> = (0..by_suffix.len())
            .map(|r| {
                if r == 0 { 0 } else { common_suffix(bases[by_suffix[r - 1]], bases[by_suffix[r]]) }
            })
            .collect();
        let mut suffix_pos = vec![0; n];
        for (r, &k) in by_suffix.iter().enumerate() {
            suffix_pos[k] = r;
        }
        let mut need = vec![0; n];
        for (t, &k) in distinct.iter().enumerate() {
            let len = bases[k].len();
            let r = suffix_pos[k];
            let shared_suffix = lcs[r].max(lcs.get(r + 1).copied().unwrap_or(0));
            let p = shared_prefix(t);
            if p < len {
                let d = shared_suffix.min(len - p - 1);
                if d >= MIN_SAVING {
                    need[k] = d;
                }
            }
        }

        let mut sweeps = Vec::new();
        let mut sweep_at = vec![None; by_suffix.len()];
        // How deep the backward pass has computed along the current suffix.
        let mut reach = 0;
        for (r, &k) in by_suffix.iter().enumerate() {
            let resume = lcs[r].min(reach);
            if need[k] > resume {
                sweep_at[r] = Some(sweeps.len());
                sweeps.push(BackwardSweep {
                    hap: k,
                    resume,
                    need: need[k],
                    start: 0,
                    writes: Vec::new(),
                });
            }
            reach = resume.max(need[k]);
        }

        // Depth `depth` of the suffix at position `r` is held by the node of the latest sweep at or
        // before `r` that computed that depth along a suffix `r` shares at least that deep.
        let mut node_of: HashMap<(usize, usize), usize> = HashMap::new();
        let mut node_for = |r: usize, depth: usize| -> usize {
            if depth == 0 {
                return 0;
            }
            let mut p = r;
            let mut shared = usize::MAX;
            loop {
                if let Some(s) = sweep_at[p] {
                    let sweep: &BackwardSweep = &sweeps[s];
                    if sweep.resume < depth && depth <= sweep.need {
                        let next = node_of.len() + 1;
                        return *node_of.entry((s, depth)).or_insert(next);
                    }
                }
                shared = shared.min(lcs[p]);
                assert!(p > 0 && shared >= depth, "backward depth {depth} is never computed");
                p -= 1;
            }
        };
        let starts: Vec<usize> =
            sweeps.iter().map(|s| node_for(suffix_pos[s.hap], s.resume)).collect();
        let mut cuts = vec![None; n];
        for &k in &distinct {
            if need[k] > 0 {
                let node = node_for(suffix_pos[k], need[k]);
                cuts[k] = Some(Cut { column: bases[k].len() - need[k], node });
            }
        }
        for (sweep, start) in sweeps.iter_mut().zip(starts) {
            sweep.start = start;
        }
        for (&(s, depth), &node) in &node_of {
            sweeps[s].writes.push((depth, node));
        }
        for sweep in &mut sweeps {
            sweep.writes.sort_unstable();
        }
        SuffixPlan { cuts, sweeps, nodes: node_of.len() + 1 }
    }
}

fn common_suffix(a: &[u8], b: &[u8]) -> usize {
    a.iter().rev().zip(b.iter().rev()).take_while(|(x, y)| x == y).count()
}

impl HapSet for SortedHaps<'_> {
    fn len(&self) -> usize {
        self.bases.len()
    }

    fn order(&self) -> &[usize] {
        &self.order
    }

    fn single(&self, k: usize) -> Self {
        SortedHaps::new(&[self.bases[k]], false)
    }

    fn lanes(key: RunnerKey) -> usize {
        with_runner(key, |runner| runner.lanes())
    }

    fn run_batch(
        &self,
        key: RunnerKey,
        reads: &[ReadRef<'_>],
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    ) {
        with_runner(key, |runner| runner.run(reads, self, out, fallback))
    }
}

thread_local! {
    /// Kernel workspaces are reused across calls on the same thread; most assembly regions are
    /// small enough that allocating them per call would dominate.
    static RUNNERS: RefCell<Vec<(RunnerKey, Box<dyn BatchRunner>)>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with this thread's cached runner for `key`, creating it on first use.
fn with_runner<R>(key: RunnerKey, f: impl FnOnce(&mut dyn BatchRunner) -> R) -> R {
    RUNNERS.with(|cell| {
        let mut runners = cell.borrow_mut();
        let index = match runners.iter().position(|(k, _)| *k == key) {
            Some(i) => i,
            None => {
                runners.push((key, make_runner(key)));
                runners.len() - 1
            }
        };
        f(runners[index].1.as_mut())
    })
}

fn make_runner(key: RunnerKey) -> Box<dyn BatchRunner> {
    use simd::{ScalarF32, ScalarF64};
    #[cfg(target_arch = "x86_64")]
    if key.backend == Backend::Avx512 && !key.wide {
        return match key.precision {
            Precision::Float => Box::new(Runner::<simd::x86::Avx512F32Narrow>::new()),
            Precision::Double => Box::new(Runner::<simd::x86::Avx512F64Narrow>::new()),
        };
    }
    match (key.backend, key.precision) {
        (Backend::Scalar, Precision::Float) => Box::new(Runner::<ScalarF32>::new()),
        (Backend::Scalar, Precision::Double) => Box::new(Runner::<ScalarF64>::new()),
        #[cfg(target_arch = "aarch64")]
        (Backend::Neon, Precision::Float) => Box::new(Runner::<simd::neon::NeonF32>::new()),
        #[cfg(target_arch = "aarch64")]
        (Backend::Neon, Precision::Double) => Box::new(Runner::<simd::neon::NeonF64>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx2, Precision::Float) => Box::new(Runner::<simd::x86::Avx2F32>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx2, Precision::Double) => Box::new(Runner::<simd::x86::Avx2F64>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx512, Precision::Float) => Box::new(Runner::<simd::x86::Avx512F32>::new()),
        #[cfg(target_arch = "x86_64")]
        (Backend::Avx512, Precision::Double) => Box::new(Runner::<simd::x86::Avx512F64>::new()),
    }
}

/// A kernel instantiated for one backend and precision, with its scratch memory.
pub(crate) trait BatchRunner: Send {
    fn lanes(&self) -> usize;

    /// Computes at most `lanes()` reads against every haplotype. Results land in
    /// `out[read * haps.len() + sorted_hap]`; pairs whose result lost precision are appended to
    /// `fallback` as `(read, sorted_hap)` with `NaN` written in their place.
    fn run(
        &mut self,
        reads: &[ReadRef<'_>],
        haps: &SortedHaps<'_>,
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    );
}

pub(crate) struct Runner<S: Simd> {
    ws: Workspace<S>,
}

impl<S: Simd> Runner<S> {
    pub fn new() -> Self {
        Runner { ws: Workspace::new() }
    }

    #[inline(always)]
    fn run_impl(
        &mut self,
        reads: &[ReadRef<'_>],
        haps: &SortedHaps<'_>,
        out: &mut [f64],
        fallback: &mut Vec<(usize, usize)>,
    ) {
        let ws = &mut self.ws;
        ws.prepare(reads, haps);
        if !haps.plan.sweeps.is_empty() {
            ws.fill_last_column_node();
            for sweep in &haps.plan.sweeps {
                ws.run_hap_backward(sweep, &haps.codes[sweep.hap], haps.bases[sweep.hap].len());
            }
        }
        let n_haps = haps.len();
        for k in 0..n_haps {
            if haps.dup[k] {
                for lane in 0..reads.len() {
                    let result = out[lane * n_haps + k - 1];
                    if result.is_nan() {
                        fallback.push((lane, k));
                    }
                    out[lane * n_haps + k] = result;
                }
                continue;
            }
            let start = haps.lcp[k];
            if k > 0 {
                ws.lcp_counts[start] -= 1;
            }
            let hap_len = haps.bases[k].len();
            let cut = haps.plan.cuts[k];
            let stop = cut.map_or(hap_len, |c| c.column - 1);
            ws.snap_cols.clear();
            for q in (start + 1)..=stop {
                if ws.lcp_counts[q] > 0 {
                    ws.snap_cols.push(q);
                }
            }
            ws.run_hap(start, stop, hap_len, &haps.codes[k], cut.is_some());
            match cut {
                None => {
                    for lane in 0..reads.len() {
                        let raw = ws.acc[lane].to_f64();
                        out[lane * n_haps + k] = finish_lane::<S::Elem>(raw, lane, k, fallback);
                    }
                }
                Some(cut) => {
                    ws.join(cut.node, hap_len);
                    for lane in 0..reads.len() {
                        let raw = ws.joined[lane];
                        out[lane * n_haps + k] = finish_lane::<S::Elem>(raw, lane, k, fallback);
                    }
                }
            }
        }
    }
}

macro_rules! runner_impl {
    ($ty:ty) => {
        impl BatchRunner for Runner<$ty> {
            fn lanes(&self) -> usize {
                <$ty as Simd>::LANES
            }
            fn run(
                &mut self,
                reads: &[ReadRef<'_>],
                haps: &SortedHaps<'_>,
                out: &mut [f64],
                fallback: &mut Vec<(usize, usize)>,
            ) {
                with_flush_to_zero(|| self.run_impl(reads, haps, out, fallback))
            }
        }
    };
    ($ty:ty, $features:literal) => {
        impl BatchRunner for Runner<$ty> {
            fn lanes(&self) -> usize {
                <$ty as Simd>::LANES
            }
            fn run(
                &mut self,
                reads: &[ReadRef<'_>],
                haps: &SortedHaps<'_>,
                out: &mut [f64],
                fallback: &mut Vec<(usize, usize)>,
            ) {
                #[target_feature(enable = $features)]
                unsafe fn go(
                    runner: &mut Runner<$ty>,
                    reads: &[ReadRef<'_>],
                    haps: &SortedHaps<'_>,
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

runner_impl!(crate::simd::ScalarF32);
runner_impl!(crate::simd::ScalarF64);
#[cfg(target_arch = "aarch64")]
runner_impl!(crate::simd::neon::NeonF32);
#[cfg(target_arch = "aarch64")]
runner_impl!(crate::simd::neon::NeonF64);
#[cfg(target_arch = "x86_64")]
runner_impl!(crate::simd::x86::Avx2F32, "avx2,fma");
#[cfg(target_arch = "x86_64")]
runner_impl!(crate::simd::x86::Avx2F64, "avx2,fma");
#[cfg(target_arch = "x86_64")]
runner_impl!(crate::simd::x86::Avx512F32, "avx512f");
#[cfg(target_arch = "x86_64")]
runner_impl!(crate::simd::x86::Avx512F64, "avx512f");
#[cfg(target_arch = "x86_64")]
runner_impl!(crate::simd::x86::Avx512F32Narrow, "avx512f");
#[cfg(target_arch = "x86_64")]
runner_impl!(crate::simd::x86::Avx512F64Narrow, "avx512f");

/// One DP row of match, insertion and deletion values for every column, lane-interleaved.
struct RowBuf<E> {
    m: AlignedVec<E>,
    x: AlignedVec<E>,
    y: AlignedVec<E>,
}

impl<E: Float> RowBuf<E> {
    fn new() -> Self {
        RowBuf { m: AlignedVec::new(), x: AlignedVec::new(), y: AlignedVec::new() }
    }

    /// Sizes the row without clearing it: `run_hap` initialises every column it reads.
    fn resize(&mut self, len: usize) {
        self.m.resize_no_fill(len);
        self.x.resize_no_fill(len);
        self.y.resize_no_fill(len);
    }
}

/// The DP state of one column for every row, kept so a later haplotype can start from it.
struct Snapshot<E> {
    m: AlignedVec<E>,
    x: AlignedVec<E>,
    y: AlignedVec<E>,
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
        self.acc.resize_no_fill(lanes);
    }
}

/// One row of backward match and insertion values for every column, lane-interleaved. Backward
/// deletion values only travel along a row, so they live in registers.
struct BackwardRow<E> {
    m: AlignedVec<E>,
    x: AlignedVec<E>,
}

impl<E: Float> BackwardRow<E> {
    fn new() -> Self {
        BackwardRow { m: AlignedVec::new(), x: AlignedVec::new() }
    }

    /// Sizes the row without clearing it: a backward sweep zeroes the columns it reads first.
    fn resize(&mut self, len: usize) {
        self.m.resize_no_fill(len);
        self.x.resize_no_fill(len);
    }
}

/// The backward values of one column of a suffix, scaled by `Float::BACKWARD_SCALE`: what a later
/// backward sweep resumes from and what a cut joins.
struct BackwardNode<E> {
    /// `[row][lane]` match values; row 0 is never read.
    m: AlignedVec<E>,
    /// `[row][lane]` deletion values.
    y: AlignedVec<E>,
    /// Per lane, the mass of the paths that start in row 0 at this column or later.
    s: AlignedVec<E>,
}

impl<E: Float> BackwardNode<E> {
    fn new() -> Self {
        BackwardNode { m: AlignedVec::new(), y: AlignedVec::new(), s: AlignedVec::new() }
    }

    /// Sizes the node without clearing it: the sweep that writes it fills every row it holds.
    fn resize(&mut self, cells: usize, lanes: usize) {
        self.m.resize_no_fill(cells);
        self.y.resize_no_fill(cells);
        self.s.resize_no_fill(lanes);
    }
}

/// Scratch memory for one batch of reads; laid out lane-interleaved so that element `i` of every
/// lane is one contiguous vector.
struct Workspace<S: Simd> {
    /// Rows of the DP excluding row zero, i.e. the longest read in the batch.
    rows: usize,
    num_codes: usize,
    /// `[row][transition][lane]`.
    trans: AlignedVec<S::Elem>,
    /// `[row][haplotype base code][lane]`.
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
    bwd_prev: BackwardRow<S::Elem>,
    bwd_cur: BackwardRow<S::Elem>,
    /// Indexed by the node numbers of the haplotype set's [`SuffixPlan`].
    bwd_nodes: Vec<BackwardNode<S::Elem>>,
    /// `[row][lane]`: the forward match and deletion values of a cut column, entered from the
    /// column before it.
    join_m: AlignedVec<S::Elem>,
    join_y: AlignedVec<S::Elem>,
    /// Per lane, the raw result of a haplotype that took a cut.
    joined: Vec<f64>,
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
            bwd_prev: BackwardRow::new(),
            bwd_cur: BackwardRow::new(),
            bwd_nodes: Vec::new(),
            join_m: AlignedVec::new(),
            join_y: AlignedVec::new(),
            joined: Vec::new(),
        }
    }

    /// Fills the per-read tables for a batch and sizes every buffer for the haplotype set.
    fn prepare(&mut self, reads: &[ReadRef<'_>], haps: &SortedHaps<'_>) {
        let lanes = S::LANES;
        assert!(reads.len() <= lanes, "batch larger than the lane count");
        let rows = reads.iter().map(|r| r.len()).max().unwrap_or(0);
        let num_codes = haps.num_codes();
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
                for code in 0..num_codes {
                    let hap_base = haps.byte_of_code(code);
                    let matches = read_base == hap_base || read_base == b'N' || hap_base == b'N';
                    let p = if matches { p_match } else { p_mismatch };
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
        // in row zero, recorded for a haplotype of length one so the length rescaling applies.
        let origin = &mut self.snapshots[0];
        origin.m.fill(S::Elem::ZERO);
        origin.x.fill(S::Elem::ZERO);
        origin.y.fill(S::Elem::ZERO);
        origin.y[..lanes].fill(S::Elem::INITIAL_CONSTANT);
        origin.acc.fill(S::Elem::ZERO);
        origin.hap_len = 1;
        self.lcp_counts.clear();
        self.lcp_counts.resize(cols, 0);
        for k in 1..haps.len() {
            if !haps.dup[k] {
                self.lcp_counts[haps.lcp[k]] += 1;
            }
        }
        if !haps.plan.sweeps.is_empty() {
            self.bwd_prev.resize(cols * lanes);
            self.bwd_cur.resize(cols * lanes);
            let nodes = haps.plan.nodes;
            if self.bwd_nodes.len() < nodes {
                self.bwd_nodes.resize_with(nodes, BackwardNode::new);
            }
            for node in &mut self.bwd_nodes[..nodes] {
                node.resize((rows + 1) * lanes, lanes);
            }
            self.join_m.resize_no_fill((rows + 1) * lanes);
            self.join_y.resize_no_fill((rows + 1) * lanes);
            self.joined.resize(lanes, 0.0);
        }
    }

    /// Fills backward node 0, the last column of every haplotype: from there only trailing
    /// insertions lead to the end of the read, whatever the haplotype.
    #[inline(always)]
    fn fill_last_column_node(&mut self) {
        let lanes = S::LANES;
        let rows = self.rows;
        let scale = S::splat(S::Elem::BACKWARD_SCALE);
        let node = &mut self.bwd_nodes[0];
        node.m.fill(S::Elem::ZERO);
        node.y.fill(S::Elem::ZERO);
        node.s.fill(S::Elem::ZERO);
        let mut x_below = S::zero();
        for i in (1..=rows).rev() {
            let sink = scale.mul(S::load(&self.end_mul[(i - 1) * lanes..]));
            let (mi, ii) = if i < rows {
                let t = Transitions::<S>::load(&self.trans, i);
                (t.mi, t.ii)
            } else {
                (S::zero(), S::zero())
            };
            x_below.mul_add(mi, sink).store(&mut node.m[i * lanes..]);
            x_below = x_below.mul_add(ii, sink);
        }
    }

    /// Runs one haplotype of length `hap_len` over columns `start + 1..=stop`, the state of
    /// column `start` coming from its snapshot. Leaves in `acc` the per-lane raw mass of the paths
    /// ending at column `stop` or before, and records the snapshots listed in `snap_cols`. With
    /// `join`, also leaves in `join_m` and `join_y` the forward values of column `stop + 1`
    /// entered from column `stop`.
    #[inline(always)]
    fn run_hap(&mut self, start: usize, stop: usize, hap_len: usize, codes: &[u8], join: bool) {
        let lanes = S::LANES;
        let rows = self.rows;
        let num_codes = self.num_codes;
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
            join_m,
            join_y,
            ..
        } = self;
        let (before, after) = snapshots.split_at_mut(start + 1);
        let src = &before[start];
        let src_m: &[S::Elem] = &src.m;
        let src_x: &[S::Elem] = &src.x;
        let src_y: &[S::Elem] = &src.y;
        let src_acc: &[S::Elem] = &src.acc;
        let trans: &[S::Elem] = trans;
        let prior: &[S::Elem] = prior;
        let end_mul: &[S::Elem] = end_mul;
        let zero = S::zero();
        let init = S::splat(S::Elem::from_f64(S::Elem::INITIAL_CONSTANT.to_f64() / hap_len as f64));
        let scale = S::splat(S::Elem::from_f64(src.hap_len as f64 / hap_len as f64));

        for j in start..=stop {
            zero.store(&mut row_prev.m[j * lanes..]);
            zero.store(&mut row_prev.x[j * lanes..]);
            init.store(&mut row_prev.y[j * lanes..]);
        }
        let mut acc_v = S::load(src_acc).mul(scale);
        for &q in snap_cols.iter() {
            let snap = &mut after[q - start - 1];
            snap.hap_len = hap_len;
            zero.store(&mut snap.m[..lanes]);
            zero.store(&mut snap.x[..lanes]);
            init.store(&mut snap.y[..lanes]);
            acc_v.store(&mut snap.acc);
        }
        if start == stop && !join {
            acc_v.store(acc);
            return;
        }

        for i in 1..=rows {
            let r = i - 1;
            let t = Transitions::<S>::load(trans, r);
            let prior_row = &prior[r * num_codes * lanes..(r + 1) * num_codes * lanes];
            let mut state = CellState {
                m_diag: S::load(&src_m[(i - 1) * lanes..]).mul(scale),
                x_diag: S::load(&src_x[(i - 1) * lanes..]).mul(scale),
                y_diag: S::load(&src_y[(i - 1) * lanes..]).mul(scale),
                m_left: S::load(&src_m[i * lanes..]).mul(scale),
                x_left: S::load(&src_x[i * lanes..]).mul(scale),
                y_left: S::load(&src_y[i * lanes..]).mul(scale),
            };
            let prev = RowView { m: &row_prev.m, x: &row_prev.x, y: &row_prev.y };
            let mut cur = RowViewMut { m: &mut row_cur.m, x: &mut row_cur.x, y: &mut row_cur.y };
            let ends_here = end_rows[r];
            let mut rowsum = zero;
            let mut lo = start;
            let mut seg = 0;
            loop {
                let (hi, is_snapshot) =
                    if seg < snap_cols.len() { (snap_cols[seg], true) } else { (stop, false) };
                if hi > lo {
                    state = if ends_here {
                        dp_segment::<S, true>(
                            &t,
                            prior_row,
                            codes,
                            lo,
                            hi,
                            &prev,
                            &mut cur,
                            state,
                            &mut rowsum,
                        )
                    } else {
                        dp_segment::<S, false>(
                            &t,
                            prior_row,
                            codes,
                            lo,
                            hi,
                            &prev,
                            &mut cur,
                            state,
                            &mut rowsum,
                        )
                    };
                }
                if is_snapshot {
                    let snap = &mut after[hi - start - 1];
                    state.m_left.store(&mut snap.m[i * lanes..]);
                    state.x_left.store(&mut snap.x[i * lanes..]);
                    state.y_left.store(&mut snap.y[i * lanes..]);
                    if ends_here {
                        let e = S::load(&end_mul[r * lanes..]);
                        e.mul_add(rowsum, S::load(&snap.acc)).store(&mut snap.acc);
                    }
                } else {
                    break;
                }
                lo = hi;
                seg += 1;
            }
            if ends_here {
                acc_v = S::load(&end_mul[r * lanes..]).mul_add(rowsum, acc_v);
            }
            if join {
                // The cells of column stop + 1 as `dp_segment` would compute them. Its insertion
                // value is not needed: insertions stay within a column, so paths enter it only
                // through match or deletion.
                let prior = S::load(&prior_row[codes[stop] as usize * lanes..]);
                let st = &state;
                let m = prior
                    .mul(st.y_diag.mul_add(t.im, st.x_diag.mul_add(t.im, st.m_diag.mul(t.mm))));
                let y = st.y_left.mul_add(t.dd, st.m_left.mul(t.md));
                m.store(&mut join_m[i * lanes..]);
                y.store(&mut join_y[i * lanes..]);
            }
            std::mem::swap(row_prev, row_cur);
        }
        acc_v.store(acc);
    }

    /// Joins the forward values `run_hap` left for a cut column with backward node `node`,
    /// leaving each lane's raw result in `joined`: the paths ending before the cut, plus those
    /// entering the cut column times their mass from there to the end, plus the paths starting at
    /// the cut column or later. In `f64`, since the product of the two scaled values can exceed
    /// the single-precision range.
    #[inline(always)]
    fn join(&mut self, node: usize, hap_len: usize) {
        let lanes = S::LANES;
        let node = &self.bwd_nodes[node];
        let crossing = &mut self.joined;
        crossing.fill(0.0);
        for i in 1..=self.rows {
            let row = i * lanes..(i + 1) * lanes;
            let terms = self.join_m[row.clone()]
                .iter()
                .zip(&node.m[row.clone()])
                .zip(self.join_y[row.clone()].iter().zip(&node.y[row]));
            for (sum, ((&fm, &bm), (&fy, &by))) in crossing.iter_mut().zip(terms) {
                *sum += fm.to_f64() * bm.to_f64() + fy.to_f64() * by.to_f64();
            }
        }
        // The rounded initial value the forward sweep used, so both halves weight starts alike.
        let init = S::Elem::from_f64(S::Elem::INITIAL_CONSTANT.to_f64() / hap_len as f64).to_f64();
        let scale = S::Elem::BACKWARD_SCALE.to_f64();
        for (lane, raw) in crossing.iter_mut().enumerate() {
            let starts_after = init * node.s[lane].to_f64();
            *raw = self.acc[lane].to_f64() + (*raw + starts_after) / scale;
        }
    }

    /// Runs one backward sweep of `sweep` over its haplotype of length `hap_len`: rows from the
    /// last up to 0, each from right to left over the columns of depths `resume + 1..=need`,
    /// storing the nodes the sweep writes.
    #[inline(always)]
    fn run_hap_backward(&mut self, sweep: &BackwardSweep, codes: &[u8], hap_len: usize) {
        let lanes = S::LANES;
        let rows = self.rows;
        let num_codes = self.num_codes;
        let right = hap_len - sweep.resume;
        let left = hap_len - sweep.need;
        // Moved out while the sweep writes other nodes; no sweep writes the node it starts from.
        let start = std::mem::replace(&mut self.bwd_nodes[sweep.start], BackwardNode::new());
        let Workspace { trans, prior, end_mul, end_rows, bwd_prev, bwd_cur, bwd_nodes, .. } = self;
        let trans: &[S::Elem] = trans;
        let zero = S::zero();
        let scale = S::splat(S::Elem::BACKWARD_SCALE);
        // Nothing lies below the last row.
        for j in left..right {
            zero.store(&mut bwd_prev.m[j * lanes..]);
            zero.store(&mut bwd_prev.x[j * lanes..]);
        }
        let mut starts_after = S::load(&start.s);
        for i in (0..=rows).rev() {
            // Moving down from row i uses the next read position's transitions and priors; the
            // last row has none, and its lower neighbours are zero, so any finite values will do.
            let below = if i < rows {
                Transitions::<S>::load(trans, i)
            } else {
                Transitions { mm: zero, im: zero, mi: zero, ii: zero, md: zero, dd: zero }
            };
            let prior_row = &prior[i.min(rows - 1) * num_codes * lanes..][..num_codes * lanes];
            // Deletions move along row i with its own read position's transitions; row 0 has none.
            let (md, dd) = if i >= 1 {
                let t = Transitions::<S>::load(trans, i - 1);
                (t.md, t.dd)
            } else {
                (zero, zero)
            };
            let sink = if i >= 1 && end_rows[i - 1] {
                Some(scale.mul(S::load(&end_mul[(i - 1) * lanes..])))
            } else {
                None
            };
            let mut state = BackwardState {
                bm_diag: if i < rows { S::load(&start.m[(i + 1) * lanes..]) } else { zero },
                by_right: S::load(&start.y[i * lanes..]),
                bm_last: zero,
            };
            let prev = BackwardRowView { m: &bwd_prev.m, x: &bwd_prev.x };
            let mut cur = BackwardRowViewMut { m: &mut bwd_cur.m, x: &mut bwd_cur.x };
            let mut hi = right;
            for w in 0..=sweep.writes.len() {
                let (lo, node) = match sweep.writes.get(w) {
                    Some(&(depth, node)) => (hap_len - depth, Some(node)),
                    None => (left, None),
                };
                if lo < hi {
                    state = if i == 0 {
                        backward_row_zero(
                            below.im,
                            prior_row,
                            codes,
                            lo,
                            hi,
                            &prev,
                            state,
                            &mut starts_after,
                        )
                    } else if let Some(sink) = sink {
                        backward_segment::<S, true>(
                            &below, md, dd, prior_row, codes, lo, hi, &prev, &mut cur, state, sink,
                        )
                    } else {
                        backward_segment::<S, false>(
                            &below, md, dd, prior_row, codes, lo, hi, &prev, &mut cur, state, zero,
                        )
                    };
                    hi = lo;
                }
                if let Some(node) = node {
                    let node = &mut bwd_nodes[node];
                    state.by_right.store(&mut node.y[i * lanes..]);
                    if i == 0 {
                        starts_after.store(&mut node.s);
                    } else {
                        state.bm_last.store(&mut node.m[i * lanes..]);
                    }
                }
            }
            if i > 0 {
                std::mem::swap(bwd_prev, bwd_cur);
            }
        }
        self.bwd_nodes[sweep.start] = start;
    }
}

/// Plain-slice views of a [`RowBuf`], resolved once per row so the hot loop indexes slices directly.
struct RowView<'a, E> {
    m: &'a [E],
    x: &'a [E],
    y: &'a [E],
}

struct RowViewMut<'a, E> {
    m: &'a mut [E],
    x: &'a mut [E],
    y: &'a mut [E],
}

/// Converts a lane's raw scaled probability for sorted haplotype `k`, computed in precision `E`,
/// into a log10 likelihood, or records the pair in `fallback` and yields `NaN` when single
/// precision lost it.
#[inline(always)]
pub(crate) fn finish_lane<E: Float>(
    raw: f64,
    lane: usize,
    k: usize,
    fallback: &mut Vec<(usize, usize)>,
) -> f64 {
    let lost = match E::MIN_ACCEPTED {
        Some(threshold) => raw.is_nan() || raw < threshold,
        None => false,
    };
    if lost {
        fallback.push((lane, k));
        f64::NAN
    } else {
        raw.log10() - E::log10_initial_constant()
    }
}

/// The six transition probabilities of one read position, one vector each.
#[derive(Clone, Copy)]
pub(crate) struct Transitions<S> {
    pub mm: S,
    pub im: S,
    pub mi: S,
    pub ii: S,
    pub md: S,
    pub dd: S,
}

impl<S: Simd> Transitions<S> {
    /// Loads read row `r` from a `[row][transition][lane]` table.
    #[inline(always)]
    pub(crate) fn load(trans: &[S::Elem], r: usize) -> Self {
        let lanes = S::LANES;
        let tb = r * NUM_TRANSITIONS * lanes;
        Transitions {
            mm: S::load(&trans[tb + MATCH_TO_MATCH * lanes..]),
            im: S::load(&trans[tb + INDEL_TO_MATCH * lanes..]),
            mi: S::load(&trans[tb + MATCH_TO_INSERTION * lanes..]),
            ii: S::load(&trans[tb + INSERTION_TO_INSERTION * lanes..]),
            md: S::load(&trans[tb + MATCH_TO_DELETION * lanes..]),
            dd: S::load(&trans[tb + DELETION_TO_DELETION * lanes..]),
        }
    }
}

/// The neighbours of the next cell in a row sweep: the previous row's values one column back and
/// the current row's values one column back.
#[derive(Clone, Copy)]
pub(crate) struct CellState<S> {
    pub m_diag: S,
    pub x_diag: S,
    pub y_diag: S,
    pub m_left: S,
    pub x_left: S,
    pub y_left: S,
}

/// Computes columns `lo+1..=hi` of the current row. With `ACC` set, also sums match plus insertion
/// over the columns into `rowsum`, which the caller applies to lanes whose read ends on this row.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dp_segment<S: Simd, const ACC: bool>(
    t: &Transitions<S>,
    prior_row: &[S::Elem],
    codes: &[u8],
    lo: usize,
    hi: usize,
    prev: &RowView<'_, S::Elem>,
    cur: &mut RowViewMut<'_, S::Elem>,
    mut st: CellState<S>,
    rowsum: &mut S,
) -> CellState<S> {
    // Bounds are checked once per segment rather than on each of the seven loads and stores per
    // cell, which otherwise cost about a quarter of the inner loop. Column `j` occupies elements
    // `j * lanes..(j + 1) * lanes` of a row, and code `c` occupies `c * lanes..(c + 1) * lanes`
    // of the prior row.
    let lanes = S::LANES;
    let end = (hi + 1) * lanes;
    assert!(prev.m.len() >= end && prev.x.len() >= end && prev.y.len() >= end);
    assert!(cur.m.len() >= end && cur.x.len() >= end && cur.y.len() >= end);
    let codes = &codes[lo..hi];
    let num_codes = prior_row.len() / lanes;
    assert!(codes.iter().all(|&c| (c as usize) < num_codes));
    let (prev_m, prev_x, prev_y) = (prev.m.as_ptr(), prev.x.as_ptr(), prev.y.as_ptr());
    let (cur_m, cur_x, cur_y) = (cur.m.as_mut_ptr(), cur.x.as_mut_ptr(), cur.y.as_mut_ptr());
    let priors = prior_row.as_ptr();
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
        let m = prior.mul(st.y_diag.mul_add(t.im, st.x_diag.mul_add(t.im, st.m_diag.mul(t.mm))));
        let x = x_up.mul_add(t.ii, m_up.mul(t.mi));
        let y = st.y_left.mul_add(t.dd, st.m_left.mul(t.md));
        // SAFETY: as above, for the current row.
        unsafe {
            m.store_ptr(cur_m.add(off));
            x.store_ptr(cur_x.add(off));
            y.store_ptr(cur_y.add(off));
        }
        if ACC {
            *rowsum = rowsum.add(m.add(x));
        }
        st =
            CellState { m_diag: m_up, x_diag: x_up, y_diag: y_up, m_left: m, x_left: x, y_left: y };
    }
    st
}

/// Views of a [`BackwardRow`], resolved once per row.
struct BackwardRowView<'a, E> {
    m: &'a [E],
    x: &'a [E],
}

struct BackwardRowViewMut<'a, E> {
    m: &'a mut [E],
    x: &'a mut [E],
}

/// What a backward row sweep carries from one cell to the next cell on its left.
#[derive(Clone, Copy)]
struct BackwardState<S> {
    /// Match value one row down and one column right of the next cell.
    bm_diag: S,
    /// Deletion value of the cell to the right of the next cell.
    by_right: S,
    /// Match value of the cell computed last.
    bm_last: S,
}

/// Computes backward columns `hi - 1` down to `lo` of row `i` from row `i + 1` (`prev`). `below`
/// holds the transitions of read position `i + 1` and `prior_row` its priors; `md` and `dd` are
/// row `i`'s own deletion transitions. With `SINK`, adds `sink`, the scaled end weight of the lanes
/// whose read ends on row `i`, to the match and insertion values.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn backward_segment<S: Simd, const SINK: bool>(
    below: &Transitions<S>,
    md: S,
    dd: S,
    prior_row: &[S::Elem],
    codes: &[u8],
    lo: usize,
    hi: usize,
    prev: &BackwardRowView<'_, S::Elem>,
    cur: &mut BackwardRowViewMut<'_, S::Elem>,
    mut st: BackwardState<S>,
    sink: S,
) -> BackwardState<S> {
    // Bounds are checked once per segment, as in `dp_segment`. Column `j` occupies elements
    // `j * lanes..(j + 1) * lanes` of a row.
    let lanes = S::LANES;
    let end = hi * lanes;
    assert!(prev.m.len() >= end && prev.x.len() >= end);
    assert!(cur.m.len() >= end && cur.x.len() >= end);
    // Moving from column j to j + 1 emits haplotype base j + 1, whose code is `codes[j]`.
    let codes = &codes[lo..hi];
    let num_codes = prior_row.len() / lanes;
    assert!(codes.iter().all(|&c| (c as usize) < num_codes));
    let (prev_m, prev_x) = (prev.m.as_ptr(), prev.x.as_ptr());
    let (cur_m, cur_x) = (cur.m.as_mut_ptr(), cur.x.as_mut_ptr());
    let priors = prior_row.as_ptr();
    for (k, &code) in codes.iter().enumerate().rev() {
        let off = (lo + k) * lanes;
        // SAFETY: `off + lanes <= end` and `(code + 1) * lanes <= prior_row.len()` by the checks
        // above, and the previous and current rows are distinct buffers.
        let (prior, bx_below, bm_below) = unsafe {
            (
                S::load_ptr(priors.add(code as usize * lanes)),
                S::load_ptr(prev_x.add(off)),
                S::load_ptr(prev_m.add(off)),
            )
        };
        let into_match = prior.mul(st.bm_diag);
        let indel_to_match = into_match.mul(below.im);
        let mut bx = bx_below.mul_add(below.ii, indel_to_match);
        let by = st.by_right.mul_add(dd, indel_to_match);
        let mut bm = st.by_right.mul_add(md, bx_below.mul_add(below.mi, into_match.mul(below.mm)));
        if SINK {
            bm = bm.add(sink);
            bx = bx.add(sink);
        }
        // SAFETY: as above, for the current row.
        unsafe {
            bm.store_ptr(cur_m.add(off));
            bx.store_ptr(cur_x.add(off));
        }
        st = BackwardState { bm_diag: bm_below, by_right: by, bm_last: bm };
    }
    st
}

/// Row 0 of a backward sweep, columns `hi - 1` down to `lo`. Row 0 has no deletion chain: a path
/// starting there in column j moves straight to the match state of row 1, column j + 1. Only the
/// deletion values are needed, and their running sum `starts_after` over the columns so far.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn backward_row_zero<S: Simd>(
    im: S,
    prior_row: &[S::Elem],
    codes: &[u8],
    lo: usize,
    hi: usize,
    prev: &BackwardRowView<'_, S::Elem>,
    mut st: BackwardState<S>,
    starts_after: &mut S,
) -> BackwardState<S> {
    let lanes = S::LANES;
    for j in (lo..hi).rev() {
        let prior = S::load(&prior_row[codes[j] as usize * lanes..]);
        let by = prior.mul(st.bm_diag).mul(im);
        *starts_after = starts_after.add(by);
        st = BackwardState {
            bm_diag: S::load(&prev.m[j * lanes..]),
            by_right: by,
            bm_last: st.bm_last,
        };
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synthetic::Rng;

    /// Replays the suffix plan of `haps` and checks that every sweep starts from, and every cut
    /// joins, a node already written for exactly its own suffix at that depth; that no node is
    /// written twice; and that a cut haplotype's forward sweep still reaches every column the
    /// next distinct haplotype resumes from.
    fn consistent_plan<'a>(haps: &[&'a [u8]]) -> SortedHaps<'a> {
        let sorted = SortedHaps::new(haps, true);
        let plan = &sorted.plan;
        let suffix =
            |k: usize, depth: usize| sorted.bases[k][sorted.bases[k].len() - depth..].to_vec();
        let mut written: HashMap<usize, Vec<u8>> = HashMap::from([(0, Vec::new())]);
        for sweep in &plan.sweeps {
            assert!(sweep.resume < sweep.need);
            assert_eq!(written.get(&sweep.start), Some(&suffix(sweep.hap, sweep.resume)));
            for &(depth, node) in &sweep.writes {
                assert!(sweep.resume < depth && depth <= sweep.need);
                assert!(written.insert(node, suffix(sweep.hap, depth)).is_none());
            }
        }
        assert_eq!(written.len(), plan.nodes);
        for (k, cut) in plan.cuts.iter().enumerate() {
            let Some(cut) = cut else { continue };
            assert!(!sorted.dup[k]);
            let depth = sorted.bases[k].len() - cut.column;
            assert!(depth >= MIN_SAVING);
            assert_eq!(written.get(&cut.node), Some(&suffix(k, depth)));
            let next = (k + 1..sorted.len()).find(|&j| !sorted.dup[j]);
            let resumes_at = next.map_or(0, |j| sorted.lcp[j]);
            assert!(cut.column > sorted.lcp[k] && cut.column > resumes_at);
        }
        sorted
    }

    fn reference_like(rng: &mut Rng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len).map(|_| alphabet[rng.below(alphabet.len())]).collect()
    }

    /// `base` with a few substitutions, insertions or deletions.
    fn edited(rng: &mut Rng, alphabet: &[u8], base: &[u8]) -> Vec<u8> {
        let mut hap = base.to_vec();
        for _ in 0..1 + rng.below(3) {
            let pos = rng.below(hap.len());
            match rng.below(3) {
                0 => hap[pos] = alphabet[rng.below(alphabet.len())],
                1 => hap.insert(pos, alphabet[rng.below(alphabet.len())]),
                _ if hap.len() > 1 => {
                    hap.remove(pos);
                }
                _ => {}
            }
        }
        hap
    }

    #[test]
    fn a_substitution_cuts_both_haplotypes_right_after_their_shared_prefix() {
        let reference = reference_like(&mut Rng::new(1), b"ACGT", 40);
        let mut snp = reference.clone();
        snp[20] = if snp[20] == b'A' { b'C' } else { b'A' };
        let sorted = consistent_plan(&[&reference, &snp]);
        // Both share columns 1..=20 forward and everything after column 21 backward, so both
        // compute column 21 alone and join the same node there.
        let cuts = &sorted.plan.cuts;
        assert_eq!(cuts[0].map(|c| c.column), Some(21));
        assert_eq!(cuts[0], cuts[1]);
        assert_eq!(sorted.plan.sweeps.len(), 1);
    }

    #[test]
    fn a_haplotype_whose_suffix_is_its_own_runs_forward_to_its_end() {
        let reference = reference_like(&mut Rng::new(2), b"ACGT", 40);
        let mut late = reference.clone();
        late[38] = if late[38] == b'A' { b'C' } else { b'A' };
        let sorted = consistent_plan(&[&reference, &late]);
        assert!(sorted.plan.cuts.iter().all(Option::is_none));
        assert!(sorted.plan.sweeps.is_empty());
    }

    #[test]
    fn duplicates_are_never_cut_and_take_their_first_copys_results() {
        let reference = reference_like(&mut Rng::new(3), b"ACGT", 30);
        let mut snp = reference.clone();
        snp[10] = if snp[10] == b'A' { b'C' } else { b'A' };
        let sorted = consistent_plan(&[&snp, &reference, &snp, &reference]);
        assert_eq!(sorted.dup, [false, true, false, true]);
        assert!(sorted.plan.cuts[0].is_some() && sorted.plan.cuts[2].is_some());
    }

    #[test]
    fn plans_of_random_haplotype_sets_are_consistent() {
        let mut rng = Rng::new(4);
        for case in 0..400 {
            let alphabet: &[u8] = if case % 2 == 0 { b"AC" } else { b"ACGT" };
            let len = 5 + rng.below(60);
            let base = reference_like(&mut rng, alphabet, len);
            let mut haps = vec![base.clone()];
            for _ in 0..rng.below(30) {
                let hap = if rng.chance(0.15) {
                    haps[rng.below(haps.len())].clone()
                } else {
                    let from = rng.below(haps.len());
                    edited(&mut rng, alphabet, &haps[from])
                };
                haps.push(hap);
            }
            let refs: Vec<&[u8]> = haps.iter().map(Vec::as_slice).collect();
            consistent_plan(&refs);
        }
    }
}
