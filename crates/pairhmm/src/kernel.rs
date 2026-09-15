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

use crate::ReadRef;
use crate::model::{
    DELETION_TO_DELETION, INDEL_TO_MATCH, INSERTION_TO_INSERTION, MATCH_TO_DELETION,
    MATCH_TO_INSERTION, MATCH_TO_MATCH, NUM_TRANSITIONS, TABLES,
};
use crate::simd::{AlignedVec, Float, Simd};

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
}

impl<'a> SortedHaps<'a> {
    pub fn new(haplotypes: &[&'a [u8]]) -> Self {
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
        SortedHaps { order, bases, codes, lcp, max_len, extra_bases }
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
            ws.run_hap(start, hap_len, &haps.codes[k]);
            for (lane, _) in reads.iter().enumerate() {
                let raw = ws.acc[lane].to_f64();
                let lost = match S::Elem::MIN_ACCEPTED {
                    Some(threshold) => raw.is_nan() || raw < threshold,
                    None => false,
                };
                out[lane * n_haps + k] = if lost {
                    fallback.push((lane, k));
                    f64::NAN
                } else {
                    raw.log10() - S::Elem::log10_initial_constant()
                };
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

/// Runs `f` with x86 flush-to-zero and denormals-are-zero set, restoring the caller's MXCSR
/// afterwards. Single-precision DP values routinely fall into the subnormal range, where x86
/// arithmetic takes microcode assists costing over a hundred cycles per operation; flushing them
/// to zero costs nothing numerically because any pair whose result is that small is recomputed
/// in double precision anyway. GKL enables the same mode.
#[cfg(target_arch = "x86_64")]
pub(crate) fn with_flush_to_zero<R>(f: impl FnOnce() -> R) -> R {
    const FTZ_DAZ: u32 = 0x8040;
    let mut saved: u32 = 0;
    // SAFETY: stmxcsr/ldmxcsr only read and write the MXCSR register through a valid u32.
    unsafe {
        core::arch::asm!("stmxcsr [{}]", in(reg) &mut saved, options(nostack, preserves_flags))
    };
    let flushed = saved | FTZ_DAZ;
    unsafe {
        core::arch::asm!("ldmxcsr [{}]", in(reg) &flushed, options(nostack, preserves_flags))
    };
    let result = f();
    unsafe { core::arch::asm!("ldmxcsr [{}]", in(reg) &saved, options(nostack, preserves_flags)) };
    result
}

/// Subnormals cost nothing extra on aarch64, so nothing to do.
#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn with_flush_to_zero<R>(f: impl FnOnce() -> R) -> R {
    f()
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
        for &p in &haps.lcp[1..] {
            self.lcp_counts[p] += 1;
        }
    }

    /// Runs one haplotype of length `hap_len` from column `start`, whose state comes from the
    /// snapshot at that column, leaving the per-lane raw result in `acc` and recording the
    /// snapshots listed in `snap_cols`.
    #[inline(always)]
    fn run_hap(&mut self, start: usize, hap_len: usize, codes: &[u8]) {
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

        for j in start..=hap_len {
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
        if start == hap_len {
            acc_v.store(acc);
            return;
        }

        for i in 1..=rows {
            let r = i - 1;
            let tb = r * NUM_TRANSITIONS * lanes;
            let t = Transitions {
                mm: S::load(&trans[tb + MATCH_TO_MATCH * lanes..]),
                im: S::load(&trans[tb + INDEL_TO_MATCH * lanes..]),
                mi: S::load(&trans[tb + MATCH_TO_INSERTION * lanes..]),
                ii: S::load(&trans[tb + INSERTION_TO_INSERTION * lanes..]),
                md: S::load(&trans[tb + MATCH_TO_DELETION * lanes..]),
                dd: S::load(&trans[tb + DELETION_TO_DELETION * lanes..]),
            };
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
                    if seg < snap_cols.len() { (snap_cols[seg], true) } else { (hap_len, false) };
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
            std::mem::swap(row_prev, row_cur);
        }
        acc_v.store(acc);
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

/// The six transition probabilities of one read position, one vector each.
#[derive(Clone, Copy)]
struct Transitions<S> {
    mm: S,
    im: S,
    mi: S,
    ii: S,
    md: S,
    dd: S,
}

/// The neighbours of the next cell in a row sweep: the previous row's values one column back and
/// the current row's values one column back.
#[derive(Clone, Copy)]
struct CellState<S> {
    m_diag: S,
    x_diag: S,
    y_diag: S,
    m_left: S,
    x_left: S,
    y_left: S,
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
