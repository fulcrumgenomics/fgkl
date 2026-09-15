//! Anti-diagonal SIMD fill of the Smith-Waterman matrix.
//!
//! Cells on one anti-diagonal (constant `i + j`) depend only on the two previous anti-diagonals,
//! so they are computed in vectors of consecutive rows. Scores for the last two anti-diagonals
//! and the gap scores for the last one live in row-indexed ring buffers with padding on both
//! ends, so the neighbour of a cell is a plain load at row offset zero or minus one and no
//! cross-lane shuffle is needed. The alternate sequence is stored reversed so that the bases
//! paired with consecutive rows are consecutive too. Traceback flags are stored anti-diagonal by
//! anti-diagonal, one byte per cell.

use std::marker::PhantomData;

use crate::simd::SimdI32;
use crate::{
    DELETION_EXTENDS, DIR_DELETION, DIR_DIAG, DIR_INSERTION, INSERTION_EXTENDS, LOW_INIT_VALUE,
    MATRIX_MIN_CUTOFF, OverhangStrategy, SwParameters, select_end,
};

/// Fills the matrix for one backend; type-erased so the aligner can hold any backend.
pub(crate) trait DiagFill: Send {
    /// Computes the matrix and returns the end cell and trailing overhang, as
    /// [`select_end`] does.
    fn fill(
        &mut self,
        reference: &[u8],
        alternate: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) -> (usize, usize, usize);

    /// Traceback flags of cell `(i, j)`, both at least 1.
    fn trace_at(&self, i: usize, j: usize) -> u8;
}

pub(crate) struct DiagWorkspace<V: SimdI32> {
    h2: Vec<i32>,
    h1: Vec<i32>,
    h0: Vec<i32>,
    e1: Vec<i32>,
    e0: Vec<i32>,
    f1: Vec<i32>,
    f0: Vec<i32>,
    ref_i32: Vec<i32>,
    alt_rev: Vec<i32>,
    trace: Vec<u8>,
    diag_start: Vec<usize>,
    last_col: Vec<i32>,
    bottom: Vec<i32>,
    alt_len: usize,
    _v: PhantomData<V>,
}

impl<V: SimdI32> DiagWorkspace<V> {
    pub fn new() -> Self {
        DiagWorkspace {
            h2: Vec::new(),
            h1: Vec::new(),
            h0: Vec::new(),
            e1: Vec::new(),
            e0: Vec::new(),
            f1: Vec::new(),
            f0: Vec::new(),
            ref_i32: Vec::new(),
            alt_rev: Vec::new(),
            trace: Vec::new(),
            diag_start: Vec::new(),
            last_col: Vec::new(),
            bottom: Vec::new(),
            alt_len: 0,
            _v: PhantomData,
        }
    }

    #[inline(always)]
    fn fill_impl(
        &mut self,
        reference: &[u8],
        alternate: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) -> (usize, usize, usize) {
        let l = V::LANES;
        let pad = l;
        let n = reference.len();
        let m = alternate.len();
        self.alt_len = m;
        for buf in [&mut self.h2, &mut self.h1, &mut self.h0] {
            buf.clear();
            buf.resize(n + 2 * l + 2, 0);
        }
        for buf in [&mut self.e1, &mut self.e0, &mut self.f1, &mut self.f0] {
            buf.clear();
            buf.resize(n + 2 * l + 2, LOW_INIT_VALUE);
        }
        self.ref_i32.clear();
        self.ref_i32.resize(n + 2 * l + 2, 0);
        for (i, &b) in reference.iter().enumerate() {
            self.ref_i32[pad + i + 1] = b as i32;
        }
        self.alt_rev.clear();
        self.alt_rev.resize(m + 2 * l + 2, -1);
        for (k, &b) in alternate.iter().rev().enumerate() {
            self.alt_rev[pad + k] = b as i32;
        }
        self.diag_start.clear();
        self.diag_start.resize(n + m + 2, 0);
        let mut total = 0usize;
        for d in 0..=n + m {
            self.diag_start[d] = total;
            if d >= 2 {
                let i_lo = d.saturating_sub(m).max(1);
                let i_hi = n.min(d - 1);
                total += i_hi + 1 - i_lo;
            }
        }
        self.diag_start[n + m + 1] = total;
        self.trace.clear();
        self.trace.resize(total + 2 * l, 0);
        self.last_col.clear();
        self.last_col.resize(n + 1, 0);
        self.bottom.clear();
        self.bottom.resize(m + 1, 0);

        let global_ends =
            matches!(strategy, OverhangStrategy::Indel | OverhangStrategy::LeadingIndel);
        let (w_open, w_extend) = (params.gap_open_penalty, params.gap_extend_penalty);
        let boundary = |k: usize| -> i32 {
            if global_ends && k > 0 { w_open + (k as i32 - 1) * w_extend } else { 0 }
        };
        // Anti-diagonals 0 and 1 hold only boundary cells.
        self.h2[pad] = 0;
        self.h1[pad] = boundary(1);
        self.h1[pad + 1] = boundary(1);
        self.e1[pad + 1] = LOW_INIT_VALUE;
        self.f1[pad] = LOW_INIT_VALUE;

        let v_match = V::splat(params.match_value);
        let v_mismatch = V::splat(params.mismatch_penalty);
        let v_open = V::splat(w_open);
        let v_extend = V::splat(w_extend);
        let v_cutoff = V::splat(MATRIX_MIN_CUTOFF);
        let v_diag = V::splat(DIR_DIAG as i32);
        let v_ins = V::splat(DIR_INSERTION as i32);
        let v_del = V::splat(DIR_DELETION as i32);
        let v_zero = V::splat(0);
        let v_del_ext = V::splat(DELETION_EXTENDS as i32);
        let v_ins_ext = V::splat(INSERTION_EXTENDS as i32);

        for d in 2..=n + m {
            let i_lo = d.saturating_sub(m).max(1);
            let i_hi = n.min(d - 1);
            let ts = self.diag_start[d];
            let mut i = i_lo;
            while i <= i_hi {
                // The base paired with row i sits at alternate index d - i - 1, which is
                // alt_rev index m - d + i; adding pad first keeps the arithmetic unsigned.
                let alt_index = pad + m + i - d;
                let h_diag = V::load(&self.h2[pad + i - 1..]);
                let h_up = V::load(&self.h1[pad + i - 1..]);
                let h_left = V::load(&self.h1[pad + i..]);
                let f_up = V::load(&self.f1[pad + i - 1..]);
                let e_left = V::load(&self.e1[pad + i..]);
                let a = V::load(&self.ref_i32[pad + i..]);
                let b = V::load(&self.alt_rev[alt_index..]);
                let diag = h_diag.add(V::select(a.eq(b), v_match, v_mismatch));
                let open_down = h_up.add(v_open);
                let ext_down = f_up.add(v_extend);
                let f_open = open_down.gt(ext_down);
                let f = V::select(f_open, open_down, ext_down);
                let open_right = h_left.add(v_open);
                let ext_right = e_left.add(v_extend);
                let e_open = open_right.gt(ext_right);
                let e = V::select(e_open, open_right, ext_right);
                let diag_best = V::mask_and(diag.ge(f), diag.ge(e));
                let e_ge_f = e.ge(f);
                let score = V::select(diag_best, diag, V::select(e_ge_f, e, f)).max(v_cutoff);
                let dir = V::select(diag_best, v_diag, V::select(e_ge_f, v_ins, v_del));
                let flags = dir
                    .add(V::select(f_open, v_zero, v_del_ext))
                    .add(V::select(e_open, v_zero, v_ins_ext));
                score.store(&mut self.h0[pad + i..]);
                e.store(&mut self.e0[pad + i..]);
                f.store(&mut self.f0[pad + i..]);
                flags.store_low_bytes(&mut self.trace[ts + (i - i_lo)..]);
                i += l;
            }
            if d <= m {
                self.h0[pad] = boundary(d);
                self.f0[pad] = LOW_INIT_VALUE;
            }
            if d <= n {
                self.h0[pad + d] = boundary(d);
                self.e0[pad + d] = LOW_INIT_VALUE;
            }
            if d > m && d - m <= n {
                self.last_col[d - m] = self.h0[pad + d - m];
            }
            if d > n && d - n <= m {
                self.bottom[d - n] = self.h0[pad + n];
            }
            std::mem::swap(&mut self.h2, &mut self.h1);
            std::mem::swap(&mut self.h1, &mut self.h0);
            std::mem::swap(&mut self.e1, &mut self.e0);
            std::mem::swap(&mut self.f1, &mut self.f0);
        }
        select_end(strategy, n, m, &self.last_col, &self.bottom)
    }

    #[inline(always)]
    fn trace_at_impl(&self, i: usize, j: usize) -> u8 {
        let d = i + j;
        let i_lo = d.saturating_sub(self.alt_len).max(1);
        self.trace[self.diag_start[d] + (i - i_lo)]
    }
}

macro_rules! diag_fill_impl {
    ($ty:ty) => {
        impl DiagFill for DiagWorkspace<$ty> {
            fn fill(
                &mut self,
                r: &[u8],
                a: &[u8],
                p: &SwParameters,
                s: OverhangStrategy,
            ) -> (usize, usize, usize) {
                self.fill_impl(r, a, p, s)
            }
            fn trace_at(&self, i: usize, j: usize) -> u8 {
                self.trace_at_impl(i, j)
            }
        }
    };
    ($ty:ty, $features:literal) => {
        impl DiagFill for DiagWorkspace<$ty> {
            fn fill(
                &mut self,
                r: &[u8],
                a: &[u8],
                p: &SwParameters,
                s: OverhangStrategy,
            ) -> (usize, usize, usize) {
                #[target_feature(enable = $features)]
                unsafe fn go(
                    w: &mut DiagWorkspace<$ty>,
                    r: &[u8],
                    a: &[u8],
                    p: &SwParameters,
                    s: OverhangStrategy,
                ) -> (usize, usize, usize) {
                    w.fill_impl(r, a, p, s)
                }
                // SAFETY: workspaces of this type are only built after runtime feature detection.
                unsafe { go(self, r, a, p, s) }
            }
            fn trace_at(&self, i: usize, j: usize) -> u8 {
                self.trace_at_impl(i, j)
            }
        }
    };
}

diag_fill_impl!(crate::simd::ScalarI32);
#[cfg(target_arch = "aarch64")]
diag_fill_impl!(crate::simd::neon::I32x4);
#[cfg(target_arch = "x86_64")]
diag_fill_impl!(crate::simd::x86::I32x8, "avx2");
#[cfg(target_arch = "x86_64")]
diag_fill_impl!(crate::simd::x86::I32x16, "avx512f,avx512bw");
