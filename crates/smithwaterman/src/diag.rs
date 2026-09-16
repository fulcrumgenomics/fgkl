//! Anti-diagonal SIMD fill of the Smith-Waterman matrix.
//!
//! Cells on one anti-diagonal (constant `i + j`) depend only on the two previous anti-diagonals,
//! so they are computed in vectors of consecutive rows. Scores for the last two anti-diagonals
//! and the gap scores for the last one live in row-indexed ring buffers with padding on both
//! ends, so the neighbour of a cell is a plain load at row offset zero or minus one and no
//! cross-lane shuffle is needed. The alternate sequence is stored reversed so that the bases
//! paired with consecutive rows are consecutive too. Traceback flags are stored anti-diagonal by
//! anti-diagonal, one byte per cell.
//!
//! Scores are stored relative to the anti-diagonal: a cell holds `s * H - (s * match / 2) * (i + j)`
//! plus the lane type's headroom constant, where `s` is 2 when the match value is odd and 1
//! otherwise. A match then costs nothing, a mismatch `s * (mismatch - match)`, and each gap step
//! its penalty minus half a match, so every step is non-positive and a cell's value never rises
//! along a path (nor above the headroom constant). That is what makes 16-bit
//! lanes exact: additions saturate at the lane floor, a saturated value can only breed saturated
//! values, so every stored value above the floor is the exact score, and the traceback only visits
//! cells whose values are at least the chosen end's. The end is chosen on absolute scores decoded
//! from the last row and column; a saturated candidate is only known to lie below a bound, and
//! when that bound is not below the best decoded candidate the fill reports failure and the
//! aligner redoes the pair in 32-bit lanes. GATK's `MATRIX_MIN_CUTOFF` clamp is not applied: it
//! would need sequences hundreds of thousands of bases long to matter.

use std::marker::PhantomData;

use crate::simd::{Lane, SimdInt};
use crate::{
    DELETION_EXTENDS, DIR_DELETION, DIR_DIAG, DIR_INSERTION, INSERTION_EXTENDS, OverhangStrategy,
    SwParameters, select_end,
};

/// Fills the matrix for one backend; type-erased so the aligner can hold any backend.
pub(crate) trait DiagFill: Send {
    /// Computes the matrix and returns the end cell and trailing overhang, as [`select_end`]
    /// does, or `None` when the lanes saturated somewhere that could change the answer.
    fn fill(
        &mut self,
        reference: &[u8],
        alternate: &[u8],
        params: &SwParameters,
        strategy: OverhangStrategy,
    ) -> Option<(usize, usize, usize)>;

    /// Traceback flags of cell `(i, j)`, both at least 1.
    fn trace_at(&self, i: usize, j: usize) -> u8;
}

/// The relative encoding of one parameter set: the per-step costs and how to decode a stored
/// value back to `s * H`.
struct Encoding {
    scale: i32,
    half_match: i32,
    mismatch: i32,
    open: i32,
    extend: i32,
    headroom: i32,
}

impl Encoding {
    fn new(params: &SwParameters, headroom: i32) -> Self {
        let scale = if params.match_value % 2 != 0 { 2 } else { 1 };
        let half_match = scale * params.match_value / 2;
        Encoding {
            scale,
            half_match,
            mismatch: scale * (params.mismatch_penalty - params.match_value),
            open: scale * params.gap_open_penalty - half_match,
            extend: scale * params.gap_extend_penalty - half_match,
            headroom,
        }
    }

    /// The stored value of a boundary cell `(k, 0)` or `(0, k)`.
    fn boundary(&self, k: usize, global_ends: bool) -> i32 {
        self.headroom
            + if global_ends && k > 0 {
                self.open + (k as i32 - 1) * self.extend
            } else {
                -self.half_match * k as i32
            }
    }

    /// The absolute score `H` of a stored value on anti-diagonal `d`, or, for a saturated value,
    /// the largest absolute score it could stand for; the flag says which.
    fn decode<E: Lane>(&self, stored: E, d: usize) -> (i32, bool) {
        let shifted = stored.to_i32() - self.headroom + self.half_match * d as i32;
        if stored == E::FLOOR {
            (shifted.div_euclid(self.scale), true)
        } else {
            (shifted / self.scale, false)
        }
    }
}

/// Whether the encoded step costs of these parameters fit 16-bit lanes with the headroom the
/// relative encoding needs. GATK's parameter sets all do; the check guards the public API.
pub(crate) fn narrow_lanes_apply(params: &SwParameters) -> bool {
    let e = Encoding::new(params, 0);
    params.match_value >= 0
        && params.mismatch_penalty <= 0
        && params.gap_open_penalty <= 0
        && params.gap_extend_penalty <= 0
        && [e.mismatch, e.open, e.extend].iter().all(|&c| c >= -16_000)
}

pub(crate) struct DiagWorkspace<V: SimdInt> {
    h2: Vec<V::Elem>,
    h1: Vec<V::Elem>,
    h0: Vec<V::Elem>,
    e1: Vec<V::Elem>,
    e0: Vec<V::Elem>,
    f1: Vec<V::Elem>,
    f0: Vec<V::Elem>,
    reference: Vec<V::Elem>,
    alt_rev: Vec<V::Elem>,
    trace: Vec<u8>,
    diag_start: Vec<usize>,
    /// Absolute scores of the last column and bottom row (a bound for saturated cells), with the
    /// saturation flags alongside.
    last_col: Vec<i32>,
    last_col_saturated: Vec<bool>,
    bottom: Vec<i32>,
    bottom_saturated: Vec<bool>,
    alt_len: usize,
    _v: PhantomData<V>,
}

impl<V: SimdInt> DiagWorkspace<V> {
    pub fn new() -> Self {
        DiagWorkspace {
            h2: Vec::new(),
            h1: Vec::new(),
            h0: Vec::new(),
            e1: Vec::new(),
            e0: Vec::new(),
            f1: Vec::new(),
            f0: Vec::new(),
            reference: Vec::new(),
            alt_rev: Vec::new(),
            trace: Vec::new(),
            diag_start: Vec::new(),
            last_col: Vec::new(),
            last_col_saturated: Vec::new(),
            bottom: Vec::new(),
            bottom_saturated: Vec::new(),
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
    ) -> Option<(usize, usize, usize)> {
        let l = V::LANES;
        let pad = l;
        let n = reference.len();
        let m = alternate.len();
        let floor = V::Elem::FLOOR;
        let enc = Encoding::new(params, V::Elem::HEADROOM);
        self.alt_len = m;
        for buf in [&mut self.h2, &mut self.h1, &mut self.h0] {
            buf.clear();
            buf.resize(n + 2 * l + 2, V::Elem::from_i32(0));
        }
        for buf in [&mut self.e1, &mut self.e0, &mut self.f1, &mut self.f0] {
            buf.clear();
            buf.resize(n + 2 * l + 2, floor);
        }
        self.reference.clear();
        self.reference.resize(n + 2 * l + 2, V::Elem::from_i32(0));
        for (i, &b) in reference.iter().enumerate() {
            self.reference[pad + i + 1] = V::Elem::from_i32(b as i32);
        }
        // Padding bases are -1 so they match nothing.
        self.alt_rev.clear();
        self.alt_rev.resize(m + 2 * l + 2, V::Elem::from_i32(-1));
        for (k, &b) in alternate.iter().rev().enumerate() {
            self.alt_rev[pad + k] = V::Elem::from_i32(b as i32);
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
        for buf in [&mut self.last_col, &mut self.bottom] {
            buf.clear();
        }
        self.last_col.resize(n + 1, 0);
        self.bottom.resize(m + 1, 0);
        for buf in [&mut self.last_col_saturated, &mut self.bottom_saturated] {
            buf.clear();
        }
        self.last_col_saturated.resize(n + 1, false);
        self.bottom_saturated.resize(m + 1, false);

        let global_ends =
            matches!(strategy, OverhangStrategy::Indel | OverhangStrategy::LeadingIndel);
        let boundary = |k: usize| -> V::Elem { V::Elem::from_i32(enc.boundary(k, global_ends)) };
        // Anti-diagonals 0 and 1 hold only boundary cells.
        self.h2[pad] = boundary(0);
        self.h1[pad] = boundary(1);
        self.h1[pad + 1] = boundary(1);
        self.e1[pad + 1] = floor;
        self.f1[pad] = floor;

        let v_zero = V::splat(V::Elem::from_i32(0));
        let v_mismatch = V::splat(V::Elem::from_i32(enc.mismatch));
        let v_open = V::splat(V::Elem::from_i32(enc.open));
        let v_extend = V::splat(V::Elem::from_i32(enc.extend));
        let v_diag = V::splat(V::Elem::from_i32(DIR_DIAG as i32));
        let v_ins = V::splat(V::Elem::from_i32(DIR_INSERTION as i32));
        let v_del = V::splat(V::Elem::from_i32(DIR_DELETION as i32));
        let v_del_ext = V::splat(V::Elem::from_i32(DELETION_EXTENDS as i32));
        let v_ins_ext = V::splat(V::Elem::from_i32(INSERTION_EXTENDS as i32));

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
                let a = V::load(&self.reference[pad + i..]);
                let b = V::load(&self.alt_rev[alt_index..]);
                let diag = h_diag.add(V::select(a.eq(b), v_zero, v_mismatch));
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
                let score = V::select(diag_best, diag, V::select(e_ge_f, e, f));
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
                self.f0[pad] = floor;
            }
            if d <= n {
                self.h0[pad + d] = boundary(d);
                self.e0[pad + d] = floor;
            }
            if d > m && d - m <= n {
                let (value, saturated) = enc.decode(self.h0[pad + d - m], d);
                self.last_col[d - m] = value;
                self.last_col_saturated[d - m] = saturated;
            }
            if d > n && d - n <= m {
                let (value, saturated) = enc.decode(self.h0[pad + n], d);
                self.bottom[d - n] = value;
                self.bottom_saturated[d - n] = saturated;
            }
            std::mem::swap(&mut self.h2, &mut self.h1);
            std::mem::swap(&mut self.h1, &mut self.h0);
            std::mem::swap(&mut self.e1, &mut self.e0);
            std::mem::swap(&mut self.f1, &mut self.f0);
        }

        let (p1, p2, trailing) = select_end(strategy, n, m, &self.last_col, &self.bottom);
        let chosen = if p2 == m { self.last_col[p1] } else { self.bottom[p2] };
        // A saturated candidate is only known to lie at or below its decoded bound. If any such
        // bound reaches the chosen score the true winner is unknown, so the pair needs wider lanes.
        let saturated_bound = |values: &[i32], flags: &[bool], count: usize| -> Option<i32> {
            values[1..=count]
                .iter()
                .zip(&flags[1..=count])
                .filter(|&(_, &saturated)| saturated)
                .map(|(&v, _)| v)
                .max()
        };
        let bound =
            match strategy {
                OverhangStrategy::Indel => self.last_col_saturated[n].then_some(self.last_col[n]),
                OverhangStrategy::LeadingIndel => {
                    saturated_bound(&self.last_col, &self.last_col_saturated, n)
                }
                OverhangStrategy::SoftClip | OverhangStrategy::Ignore => {
                    saturated_bound(&self.last_col, &self.last_col_saturated, n)
                        .max(saturated_bound(&self.bottom, &self.bottom_saturated, m))
                }
            };
        if bound.is_some_and(|b| b >= chosen) {
            return None;
        }
        Some((p1, p2, trailing))
    }

    #[inline(always)]
    fn trace_at_impl(&self, i: usize, j: usize) -> u8 {
        debug_assert!(i >= 1 && j >= 1, "traceback never reads the boundary row or column");
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
            ) -> Option<(usize, usize, usize)> {
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
            ) -> Option<(usize, usize, usize)> {
                #[target_feature(enable = $features)]
                unsafe fn go(
                    w: &mut DiagWorkspace<$ty>,
                    r: &[u8],
                    a: &[u8],
                    p: &SwParameters,
                    s: OverhangStrategy,
                ) -> Option<(usize, usize, usize)> {
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

#[cfg(target_arch = "aarch64")]
diag_fill_impl!(crate::simd::neon::I32x4);
#[cfg(target_arch = "aarch64")]
diag_fill_impl!(crate::simd::neon::I16x8);
#[cfg(target_arch = "x86_64")]
diag_fill_impl!(crate::simd::x86::I32x8, "avx2");
#[cfg(target_arch = "x86_64")]
diag_fill_impl!(crate::simd::x86::I16x16, "avx2");
#[cfg(target_arch = "x86_64")]
diag_fill_impl!(crate::simd::x86::I32x16, "avx512f,avx512bw");
#[cfg(target_arch = "x86_64")]
diag_fill_impl!(crate::simd::x86::I16x32, "avx512f,avx512bw");
