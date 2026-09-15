//! A line-by-line port of GATK's `SmithWatermanJavaAligner`, with full integer score and
//! traceback matrices; the oracle for [`crate::Aligner`].

use crate::{Alignment, CigarElement, CigarOp, OverhangStrategy, SwParameters, last_index_of};

const MATRIX_MIN_CUTOFF: i32 = -100_000_000;
const LOW_INIT_VALUE: i32 = i32::MIN / 2;

/// Aligns `alternate` to `reference`; both must be non-empty.
pub fn align(
    reference: &[u8],
    alternate: &[u8],
    params: &SwParameters,
    strategy: OverhangStrategy,
) -> Alignment {
    assert!(!reference.is_empty() && !alternate.is_empty());
    if matches!(strategy, OverhangStrategy::SoftClip | OverhangStrategy::Ignore)
        && let Some(offset) = last_index_of(reference, alternate)
    {
        return Alignment {
            cigar: vec![CigarElement { len: alternate.len() as u32, op: CigarOp::M }],
            offset: offset as i32,
        };
    }
    let n = reference.len() + 1;
    let m = alternate.len() + 1;
    let mut sw = vec![0i32; n * m];
    let mut btrack = vec![0i32; n * m];
    calculate_matrix(reference, alternate, &mut sw, &mut btrack, strategy, params);
    calculate_cigar(&sw, &btrack, n, m, strategy)
}

#[allow(clippy::needless_range_loop)]
fn calculate_matrix(
    reference: &[u8],
    alternate: &[u8],
    sw: &mut [i32],
    btrack: &mut [i32],
    strategy: OverhangStrategy,
    params: &SwParameters,
) {
    let ncol = alternate.len() + 1;
    let nrow = reference.len() + 1;
    let mut best_gap_v = vec![LOW_INIT_VALUE; ncol + 1];
    let mut gap_size_v = vec![0i32; ncol + 1];
    let mut best_gap_h = vec![LOW_INIT_VALUE; nrow + 1];
    let mut gap_size_h = vec![0i32; nrow + 1];
    if matches!(strategy, OverhangStrategy::Indel | OverhangStrategy::LeadingIndel) {
        sw[1] = params.gap_open_penalty;
        let mut current = params.gap_open_penalty;
        for j in 2..ncol {
            current += params.gap_extend_penalty;
            sw[j] = current;
        }
        sw[ncol] = params.gap_open_penalty;
        current = params.gap_open_penalty;
        for i in 2..nrow {
            current += params.gap_extend_penalty;
            sw[i * ncol] = current;
        }
    }
    let w_open = params.gap_open_penalty;
    let w_extend = params.gap_extend_penalty;
    let w_match = params.match_value;
    let w_mismatch = params.mismatch_penalty;
    for i in 1..nrow {
        let a_base = reference[i - 1];
        for j in 1..ncol {
            let b_base = alternate[j - 1];
            let step_diag =
                sw[(i - 1) * ncol + j - 1] + if a_base == b_base { w_match } else { w_mismatch };
            let mut prev_gap = sw[(i - 1) * ncol + j] + w_open;
            best_gap_v[j] += w_extend;
            if prev_gap > best_gap_v[j] {
                best_gap_v[j] = prev_gap;
                gap_size_v[j] = 1;
            } else {
                gap_size_v[j] += 1;
            }
            let step_down = best_gap_v[j];
            let kd = gap_size_v[j];
            prev_gap = sw[i * ncol + j - 1] + w_open;
            best_gap_h[i] += w_extend;
            if prev_gap > best_gap_h[i] {
                best_gap_h[i] = prev_gap;
                gap_size_h[i] = 1;
            } else {
                gap_size_h[i] += 1;
            }
            let step_right = best_gap_h[i];
            let ki = gap_size_h[i];
            let diag_highest_or_equal = step_diag >= step_down && step_diag >= step_right;
            let cell = i * ncol + j;
            if diag_highest_or_equal {
                sw[cell] = MATRIX_MIN_CUTOFF.max(step_diag);
                btrack[cell] = 0;
            } else if step_right >= step_down {
                sw[cell] = MATRIX_MIN_CUTOFF.max(step_right);
                btrack[cell] = -ki;
            } else {
                sw[cell] = MATRIX_MIN_CUTOFF.max(step_down);
                btrack[cell] = kd;
            }
        }
    }
}

fn calculate_cigar(
    sw: &[i32],
    btrack: &[i32],
    n: usize,
    m: usize,
    strategy: OverhangStrategy,
) -> Alignment {
    let ref_length = n - 1;
    let alt_length = m - 1;
    let mut p1 = 0usize;
    let mut p2 = alt_length;
    let mut max_score = i32::MIN;
    let mut segment_length = 0usize;
    if strategy == OverhangStrategy::Indel {
        p1 = ref_length;
    } else {
        for i in 1..n {
            let cur = sw[i * m + alt_length];
            if cur >= max_score {
                p1 = i;
                max_score = cur;
            }
        }
        if strategy != OverhangStrategy::LeadingIndel {
            for j in 1..m {
                let cur = sw[ref_length * m + j];
                if cur > max_score
                    || (cur == max_score
                        && (ref_length as i32 - j as i32).abs() < (p1 as i32 - p2 as i32).abs())
                {
                    p1 = ref_length;
                    p2 = j;
                    max_score = cur;
                    segment_length = alt_length - j;
                }
            }
        }
    }
    let mut lce: Vec<CigarElement> = Vec::new();
    if segment_length > 0 && strategy == OverhangStrategy::SoftClip {
        lce.push(CigarElement { len: segment_length as u32, op: CigarOp::S });
        segment_length = 0;
    }
    let mut state = CigarOp::M;
    loop {
        let btr = btrack[p1 * m + p2];
        let (new_state, step_length) = if btr > 0 {
            (CigarOp::D, btr as usize)
        } else if btr < 0 {
            (CigarOp::I, (-btr) as usize)
        } else {
            (CigarOp::M, 1)
        };
        match new_state {
            CigarOp::M => {
                p1 -= 1;
                p2 -= 1;
            }
            CigarOp::I => p2 -= step_length,
            CigarOp::D => p1 -= step_length,
            CigarOp::S => unreachable!(),
        }
        if new_state == state {
            segment_length += step_length;
        } else {
            if segment_length > 0 {
                lce.push(CigarElement { len: segment_length as u32, op: state });
            }
            segment_length = step_length;
            state = new_state;
        }
        if !(p1 > 0 && p2 > 0) {
            break;
        }
    }
    let alignment_offset;
    if strategy == OverhangStrategy::SoftClip {
        lce.push(CigarElement { len: segment_length as u32, op: state });
        if p2 > 0 {
            lce.push(CigarElement { len: p2 as u32, op: CigarOp::S });
        }
        alignment_offset = p1 as i32;
    } else if strategy == OverhangStrategy::Ignore {
        lce.push(CigarElement { len: (segment_length + p2) as u32, op: state });
        alignment_offset = p1 as i32 - p2 as i32;
    } else {
        lce.push(CigarElement { len: segment_length as u32, op: state });
        if p1 > 0 {
            lce.push(CigarElement { len: p1 as u32, op: CigarOp::D });
        } else if p2 > 0 {
            lce.push(CigarElement { len: p2 as u32, op: CigarOp::I });
        }
        alignment_offset = 0;
    }
    lce.reverse();
    Alignment { cigar: lce, offset: alignment_offset }
}
