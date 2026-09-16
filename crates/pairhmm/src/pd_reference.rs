//! The partially determined PairHMM of GATK's DRAGEN mode, in which a haplotype carries per-base
//! flags describing alternate alleles: `SNP` with the allowed bases, and `DEL_START`/`DEL_END`
//! marking a span the read may skip. GATK's `LoglessPDPairHMM` adds three "branch" matrices that
//! remember the state before a deletion so the recurrence can take the maximum of skipping or
//! not skipping it. This module holds the scalar double-precision port used as the oracle.
//!
//! GATK carries the deletion state from the end of one row into the start of the next, so the
//! values in a column depend on flags to the right of it; a vectorized PD kernel therefore cannot
//! share columns between haplotypes the way the plain kernel does.

use crate::ReadRef;
use crate::model::{
    DELETION_TO_DELETION, INDEL_TO_MATCH, INSERTION_TO_INSERTION, MATCH_TO_DELETION,
    MATCH_TO_INSERTION, MATCH_TO_MATCH, NUM_TRANSITIONS, TABLES,
};
use crate::pd::{DEL_END, DEL_START, base_matches_pd};
use crate::reference::{INITIAL_CONDITION, initial_condition_log10};

#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    Normal,
    InsideDel,
    AfterDel,
}

/// The log10 likelihood of `read` given the partially determined haplotype `(bases, flags)`,
/// computed exactly as GATK's `LoglessPDPairHMM` does for a haplotype computed from its first
/// column, including the deletion state carried from the end of one row into the next.
pub fn log10_likelihood(bases: &[u8], flags: &[u8], read: &ReadRef<'_>) -> f64 {
    assert_eq!(bases.len(), flags.len());
    let rows = read.len() + 1;
    let cols = bases.len() + 1;
    let idx = |i: usize, j: usize| i * cols + j;
    let mut m = vec![0.0f64; rows * cols];
    let mut x = vec![0.0f64; rows * cols];
    let mut y = vec![0.0f64; rows * cols];
    let mut bm = vec![0.0f64; rows * cols];
    let mut bx = vec![0.0f64; rows * cols];
    let mut by = vec![0.0f64; rows * cols];
    let initial = INITIAL_CONDITION / bases.len() as f64;
    y[..cols].fill(initial);
    let transitions: Vec<[f64; NUM_TRANSITIONS]> = (0..read.len())
        .map(|r| TABLES.transitions(read.ins_gop[r], read.del_gop[r], read.gcp[r]))
        .collect();

    let mut state = State::Normal;
    for i in 1..rows {
        let t = &transitions[i - 1];
        let (p_match, p_mismatch) = TABLES.priors(read.quals[i - 1]);
        let rb = read.bases[i - 1];
        for j in 1..cols {
            let hb = bases[j - 1];
            let flag = flags[j - 1];
            let prior = if rb == hb || rb == b'N' || hb == b'N' || base_matches_pd(rb, flag) {
                p_match
            } else {
                p_mismatch
            };
            let (diag, up, left, cur) =
                (idx(i - 1, j - 1), idx(i - 1, j), idx(i, j - 1), idx(i, j));
            let del_end = flag & DEL_END != 0;
            match state {
                State::Normal => {
                    bm[cur] = m[left];
                    by[cur] = y[left];
                    bx[cur] = x[left];
                    m[cur] = prior
                        * (m[diag] * t[MATCH_TO_MATCH]
                            + x[diag] * t[INDEL_TO_MATCH]
                            + y[diag] * t[INDEL_TO_MATCH]);
                    y[cur] = m[left] * t[MATCH_TO_DELETION] + y[left] * t[DELETION_TO_DELETION];
                }
                State::InsideDel => {
                    bm[cur] = bm[left];
                    by[cur] = by[left];
                    bx[cur] = bx[left];
                    m[cur] = prior
                        * (m[diag] * t[MATCH_TO_MATCH]
                            + x[diag] * t[INDEL_TO_MATCH]
                            + y[diag] * t[INDEL_TO_MATCH]);
                    y[cur] = m[left] * t[MATCH_TO_DELETION] + y[left] * t[DELETION_TO_DELETION];
                }
                State::AfterDel => {
                    bm[cur] = bm[left].max(m[left]);
                    by[cur] = by[left].max(y[left]);
                    bx[cur] = bx[left].max(x[left]);
                    m[cur] = prior
                        * (bm[diag].max(m[diag]) * t[MATCH_TO_MATCH]
                            + bx[diag].max(x[diag]) * t[INDEL_TO_MATCH]
                            + by[diag].max(y[diag]) * t[INDEL_TO_MATCH]);
                    y[cur] = bm[left].max(m[left]) * t[MATCH_TO_DELETION]
                        + by[left].max(y[left]) * t[DELETION_TO_DELETION];
                }
            }
            x[cur] = if del_end {
                bm[up].max(m[up]) * t[MATCH_TO_INSERTION]
                    + bx[up].max(x[up]) * t[INSERTION_TO_INSERTION]
            } else {
                m[up] * t[MATCH_TO_INSERTION] + x[up] * t[INSERTION_TO_INSERTION]
            };
            if state == State::AfterDel {
                state = State::Normal;
            }
            if flag & DEL_START != 0 {
                state = State::InsideDel;
            }
            if del_end {
                state = State::AfterDel;
            }
        }
    }
    let last = rows - 1;
    let mut sum = 0.0;
    for j in 1..cols {
        sum += m[idx(last, j)] + x[idx(last, j)];
    }
    sum.log10() - initial_condition_log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pd::{ALT_A, ALT_C, ALT_G, ALT_T, SNP};
    use crate::reference;
    use crate::synthetic::Read;

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
    fn without_flags_equals_the_plain_pairhmm() {
        let hap = b"ACGTTGCAAGGCTTAGGCTTACG";
        let r = read(b"GTTGCAAGGCTTAG");
        let flags = vec![0u8; hap.len()];
        let pd = log10_likelihood(hap, &flags, &r.as_ref());
        let plain = reference::log10_likelihood(hap, &r.as_ref());
        assert!((pd - plain).abs() < 1e-12, "{pd} vs {plain}");
    }

    #[test]
    fn snp_alternate_makes_the_alternate_base_match() {
        let hap = b"ACGTTGCAAGGCTTAGGCTTACG";
        let alt_read = read(b"GTTGCATGGCTTAG");
        let mut flags = vec![0u8; hap.len()];
        // Position 8 (the second A of CAAG) may also be T.
        flags[8] = SNP | ALT_T;
        let with = log10_likelihood(hap, &flags, &alt_read.as_ref());
        let without = log10_likelihood(hap, &vec![0u8; hap.len()], &alt_read.as_ref());
        let exact = log10_likelihood(hap, &flags, &read(b"GTTGCAAGGCTTAG").as_ref());
        // With the flag the mismatch penalty disappears: the alternate read scores within a
        // fraction of a log10 unit of the exact read, instead of several units below it.
        assert!(with > without + 1.0, "{with} vs {without}");
        assert!((with - exact).abs() < 0.2, "{with} vs {exact}");
    }

    #[test]
    fn deletion_span_lets_the_read_skip_bases() {
        let hap = b"ACGTTGCAAGGCTTAGGCTTACG";
        // A read missing "AGG" (positions 8..11 of the haplotype).
        let r = read(b"GTTGCACTTAGGC");
        let mut flags = vec![0u8; hap.len()];
        flags[7] = DEL_START;
        flags[10] = DEL_END;
        let with = log10_likelihood(hap, &flags, &r.as_ref());
        let without = log10_likelihood(hap, &vec![0u8; hap.len()], &r.as_ref());
        assert!(with > without, "{with} vs {without}");
    }

    #[test]
    fn base_matching_honours_only_flagged_alternates() {
        assert!(base_matches_pd(b'A', SNP | ALT_A));
        assert!(base_matches_pd(b'a', SNP | ALT_A | ALT_G));
        assert!(!base_matches_pd(b'C', SNP | ALT_A));
        assert!(!base_matches_pd(b'A', ALT_A), "no SNP flag, no alternate");
        assert!(!base_matches_pd(b'N', SNP | ALT_A | ALT_C | ALT_G | ALT_T));
    }
}
