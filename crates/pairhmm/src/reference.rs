//! Scalar double-precision PairHMM, a direct port of GATK's `LoglessPairHMM`. It is the oracle for
//! the vectorized kernels and is deliberately written to mirror the Java operation order.

use crate::ReadRef;
use crate::model::{
    DELETION_TO_DELETION, INDEL_TO_MATCH, INSERTION_TO_INSERTION, MATCH_TO_DELETION,
    MATCH_TO_INSERTION, MATCH_TO_MATCH, NUM_TRANSITIONS, TABLES,
};

/// GATK's `LoglessPairHMM.INITIAL_CONDITION`: `2^1020`.
pub const INITIAL_CONDITION: f64 = f64::from_bits(0x7FB0_0000_0000_0000);

/// `log10(INITIAL_CONDITION)`.
pub fn initial_condition_log10() -> f64 {
    INITIAL_CONDITION.log10()
}

/// The log10 likelihood of `read` given `haplotype`. Both must be non-empty.
pub fn log10_likelihood(haplotype: &[u8], read: &ReadRef<'_>) -> f64 {
    let rows = read.len() + 1;
    let cols = haplotype.len() + 1;
    let mut m = vec![0.0f64; rows * cols];
    let mut x = vec![0.0f64; rows * cols];
    let mut y = vec![0.0f64; rows * cols];

    let initial = INITIAL_CONDITION / haplotype.len() as f64;
    y[..cols].fill(initial);

    let transitions: Vec<[f64; NUM_TRANSITIONS]> = (0..read.len())
        .map(|r| TABLES.transitions(read.ins_gop[r], read.del_gop[r], read.gcp[r]))
        .collect();

    for i in 1..rows {
        let t = &transitions[i - 1];
        let (p_match, p_mismatch) = TABLES.priors(read.quals[i - 1]);
        let read_base = read.bases[i - 1];
        for j in 1..cols {
            let hap_base = haplotype[j - 1];
            let prior = if read_base == hap_base || read_base == b'N' || hap_base == b'N' {
                p_match
            } else {
                p_mismatch
            };
            let diag = (i - 1) * cols + (j - 1);
            let up = (i - 1) * cols + j;
            let left = i * cols + (j - 1);
            let cur = i * cols + j;
            m[cur] = prior
                * (m[diag] * t[MATCH_TO_MATCH]
                    + x[diag] * t[INDEL_TO_MATCH]
                    + y[diag] * t[INDEL_TO_MATCH]);
            x[cur] = m[up] * t[MATCH_TO_INSERTION] + x[up] * t[INSERTION_TO_INSERTION];
            y[cur] = m[left] * t[MATCH_TO_DELETION] + y[left] * t[DELETION_TO_DELETION];
        }
    }

    let last = (rows - 1) * cols;
    let mut sum = 0.0;
    for j in 1..cols {
        sum += m[last + j] + x[last + j];
    }
    sum.log10() - initial_condition_log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read<'a>(
        bases: &'a [u8],
        q: &'a [u8],
        i: &'a [u8],
        d: &'a [u8],
        g: &'a [u8],
    ) -> ReadRef<'a> {
        ReadRef { bases, quals: q, ins_gop: i, del_gop: d, gcp: g }
    }

    #[test]
    fn initial_condition_is_two_to_the_1020() {
        assert_eq!(INITIAL_CONDITION, 2f64.powi(1020));
    }

    #[test]
    fn identical_single_base_has_likelihood_near_log10_of_match_probability() {
        let lk = log10_likelihood(b"A", &read(b"A", &[30], &[45], &[45], &[10]));
        // One match cell: prior(1 - 1e-3) * initial * indelToMatch(0.9), normalized by initial.
        let expected = ((1.0 - 1e-3) * 0.9f64).log10();
        assert!((lk - expected).abs() < 1e-12, "{lk} vs {expected}");
    }

    #[test]
    fn mismatch_is_less_likely_than_match() {
        let matched =
            log10_likelihood(b"ACGT", &read(b"ACGT", &[30; 4], &[45; 4], &[45; 4], &[10; 4]));
        let mismatched =
            log10_likelihood(b"ACGT", &read(b"ACTT", &[30; 4], &[45; 4], &[45; 4], &[10; 4]));
        assert!(matched > mismatched);
        assert!(matched <= 0.0);
    }

    #[test]
    fn n_matches_every_base_wherever_it_is_aligned() {
        let (q, i, d, g) = (&[30u8; 4], &[45u8; 4], &[45u8; 4], &[10u8; 4]);
        let exact = log10_likelihood(b"ACGT", &read(b"ACGT", q, i, d, g));
        // An all-N read cannot tell haplotypes apart, and an all-N haplotype cannot tell reads
        // apart; either is at least as likely as the exact match.
        let n_read = log10_likelihood(b"ACGT", &read(b"NNNN", q, i, d, g));
        assert_eq!(n_read, log10_likelihood(b"TTTT", &read(b"NNNN", q, i, d, g)));
        let n_hap = log10_likelihood(b"NNNN", &read(b"ACGT", q, i, d, g));
        assert_eq!(n_hap, log10_likelihood(b"NNNN", &read(b"GGGG", q, i, d, g)));
        assert!(n_read >= exact && n_hap >= exact);
    }
}
