//! Probability tables shared by every PairHMM implementation in this crate.
//!
//! Values are computed in `f64` exactly the way GATK computes them in `QualityUtils`,
//! `PairHMMModel` and `MathUtils.approximateLog10SumLog10`, including the quantized Jacobian
//! log-sum table, so that likelihoods agree with GATK's Java `LoglessPairHMM` and with Intel GKL.
//! Single-precision kernels narrow these values at batch setup.

use std::sync::LazyLock;

/// Largest quality GATK tabulates; larger inputs are clamped (GATK would throw).
pub const MAX_QUAL: usize = 254;
/// Divisor applied to the mismatch probability so that the three possible wrong bases share it.
pub const TRISTATE_CORRECTION: f64 = 3.0;

pub const NUM_TRANSITIONS: usize = 6;
pub const MATCH_TO_MATCH: usize = 0;
pub const INDEL_TO_MATCH: usize = 1;
pub const MATCH_TO_INSERTION: usize = 2;
pub const INSERTION_TO_INSERTION: usize = 3;
pub const MATCH_TO_DELETION: usize = 4;
pub const DELETION_TO_DELETION: usize = 5;

const JACOBIAN_MAX_TOLERANCE: f64 = 8.0;
const JACOBIAN_TABLE_STEP: f64 = 0.0001;
const JACOBIAN_INV_STEP: f64 = 1.0 / JACOBIAN_TABLE_STEP;
const INV_LN10: f64 = 1.0 / std::f64::consts::LN_10;

pub static TABLES: LazyLock<Tables> = LazyLock::new(Tables::new);

/// Phred-quality lookup tables. Built once per process through [`TABLES`].
pub struct Tables {
    error_prob: Vec<f64>,
    match_to_match: Vec<f64>,
}

impl Tables {
    fn new() -> Self {
        let error_prob: Vec<f64> = (0..=MAX_QUAL).map(|q| 10f64.powf(q as f64 / -10.0)).collect();
        let mut match_to_match = vec![0.0; ((MAX_QUAL + 1) * (MAX_QUAL + 2)) / 2];
        let mut offset = 0;
        for i in 0..=MAX_QUAL {
            for j in 0..=i {
                let log10_sum = approximate_log10_sum_log10(-0.1 * i as f64, -0.1 * j as f64);
                let m2m_log10 = (-(10f64.powf(log10_sum)).min(1.0)).ln_1p() * INV_LN10;
                match_to_match[offset + j] = 10f64.powf(m2m_log10);
            }
            offset += i + 1;
        }
        Tables { error_prob, match_to_match }
    }

    /// Probability that a base with the given phred quality is wrong.
    pub fn error_prob(&self, qual: u8) -> f64 {
        self.error_prob[(qual as usize).min(MAX_QUAL)]
    }

    /// Match-to-match transition probability for the given insertion and deletion gap-open
    /// qualities; symmetric in its arguments.
    pub fn match_to_match(&self, ins_qual: u8, del_qual: u8) -> f64 {
        let a = (ins_qual as usize).min(MAX_QUAL);
        let b = (del_qual as usize).min(MAX_QUAL);
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        self.match_to_match[((hi * (hi + 1)) >> 1) + lo]
    }

    /// The six transition probabilities for one read position, indexed by the `*_TO_*` constants.
    pub fn transitions(&self, ins_qual: u8, del_qual: u8, gcp: u8) -> [f64; NUM_TRANSITIONS] {
        let gap_continue = self.error_prob(gcp);
        let mut t = [0.0; NUM_TRANSITIONS];
        t[MATCH_TO_MATCH] = self.match_to_match(ins_qual, del_qual);
        t[MATCH_TO_INSERTION] = self.error_prob(ins_qual);
        t[MATCH_TO_DELETION] = self.error_prob(del_qual);
        t[INDEL_TO_MATCH] = 1.0 - gap_continue;
        t[INSERTION_TO_INSERTION] = gap_continue;
        t[DELETION_TO_DELETION] = gap_continue;
        t
    }

    /// Emission priors for one read position: `(matching base, each mismatching base)`.
    pub fn priors(&self, qual: u8) -> (f64, f64) {
        let e = self.error_prob(qual);
        (1.0 - e, e / TRISTATE_CORRECTION)
    }
}

/// GATK's `MathUtils.approximateLog10SumLog10`: `log10(10^a + 10^b)` through a quantized table.
fn approximate_log10_sum_log10(a: f64, b: f64) -> f64 {
    if a > b {
        return approximate_log10_sum_log10(b, a);
    }
    if a == f64::NEG_INFINITY {
        return b;
    }
    let diff = b - a;
    if diff < JACOBIAN_MAX_TOLERANCE { b + jacobian(diff) } else { b }
}

/// `log10(1 + 10^-diff)` evaluated at the table point GATK would look up.
fn jacobian(diff: f64) -> f64 {
    let k = fast_round(diff * JACOBIAN_INV_STEP);
    (1.0 + 10f64.powf(-(k as f64) * JACOBIAN_TABLE_STEP)).log10()
}

fn fast_round(d: f64) -> i64 {
    if d > 0.0 { (d + 0.5) as i64 } else { (d - 0.5) as i64 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_prob_matches_phred_definition() {
        assert!((TABLES.error_prob(0) - 1.0).abs() < 1e-15);
        assert!((TABLES.error_prob(10) - 0.1).abs() < 1e-15);
        assert!((TABLES.error_prob(30) - 0.001).abs() < 1e-15);
    }

    #[test]
    fn quals_above_max_are_clamped() {
        assert_eq!(TABLES.error_prob(255), TABLES.error_prob(254));
        assert_eq!(TABLES.match_to_match(255, 10), TABLES.match_to_match(254, 10));
    }

    #[test]
    fn match_to_match_is_symmetric_and_close_to_one_minus_gap_opens() {
        assert_eq!(TABLES.match_to_match(45, 30), TABLES.match_to_match(30, 45));
        let expected = 1.0 - (TABLES.error_prob(45) + TABLES.error_prob(30));
        assert!((TABLES.match_to_match(45, 30) - expected).abs() < 1e-6);
    }

    #[test]
    fn match_to_match_at_zero_quals_is_zero() {
        assert_eq!(TABLES.match_to_match(0, 0), 0.0);
    }

    #[test]
    fn transitions_use_gap_continuation_for_extension() {
        let t = TABLES.transitions(45, 40, 10);
        assert!((t[INSERTION_TO_INSERTION] - 0.1).abs() < 1e-15);
        assert!((t[DELETION_TO_DELETION] - 0.1).abs() < 1e-15);
        assert!((t[INDEL_TO_MATCH] - 0.9).abs() < 1e-15);
        assert!((t[MATCH_TO_INSERTION] - TABLES.error_prob(45)).abs() < 1e-15);
        assert!((t[MATCH_TO_DELETION] - TABLES.error_prob(40)).abs() < 1e-15);
    }

    #[test]
    fn approximate_log10_sum_matches_exact_for_moderate_differences() {
        let (a, b) = (-3.0, -2.5);
        let exact = (10f64.powf(a) + 10f64.powf(b)).log10();
        assert!((approximate_log10_sum_log10(a, b) - exact).abs() < 1e-4);
        assert_eq!(approximate_log10_sum_log10(a, b), approximate_log10_sum_log10(b, a));
        assert_eq!(approximate_log10_sum_log10(-20.0, -2.0), -2.0);
    }
}
