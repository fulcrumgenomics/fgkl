package com.fulcrumgenomics.fgkl.pairhmm;

import static com.fulcrumgenomics.fgkl.PairHmmModel.INITIAL_CONDITION;
import static com.fulcrumgenomics.fgkl.PairHmmModel.INITIAL_CONDITION_LOG10;
import static com.fulcrumgenomics.fgkl.PairHmmModel.TRISTATE_CORRECTION;
import static com.fulcrumgenomics.fgkl.PairHmmModel.errorProb;
import static com.fulcrumgenomics.fgkl.PairHmmModel.matchToMatch;

/**
 * Scalar double-precision PairHMM ported from GATK's {@code LoglessPairHMM}, {@code PairHMMModel}
 * and {@code QualityUtils}; the oracle the native binding is tested against.
 */
final class ReferencePairHmm {
    private ReferencePairHmm() {}

    /** The log10 likelihood of the read given the haplotype; both must be non-empty. */
    static double log10Likelihood(byte[] hap, byte[] read, byte[] quals, byte[] insGop, byte[] delGop, byte[] gcp) {
        int rows = read.length + 1;
        int cols = hap.length + 1;
        double[][] m = new double[rows][cols];
        double[][] x = new double[rows][cols];
        double[][] y = new double[rows][cols];
        double initial = INITIAL_CONDITION / hap.length;
        for (int j = 0; j < cols; j++) y[0][j] = initial;
        for (int i = 1; i < rows; i++) {
            double tMM = matchToMatch(insGop[i - 1], delGop[i - 1]);
            double tMI = errorProb(insGop[i - 1]);
            double tMD = errorProb(delGop[i - 1]);
            double tGap = errorProb(gcp[i - 1]);
            double tIM = 1.0 - tGap;
            double pMatch = 1.0 - errorProb(quals[i - 1]);
            double pMismatch = errorProb(quals[i - 1]) / TRISTATE_CORRECTION;
            byte rb = read[i - 1];
            for (int j = 1; j < cols; j++) {
                byte hb = hap[j - 1];
                double prior = (rb == hb || rb == 'N' || hb == 'N') ? pMatch : pMismatch;
                m[i][j] = prior * (m[i - 1][j - 1] * tMM + x[i - 1][j - 1] * tIM + y[i - 1][j - 1] * tIM);
                x[i][j] = m[i - 1][j] * tMI + x[i - 1][j] * tGap;
                y[i][j] = m[i][j - 1] * tMD + y[i][j - 1] * tGap;
            }
        }
        double sum = 0.0;
        for (int j = 1; j < cols; j++) sum += m[rows - 1][j] + x[rows - 1][j];
        return Math.log10(sum) - INITIAL_CONDITION_LOG10;
    }
}
