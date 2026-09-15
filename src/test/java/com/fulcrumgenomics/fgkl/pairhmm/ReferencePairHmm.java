package com.fulcrumgenomics.fgkl.pairhmm;

/**
 * Scalar double-precision PairHMM ported from GATK's {@code LoglessPairHMM}, {@code PairHMMModel}
 * and {@code QualityUtils}; the oracle the native binding is tested against.
 */
final class ReferencePairHmm {
    private static final int MAX_QUAL = 254;
    private static final double INITIAL_CONDITION = Math.pow(2, 1020);
    private static final double INITIAL_CONDITION_LOG10 = Math.log10(INITIAL_CONDITION);
    private static final double TRISTATE_CORRECTION = 3.0;
    private static final double INV_LN10 = 1.0 / Math.log(10);
    private static final double JACOBIAN_MAX_TOLERANCE = 8.0;
    private static final double JACOBIAN_TABLE_STEP = 0.0001;
    private static final double[] ERROR_PROB = new double[MAX_QUAL + 1];
    private static final double[] MATCH_TO_MATCH = new double[((MAX_QUAL + 1) * (MAX_QUAL + 2)) >> 1];

    static {
        for (int q = 0; q <= MAX_QUAL; q++) {
            ERROR_PROB[q] = Math.pow(10.0, q / -10.0);
        }
        for (int i = 0, offset = 0; i <= MAX_QUAL; offset += ++i) {
            for (int j = 0; j <= i; j++) {
                double log10Sum = approximateLog10SumLog10(-0.1 * i, -0.1 * j);
                double m2mLog10 = Math.log1p(-Math.min(1, Math.pow(10, log10Sum))) * INV_LN10;
                MATCH_TO_MATCH[offset + j] = Math.pow(10, m2mLog10);
            }
        }
    }

    private ReferencePairHmm() {}

    static double errorProb(byte qual) {
        return ERROR_PROB[Math.min(qual & 0xff, MAX_QUAL)];
    }

    static double matchToMatch(byte insQual, byte delQual) {
        int a = Math.min(insQual & 0xff, MAX_QUAL);
        int b = Math.min(delQual & 0xff, MAX_QUAL);
        int lo = Math.min(a, b);
        int hi = Math.max(a, b);
        return MATCH_TO_MATCH[((hi * (hi + 1)) >> 1) + lo];
    }

    private static double approximateLog10SumLog10(double a, double b) {
        if (a > b) return approximateLog10SumLog10(b, a);
        if (a == Double.NEGATIVE_INFINITY) return b;
        double diff = b - a;
        if (diff >= JACOBIAN_MAX_TOLERANCE) return b;
        int k = diff * (1.0 / JACOBIAN_TABLE_STEP) > 0.0 ? (int) (diff * (1.0 / JACOBIAN_TABLE_STEP) + 0.5) : (int) (diff * (1.0 / JACOBIAN_TABLE_STEP) - 0.5);
        return b + Math.log10(1.0 + Math.pow(10.0, -k * JACOBIAN_TABLE_STEP));
    }

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
