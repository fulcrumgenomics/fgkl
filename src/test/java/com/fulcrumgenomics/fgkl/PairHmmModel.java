package com.fulcrumgenomics.fgkl;

/**
 * GATK's {@code PairHMMModel} and {@code QualityUtils} probability tables, ported for the Java
 * reference implementations the native bindings are tested against.
 */
public final class PairHmmModel {
    public static final int MAX_QUAL = 254;
    public static final double INITIAL_CONDITION = Math.pow(2, 1020);
    public static final double INITIAL_CONDITION_LOG10 = Math.log10(INITIAL_CONDITION);
    public static final double TRISTATE_CORRECTION = 3.0;
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

    private PairHmmModel() {}

    /** Probability that a base with the given phred quality is wrong. */
    public static double errorProb(byte qual) {
        return ERROR_PROB[Math.min(qual & 0xff, MAX_QUAL)];
    }

    /** Match-to-match transition probability for the two gap-open qualities; symmetric. */
    public static double matchToMatch(byte insQual, byte delQual) {
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
}
