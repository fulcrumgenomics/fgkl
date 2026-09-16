package com.fulcrumgenomics.fgkl.pdhmm;

import static com.fulcrumgenomics.fgkl.PairHmmModel.INITIAL_CONDITION;
import static com.fulcrumgenomics.fgkl.PairHmmModel.INITIAL_CONDITION_LOG10;
import static com.fulcrumgenomics.fgkl.PairHmmModel.TRISTATE_CORRECTION;
import static com.fulcrumgenomics.fgkl.PairHmmModel.errorProb;
import static com.fulcrumgenomics.fgkl.PairHmmModel.matchToMatch;

/**
 * Scalar double-precision partially determined PairHMM ported from GATK's {@code LoglessPDPairHMM}
 * (with {@code PairHMMModel} and {@code QualityUtils}); the oracle the native binding is tested
 * against. The deletion state is carried from the end of one row into the next exactly as GATK
 * does.
 */
final class ReferencePdHmm {
    static final byte SNP = 1;
    static final byte DEL_START = 2;
    static final byte DEL_END = 4;
    static final byte ALT_A = 8;
    static final byte ALT_C = 16;
    static final byte ALT_G = 32;
    static final byte ALT_T = 64;

    private enum State { NORMAL, INSIDE_DEL, AFTER_DEL }

    private ReferencePdHmm() {}

    static boolean baseMatchesPd(byte readBase, byte flag) {
        if ((flag & SNP) == 0) return false;
        switch (readBase) {
            case 'A': case 'a': return (flag & ALT_A) != 0;
            case 'C': case 'c': return (flag & ALT_C) != 0;
            case 'G': case 'g': return (flag & ALT_G) != 0;
            case 'T': case 't': return (flag & ALT_T) != 0;
            default: return false;
        }
    }

    /** The log10 likelihood of the read given the partially determined haplotype; both non-empty. */
    static double log10Likelihood(byte[] hap, byte[] flags, byte[] read, byte[] quals, byte[] insGop, byte[] delGop, byte[] gcp) {
        int rows = read.length + 1;
        int cols = hap.length + 1;
        double[][] m = new double[rows][cols];
        double[][] x = new double[rows][cols];
        double[][] y = new double[rows][cols];
        double[][] bm = new double[rows][cols];
        double[][] bx = new double[rows][cols];
        double[][] by = new double[rows][cols];
        double initial = INITIAL_CONDITION / hap.length;
        for (int j = 0; j < cols; j++) y[0][j] = initial;
        State state = State.NORMAL;
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
                byte flag = flags[j - 1];
                double prior = (rb == hb || rb == 'N' || hb == 'N' || baseMatchesPd(rb, flag)) ? pMatch : pMismatch;
                boolean delEnd = (flag & DEL_END) != 0;
                switch (state) {
                    case NORMAL:
                        bm[i][j] = m[i][j - 1];
                        by[i][j] = y[i][j - 1];
                        bx[i][j] = x[i][j - 1];
                        m[i][j] = prior * (m[i - 1][j - 1] * tMM + x[i - 1][j - 1] * tIM + y[i - 1][j - 1] * tIM);
                        y[i][j] = m[i][j - 1] * tMD + y[i][j - 1] * tGap;
                        break;
                    case INSIDE_DEL:
                        bm[i][j] = bm[i][j - 1];
                        by[i][j] = by[i][j - 1];
                        bx[i][j] = bx[i][j - 1];
                        m[i][j] = prior * (m[i - 1][j - 1] * tMM + x[i - 1][j - 1] * tIM + y[i - 1][j - 1] * tIM);
                        y[i][j] = m[i][j - 1] * tMD + y[i][j - 1] * tGap;
                        break;
                    default:
                        bm[i][j] = Math.max(bm[i][j - 1], m[i][j - 1]);
                        by[i][j] = Math.max(by[i][j - 1], y[i][j - 1]);
                        bx[i][j] = Math.max(bx[i][j - 1], x[i][j - 1]);
                        m[i][j] = prior * (Math.max(bm[i - 1][j - 1], m[i - 1][j - 1]) * tMM
                                + Math.max(bx[i - 1][j - 1], x[i - 1][j - 1]) * tIM
                                + Math.max(by[i - 1][j - 1], y[i - 1][j - 1]) * tIM);
                        y[i][j] = Math.max(bm[i][j - 1], m[i][j - 1]) * tMD + Math.max(by[i][j - 1], y[i][j - 1]) * tGap;
                        state = State.NORMAL;
                        break;
                }
                if (delEnd) {
                    x[i][j] = Math.max(bm[i - 1][j], m[i - 1][j]) * tMI + Math.max(bx[i - 1][j], x[i - 1][j]) * tGap;
                } else {
                    x[i][j] = m[i - 1][j] * tMI + x[i - 1][j] * tGap;
                }
                if ((flag & DEL_START) != 0) state = State.INSIDE_DEL;
                if (delEnd) state = State.AFTER_DEL;
            }
        }
        double sum = 0.0;
        for (int j = 1; j < cols; j++) sum += m[rows - 1][j] + x[rows - 1][j];
        return Math.log10(sum) - INITIAL_CONDITION_LOG10;
    }
}
