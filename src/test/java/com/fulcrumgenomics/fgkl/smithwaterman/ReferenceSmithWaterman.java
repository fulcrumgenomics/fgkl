package com.fulcrumgenomics.fgkl.smithwaterman;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWOverhangStrategy;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWParameters;

/**
 * A port of GATK's {@code SmithWatermanJavaAligner} (BSD-3, Broad Institute) producing the CIGAR
 * as a string and the alignment offset; the oracle for the native binding.
 */
final class ReferenceSmithWaterman {
    private ReferenceSmithWaterman() {}

    static final class Result {
        final String cigar;
        final int offset;

        Result(String cigar, int offset) {
            this.cigar = cigar;
            this.offset = offset;
        }
    }

    private enum State { MATCH, INSERTION, DELETION, CLIP }

    static Result align(byte[] reference, byte[] alternate, SWParameters parameters, SWOverhangStrategy overhangStrategy) {
        if (reference.length == 0 || alternate.length == 0) throw new IllegalArgumentException("empty");
        int matchIndex = -1;
        if (overhangStrategy == SWOverhangStrategy.SOFTCLIP || overhangStrategy == SWOverhangStrategy.IGNORE) {
            matchIndex = lastIndexOf(reference, alternate);
        }
        if (matchIndex != -1) {
            return new Result(alternate.length + "M", matchIndex);
        }
        int n = reference.length + 1;
        int m = alternate.length + 1;
        int[][] sw = new int[n][m];
        int[][] btrack = new int[n][m];
        calculateMatrix(reference, alternate, sw, btrack, overhangStrategy, parameters);
        return calculateCigar(sw, btrack, overhangStrategy);
    }

    static int lastIndexOf(byte[] reference, byte[] query) {
        int queryLength = query.length;
        for (int r = reference.length - queryLength; r >= 0; r--) {
            int q = 0;
            while (q < queryLength && reference[r + q] == query[q]) q++;
            if (q == queryLength) return r;
        }
        return -1;
    }

    private static void calculateMatrix(byte[] reference, byte[] alternate, int[][] sw, int[][] btrack, SWOverhangStrategy overhangStrategy, SWParameters parameters) {
        int ncol = sw[0].length;
        int nrow = sw.length;
        int MATRIX_MIN_CUTOFF = (int) -1.0e8;
        int lowInitValue = Integer.MIN_VALUE / 2;
        int[] best_gap_v = new int[ncol + 1];
        Arrays.fill(best_gap_v, lowInitValue);
        int[] gap_size_v = new int[ncol + 1];
        int[] best_gap_h = new int[nrow + 1];
        Arrays.fill(best_gap_h, lowInitValue);
        int[] gap_size_h = new int[nrow + 1];
        if (overhangStrategy == SWOverhangStrategy.INDEL || overhangStrategy == SWOverhangStrategy.LEADING_INDEL) {
            int[] topRow = sw[0];
            topRow[1] = parameters.getGapOpenPenalty();
            int currentValue = parameters.getGapOpenPenalty();
            for (int i = 2; i < topRow.length; i++) {
                currentValue += parameters.getGapExtendPenalty();
                topRow[i] = currentValue;
            }
            sw[1][0] = parameters.getGapOpenPenalty();
            currentValue = parameters.getGapOpenPenalty();
            for (int i = 2; i < sw.length; i++) {
                currentValue += parameters.getGapExtendPenalty();
                sw[i][0] = currentValue;
            }
        }
        int[] curRow = sw[0];
        int w_open = parameters.getGapOpenPenalty();
        int w_extend = parameters.getGapExtendPenalty();
        int w_match = parameters.getMatchValue();
        int w_mismatch = parameters.getMismatchPenalty();
        for (int i = 1; i < sw.length; i++) {
            byte a_base = reference[i - 1];
            int[] lastRow = curRow;
            curRow = sw[i];
            int[] curBackTrackRow = btrack[i];
            for (int j = 1; j < curRow.length; j++) {
                byte b_base = alternate[j - 1];
                int step_diag = lastRow[j - 1] + (a_base == b_base ? w_match : w_mismatch);
                int prev_gap = lastRow[j] + w_open;
                best_gap_v[j] += w_extend;
                if (prev_gap > best_gap_v[j]) {
                    best_gap_v[j] = prev_gap;
                    gap_size_v[j] = 1;
                } else {
                    gap_size_v[j]++;
                }
                int step_down = best_gap_v[j];
                int kd = gap_size_v[j];
                prev_gap = curRow[j - 1] + w_open;
                best_gap_h[i] += w_extend;
                if (prev_gap > best_gap_h[i]) {
                    best_gap_h[i] = prev_gap;
                    gap_size_h[i] = 1;
                } else {
                    gap_size_h[i]++;
                }
                int step_right = best_gap_h[i];
                int ki = gap_size_h[i];
                boolean diagHighestOrEqual = (step_diag >= step_down) && (step_diag >= step_right);
                if (diagHighestOrEqual) {
                    curRow[j] = Math.max(MATRIX_MIN_CUTOFF, step_diag);
                    curBackTrackRow[j] = 0;
                } else if (step_right >= step_down) {
                    curRow[j] = Math.max(MATRIX_MIN_CUTOFF, step_right);
                    curBackTrackRow[j] = -ki;
                } else {
                    curRow[j] = Math.max(MATRIX_MIN_CUTOFF, step_down);
                    curBackTrackRow[j] = kd;
                }
            }
        }
    }

    private static Result calculateCigar(int[][] sw, int[][] btrack, SWOverhangStrategy overhangStrategy) {
        int p1 = 0, p2 = 0;
        int refLength = sw.length - 1;
        int altLength = sw[0].length - 1;
        int maxscore = Integer.MIN_VALUE;
        int segment_length = 0;
        if (overhangStrategy == SWOverhangStrategy.INDEL) {
            p1 = refLength;
            p2 = altLength;
        } else {
            p2 = altLength;
            for (int i = 1; i < sw.length; i++) {
                int curScore = sw[i][altLength];
                if (curScore >= maxscore) {
                    p1 = i;
                    maxscore = curScore;
                }
            }
            if (overhangStrategy != SWOverhangStrategy.LEADING_INDEL) {
                int[] bottomRow = sw[refLength];
                for (int j = 1; j < bottomRow.length; j++) {
                    int curScore = bottomRow[j];
                    if (curScore > maxscore || (curScore == maxscore && Math.abs(refLength - j) < Math.abs(p1 - p2))) {
                        p1 = refLength;
                        p2 = j;
                        maxscore = curScore;
                        segment_length = altLength - j;
                    }
                }
            }
        }
        List<String> lce = new ArrayList<>();
        if (segment_length > 0 && overhangStrategy == SWOverhangStrategy.SOFTCLIP) {
            lce.add(element(State.CLIP, segment_length));
            segment_length = 0;
        }
        State state = State.MATCH;
        do {
            int btr = btrack[p1][p2];
            State new_state;
            int step_length = 1;
            if (btr > 0) {
                new_state = State.DELETION;
                step_length = btr;
            } else if (btr < 0) {
                new_state = State.INSERTION;
                step_length = -btr;
            } else {
                new_state = State.MATCH;
            }
            switch (new_state) {
                case MATCH: p1--; p2--; break;
                case INSERTION: p2 -= step_length; break;
                case DELETION: p1 -= step_length; break;
                default: break;
            }
            if (new_state == state) {
                segment_length += step_length;
            } else {
                if (segment_length > 0) lce.add(element(state, segment_length));
                segment_length = step_length;
                state = new_state;
            }
        } while (p1 > 0 && p2 > 0);
        int alignment_offset;
        if (overhangStrategy == SWOverhangStrategy.SOFTCLIP) {
            lce.add(element(state, segment_length));
            if (p2 > 0) lce.add(element(State.CLIP, p2));
            alignment_offset = p1;
        } else if (overhangStrategy == SWOverhangStrategy.IGNORE) {
            lce.add(element(state, segment_length + p2));
            alignment_offset = p1 - p2;
        } else {
            lce.add(element(state, segment_length));
            if (p1 > 0) lce.add(element(State.DELETION, p1));
            else if (p2 > 0) lce.add(element(State.INSERTION, p2));
            alignment_offset = 0;
        }
        Collections.reverse(lce);
        return new Result(String.join("", lce), alignment_offset);
    }

    private static String element(State state, int length) {
        char op;
        switch (state) {
            case MATCH: op = 'M'; break;
            case INSERTION: op = 'I'; break;
            case DELETION: op = 'D'; break;
            default: op = 'S'; break;
        }
        return length + "" + op;
    }
}
