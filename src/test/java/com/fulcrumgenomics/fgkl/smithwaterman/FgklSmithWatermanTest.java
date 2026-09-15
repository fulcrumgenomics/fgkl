package com.fulcrumgenomics.fgkl.smithwaterman;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Random;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWNativeAlignerResult;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWOverhangStrategy;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWParameters;
import org.junit.jupiter.api.Test;

class FgklSmithWatermanTest {
    private static final byte[] BASES = {'A', 'C', 'G', 'T'};
    private static final SWParameters HAP_TO_REF = new SWParameters(200, -150, -260, -11);
    private static final SWParameters READ_TO_HAP = new SWParameters(10, -15, -30, -5);

    private static FgklSmithWaterman aligner() {
        FgklSmithWaterman sw = new FgklSmithWaterman();
        assertTrue(sw.load(null));
        return sw;
    }

    /** A random reference and an edited window of it, optionally with unrelated overhangs. */
    private static byte[][] relatedPair(Random rng, int refLen, int altLen) {
        byte[] ref = new byte[refLen];
        for (int i = 0; i < refLen; i++) ref[i] = BASES[rng.nextInt(4)];
        int start = rng.nextInt(Math.max(1, refLen - altLen + 1));
        byte[] alt = Arrays.copyOfRange(ref, start, Math.min(refLen, start + altLen));
        for (int e = 0, n = rng.nextInt(4); e < n && alt.length > 0; e++) {
            int pos = rng.nextInt(alt.length);
            switch (rng.nextInt(3)) {
                case 0 -> alt[pos] = BASES[rng.nextInt(4)];
                case 1 -> {
                    byte[] longer = new byte[alt.length + 1 + rng.nextInt(5)];
                    System.arraycopy(alt, 0, longer, 0, pos);
                    for (int k = pos; k < pos + longer.length - alt.length; k++) longer[k] = BASES[rng.nextInt(4)];
                    System.arraycopy(alt, pos, longer, pos + longer.length - alt.length, alt.length - pos);
                    alt = longer;
                }
                default -> {
                    int end = Math.min(alt.length, pos + 1 + rng.nextInt(5));
                    if (alt.length > end - pos) {
                        byte[] shorter = new byte[alt.length - (end - pos)];
                        System.arraycopy(alt, 0, shorter, 0, pos);
                        System.arraycopy(alt, end, shorter, pos, alt.length - end);
                        alt = shorter;
                    }
                }
            }
        }
        if (rng.nextDouble() < 0.3) {
            byte[] extra = new byte[1 + rng.nextInt(20)];
            for (int i = 0; i < extra.length; i++) extra[i] = BASES[rng.nextInt(4)];
            byte[] joined = new byte[alt.length + extra.length];
            if (rng.nextBoolean()) {
                System.arraycopy(extra, 0, joined, 0, extra.length);
                System.arraycopy(alt, 0, joined, extra.length, alt.length);
            } else {
                System.arraycopy(alt, 0, joined, 0, alt.length);
                System.arraycopy(extra, 0, joined, alt.length, extra.length);
            }
            alt = joined;
        }
        if (alt.length == 0) alt = new byte[] {'A'};
        return new byte[][] {ref, alt};
    }

    private static void assertSame(FgklSmithWaterman sw, byte[] ref, byte[] alt, SWParameters params, SWOverhangStrategy strategy) {
        ReferenceSmithWaterman.Result expected = ReferenceSmithWaterman.align(ref, alt, params, strategy);
        SWNativeAlignerResult actual = sw.align(ref, alt, params, strategy);
        String what = strategy + " ref=" + new String(ref) + " alt=" + new String(alt);
        assertEquals(expected.cigar, actual.cigar, what);
        assertEquals(expected.offset, actual.alignment_offset, what);
    }

    @Test
    void matchesReferenceOnRandomPairs() {
        FgklSmithWaterman sw = aligner();
        Random rng = new Random(17);
        for (int i = 0; i < 300; i++) {
            byte[][] pair = i % 2 == 0 ? relatedPair(rng, 40 + rng.nextInt(300), 20 + rng.nextInt(200)) : relatedPair(rng, 5 + rng.nextInt(30), 3 + rng.nextInt(30));
            for (SWParameters params : new SWParameters[] {HAP_TO_REF, READ_TO_HAP}) {
                for (SWOverhangStrategy strategy : SWOverhangStrategy.values()) {
                    assertSame(sw, pair[0], pair[1], params, strategy);
                }
            }
        }
    }

    @Test
    void exactSubstringIsAllMatch() {
        SWNativeAlignerResult r = aligner().align("AAACGTACGTAAA".getBytes(), "ACGTACGT".getBytes(), HAP_TO_REF, SWOverhangStrategy.SOFTCLIP);
        assertEquals("8M", r.cigar);
        assertEquals(2, r.alignment_offset);
    }

    @Test
    void readOverhangsAreSoftClipped() {
        SWNativeAlignerResult r = aligner().align("ACGTACGTAGGCCTTAGCA".getBytes(), "GGGGACGTACGTAGGCCTTAGCAGGGG".getBytes(), READ_TO_HAP, SWOverhangStrategy.SOFTCLIP);
        assertEquals("4S19M4S", r.cigar);
        assertEquals(0, r.alignment_offset);
    }

    @Test
    void invalidInputsThrow() {
        FgklSmithWaterman sw = aligner();
        byte[] seq = "ACGT".getBytes();
        assertThrows(NullPointerException.class, () -> sw.align(null, seq, HAP_TO_REF, SWOverhangStrategy.INDEL));
        assertThrows(NullPointerException.class, () -> sw.align(seq, null, HAP_TO_REF, SWOverhangStrategy.INDEL));
        assertThrows(NullPointerException.class, () -> sw.align(seq, seq, null, SWOverhangStrategy.INDEL));
        assertThrows(NullPointerException.class, () -> sw.align(seq, seq, HAP_TO_REF, null));
        assertThrows(IllegalArgumentException.class, () -> sw.align(new byte[0], seq, HAP_TO_REF, SWOverhangStrategy.INDEL));
        assertThrows(IllegalArgumentException.class, () -> sw.align(seq, new byte[0], HAP_TO_REF, SWOverhangStrategy.INDEL));
    }
}
