package com.fulcrumgenomics.fgkl.smithwaterman;

import com.fulcrumgenomics.fgkl.NativeLoader;
import java.io.File;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWAlignerNativeBinding;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWNativeAlignerResult;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWOverhangStrategy;
import org.broadinstitute.gatk.nativebindings.smithwaterman.SWParameters;

/**
 * GATK's {@link SWAlignerNativeBinding} backed by the fgkl Smith-Waterman aligner, which
 * reproduces {@code SmithWatermanJavaAligner} exactly: the same affine gap model, tie-breaking,
 * overhang strategies, CIGARs and alignment offsets.
 *
 * <p>Each call runs on the calling thread and reuses a per-thread native buffer set; instances
 * may be shared between threads.
 */
public final class FgklSmithWaterman implements SWAlignerNativeBinding {
    /** Loads the native library; returns false (rather than throwing) when this platform has none. */
    @Override
    public boolean load(File tempDir) {
        try {
            NativeLoader.load(tempDir);
            return true;
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    @Override
    public SWNativeAlignerResult align(byte[] reference, byte[] alternate, SWParameters parameters, SWOverhangStrategy overhangStrategy) {
        if (reference == null) throw new NullPointerException("Reference data array is null.");
        if (alternate == null) throw new NullPointerException("Alternate data array is null.");
        if (parameters == null) throw new NullPointerException("Parameter structure is null.");
        if (overhangStrategy == null) throw new NullPointerException("OverhangStrategy is null.");
        if (reference.length == 0 || alternate.length == 0) {
            throw new IllegalArgumentException("Cannot align empty sequences");
        }
        int[] offset = new int[1];
        String cigar = alignNative(reference, alternate, parameters.getMatchValue(), parameters.getMismatchPenalty(), parameters.getGapOpenPenalty(), parameters.getGapExtendPenalty(), overhangStrategy.ordinal(), offset);
        return new SWNativeAlignerResult(cigar, offset[0]);
    }

    @Override
    public void close() {}

    /**
     * Returns the CIGAR string and stores the alignment offset in {@code offset[0]}. The strategy
     * is the ordinal of {@link SWOverhangStrategy}: SOFTCLIP, INDEL, LEADING_INDEL, IGNORE.
     */
    private static native String alignNative(byte[] reference, byte[] alternate, int match, int mismatch, int gapOpen, int gapExtend, int strategy, int[] offset);
}
