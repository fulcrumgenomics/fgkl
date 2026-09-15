package com.fulcrumgenomics.fgkl.pairhmm;

import com.fulcrumgenomics.fgkl.NativeLoader;
import java.io.File;
import org.broadinstitute.gatk.nativebindings.pairhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pairhmm.ReadDataHolder;

/**
 * GATK's {@link PairHMMNativeBinding} backed by the fgkl vectorized PairHMM.
 *
 * <p>Results match GATK's Java {@code LoglessPairHMM} and Intel GKL: single precision by default,
 * with any read-haplotype pair whose probability underflows recomputed in double precision, or
 * double precision throughout when {@link PairHMMNativeArguments#useDoublePrecision} is set.
 *
 * <p>Every call to {@link #computeLikelihoods} runs entirely on the calling thread;
 * {@link PairHMMNativeArguments#maxNumberOfThreads} is accepted and ignored. Callers that want
 * parallelism run independent calls concurrently, which is safe.
 */
public final class FgklPairHmm implements PairHMMNativeBinding {
    private volatile boolean doublePrecision = false;

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
    public void initialize(PairHMMNativeArguments args) {
        doublePrecision = args != null && args.useDoublePrecision;
    }

    /** Name of the vector instruction set the native kernel will use on this CPU. */
    public String backend() {
        return backendNative();
    }

    @Override
    public void computeLikelihoods(ReadDataHolder[] readDataArray, HaplotypeDataHolder[] haplotypeDataArray, double[] likelihoodArray) {
        if (readDataArray == null || haplotypeDataArray == null || likelihoodArray == null) {
            throw new NullPointerException("Input is null");
        }
        if (likelihoodArray.length != readDataArray.length * haplotypeDataArray.length) {
            throw new IllegalArgumentException("likelihoodArray length must equal reads x haplotypes");
        }
        if (readDataArray.length == 0 || haplotypeDataArray.length == 0) {
            return;
        }

        int[] readOffsets = new int[readDataArray.length + 1];
        for (int i = 0; i < readDataArray.length; i++) {
            ReadDataHolder read = readDataArray[i];
            int len = read.readBases.length;
            if (read.readQuals.length != len || read.insertionGOP.length != len || read.deletionGOP.length != len || read.overallGCP.length != len) {
                throw new IllegalArgumentException("Read " + i + ": bases, qualities and penalties differ in length");
            }
            readOffsets[i + 1] = readOffsets[i] + len;
        }
        int totalReadBases = readOffsets[readDataArray.length];
        byte[] bases = new byte[totalReadBases];
        byte[] quals = new byte[totalReadBases];
        byte[] insGop = new byte[totalReadBases];
        byte[] delGop = new byte[totalReadBases];
        byte[] gcp = new byte[totalReadBases];
        for (int i = 0; i < readDataArray.length; i++) {
            ReadDataHolder read = readDataArray[i];
            int off = readOffsets[i];
            System.arraycopy(read.readBases, 0, bases, off, read.readBases.length);
            System.arraycopy(read.readQuals, 0, quals, off, read.readQuals.length);
            System.arraycopy(read.insertionGOP, 0, insGop, off, read.insertionGOP.length);
            System.arraycopy(read.deletionGOP, 0, delGop, off, read.deletionGOP.length);
            System.arraycopy(read.overallGCP, 0, gcp, off, read.overallGCP.length);
        }

        int[] hapOffsets = new int[haplotypeDataArray.length + 1];
        for (int i = 0; i < haplotypeDataArray.length; i++) {
            hapOffsets[i + 1] = hapOffsets[i] + haplotypeDataArray[i].haplotypeBases.length;
        }
        byte[] hapBases = new byte[hapOffsets[haplotypeDataArray.length]];
        for (int i = 0; i < haplotypeDataArray.length; i++) {
            byte[] h = haplotypeDataArray[i].haplotypeBases;
            System.arraycopy(h, 0, hapBases, hapOffsets[i], h.length);
        }

        computeNative(bases, quals, insGop, delGop, gcp, readOffsets, hapBases, hapOffsets, doublePrecision, likelihoodArray);
    }

    @Override
    public void done() {}

    private static native String backendNative();

    /**
     * Reads and haplotypes are passed concatenated, with {@code readOffsets[i]..readOffsets[i+1]}
     * delimiting read {@code i} in each of the five read arrays and likewise for haplotypes. The
     * result for read {@code r} and haplotype {@code h} is written to
     * {@code likelihoods[r * numHaplotypes + h]}.
     */
    private static native void computeNative(byte[] bases, byte[] quals, byte[] insGop, byte[] delGop, byte[] gcp, int[] readOffsets, byte[] hapBases, int[] hapOffsets, boolean doublePrecision, double[] likelihoods);
}
