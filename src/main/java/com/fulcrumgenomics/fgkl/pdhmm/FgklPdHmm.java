package com.fulcrumgenomics.fgkl.pdhmm;

import com.fulcrumgenomics.fgkl.NativeLoader;
import java.io.File;
import org.broadinstitute.gatk.nativebindings.pdhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeBinding;
import org.broadinstitute.gatk.nativebindings.pdhmm.ReadDataHolder;

/**
 * The fgkl partially determined PairHMM behind GATK's {@link PDHMMNativeBinding} contract, a
 * drop-in replacement for GKL's {@code IntelPDHMM} in DRAGEN 3.7.8 concordance mode.
 *
 * <p>Likelihoods agree with GATK's Java {@code LoglessPDPairHMM} to within 1e-9 in double
 * precision and 1e-4 in the default single precision, which recomputes in double any pair whose
 * probability underflows. The DRAGEN-mode GATK code path always requests the full
 * {@code PDHMMNativeArguments}; only {@code maxNumberOfThreads}, {@code avxLevel},
 * {@code openMPSetting} and the memory limit are accepted and ignored, since every call to
 * {@link #computeLikelihoods} runs entirely on the calling thread and the kernel picks the widest
 * vector unit the CPU has. Callers that want parallelism run independent calls concurrently, which
 * is safe.
 */
public final class FgklPdHmm implements PDHMMNativeBinding {
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
    public void initialize(PDHMMNativeArguments args) {}

    /**
     * Selects double precision throughout instead of single precision with double recomputation
     * of underflowing pairs. GKL's PD kernel is double-only, so this is the setting that matches
     * it bit for bit in spirit; the default is faster and agrees to 1e-4.
     */
    public void setDoublePrecision(boolean doublePrecision) {
        this.doublePrecision = doublePrecision;
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
            HaplotypeDataHolder hap = haplotypeDataArray[i];
            if (hap.haplotypePDBases == null || hap.haplotypePDBases.length != hap.haplotypeBases.length) {
                throw new IllegalArgumentException("Haplotype " + i + ": bases and PD flags differ in length");
            }
            hapOffsets[i + 1] = hapOffsets[i] + hap.haplotypeBases.length;
        }
        byte[] hapBases = new byte[hapOffsets[haplotypeDataArray.length]];
        byte[] hapFlags = new byte[hapBases.length];
        for (int i = 0; i < haplotypeDataArray.length; i++) {
            HaplotypeDataHolder hap = haplotypeDataArray[i];
            System.arraycopy(hap.haplotypeBases, 0, hapBases, hapOffsets[i], hap.haplotypeBases.length);
            System.arraycopy(hap.haplotypePDBases, 0, hapFlags, hapOffsets[i], hap.haplotypePDBases.length);
        }

        computeNative(bases, quals, insGop, delGop, gcp, readOffsets, hapBases, hapFlags, hapOffsets, doublePrecision, likelihoodArray);
    }

    @Override
    public void done() {}

    private static native String backendNative();

    /**
     * Reads and haplotypes are passed concatenated, with {@code readOffsets[i]..readOffsets[i+1]}
     * delimiting read {@code i} in each of the five read arrays and likewise for the haplotype
     * bases and their flags. The result for read {@code r} and haplotype {@code h} is written to
     * {@code likelihoods[r * numHaplotypes + h]}.
     */
    private static native void computeNative(byte[] bases, byte[] quals, byte[] insGop, byte[] delGop, byte[] gcp, int[] readOffsets, byte[] hapBases, byte[] hapFlags, int[] hapOffsets, boolean doublePrecision, double[] likelihoods);
}
