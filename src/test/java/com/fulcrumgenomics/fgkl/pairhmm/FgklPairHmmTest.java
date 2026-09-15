package com.fulcrumgenomics.fgkl.pairhmm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.broadinstitute.gatk.nativebindings.pairhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pairhmm.PairHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pairhmm.ReadDataHolder;
import org.junit.jupiter.api.Test;

class FgklPairHmmTest {
    private static final byte[] BASES = {'A', 'C', 'G', 'T'};

    /** Reads and haplotypes shaped like an assembly region: haplotypes are edits of one reference. */
    private static final class Region {
        final ReadDataHolder[] reads;
        final HaplotypeDataHolder[] haps;

        Region(long seed, int numReads, int readLen, int numHaps, int hapLen) {
            Random rng = new Random(seed);
            byte[] reference = new byte[hapLen];
            for (int i = 0; i < hapLen; i++) reference[i] = BASES[rng.nextInt(4)];
            List<byte[]> hapList = new ArrayList<>();
            hapList.add(reference);
            while (hapList.size() < numHaps) {
                byte[] hap = reference.clone();
                for (int e = 0, n = 1 + rng.nextInt(3); e < n; e++) {
                    int pos = rng.nextInt(hap.length);
                    switch (rng.nextInt(3)) {
                        case 0 -> hap[pos] = BASES[(Arrays.binarySearch(BASES, hap[pos]) + 1 + rng.nextInt(3)) % 4];
                        case 1 -> {
                            byte[] longer = new byte[hap.length + 1];
                            System.arraycopy(hap, 0, longer, 0, pos);
                            longer[pos] = BASES[rng.nextInt(4)];
                            System.arraycopy(hap, pos, longer, pos + 1, hap.length - pos);
                            hap = longer;
                        }
                        default -> {
                            if (hap.length > 1) {
                                byte[] shorter = new byte[hap.length - 1];
                                System.arraycopy(hap, 0, shorter, 0, pos);
                                System.arraycopy(hap, pos + 1, shorter, pos, hap.length - pos - 1);
                                hap = shorter;
                            }
                        }
                    }
                }
                hapList.add(hap);
            }
            haps = new HaplotypeDataHolder[numHaps];
            for (int h = 0; h < numHaps; h++) {
                haps[h] = new HaplotypeDataHolder();
                haps[h].haplotypeBases = hapList.get(h);
            }
            reads = new ReadDataHolder[numReads];
            for (int r = 0; r < numReads; r++) {
                byte[] hap = hapList.get(rng.nextInt(numHaps));
                int len = Math.min(rng.nextDouble() < 0.3 ? 1 + rng.nextInt(readLen) : readLen, hap.length);
                int start = rng.nextInt(hap.length - len + 1);
                ReadDataHolder read = new ReadDataHolder();
                read.readBases = Arrays.copyOfRange(hap, start, start + len);
                read.readQuals = new byte[len];
                read.insertionGOP = new byte[len];
                read.deletionGOP = new byte[len];
                read.overallGCP = new byte[len];
                for (int i = 0; i < len; i++) {
                    if (rng.nextDouble() < 0.002) read.readBases[i] = 'N';
                    else if (rng.nextDouble() < 0.01) read.readBases[i] = BASES[rng.nextInt(4)];
                    read.readQuals[i] = (byte) (rng.nextDouble() < 0.05 ? 2 + rng.nextInt(15) : 20 + rng.nextInt(21));
                    read.insertionGOP[i] = (byte) (rng.nextDouble() < 0.1 ? 20 + rng.nextInt(30) : 45);
                    read.deletionGOP[i] = (byte) (rng.nextDouble() < 0.1 ? 20 + rng.nextInt(30) : 45);
                    read.overallGCP[i] = 10;
                }
                reads[r] = read;
            }
        }

        double[] reference() {
            double[] out = new double[reads.length * haps.length];
            for (int r = 0; r < reads.length; r++) {
                ReadDataHolder read = reads[r];
                for (int h = 0; h < haps.length; h++) {
                    out[r * haps.length + h] = ReferencePairHmm.log10Likelihood(haps[h].haplotypeBases, read.readBases, read.readQuals, read.insertionGOP, read.deletionGOP, read.overallGCP);
                }
            }
            return out;
        }

        double[] compute(FgklPairHmm hmm) {
            double[] out = new double[reads.length * haps.length];
            hmm.computeLikelihoods(reads, haps, out);
            return out;
        }
    }

    private static FgklPairHmm hmm(boolean doublePrecision) {
        FgklPairHmm hmm = new FgklPairHmm();
        assertTrue(hmm.load(null), "native library should load on " + com.fulcrumgenomics.fgkl.NativeLoader.detectPlatform());
        PairHMMNativeArguments args = new PairHMMNativeArguments();
        args.useDoublePrecision = doublePrecision;
        args.maxNumberOfThreads = 4;
        hmm.initialize(args);
        return hmm;
    }

    private static void assertClose(double[] actual, double[] expected, double tolerance) {
        assertEquals(expected.length, actual.length);
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance, "pair " + i);
        }
    }

    @Test
    void backendIsReported() {
        assertNotNull(hmm(false).backend());
    }

    @Test
    void doublePrecisionMatchesReference() {
        Region region = new Region(1, 60, 120, 20, 300);
        assertClose(region.compute(hmm(true)), region.reference(), 1e-9);
    }

    @Test
    void singlePrecisionMatchesReferenceWithinFloatTolerance() {
        Region region = new Region(2, 60, 120, 20, 300);
        assertClose(region.compute(hmm(false)), region.reference(), 1e-4);
    }

    @Test
    void underflowingPairsAreRecomputedInDouble() {
        ReadDataHolder read = new ReadDataHolder();
        read.readBases = new byte[80];
        Arrays.fill(read.readBases, (byte) 'A');
        read.readQuals = new byte[80];
        Arrays.fill(read.readQuals, (byte) 40);
        read.insertionGOP = new byte[80];
        Arrays.fill(read.insertionGOP, (byte) 45);
        read.deletionGOP = read.insertionGOP.clone();
        read.overallGCP = new byte[80];
        Arrays.fill(read.overallGCP, (byte) 10);
        HaplotypeDataHolder hap = new HaplotypeDataHolder();
        hap.haplotypeBases = new byte[80];
        Arrays.fill(hap.haplotypeBases, (byte) 'C');
        double expected = ReferencePairHmm.log10Likelihood(hap.haplotypeBases, read.readBases, read.readQuals, read.insertionGOP, read.deletionGOP, read.overallGCP);
        assertTrue(expected < -50);
        double[] out = new double[1];
        hmm(false).computeLikelihoods(new ReadDataHolder[] {read}, new HaplotypeDataHolder[] {hap}, out);
        assertEquals(expected, out[0], 1e-9);
    }

    @Test
    void emptyInputsProduceNoOutput() {
        Region region = new Region(3, 5, 50, 4, 80);
        hmm(false).computeLikelihoods(new ReadDataHolder[0], region.haps, new double[0]);
        hmm(false).computeLikelihoods(region.reads, new HaplotypeDataHolder[0], new double[0]);
    }

    @Test
    void nullInputsThrow() {
        Region region = new Region(4, 5, 50, 4, 80);
        FgklPairHmm hmm = hmm(false);
        assertThrows(NullPointerException.class, () -> hmm.computeLikelihoods(null, region.haps, new double[20]));
        assertThrows(NullPointerException.class, () -> hmm.computeLikelihoods(region.reads, null, new double[20]));
        assertThrows(NullPointerException.class, () -> hmm.computeLikelihoods(region.reads, region.haps, null));
    }

    @Test
    void malformedInputsThrowIllegalArgument() {
        Region region = new Region(5, 5, 50, 4, 80);
        FgklPairHmm hmm = hmm(false);
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(region.reads, region.haps, new double[19]));
        Region broken = new Region(6, 5, 50, 4, 80);
        broken.reads[2].readQuals = new byte[3];
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(broken.reads, broken.haps, new double[20]));
        Region emptyHap = new Region(7, 5, 50, 4, 80);
        emptyHap.haps[1].haplotypeBases = new byte[0];
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(emptyHap.reads, emptyHap.haps, new double[20]));
    }

    @Test
    void concurrentCallsGiveIdenticalResults() throws Exception {
        Region region = new Region(8, 200, 150, 30, 400);
        FgklPairHmm hmm = hmm(false);
        double[] expected = region.compute(hmm);
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try {
            List<Future<double[]>> futures = new ArrayList<>();
            for (int i = 0; i < 8; i++) {
                Callable<double[]> task = () -> region.compute(hmm);
                futures.add(pool.submit(task));
            }
            for (Future<double[]> f : futures) {
                assertArrayEquals(expected, f.get());
            }
        } finally {
            pool.shutdownNow();
        }
    }
}
