package com.fulcrumgenomics.fgkl.pdhmm;

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
import org.broadinstitute.gatk.nativebindings.pdhmm.HaplotypeDataHolder;
import org.broadinstitute.gatk.nativebindings.pdhmm.PDHMMNativeArguments;
import org.broadinstitute.gatk.nativebindings.pdhmm.ReadDataHolder;
import org.junit.jupiter.api.Test;

class FgklPdHmmTest {
    private static final byte[] BASES = {'A', 'C', 'G', 'T'};
    private static final byte[] ALT_FLAGS = {ReferencePdHmm.ALT_A, ReferencePdHmm.ALT_C, ReferencePdHmm.ALT_G, ReferencePdHmm.ALT_T};

    /**
     * Reads and partially determined haplotypes shaped like a DRAGEN-mode assembly region: every
     * haplotype is the reference with a few determined SNPs plus flagged undetermined SNPs and
     * deletions, and reads come from realized alleles that take those events or from the bases.
     */
    private static final class Region {
        final ReadDataHolder[] reads;
        final HaplotypeDataHolder[] haps;

        Region(long seed, int numReads, int readLen, int numHaps, int hapLen) {
            Random rng = new Random(seed);
            byte[] reference = new byte[hapLen];
            for (int i = 0; i < hapLen; i++) reference[i] = BASES[rng.nextInt(4)];
            haps = new HaplotypeDataHolder[numHaps];
            List<byte[]> realized = new ArrayList<>();
            for (int h = 0; h < numHaps; h++) {
                byte[] bases = reference.clone();
                for (int e = 0, n = h == 0 ? 0 : rng.nextInt(3); e < n; e++) {
                    bases[rng.nextInt(hapLen)] = BASES[rng.nextInt(4)];
                }
                byte[] flags = new byte[hapLen];
                byte[] alt = bases.clone();
                boolean[] deleted = new boolean[hapLen];
                for (int e = 0, n = 1 + rng.nextInt(4); e < n; e++) {
                    int pos = rng.nextInt(hapLen);
                    if (flags[pos] != 0) continue;
                    if (rng.nextBoolean()) {
                        int mask = 1 + rng.nextInt(15);
                        byte flag = ReferencePdHmm.SNP;
                        List<Byte> choices = new ArrayList<>();
                        for (int bit = 0; bit < 4; bit++) {
                            if ((mask & (1 << bit)) != 0) {
                                flag |= ALT_FLAGS[bit];
                                choices.add(BASES[bit]);
                            }
                        }
                        flags[pos] = flag;
                        alt[pos] = choices.get(rng.nextInt(choices.size()));
                    } else {
                        int len = Math.min(1 + rng.nextInt(8), hapLen - pos);
                        boolean clear = true;
                        for (int k = pos; k < pos + len; k++) clear &= flags[k] == 0;
                        if (!clear) continue;
                        flags[pos] |= ReferencePdHmm.DEL_START;
                        flags[pos + len - 1] |= ReferencePdHmm.DEL_END;
                        for (int k = pos; k < pos + len; k++) deleted[k] = true;
                    }
                }
                int kept = 0;
                for (boolean d : deleted) if (!d) kept++;
                byte[] allele = new byte[kept];
                for (int k = 0, out = 0; k < hapLen; k++) if (!deleted[k]) allele[out++] = alt[k];
                realized.add(allele.length == 0 ? bases : allele);
                haps[h] = new HaplotypeDataHolder();
                haps[h].haplotypeBases = bases;
                haps[h].haplotypePDBases = flags;
            }
            reads = new ReadDataHolder[numReads];
            for (int r = 0; r < numReads; r++) {
                int h = rng.nextInt(numHaps);
                byte[] source = rng.nextBoolean() ? realized.get(h) : haps[h].haplotypeBases;
                int len = Math.min(rng.nextDouble() < 0.3 ? 1 + rng.nextInt(readLen) : readLen, source.length);
                int start = rng.nextInt(source.length - len + 1);
                ReadDataHolder read = new ReadDataHolder();
                read.readBases = Arrays.copyOfRange(source, start, start + len);
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
                    out[r * haps.length + h] = ReferencePdHmm.log10Likelihood(haps[h].haplotypeBases, haps[h].haplotypePDBases, read.readBases, read.readQuals, read.insertionGOP, read.deletionGOP, read.overallGCP);
                }
            }
            return out;
        }

        double[] compute(FgklPdHmm hmm) {
            double[] out = new double[reads.length * haps.length];
            hmm.computeLikelihoods(reads, haps, out);
            return out;
        }
    }

    private static FgklPdHmm hmm(boolean doublePrecision) {
        FgklPdHmm hmm = new FgklPdHmm();
        assertTrue(hmm.load(null), "native library should load on " + com.fulcrumgenomics.fgkl.NativeLoader.detectPlatform());
        PDHMMNativeArguments args = new PDHMMNativeArguments();
        args.maxNumberOfThreads = 4;
        args.setMaxMemoryInMB(512);
        hmm.initialize(args);
        hmm.setDoublePrecision(doublePrecision);
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
    void flagsChangeTheLikelihoods() {
        Region region = new Region(9, 60, 120, 20, 300);
        double[] flagged = region.compute(hmm(true));
        HaplotypeDataHolder[] plain = new HaplotypeDataHolder[region.haps.length];
        for (int h = 0; h < plain.length; h++) {
            plain[h] = new HaplotypeDataHolder();
            plain[h].haplotypeBases = region.haps[h].haplotypeBases;
            plain[h].haplotypePDBases = new byte[plain[h].haplotypeBases.length];
        }
        double[] unflagged = new double[flagged.length];
        hmm(true).computeLikelihoods(region.reads, plain, unflagged);
        int changed = 0;
        for (int i = 0; i < flagged.length; i++) {
            assertTrue(flagged[i] >= unflagged[i], "flags only add paths, pair " + i);
            if (flagged[i] - unflagged[i] > 1.0) changed++;
        }
        assertTrue(changed > 10, "only " + changed + " pairs changed");
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
        hap.haplotypePDBases = new byte[80];
        hap.haplotypePDBases[10] = ReferencePdHmm.DEL_START;
        hap.haplotypePDBases[20] = ReferencePdHmm.DEL_END;
        double expected = ReferencePdHmm.log10Likelihood(hap.haplotypeBases, hap.haplotypePDBases, read.readBases, read.readQuals, read.insertionGOP, read.deletionGOP, read.overallGCP);
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
        FgklPdHmm hmm = hmm(false);
        assertThrows(NullPointerException.class, () -> hmm.computeLikelihoods(null, region.haps, new double[20]));
        assertThrows(NullPointerException.class, () -> hmm.computeLikelihoods(region.reads, null, new double[20]));
        assertThrows(NullPointerException.class, () -> hmm.computeLikelihoods(region.reads, region.haps, null));
    }

    @Test
    void malformedInputsThrowIllegalArgument() {
        Region region = new Region(5, 5, 50, 4, 80);
        FgklPdHmm hmm = hmm(false);
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(region.reads, region.haps, new double[19]));
        Region broken = new Region(6, 5, 50, 4, 80);
        broken.reads[2].readQuals = new byte[3];
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(broken.reads, broken.haps, new double[20]));
        Region shortFlags = new Region(7, 5, 50, 4, 80);
        shortFlags.haps[1].haplotypePDBases = new byte[3];
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(shortFlags.reads, shortFlags.haps, new double[20]));
        Region emptyHap = new Region(8, 5, 50, 4, 80);
        emptyHap.haps[1].haplotypeBases = new byte[0];
        emptyHap.haps[1].haplotypePDBases = new byte[0];
        assertThrows(IllegalArgumentException.class, () -> hmm.computeLikelihoods(emptyHap.reads, emptyHap.haps, new double[20]));
    }

    @Test
    void concurrentCallsGiveIdenticalResults() throws Exception {
        Region region = new Region(10, 200, 150, 30, 400);
        FgklPdHmm hmm = hmm(false);
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
