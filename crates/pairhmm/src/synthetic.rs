//! Deterministic synthetic reads and haplotypes shaped like a HaplotypeCaller assembly region, for
//! tests and benchmarks.

use crate::ReadRef;

const BASES: [u8; 4] = *b"ACGT";

/// SplitMix64: tiny, seedable and reproducible across platforms.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..n`; `n` must be positive.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    pub fn chance(&mut self, p: f64) -> bool {
        ((self.next_u64() >> 11) as f64 / (1u64 << 53) as f64) < p
    }

    pub fn base(&mut self) -> u8 {
        BASES[self.below(4)]
    }
}

/// An owned read with its per-base penalties.
#[derive(Clone, Debug)]
pub struct Read {
    pub bases: Vec<u8>,
    pub quals: Vec<u8>,
    pub ins_gop: Vec<u8>,
    pub del_gop: Vec<u8>,
    pub gcp: Vec<u8>,
}

impl Read {
    pub fn as_ref(&self) -> ReadRef<'_> {
        ReadRef {
            bases: &self.bases,
            quals: &self.quals,
            ins_gop: &self.ins_gop,
            del_gop: &self.del_gop,
            gcp: &self.gcp,
        }
    }
}

/// Reads and haplotypes of one synthetic region. Haplotypes are edits of one reference sequence,
/// so they share prefixes the way real assembly haplotypes do; reads are error-bearing substrings
/// of the haplotypes with realistic quality distributions.
#[derive(Clone, Debug)]
pub struct Region {
    pub reads: Vec<Read>,
    pub haplotypes: Vec<Vec<u8>>,
}

impl Region {
    /// Generates a region with `num_reads` reads of up to `read_len` bases and `num_haplotypes`
    /// haplotypes derived from a reference of `hap_len` bases.
    pub fn generate(
        seed: u64,
        num_reads: usize,
        read_len: usize,
        num_haplotypes: usize,
        hap_len: usize,
    ) -> Region {
        assert!(read_len >= 1 && hap_len >= 1 && num_haplotypes >= 1);
        let mut rng = Rng::new(seed);
        let reference: Vec<u8> = (0..hap_len).map(|_| rng.base()).collect();
        let mut haplotypes = vec![reference.clone()];
        while haplotypes.len() < num_haplotypes {
            if haplotypes.len() > 1 && rng.chance(0.1) {
                let dup = haplotypes[rng.below(haplotypes.len())].clone();
                haplotypes.push(dup);
                continue;
            }
            let mut hap = reference.clone();
            for _ in 0..(1 + rng.below(3)) {
                let pos = rng.below(hap.len());
                match rng.below(3) {
                    0 => {
                        let old = hap[pos];
                        let mut new = rng.base();
                        while new == old {
                            new = rng.base();
                        }
                        hap[pos] = new;
                    }
                    1 => {
                        let ins: Vec<u8> = (0..(1 + rng.below(5))).map(|_| rng.base()).collect();
                        hap.splice(pos..pos, ins);
                    }
                    _ => {
                        let end = (pos + 1 + rng.below(5)).min(hap.len());
                        if hap.len() - (end - pos) >= 1 {
                            hap.drain(pos..end);
                        }
                    }
                }
            }
            haplotypes.push(hap);
        }

        let mut reads = Vec::with_capacity(num_reads);
        for _ in 0..num_reads {
            let hap = &haplotypes[rng.below(haplotypes.len())];
            let mut len = read_len;
            if rng.chance(0.3) {
                len = 1 + rng.below(read_len);
            }
            let len = len.min(hap.len());
            let start = rng.below(hap.len() - len + 1);
            let mut bases = hap[start..start + len].to_vec();
            let mut quals = Vec::with_capacity(len);
            let mut ins_gop = Vec::with_capacity(len);
            let mut del_gop = Vec::with_capacity(len);
            for b in bases.iter_mut() {
                if rng.chance(0.002) {
                    *b = b'N';
                } else if rng.chance(0.01) {
                    *b = rng.base();
                }
                quals.push(
                    if rng.chance(0.05) { 2 + rng.below(15) } else { 20 + rng.below(21) } as u8
                );
                let gop = if rng.chance(0.1) { 20 + rng.below(30) } else { 45 } as u8;
                ins_gop.push(gop);
                del_gop.push(if rng.chance(0.1) { 20 + rng.below(30) } else { 45 } as u8);
            }
            let gcp = vec![10u8; len];
            reads.push(Read { bases, quals, ins_gop, del_gop, gcp });
        }
        Region { reads, haplotypes }
    }

    pub fn read_refs(&self) -> Vec<ReadRef<'_>> {
        self.reads.iter().map(Read::as_ref).collect()
    }

    pub fn haplotype_refs(&self) -> Vec<&[u8]> {
        self.haplotypes.iter().map(Vec::as_slice).collect()
    }

    /// Total number of DP cells across all read-haplotype pairs.
    pub fn cells(&self) -> u64 {
        let read_bases: u64 = self.reads.iter().map(|r| r.bases.len() as u64).sum();
        let hap_bases: u64 = self.haplotypes.iter().map(|h| h.len() as u64).sum();
        read_bases * hap_bases
    }
}
