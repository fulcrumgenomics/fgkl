//! Throughput of the aligner on synthetic haplotype-to-reference and read-to-haplotype pairs.
//! Usage: sw-bench [--pairs N] [--ref-len N] [--alt-len N] [--iters N]

use std::time::Instant;

use fgkl_smithwaterman::{Aligner, OverhangStrategy, SwParameters, reference};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn main() {
    let mut pairs = 2000usize;
    let mut ref_len = 500usize;
    let mut alt_len = 500usize;
    let mut iters = 3usize;
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let v = it.next().expect("value");
        match flag.as_str() {
            "--pairs" => pairs = v.parse().unwrap(),
            "--ref-len" => ref_len = v.parse().unwrap(),
            "--alt-len" => alt_len = v.parse().unwrap(),
            "--iters" => iters = v.parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
    }
    let mut rng = Rng(5);
    let mut data = Vec::with_capacity(pairs);
    for _ in 0..pairs {
        let r: Vec<u8> = (0..ref_len).map(|_| b"ACGT"[rng.below(4)]).collect();
        let mut a = r[..alt_len.min(ref_len)].to_vec();
        for _ in 0..3 {
            let p = rng.below(a.len());
            a[p] = b"ACGT"[rng.below(4)];
        }
        let p = rng.below(a.len());
        a.remove(p);
        data.push((r, a));
    }
    let params = SwParameters::new(200, -150, -260, -11);
    let cells: u64 = data.iter().map(|(r, a)| (r.len() * a.len()) as u64).sum();
    let mut aligner = Aligner::new();
    let mut best = f64::INFINITY;
    for _ in 0..iters {
        let start = Instant::now();
        for (r, a) in &data {
            std::hint::black_box(aligner.align(r, a, &params, OverhangStrategy::Indel).unwrap());
        }
        best = best.min(start.elapsed().as_secs_f64());
    }
    println!("aligner:   {:8.1} ms  {:8.1} Mcells/s", best * 1e3, cells as f64 / best / 1e6);
    let start = Instant::now();
    for (r, a) in data.iter().take(pairs / 4) {
        std::hint::black_box(reference::align(r, a, &params, OverhangStrategy::Indel));
    }
    let t = start.elapsed().as_secs_f64();
    println!("reference: {:8.1} ms  {:8.1} Mcells/s", t * 1e3, cells as f64 / 4.0 / t / 1e6);
}
