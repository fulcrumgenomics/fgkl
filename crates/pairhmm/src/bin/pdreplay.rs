//! Replays a GATK `--pdhmm-results-file` dump (HaplotypeCaller in `--dragen-378-concordance-mode`)
//! through the partially determined kernel: timing per precision, the worst deviation from GATK's
//! recorded likelihoods, the fallback count, and how many cells prefix sharing between PD
//! haplotypes could save. GATK writes the likelihoods with `%e`, i.e. seven significant digits,
//! so the deviation bottoms out around `5e-7 * |likelihood|` whatever the kernel does.
//!
//! Usage: pdhmm-replay <pdpairs.txt> [--backend NAME] [--iters N] [--max-regions N]

use std::io::{BufRead, BufReader};
use std::time::Instant;

use fgkl_pairhmm::synthetic::Read;
use fgkl_pairhmm::{Backend, Config, PdHaplotype, PdPairHmm, Precision, ReadRef, pdhmm};

struct Hap {
    bases: Vec<u8>,
    flags: Vec<u8>,
}

struct Region {
    reads: Vec<Read>,
    haps: Vec<Hap>,
    /// `expected[read * haps.len() + hap]`, as GATK computed it.
    expected: Vec<f64>,
}

fn phred(s: &str) -> Vec<u8> {
    s.bytes().map(|b| b - 33).collect()
}

/// Parses GATK's `Arrays.toString(byte[])` rendering of the flag bytes.
fn flags(s: &str) -> Vec<u8> {
    s.trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .filter(|t| !t.trim().is_empty())
        .map(|t| t.trim().parse::<i16>().expect("flag byte") as u8)
        .collect()
}

/// GATK writes one line per (read, haplotype), reads outer and haplotypes inner, region after
/// region. A region's haplotype list is learnt from its first read and must repeat for every
/// later read; a line that breaks the pattern starts a new region.
fn parse(path: &str, max_regions: usize) -> Vec<Region> {
    let file = std::fs::File::open(path).expect("open dump");
    let mut regions: Vec<Region> = Vec::new();
    let mut cur = Region { reads: Vec::new(), haps: Vec::new(), expected: Vec::new() };
    let mut list_known = false;
    let mut next_hap = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 8 {
            continue;
        }
        let hap = Hap { bases: f[0].as_bytes().to_vec(), flags: flags(f[1]) };
        let lk: f64 = f[7].parse().unwrap();
        let read = || Read {
            bases: f[2].as_bytes().to_vec(),
            quals: phred(f[3]),
            ins_gop: phred(f[4]),
            del_gop: phred(f[5]),
            gcp: phred(f[6]),
        };
        let same = |a: &Hap, b: &Hap| a.bases == b.bases && a.flags == b.flags;
        let starts_over = cur.haps.first().is_some_and(|h| same(h, &hap));
        if cur.reads.is_empty() {
            cur.reads.push(read());
            cur.haps.push(hap);
            next_hap = 1;
        } else if !list_known {
            if starts_over {
                list_known = true;
                cur.reads.push(read());
                next_hap = 1;
            } else {
                cur.haps.push(hap);
                next_hap += 1;
            }
        } else if next_hap == cur.haps.len() {
            if starts_over {
                cur.reads.push(read());
                next_hap = 1;
            } else {
                regions.push(std::mem::replace(
                    &mut cur,
                    Region { reads: Vec::new(), haps: Vec::new(), expected: Vec::new() },
                ));
                if regions.len() >= max_regions {
                    break;
                }
                cur.reads.push(read());
                cur.haps.push(hap);
                list_known = false;
                next_hap = 1;
            }
        } else if same(&cur.haps[next_hap], &hap) {
            next_hap += 1;
        } else {
            cur = Region { reads: vec![read()], haps: vec![hap], expected: Vec::new() };
            list_known = false;
            next_hap = 1;
        }
        cur.expected.push(lk);
    }
    if !cur.reads.is_empty() {
        regions.push(cur);
    }
    regions.retain(|r| r.expected.len() == r.reads.len() * r.haps.len());
    regions
}

/// The deletion state GATK carries out of a row, which every row after the first starts in.
fn row_end_state(flags: &[u8]) -> u8 {
    let mut state = 0u8;
    for &f in flags {
        if state == 2 {
            state = 0;
        }
        if f & pdhmm::DEL_START != 0 {
            state = 1;
        }
        if f & pdhmm::DEL_END != 0 {
            state = 2;
        }
    }
    state
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: pdhmm-replay <pdpairs.txt>");
    let mut backend: Option<Backend> = None;
    let mut iters = 2usize;
    let mut max_regions = usize::MAX;
    let mut i = 1;
    while i + 1 < args.len() {
        match args[i].as_str() {
            "--backend" => backend = Some(args[i + 1].parse().unwrap()),
            "--iters" => iters = args[i + 1].parse().unwrap(),
            "--max-regions" => max_regions = args[i + 1].parse().unwrap(),
            other => panic!("unknown flag {other}"),
        }
        i += 2;
    }
    let regions = parse(path, max_regions);
    let pairs: usize = regions.iter().map(|r| r.expected.len()).sum();
    let mut cells = 0u64;
    let mut shareable = 0u64;
    let mut flagged_haps = 0usize;
    let mut flagged_cols = 0usize;
    let mut total_cols = 0usize;
    for r in &regions {
        let rb: u64 = r.reads.iter().map(|x| x.bases.len() as u64).sum();
        let hb: u64 = r.haps.iter().map(|h| h.bases.len() as u64).sum();
        cells += rb * hb;
        for h in &r.haps {
            total_cols += h.flags.len();
            let n = h.flags.iter().filter(|&&f| f != 0).count();
            flagged_cols += n;
            flagged_haps += (n > 0) as usize;
        }
        // Columns two haplotypes share (same base and flag) while both rows start in the same
        // state could be computed once, as the plain kernel does for plain haplotypes.
        let mut keys: Vec<(u8, Vec<(u8, u8)>)> = r
            .haps
            .iter()
            .map(|h| {
                (
                    row_end_state(&h.flags),
                    h.bases.iter().copied().zip(h.flags.iter().copied()).collect(),
                )
            })
            .collect();
        keys.sort();
        let lcp: u64 = keys
            .windows(2)
            .filter(|w| w[0].0 == w[1].0)
            .map(|w| w[0].1.iter().zip(&w[1].1).take_while(|(a, b)| a == b).count() as u64)
            .sum();
        shareable += rb * lcp;
    }
    println!(
        "regions={} pairs={} cells={} shareable cells={} ({:.1}%) mean haps/region={:.1} mean reads/region={:.1} flagged haps={:.1}% flagged columns={:.2}%",
        regions.len(),
        pairs,
        cells,
        shareable,
        100.0 * shareable as f64 / cells as f64,
        regions.iter().map(|r| r.haps.len()).sum::<usize>() as f64 / regions.len() as f64,
        regions.iter().map(|r| r.reads.len()).sum::<usize>() as f64 / regions.len() as f64,
        100.0 * flagged_haps as f64 / regions.iter().map(|r| r.haps.len()).sum::<usize>() as f64,
        100.0 * flagged_cols as f64 / total_cols as f64,
    );
    println!(
        "{:<8} {:<7} {:>10} {:>12} {:>12} {:>10}",
        "backend", "prec", "ms", "Mcells/s", "max|err|", "fallbacks"
    );
    for precision in [Precision::Float, Precision::Double] {
        let config = Config { precision, backend, double_fallback: true };
        let hmm = PdPairHmm::new(&config).unwrap();
        let mut worst = 0.0f64;
        let mut best = f64::INFINITY;
        let mut fallbacks = 0u64;
        for it in 0..iters {
            let before = hmm.fallback_pairs();
            let start = Instant::now();
            for r in &regions {
                let reads: Vec<ReadRef<'_>> = r.reads.iter().map(Read::as_ref).collect();
                let haps: Vec<PdHaplotype<'_>> = r
                    .haps
                    .iter()
                    .map(|h| PdHaplotype { bases: &h.bases, flags: &h.flags })
                    .collect();
                let mut out = vec![0.0; r.expected.len()];
                hmm.compute_log10_likelihoods(&reads, &haps, &mut out).unwrap();
                if it == 0 {
                    for (a, e) in out.iter().zip(&r.expected) {
                        // GATK writes -Infinity for pairs it discards after the kernel.
                        if e.is_finite() {
                            worst = worst.max((a - e).abs());
                        }
                    }
                }
            }
            best = best.min(start.elapsed().as_secs_f64());
            fallbacks = hmm.fallback_pairs() - before;
        }
        let prec = match precision {
            Precision::Float => "float",
            Precision::Double => "double",
        };
        println!(
            "{:<8} {:<7} {:>10.1} {:>12.1} {:>12.2e} {:>10}",
            hmm.backend().name(),
            prec,
            best * 1e3,
            cells as f64 / best / 1e6,
            worst,
            fallbacks
        );
    }
}
