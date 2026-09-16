//! Replays a GATK `--pair-hmm-results-file` dump: reconstructs the per-region read x haplotype
//! batches, checks the kernel against GATK's recorded likelihoods, and reports throughput, the
//! single-precision fallback rate (and what the double-precision recomputation costs), and how
//! much work prefix sharing removes.
//!
//! Usage: pairhmm-replay <pairs.txt> [--backend NAME] [--iters N] [--max-regions N]
//!        [--only-region N] [--dump-region N FILE] [--write-results DIR]
//!
//! `--write-results DIR` writes every pair's result as little-endian f64 in region order to
//! `DIR/<dump stem>.<mode>.f64` (and the recorded GATK values to `<stem>.expected.f64`) so two
//! platforms' results can be compared pair by pair.

use std::io::{BufRead, BufReader};
use std::time::Instant;

use fgkl_pairhmm::synthetic::Read;
use fgkl_pairhmm::{Backend, Config, PairHmm, Precision, ReadRef};

struct Region {
    reads: Vec<Read>,
    haps: Vec<Vec<u8>>,
    expected: Vec<f64>,
}

fn phred(s: &str) -> Vec<u8> {
    s.bytes().map(|b| b - 33).collect()
}

fn fastq(quals: &[u8]) -> String {
    quals.iter().map(|&q| (q + 33) as char).collect()
}

/// Writes one region back out in the dump format, so it can be replayed on its own.
fn write_region_dump(r: &Region, file: &str) {
    let mut out = String::from(
        "# hap-bases read-bases read-qual read-ins-qual read-del-qual gcp expected-result\n",
    );
    for (ri, read) in r.reads.iter().enumerate() {
        for (hi, hap) in r.haps.iter().enumerate() {
            out.push_str(&format!(
                "{} {} {} {} {} {} {:e}\n",
                String::from_utf8_lossy(hap),
                String::from_utf8_lossy(&read.bases),
                fastq(&read.quals),
                fastq(&read.ins_gop),
                fastq(&read.del_gop),
                fastq(&read.gcp),
                r.expected[ri * r.haps.len() + hi]
            ));
        }
    }
    std::fs::write(file, out).expect("write region dump");
}

/// Prints every pair of `r` whose kernel result is non-finite or further than `threshold` from
/// GATK's, with the inputs needed to reproduce it.
fn report_bad_pairs(region_index: usize, r: &Region, out: &[f64], threshold: f64) {
    let n = r.haps.len();
    for (k, (a, e)) in out.iter().zip(&r.expected).enumerate() {
        if a.is_finite() && (a - e).abs() <= threshold {
            continue;
        }
        let (ri, hi) = (k / n, k % n);
        let read = &r.reads[ri];
        eprintln!(
            "BAD region={region_index} read={ri} hap={hi} got={a} expected={e} nreads={} nhaps={n}\nHAP {}\nBASES {}\nQUALS {}\nINS {}\nDEL {}\nGCP {}",
            r.reads.len(),
            String::from_utf8_lossy(&r.haps[hi]),
            String::from_utf8_lossy(&read.bases),
            fastq(&read.quals),
            fastq(&read.ins_gop),
            fastq(&read.del_gop),
            fastq(&read.gcp)
        );
    }
}

fn parse(path: &str, max_regions: usize) -> Vec<Region> {
    let file = std::fs::File::open(path).expect("open dump");
    let mut regions: Vec<Region> = Vec::new();
    // GATK writes one line per (read, haplotype) with reads outer and haplotypes inner, region
    // after region. The first read of a region lists the region's haplotypes and every later
    // read repeats that list. Duplicate reads are dumped back to back, so a read boundary is
    // recognised by the haplotype list starting over, never by the read changing.
    let mut cur = Region { reads: Vec::new(), haps: Vec::new(), expected: Vec::new() };
    let mut list_known = false;
    let mut next_hap = 0usize;
    for line in BufReader::new(file).lines() {
        let line = line.unwrap();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let f: Vec<&str> = line.split(' ').collect();
        if f.len() < 7 {
            continue;
        }
        let hap = f[0].as_bytes();
        let lk: f64 = f[6].parse().unwrap();
        let read = || Read {
            bases: f[1].as_bytes().to_vec(),
            quals: phred(f[2]),
            ins_gop: phred(f[3]),
            del_gop: phred(f[4]),
            gcp: phred(f[5]),
        };
        let starts_over = cur.haps.first().map(Vec::as_slice) == Some(hap);
        if cur.reads.is_empty() {
            cur.reads.push(read());
            cur.haps.push(hap.to_vec());
            next_hap = 1;
        } else if !list_known {
            // Still collecting the first read's haplotypes; the list is complete once it repeats.
            if starts_over {
                list_known = true;
                cur.reads.push(read());
                next_hap = 1;
            } else {
                cur.haps.push(hap.to_vec());
                next_hap += 1;
            }
        } else if next_hap == cur.haps.len() {
            // A read boundary: the same region if the list starts over, otherwise a new one.
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
                cur.haps.push(hap.to_vec());
                list_known = false;
                next_hap = 1;
            }
        } else if cur.haps[next_hap].as_slice() == hap {
            next_hap += 1;
        } else {
            // The list disagrees mid-read: the region so far is unusable, start over from here.
            cur = Region { reads: vec![read()], haps: vec![hap.to_vec()], expected: Vec::new() };
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let path = args.first().expect("usage: pairhmm-replay <pairs.txt>");
    let mut backend: Option<Backend> = None;
    let mut iters = 2usize;
    let mut max_regions = usize::MAX;
    let mut only_region: Option<usize> = None;
    let mut dump_region: Option<(usize, String)> = None;
    let mut write_results: Option<String> = None;
    let mut rest = args[1..].iter();
    let value = |flag: &str, v: Option<&String>| -> String {
        v.cloned().unwrap_or_else(|| panic!("{flag} needs a value"))
    };
    while let Some(flag) = rest.next() {
        match flag.as_str() {
            "--backend" => backend = Some(value(flag, rest.next()).parse().unwrap()),
            "--iters" => iters = value(flag, rest.next()).parse().unwrap(),
            "--max-regions" => max_regions = value(flag, rest.next()).parse().unwrap(),
            "--only-region" => only_region = Some(value(flag, rest.next()).parse().unwrap()),
            "--write-results" => write_results = Some(value(flag, rest.next())),
            "--dump-region" => {
                let index = value(flag, rest.next()).parse().unwrap();
                dump_region = Some((index, value(flag, rest.next())));
            }
            other => panic!("unknown flag {other}"),
        }
    }
    // `FGKL_REPLAY_DEBUG[=threshold]` prints every pair whose result is non-finite or further
    // than the threshold (default 0.5 log10) from GATK's.
    let debug_threshold: Option<f64> =
        std::env::var_os("FGKL_REPLAY_DEBUG").map(|v| v.to_string_lossy().parse().unwrap_or(0.5));
    let mut regions = parse(path, max_regions);
    if let Some((n, file)) = dump_region {
        write_region_dump(&regions[n], &file);
        eprintln!("wrote region {n} to {file}");
    }
    if let Some(n) = only_region {
        regions = vec![regions.swap_remove(n)];
    }
    if regions.is_empty() {
        eprintln!("no complete regions found in {path}");
        std::process::exit(1);
    }
    let pairs: usize = regions.iter().map(|r| r.expected.len()).sum();
    let cells: u64 = regions
        .iter()
        .map(|r| {
            let rb: u64 = r.reads.iter().map(|x| x.bases.len() as u64).sum();
            let hb: u64 = r.haps.iter().map(|h| h.len() as u64).sum();
            rb * hb
        })
        .sum();
    let shared: u64 = regions
        .iter()
        .map(|r| {
            let mut haps: Vec<&[u8]> = r.haps.iter().map(Vec::as_slice).collect();
            haps.sort();
            let rb: u64 = r.reads.iter().map(|x| x.bases.len() as u64).sum();
            let lcp: u64 = haps
                .windows(2)
                .map(|w| w[0].iter().zip(w[1]).take_while(|(a, b)| a == b).count() as u64)
                .sum();
            rb * lcp
        })
        .sum();
    println!(
        "regions={} pairs={} cells={} prefix-shared cells={} ({:.1}%) mean haps/region={:.1} mean reads/region={:.1}",
        regions.len(),
        pairs,
        cells,
        shared,
        100.0 * shared as f64 / cells as f64,
        regions.iter().map(|r| r.haps.len()).sum::<usize>() as f64 / regions.len() as f64,
        regions.iter().map(|r| r.reads.len()).sum::<usize>() as f64 / regions.len() as f64
    );
    struct Mode {
        label: &'static str,
        precision: Precision,
        double_fallback: bool,
    }
    let modes = [
        Mode { label: "float-nofb", precision: Precision::Float, double_fallback: false },
        Mode { label: "float", precision: Precision::Float, double_fallback: true },
        Mode { label: "double", precision: Precision::Double, double_fallback: true },
    ];
    for mode in modes {
        let config =
            Config { precision: mode.precision, backend, double_fallback: mode.double_fallback };
        let hmm = PairHmm::new(&config).unwrap();
        let mut worst = 0.0f64;
        let mut best = f64::INFINITY;
        let mut low = 0usize;
        let mut fallback_pairs = 0u64;
        // FNV-1a over the result bit patterns, so kernel changes can be checked for bit identity;
        // the second one skips pairs GKL would recompute in double, so a change to how those are
        // recomputed does not disturb it; the third rounds those to 1e-6 log10 so last-bit libm
        // differences between platforms drop out and only real kernel differences remain.
        let mut checksum = 0xcbf2_9ce4_8422_2325u64;
        let mut checksum_kept = 0xcbf2_9ce4_8422_2325u64;
        let mut checksum_rounded = 0xcbf2_9ce4_8422_2325u64;
        let mut results: Vec<f64> = Vec::new();
        for it in 0..iters {
            let counted_before = hmm.fallback_pairs();
            let start = Instant::now();
            for (region_index, r) in regions.iter().enumerate() {
                let reads: Vec<ReadRef<'_>> = r.reads.iter().map(Read::as_ref).collect();
                let haps: Vec<&[u8]> = r.haps.iter().map(Vec::as_slice).collect();
                let mut out = vec![0.0; r.expected.len()];
                hmm.compute_log10_likelihoods(&reads, &haps, &mut out).unwrap();
                if it == 0
                    && let Some(threshold) = debug_threshold
                {
                    report_bad_pairs(region_index, r, &out, threshold);
                }
                if it == 0 {
                    if write_results.is_some() {
                        results.extend_from_slice(&out);
                    }
                    for (a, e) in out.iter().zip(&r.expected) {
                        checksum = (checksum ^ a.to_bits()).wrapping_mul(0x0100_0000_01b3);
                        if *e >= -64.0 {
                            checksum_kept =
                                (checksum_kept ^ a.to_bits()).wrapping_mul(0x0100_0000_01b3);
                            let rounded = (a * 1e6).round() as i64 as u64;
                            checksum_rounded =
                                (checksum_rounded ^ rounded).wrapping_mul(0x0100_0000_01b3);
                        }
                        // Without the double fallback, underflowed pairs come back as NaN.
                        if a.is_finite() {
                            worst = worst.max((a - e).abs());
                        }
                        // GKL recomputes in double below 1e-28 of the 2^120-scaled value.
                        if *e < -64.0 {
                            low += 1;
                        }
                    }
                }
            }
            best = best.min(start.elapsed().as_secs_f64());
            fallback_pairs = hmm.fallback_pairs() - counted_before;
        }
        if let Some(dir) = &write_results {
            let stem = std::path::Path::new(path).file_stem().unwrap().to_string_lossy();
            std::fs::create_dir_all(dir).unwrap();
            let write = |name: &str, values: &[f64]| {
                let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(format!("{dir}/{stem}.{name}.f64"), bytes).unwrap();
            };
            write(mode.label, &results);
            let expected: Vec<f64> =
                regions.iter().flat_map(|r| r.expected.iter().copied()).collect();
            write("expected", &expected);
        }
        println!(
            "{:<6} {:<10} {:8.1} ms {:8.1} Mcells/s  max|err| vs GATK {:.2e}  kernel fallback pairs {} ({:.2}%)  GATK results below f32 threshold {} ({:.2}%)  checksum {:016x} (above threshold {:016x}, rounded 1e-6 {:016x})",
            hmm.backend().name(),
            mode.label,
            best * 1e3,
            cells as f64 / best / 1e6,
            worst,
            fallback_pairs,
            100.0 * fallback_pairs as f64 / pairs as f64,
            low,
            100.0 * low as f64 / pairs as f64,
            checksum,
            checksum_kept,
            checksum_rounded
        );
    }
}
