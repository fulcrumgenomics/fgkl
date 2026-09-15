//! Throughput benchmark of the partially determined kernel on synthetic DRAGEN-mode regions:
//! cells per second per backend and precision, plus the worst deviation from the scalar
//! double-precision PD reference on a sample of pairs, and the plain kernel on the same region
//! with the flags stripped for comparison.
//!
//! Usage: pdhmm-bench [--reads N] [--read-len N] [--haps N] [--hap-len N] [--iters N]
//!                    [--seed N] [--backend NAME] [--precision float|double]

use std::time::Instant;

use fgkl_pairhmm::synthetic::PdRegion;
use fgkl_pairhmm::{Backend, Config, PairHmm, PdPairHmm, Precision, pdhmm};

struct Args {
    reads: usize,
    read_len: usize,
    haps: usize,
    hap_len: usize,
    iters: usize,
    seed: u64,
    backend: Option<Backend>,
    precision: Option<Precision>,
}

impl Args {
    fn parse() -> Args {
        let mut args = Args {
            reads: 2000,
            read_len: 150,
            haps: 64,
            hap_len: 500,
            iters: 3,
            seed: 7,
            backend: None,
            precision: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let value = it.next().unwrap_or_else(|| panic!("missing value for {flag}"));
            match flag.as_str() {
                "--reads" => args.reads = value.parse().unwrap(),
                "--read-len" => args.read_len = value.parse().unwrap(),
                "--haps" => args.haps = value.parse().unwrap(),
                "--hap-len" => args.hap_len = value.parse().unwrap(),
                "--iters" => args.iters = value.parse().unwrap(),
                "--seed" => args.seed = value.parse().unwrap(),
                "--backend" => args.backend = Some(value.parse().unwrap()),
                "--precision" => {
                    args.precision = Some(match value.as_str() {
                        "float" => Precision::Float,
                        "double" => Precision::Double,
                        other => panic!("unknown precision {other}"),
                    })
                }
                other => panic!("unknown flag {other}"),
            }
        }
        args
    }
}

fn main() {
    let args = Args::parse();
    let region = PdRegion::generate(args.seed, args.reads, args.read_len, args.haps, args.hap_len);
    let reads = region.read_refs();
    let haps = region.haplotype_refs();
    let plain: Vec<&[u8]> = haps.iter().map(|h| h.bases).collect();
    let cells = region.cells();
    let n_haps = haps.len();

    let sample = reads.len().min(64);
    let mut expected = Vec::with_capacity(sample * n_haps);
    for read in &reads[..sample] {
        for hap in &haps {
            expected.push(pdhmm::reference_log10_likelihood(hap.bases, hap.flags, read));
        }
    }

    let backends = match args.backend {
        Some(b) => vec![b],
        None => Backend::available(),
    };
    let precisions = match args.precision {
        Some(p) => vec![p],
        None => vec![Precision::Float, Precision::Double],
    };
    println!("reads={} haps={} cells={} iters={}", reads.len(), n_haps, cells, args.iters);
    println!(
        "{:<8} {:<7} {:<6} {:>10} {:>12} {:>12} {:>10}",
        "backend", "prec", "kernel", "ms", "Mcells/s", "max|err|", "fallbacks"
    );
    let mut out = vec![0.0; reads.len() * n_haps];
    for &backend in &backends {
        for &precision in &precisions {
            let config = Config { precision, backend: Some(backend), double_fallback: true };
            let prec = match precision {
                Precision::Float => "float",
                Precision::Double => "double",
            };
            let hmm = PdPairHmm::new(&config).unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..=args.iters {
                let start = Instant::now();
                hmm.compute_log10_likelihoods(&reads, &haps, &mut out).unwrap();
                best = best.min(start.elapsed().as_secs_f64());
            }
            let err = out[..expected.len()]
                .iter()
                .zip(&expected)
                .map(|(a, e)| (a - e).abs())
                .fold(0.0f64, f64::max);
            let fallbacks = hmm.fallback_pairs() / (args.iters as u64 + 1);
            println!(
                "{:<8} {:<7} {:<6} {:>10.2} {:>12.1} {:>12.2e} {:>10}",
                backend.name(),
                prec,
                "pd",
                best * 1e3,
                cells as f64 / best / 1e6,
                err,
                fallbacks
            );
            let plain_hmm = PairHmm::new(&config).unwrap();
            let mut best = f64::INFINITY;
            for _ in 0..=args.iters {
                let start = Instant::now();
                plain_hmm.compute_log10_likelihoods(&reads, &plain, &mut out).unwrap();
                best = best.min(start.elapsed().as_secs_f64());
            }
            println!(
                "{:<8} {:<7} {:<6} {:>10.2} {:>12.1} {:>12} {:>10}",
                backend.name(),
                prec,
                "plain",
                best * 1e3,
                cells as f64 / best / 1e6,
                "-",
                plain_hmm.fallback_pairs() / (args.iters as u64 + 1)
            );
        }
    }
}
