//! Planning suffix sharing for a sorted haplotype set: which haplotypes stop their forward sweep
//! at a cut, and the backward sweeps that compute, once per shared suffix, the values the cuts
//! join. The kernel that follows the plan is in `kernel.rs`.

use std::collections::HashMap;

/// A backward sweep must skip at least this many columns of a haplotype's forward sweep to pay
/// for its join, which costs about two columns.
const MIN_SAVING: usize = 4;

/// Where a haplotype's forward sweep stops: it computes columns up to `column - 1`, then joins
/// the backward values of `column`, held in backward node `node`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cut {
    pub column: usize,
    pub node: usize,
}

/// One backward sweep along a haplotype's suffix, where depth `d` is the haplotype's column
/// `len - d`. It computes depths `resume + 1..=need` starting from node `start`, which holds depth
/// `resume`, and stores each `(depth, node)` of `writes` (ascending) for later sweeps and cuts.
#[derive(Debug)]
pub(crate) struct BackwardSweep {
    /// Sorted index of the haplotype whose bases the sweep follows.
    pub hap: usize,
    pub resume: usize,
    pub need: usize,
    pub start: usize,
    pub writes: Vec<(usize, usize)>,
}

/// How a sorted haplotype set shares suffixes: which haplotypes stop their forward sweep at a cut,
/// and the backward sweeps that compute the values those cuts join, once per shared suffix.
///
/// Backward values at a column depend only on the haplotype's bases from that column to its end,
/// not on its length or prefix. A haplotype whose longest prefix shared with another distinct
/// haplotype is `p` and longest shared suffix is `s` cuts at column `len - d`, with
/// `d = min(s, len - p - 1)`: the forward sweep then covers exactly its shared prefix and any
/// private middle, and the backward pass only goes as deep as the cuts read. Sweeps visit the
/// haplotypes ordered by reversed bases, each resuming from the deepest depth an earlier sweep
/// computed along its suffix, so each (suffix, depth) is computed once.
#[derive(Debug)]
pub(crate) struct SuffixPlan {
    /// Per sorted haplotype; `None` runs the forward sweep to the end.
    pub cuts: Vec<Option<Cut>>,
    /// In the order they must run.
    pub sweeps: Vec<BackwardSweep>,
    /// Backward nodes the plan uses. Node 0 is depth 0, the last column, whose backward values
    /// depend on the reads alone.
    pub nodes: usize,
}

impl SuffixPlan {
    /// A plan with no cuts: every haplotype runs forward to its end.
    pub(crate) fn forward_only(num_haps: usize) -> Self {
        SuffixPlan { cuts: vec![None; num_haps], sweeps: Vec::new(), nodes: 1 }
    }

    /// Plans suffix sharing for haplotypes in sorted order, given their common-prefix lengths with
    /// their predecessors; `dup[k]` marks a copy of haplotype `k - 1`, which plays no part.
    pub(crate) fn new(bases: &[&[u8]], lcp: &[usize], dup: &[bool]) -> Self {
        let distinct: Vec<usize> = (0..bases.len()).filter(|&k| !dup[k]).collect();
        let suffixes = SuffixOrder::new(bases, &distinct);
        let depth = cut_depths(bases, lcp, &distinct, &suffixes);
        let (mut sweeps, sweep_at) = backward_sweeps(&suffixes, &depth);
        let mut numbering = NodeNumbering::default();
        let mut node_for = |r: usize, d: usize, sweeps: &[BackwardSweep]| {
            numbering.node_for(sweeps, &sweep_at, &suffixes.lcs, r, d)
        };
        let starts: Vec<usize> =
            sweeps.iter().map(|s| node_for(suffixes.pos[s.hap], s.resume, &sweeps)).collect();
        let mut cuts = vec![None; bases.len()];
        for &k in &distinct {
            if depth[k] > 0 {
                let node = node_for(suffixes.pos[k], depth[k], &sweeps);
                cuts[k] = Some(Cut { column: bases[k].len() - depth[k], node });
            }
        }
        for (sweep, start) in sweeps.iter_mut().zip(starts) {
            sweep.start = start;
        }
        for (&(s, d), &node) in &numbering.ids {
            sweeps[s].writes.push((d, node));
        }
        for sweep in &mut sweeps {
            sweep.writes.sort_unstable();
        }
        SuffixPlan { cuts, sweeps, nodes: numbering.ids.len() + 1 }
    }

    /// The haplotype columns this plan computes per read base, and those prefix sharing alone
    /// would, for the set it was made for.
    pub(crate) fn columns(&self, bases: &[&[u8]], lcp: &[usize], dup: &[bool]) -> SharedColumns {
        let mut columns = SharedColumns::default();
        for k in (0..bases.len()).filter(|&k| !dup[k]) {
            let len = bases[k].len();
            let stop = self.cuts[k].map_or(len, |c| c.column - 1);
            columns.prefix_only += (len - lcp[k]) as u64;
            columns.forward += (stop - lcp[k]) as u64;
        }
        columns.backward = self.sweeps.iter().map(|s| (s.need - s.resume) as u64).sum();
        columns
    }
}

/// Haplotype columns a set computes per read base, summed over its distinct haplotypes: with
/// prefix sharing alone, and with suffix sharing too, split into forward and backward columns.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SharedColumns {
    pub prefix_only: u64,
    pub forward: u64,
    pub backward: u64,
}

/// Distinct haplotypes ordered by reversed bases.
struct SuffixOrder {
    /// Sorted haplotype indices in reversed-bases order.
    order: Vec<usize>,
    /// `lcs[r]`: common-suffix length of `order[r - 1]` and `order[r]`; `lcs[0]` is 0.
    lcs: Vec<usize>,
    /// Per sorted haplotype, its position in `order`; meaningless for duplicates.
    pos: Vec<usize>,
}

impl SuffixOrder {
    fn new(bases: &[&[u8]], distinct: &[usize]) -> Self {
        let mut order = distinct.to_vec();
        order.sort_by(|&a, &b| bases[a].iter().rev().cmp(bases[b].iter().rev()));
        let lcs = (0..order.len())
            .map(|r| if r == 0 { 0 } else { common_suffix(bases[order[r - 1]], bases[order[r]]) })
            .collect();
        let mut pos = vec![0; bases.len()];
        for (r, &k) in order.iter().enumerate() {
            pos[k] = r;
        }
        SuffixOrder { order, lcs, pos }
    }

    /// The longest suffix the haplotype at position `r` shares with another distinct haplotype,
    /// which is reached at one of its neighbours in this order.
    fn longest_shared(&self, r: usize) -> usize {
        self.lcs[r].max(self.lcs.get(r + 1).copied().unwrap_or(0))
    }
}

/// Per sorted haplotype, how deep into the backward values its cut reads: `min(s, len - p - 1)`,
/// or 0 (no cut) when that would save fewer than [`MIN_SAVING`] columns.
fn cut_depths(
    bases: &[&[u8]],
    lcp: &[usize],
    distinct: &[usize],
    suffixes: &SuffixOrder,
) -> Vec<usize> {
    let mut depth = vec![0; bases.len()];
    for (t, &k) in distinct.iter().enumerate() {
        // `lcp` of a first copy is with the previous distinct haplotype or a copy of it, which
        // has the same bases.
        let shared_prefix = lcp[k].max(distinct.get(t + 1).map_or(0, |&next| lcp[next]));
        let len = bases[k].len();
        if shared_prefix < len {
            let d = suffixes.longest_shared(suffixes.pos[k]).min(len - shared_prefix - 1);
            if d >= MIN_SAVING {
                depth[k] = d;
            }
        }
    }
    depth
}

/// The backward sweeps in reversed-bases order, each resuming from the deepest depth computed
/// along its suffix so far and continuing to its haplotype's cut depth if that is deeper; also,
/// per position in that order, the sweep there if any.
fn backward_sweeps(
    suffixes: &SuffixOrder,
    depth: &[usize],
) -> (Vec<BackwardSweep>, Vec<Option<usize>>) {
    let mut sweeps = Vec::new();
    let mut sweep_at = vec![None; suffixes.order.len()];
    // How deep the backward pass has computed along the current suffix.
    let mut reach = 0;
    for (r, &k) in suffixes.order.iter().enumerate() {
        let resume = suffixes.lcs[r].min(reach);
        if depth[k] > resume {
            sweep_at[r] = Some(sweeps.len());
            sweeps.push(BackwardSweep {
                hap: k,
                resume,
                need: depth[k],
                start: 0,
                writes: Vec::new(),
            });
        }
        reach = resume.max(depth[k]);
    }
    (sweeps, sweep_at)
}

/// Numbers the backward nodes a plan uses: node 0 is depth 0, and every other node is a depth a
/// sweep computes and a cut or a later sweep reads.
#[derive(Default)]
struct NodeNumbering {
    /// `(sweep, depth)` to node.
    ids: HashMap<(usize, usize), usize>,
}

impl NodeNumbering {
    /// The node holding depth `depth` of the suffix at position `r` of the reversed-bases order:
    /// the one written by the latest sweep at or before `r` that computed that depth along a
    /// suffix `r` shares at least that deep.
    fn node_for(
        &mut self,
        sweeps: &[BackwardSweep],
        sweep_at: &[Option<usize>],
        lcs: &[usize],
        r: usize,
        depth: usize,
    ) -> usize {
        if depth == 0 {
            return 0;
        }
        let mut p = r;
        let mut shared = usize::MAX;
        loop {
            if let Some(s) = sweep_at[p]
                && sweeps[s].resume < depth
                && depth <= sweeps[s].need
            {
                let next = self.ids.len() + 1;
                return *self.ids.entry((s, depth)).or_insert(next);
            }
            shared = shared.min(lcs[p]);
            assert!(p > 0 && shared >= depth, "backward depth {depth} is never computed");
            p -= 1;
        }
    }
}

fn common_suffix(a: &[u8], b: &[u8]) -> usize {
    a.iter().rev().zip(b.iter().rev()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::SortedHaps;
    use crate::synthetic::{Rng, edited};

    /// Replays the suffix plan of `haps` and checks that every sweep starts from, and every cut
    /// joins, a node already written for exactly its own suffix at that depth; that no node is
    /// written twice; and that a cut haplotype's forward sweep still reaches every column the
    /// next distinct haplotype resumes from.
    fn consistent_plan<'a>(haps: &[&'a [u8]]) -> SortedHaps<'a> {
        let sorted = SortedHaps::new(haps, true);
        let plan = &sorted.plan;
        let suffix =
            |k: usize, depth: usize| sorted.bases[k][sorted.bases[k].len() - depth..].to_vec();
        let mut written: HashMap<usize, Vec<u8>> = HashMap::from([(0, Vec::new())]);
        for sweep in &plan.sweeps {
            assert!(sweep.resume < sweep.need);
            assert_eq!(written.get(&sweep.start), Some(&suffix(sweep.hap, sweep.resume)));
            for &(depth, node) in &sweep.writes {
                assert!(sweep.resume < depth && depth <= sweep.need);
                assert!(written.insert(node, suffix(sweep.hap, depth)).is_none());
            }
        }
        assert_eq!(written.len(), plan.nodes);
        for (k, cut) in plan.cuts.iter().enumerate() {
            let Some(cut) = cut else { continue };
            assert!(!sorted.dup[k]);
            let depth = sorted.bases[k].len() - cut.column;
            assert!(depth >= MIN_SAVING);
            assert_eq!(written.get(&cut.node), Some(&suffix(k, depth)));
            let next = (k + 1..sorted.len()).find(|&j| !sorted.dup[j]);
            let resumes_at = next.map_or(0, |j| sorted.lcp[j]);
            assert!(cut.column > sorted.lcp[k] && cut.column > resumes_at);
        }
        sorted
    }

    fn reference_like(rng: &mut Rng, alphabet: &[u8], len: usize) -> Vec<u8> {
        (0..len).map(|_| alphabet[rng.below(alphabet.len())]).collect()
    }

    #[test]
    fn a_substitution_cuts_both_haplotypes_right_after_their_shared_prefix() {
        let reference = reference_like(&mut Rng::new(1), b"ACGT", 40);
        let mut snp = reference.clone();
        snp[20] = if snp[20] == b'A' { b'C' } else { b'A' };
        let sorted = consistent_plan(&[&reference, &snp]);
        // Both share columns 1..=20 forward and everything after column 21 backward, so both
        // compute column 21 alone and join the same node there.
        let cuts = &sorted.plan.cuts;
        assert_eq!(cuts[0].map(|c| c.column), Some(21));
        assert_eq!(cuts[0], cuts[1]);
        assert_eq!(sorted.plan.sweeps.len(), 1);
    }

    #[test]
    fn a_haplotype_whose_suffix_is_its_own_runs_forward_to_its_end() {
        let reference = reference_like(&mut Rng::new(2), b"ACGT", 40);
        let mut late = reference.clone();
        late[38] = if late[38] == b'A' { b'C' } else { b'A' };
        let sorted = consistent_plan(&[&reference, &late]);
        assert!(sorted.plan.cuts.iter().all(Option::is_none));
        assert!(sorted.plan.sweeps.is_empty());
    }

    #[test]
    fn duplicates_are_flagged_and_never_cut() {
        let reference = reference_like(&mut Rng::new(3), b"ACGT", 30);
        let mut snp = reference.clone();
        snp[10] = if snp[10] == b'A' { b'C' } else { b'A' };
        let sorted = consistent_plan(&[&snp, &reference, &snp, &reference]);
        assert_eq!(sorted.dup, [false, true, false, true]);
        assert!(sorted.plan.cuts[0].is_some() && sorted.plan.cuts[2].is_some());
    }

    #[test]
    fn plans_of_random_haplotype_sets_are_consistent() {
        let mut rng = Rng::new(4);
        for case in 0..400 {
            let alphabet: &[u8] = if case % 2 == 0 { b"AC" } else { b"ACGT" };
            let len = 5 + rng.below(60);
            let base = reference_like(&mut rng, alphabet, len);
            let mut haps = vec![base.clone()];
            for _ in 0..rng.below(30) {
                let hap = if rng.chance(0.15) {
                    haps[rng.below(haps.len())].clone()
                } else {
                    let from = rng.below(haps.len());
                    edited(&mut rng, alphabet, &haps[from])
                };
                haps.push(hap);
            }
            let refs: Vec<&[u8]> = haps.iter().map(Vec::as_slice).collect();
            consistent_plan(&refs);
        }
    }
}
