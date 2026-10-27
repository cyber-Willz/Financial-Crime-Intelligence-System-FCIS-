//! One time window of the transaction stream as a graph *and* a hypergraph.
//!
//! * **Transaction graph.** Nodes are the accounts active in the window; the distinct directed pairs
//!   `sender -> receiver` give per-node in/out neighbour lists (CSR).
//! * **Hypergraph.** Money-laundering structure is group structure: a fan-in is one receiver and *all*
//!   of its senders at once, not a set of unrelated pairs. For every account with at least `min_star`
//!   distinct senders (receivers) in the window we create an *in-star* (*out-star*) hyperedge
//!   `{account} ∪ senders` (`{account} ∪ receivers`). Hyperedge diffusion
//!   `(Ĥx)_v = 1/deg(v) Σ_{e∋v} mean_{u∈e} x_u` is the random-walk-normalised hypergraph convolution.
//!
//! The adjoint operators (`*_t`) are what back-propagation needs; they are exact transposes, which the
//! unit tests check.

use crate::types::GTxn;
use std::collections::HashMap;

#[derive(Debug, Clone, Default)]
pub struct WindowGraph {
    pub win: i64,
    /// local node -> account id
    pub accounts: Vec<i64>,
    /// per transaction (in the window's slice order): local sender / receiver
    pub s: Vec<u32>,
    pub r: Vec<u32>,
    pub in_ptr: Vec<usize>,
    pub in_idx: Vec<u32>,
    pub out_ptr: Vec<usize>,
    pub out_idx: Vec<u32>,
    pub he_ptr: Vec<usize>,
    pub he_idx: Vec<u32>,
    /// number of hyperedges containing each node
    pub hdeg: Vec<u32>,
}

impl WindowGraph {
    pub fn n(&self) -> usize {
        self.accounts.len()
    }
    pub fn n_hyperedges(&self) -> usize {
        self.he_ptr.len().saturating_sub(1)
    }
    pub fn in_nbrs(&self, v: usize) -> &[u32] {
        &self.in_idx[self.in_ptr[v]..self.in_ptr[v + 1]]
    }
    pub fn out_nbrs(&self, v: usize) -> &[u32] {
        &self.out_idx[self.out_ptr[v]..self.out_ptr[v + 1]]
    }
    pub fn hyperedge(&self, e: usize) -> &[u32] {
        &self.he_idx[self.he_ptr[e]..self.he_ptr[e + 1]]
    }

    pub fn build(win: i64, txns: &[GTxn], min_star: usize) -> Self {
        Self::build_with(win, txns, min_star, None)
    }

    /// Like [`build`](Self::build), plus one *ownership-group* hyperedge for every group (a non-zero id returned
    /// by `group_of`) that has at least two accounts active in the window.
    pub fn build_with(win: i64, txns: &[GTxn], min_star: usize, group_of: Option<&dyn Fn(i64) -> u32>) -> Self {
        let mut map: HashMap<i64, u32> = HashMap::with_capacity(txns.len() * 2);
        let mut accounts: Vec<i64> = Vec::new();
        let mut id = |a: i64, accounts: &mut Vec<i64>| -> u32 {
            *map.entry(a).or_insert_with(|| {
                accounts.push(a);
                (accounts.len() - 1) as u32
            })
        };
        let mut s = Vec::with_capacity(txns.len());
        let mut r = Vec::with_capacity(txns.len());
        for t in txns {
            let a = id(t.sender, &mut accounts);
            let b = id(t.receiver, &mut accounts);
            s.push(a);
            r.push(b);
        }
        let n = accounts.len();
        let mut pairs: Vec<(u32, u32)> = s.iter().cloned().zip(r.iter().cloned()).filter(|(a, b)| a != b).collect();
        pairs.sort_unstable();
        pairs.dedup();
        // out CSR: pairs sorted by (sender, receiver)
        let (mut out_ptr, mut out_idx) = (vec![0usize; n + 1], Vec::with_capacity(pairs.len()));
        for &(a, b) in &pairs {
            out_ptr[a as usize + 1] += 1;
            out_idx.push(b);
        }
        for i in 0..n {
            out_ptr[i + 1] += out_ptr[i];
        }
        let mut rev: Vec<(u32, u32)> = pairs.iter().map(|&(a, b)| (b, a)).collect();
        rev.sort_unstable();
        let (mut in_ptr, mut in_idx) = (vec![0usize; n + 1], Vec::with_capacity(rev.len()));
        for &(b, a) in &rev {
            in_ptr[b as usize + 1] += 1;
            in_idx.push(a);
        }
        for i in 0..n {
            in_ptr[i + 1] += in_ptr[i];
        }
        let mut g = WindowGraph { win, accounts, s, r, in_ptr, in_idx, out_ptr, out_idx, he_ptr: vec![0], he_idx: vec![], hdeg: vec![0; n] };
        for v in 0..n {
            for nb in [g.in_nbrs(v).to_vec(), g.out_nbrs(v).to_vec()] {
                if nb.len() >= min_star.max(2) {
                    g.he_idx.push(v as u32);
                    g.hdeg[v] += 1;
                    for &u in &nb {
                        g.he_idx.push(u);
                        g.hdeg[u as usize] += 1;
                    }
                    g.he_ptr.push(g.he_idx.len());
                }
            }
        }
        if let Some(gf) = group_of {
            let mut by: HashMap<u32, Vec<u32>> = HashMap::new();
            for v in 0..n {
                let gid = gf(g.accounts[v]);
                if gid != 0 {
                    by.entry(gid).or_default().push(v as u32);
                }
            }
            let mut keys: Vec<u32> = by.keys().cloned().collect();
            keys.sort_unstable();
            for k in keys {
                let m = &by[&k];
                if m.len() >= 2 {
                    for &u in m {
                        g.he_idx.push(u);
                        g.hdeg[u as usize] += 1;
                    }
                    g.he_ptr.push(g.he_idx.len());
                }
            }
        }
        g
    }

    // ---- neighbour mean operators (forward and adjoint). `x` is n x d row-major. ----

    fn mean_apply(ptr: &[usize], idx: &[u32], n: usize, x: &[f64], d: usize, out: &mut [f64]) {
        for v in 0..n {
            let nb = &idx[ptr[v]..ptr[v + 1]];
            let o = &mut out[v * d..(v + 1) * d];
            o.iter_mut().for_each(|a| *a = 0.0);
            if nb.is_empty() {
                continue;
            }
            for &u in nb {
                let xu = &x[u as usize * d..(u as usize + 1) * d];
                for k in 0..d {
                    o[k] += xu[k];
                }
            }
            let inv = 1.0 / nb.len() as f64;
            o.iter_mut().for_each(|a| *a *= inv);
        }
    }

    /// out_u += g_v / |nbrs(v)|  for u in nbrs(v)   (adjoint of the neighbour mean)
    fn mean_adjoint(ptr: &[usize], idx: &[u32], n: usize, g: &[f64], d: usize, out: &mut [f64]) {
        for v in 0..n {
            let nb = &idx[ptr[v]..ptr[v + 1]];
            if nb.is_empty() {
                continue;
            }
            let inv = 1.0 / nb.len() as f64;
            let gv = &g[v * d..(v + 1) * d];
            for &u in nb {
                let o = &mut out[u as usize * d..(u as usize + 1) * d];
                for k in 0..d {
                    o[k] += gv[k] * inv;
                }
            }
        }
    }

    /// Mean over distinct senders.
    pub fn mean_in(&self, x: &[f64], d: usize, out: &mut [f64]) {
        Self::mean_apply(&self.in_ptr, &self.in_idx, self.n(), x, d, out)
    }
    pub fn mean_in_t(&self, g: &[f64], d: usize, out: &mut [f64]) {
        Self::mean_adjoint(&self.in_ptr, &self.in_idx, self.n(), g, d, out)
    }
    /// Mean over distinct receivers.
    pub fn mean_out(&self, x: &[f64], d: usize, out: &mut [f64]) {
        Self::mean_apply(&self.out_ptr, &self.out_idx, self.n(), x, d, out)
    }
    pub fn mean_out_t(&self, g: &[f64], d: usize, out: &mut [f64]) {
        Self::mean_adjoint(&self.out_ptr, &self.out_idx, self.n(), g, d, out)
    }

    /// Random-walk hypergraph diffusion `(Ĥx)_v = 1/deg(v) Σ_{e∋v} mean_{u∈e} x_u` (zero when deg = 0).
    pub fn hyper(&self, x: &[f64], d: usize, out: &mut [f64]) {
        out.iter_mut().for_each(|a| *a = 0.0);
        let mut m = vec![0.0; d];
        for e in 0..self.n_hyperedges() {
            let mem = self.hyperedge(e);
            m.iter_mut().for_each(|a| *a = 0.0);
            for &u in mem {
                for k in 0..d {
                    m[k] += x[u as usize * d + k];
                }
            }
            let inv = 1.0 / mem.len() as f64;
            for &v in mem {
                let w = inv / self.hdeg[v as usize] as f64;
                for k in 0..d {
                    out[v as usize * d + k] += m[k] * w;
                }
            }
        }
    }

    /// Exact transpose of [`hyper`](Self::hyper): `out_u += Σ_{e∋u} 1/|e| Σ_{v∈e} g_v / deg(v)`.
    pub fn hyper_t(&self, g: &[f64], d: usize, out: &mut [f64]) {
        let mut s = vec![0.0; d];
        for e in 0..self.n_hyperedges() {
            let mem = self.hyperedge(e);
            s.iter_mut().for_each(|a| *a = 0.0);
            for &v in mem {
                let w = 1.0 / self.hdeg[v as usize] as f64;
                for k in 0..d {
                    s[k] += g[v as usize * d + k] * w;
                }
            }
            let inv = 1.0 / mem.len() as f64;
            for &u in mem {
                for k in 0..d {
                    out[u as usize * d + k] += s[k] * inv;
                }
            }
        }
    }

    /// Size of the largest hyperedge each node belongs to (0 when none).
    pub fn max_hyperedge_size(&self) -> Vec<usize> {
        let mut m = vec![0usize; self.n()];
        for e in 0..self.n_hyperedges() {
            let mem = self.hyperedge(e);
            for &v in mem {
                m[v as usize] = m[v as usize].max(mem.len());
            }
        }
        m
    }

    /// Non-backtracking (Hashimoto) walk mass on the *directed* money-flow graph of this window.
    ///
    /// `B` acts on directed edges: `(u→v)` feeds `(v→w)` for `w ≠ u`. The returned value for a node is
    /// the number of non-backtracking walks of `steps + 1` edges that end at it. On an acyclic flow
    /// graph `B` is nilpotent, so the mass is non-zero only at the end of chains at least `steps + 1`
    /// edges long and in/after directed cycles (round-tripping): a spectral cycle-and-depth signal that
    /// plain degree statistics cannot see. Counts are capped to keep dense windows finite.
    pub fn nb_walk_mass(&self, steps: usize) -> Vec<f64> {
        let n = self.n();
        // distinct directed edges, sorted by (u, v): rebuild from the out CSR
        let mut eu = Vec::with_capacity(self.out_idx.len());
        let mut ev = Vec::with_capacity(self.out_idx.len());
        for u in 0..n {
            for &v in self.out_nbrs(u) {
                eu.push(u as u32);
                ev.push(v);
            }
        }
        let m = eu.len();
        let reverse: Vec<Option<usize>> = (0..m)
            .map(|k| {
                let (u, v) = (eu[k], ev[k]);
                let (lo, hi) = (self.out_ptr[v as usize], self.out_ptr[v as usize + 1]);
                self.out_idx[lo..hi].binary_search(&u).ok().map(|p| lo + p)
            })
            .collect();
        let mut x = vec![1.0f64; m];
        let mut inmass = vec![0.0f64; n];
        for _ in 0..steps {
            inmass.iter_mut().for_each(|a| *a = 0.0);
            for k in 0..m {
                inmass[ev[k] as usize] += x[k];
            }
            let mut nx = vec![0.0f64; m];
            for k in 0..m {
                let mut v = inmass[eu[k] as usize];
                if let Some(rk) = reverse[k] {
                    v -= x[rk];
                }
                nx[k] = v.clamp(0.0, 1e9);
            }
            x = nx;
        }
        let mut score = vec![0.0; n];
        for k in 0..m {
            score[ev[k] as usize] += x[k];
        }
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aml_engine::{Country, PaymentType};

    fn tx(s: i64, r: i64) -> GTxn {
        GTxn {
            ts: 0, sender: s, receiver: r, amount: 100.0, ptype: PaymentType::Ach,
            sc: Country::parse("GB").unwrap(), rc: Country::parse("GB").unwrap(),
            cross_ccy: false, rf: 0, label: false, typology: 0, rule: 0.0, rule_retro: 0.0,
        }
    }

    fn toy() -> WindowGraph {
        // fan-in 1..5 -> 10, a chain 20->21->22->23->24->25, a 3-cycle 30->31->32->30
        let mut t: Vec<GTxn> = (1..=5).map(|s| tx(s, 10)).collect();
        for (a, b) in [(20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (30, 31), (31, 32), (32, 30), (10, 40), (10, 41), (10, 42)] {
            t.push(tx(a, b));
        }
        WindowGraph::build(0, &t, 3)
    }

    #[test]
    fn hyperedges_cover_fan_in_and_fan_out() {
        let g = toy();
        assert_eq!(g.n_hyperedges(), 2);
        let sizes: Vec<usize> = (0..2).map(|e| g.hyperedge(e).len()).collect();
        assert!(sizes.iter().all(|&s| s == 6 || s == 4), "{sizes:?}");
    }

    fn rand_vec(n: usize, seed: u64) -> Vec<f64> {
        let mut r = hcrnn::rng::Rng::new(seed);
        (0..n).map(|_| r.normal()).collect()
    }

    #[test]
    fn adjoints_are_exact_transposes() {
        let g = toy();
        let (n, d) = (g.n(), 3);
        let x = rand_vec(n * d, 1);
        let y = rand_vec(n * d, 2);
        type Op = fn(&WindowGraph, &[f64], usize, &mut [f64]);
        let ops: [(Op, Op); 3] = [
            (WindowGraph::mean_in, WindowGraph::mean_in_t),
            (WindowGraph::mean_out, WindowGraph::mean_out_t),
            (WindowGraph::hyper, WindowGraph::hyper_t),
        ];
        for (f, ft) in ops {
            let mut fx = vec![0.0; n * d];
            f(&g, &x, d, &mut fx);
            let mut fty = vec![0.0; n * d];
            ft(&g, &y, d, &mut fty);
            let a: f64 = fx.iter().zip(&y).map(|(p, q)| p * q).sum();
            let b: f64 = x.iter().zip(&fty).map(|(p, q)| p * q).sum();
            assert!((a - b).abs() < 1e-10, "<Fx,y>={a} vs <x,F^T y>={b}");
        }
    }

    #[test]
    fn nb_mass_sees_chains_and_cycles_but_not_fan_in() {
        let g = toy();
        let m = g.nb_walk_mass(3);
        let at = |acct: i64| m[g.accounts.iter().position(|&a| a == acct).unwrap()];
        assert_eq!(at(10), 0.0, "a fan-in hub is the end of walks of length 1 only");
        assert!(at(25) > 0.0, "end of a 5-edge chain carries mass");
        assert_eq!(at(22), 0.0, "middle of a short chain does not");
        assert!(at(30) > 0.0 && at(31) > 0.0 && at(32) > 0.0, "cycle nodes carry mass");
        // a 2-cycle (pure backtracking) must carry no mass
        let g2 = WindowGraph::build(0, &[tx(1, 2), tx(2, 1)], 3);
        assert!(g2.nb_walk_mass(3).iter().all(|&v| v == 0.0));
    }
}
