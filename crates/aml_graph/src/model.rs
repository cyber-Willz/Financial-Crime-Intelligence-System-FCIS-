//! The connectome-informed recurrent graph neural network.
//!
//! ```text
//!   transaction window ──► temporal hypergraph snapshot (stars + pair graph)
//!        features x_v          │
//!                              ▼
//!   stage 1  u_v = tanh(W1 [x_v ‖ Ĥx_v ‖ A_in x_v ‖ A_out x_v] + b1)        hypergraph + graph convolution
//!                              ▼
//!   stage 2  y_v = connectome-RNN memory step(u_v)                           one persistent state per account
//!                              ▼
//!   stage 3  g_v = tanh(W3 [y_v ‖ Ĥy_v ‖ A_in y_v ‖ A_out y_v] + b3)        GNN over the *memory read-outs*
//!                              ▼
//!   head     p_tx = σ( v·tanh(Ws g_s + Wr g_r + We f_tx + bh) + c )          risk of each transaction
//! ```
//!
//! Everything between the graph layers is the recurrent network from `human_connectome_rnn`: wiring from a
//! structural connectome, conduction delays from fibre length, Dale's-law excitatory/inhibitory units,
//! input through the E units of the input regions and read-out from the E units of the output regions.

use crate::features::{WindowData, EDGE_FEATS, NODE_FEATS};
use crate::memory::{Cell, NodeSeq, NodeState};
use crate::window::WindowGraph;
use hcrnn::network::{default_roles, Calibration, ConnectomeRnn, EdgeKind, MaskMode, RnnConfig};
use hcrnn::rng::Rng;
use hcrnn::Connectome;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CirGnnConfig {
    // ---- windows / temporal hypergraph ----
    pub window_secs: i64,
    pub min_star: usize,
    pub horizon: usize,
    pub nb_steps: usize,
    pub warmup_windows: usize,
    // ---- layer sizes ----
    pub u_dim: usize,
    pub o_dim: usize,
    pub g_dim: usize,
    pub h_dim: usize,
    // ---- ablation switches ----
    pub use_hypergraph: bool,
    pub use_graph: bool,
    pub use_memory: bool,
    pub use_rules: bool,
    /// use beneficial-ownership context when the feature pipeline is given one (see `RiskTable::with_ownership`)
    pub use_ownership: bool,
    // ---- connectome memory ----
    pub per_hemi: usize,
    pub density: f64,
    pub conn_seed: u64,
    pub tvb_dir: Option<String>,
    /// connectome | degree-preserving | density-matched | dense
    pub wiring: String,
    /// leak per step (= dt / tau)
    pub alpha: f64,
    pub gain: f64,
    pub max_delay: usize,
    pub radius_cap: f64,
    pub seed: u64,
    /// forget accounts idle for more than this many windows
    pub evict_windows: i64,
}

impl Default for CirGnnConfig {
    fn default() -> Self {
        CirGnnConfig {
            window_secs: 86_400,
            min_star: 3,
            horizon: 7,
            nb_steps: 3,
            warmup_windows: 14,
            u_dim: 16,
            o_dim: 8,
            g_dim: 16,
            h_dim: 16,
            use_hypergraph: true,
            use_graph: true,
            use_memory: true,
            use_rules: true,
            use_ownership: true,
            per_hemi: 33,
            density: 0.2,
            conn_seed: 1,
            tvb_dir: None,
            wiring: "connectome".into(),
            alpha: 0.4,
            gain: 0.9,
            max_delay: 3,
            radius_cap: 1.05,
            seed: 1,
            evict_windows: 90,
        }
    }
}

/// Parameter gradient, one vector per parameter group (same shapes as the parameters).
#[derive(Clone)]
pub struct Grad {
    pub rnn: Vec<f64>,
    pub p1: Vec<f64>,
    pub p3: Vec<f64>,
    pub ph: Vec<f64>,
}

impl Grad {
    pub fn zeros_like(m: &CirGnn) -> Self {
        Grad { rnn: vec![0.0; m.rnn.params.len()], p1: vec![0.0; m.p1.len()], p3: vec![0.0; m.p3.len()], ph: vec![0.0; m.ph.len()] }
    }
    pub fn norm(&self) -> f64 {
        [&self.rnn, &self.p1, &self.p3, &self.ph].iter().map(|v| v.iter().map(|x| x * x).sum::<f64>()).sum::<f64>().sqrt()
    }
}

#[derive(Clone, Debug)]
pub struct LossCfg {
    pub pos_weight: f64,
    /// fraction of negative transactions that contribute to the loss
    pub neg_keep: f64,
    pub seed: u64,
}

pub enum Mode<'a> {
    Eval,
    Train { grad: &'a mut Grad, loss: &'a LossCfg },
}

#[derive(Default)]
pub struct MemStore {
    pub map: HashMap<i64, NodeState>,
}

impl MemStore {
    pub fn evict(&mut self, now_win: i64, windows: i64) {
        self.map.retain(|_, s| now_win - s.last_win <= windows);
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[derive(Default)]
pub struct ChunkOut {
    /// mean weighted loss over the chunk's sampled transactions (train mode)
    pub loss: f64,
    pub weight: f64,
    /// (global transaction index, probability) for every transaction of a counted window
    pub preds: Vec<(usize, f32)>,
}

pub struct CirGnn {
    pub cfg: CirGnnConfig,
    pub rnn: ConnectomeRnn,
    pub p1: Vec<f64>,
    pub p3: Vec<f64>,
    pub ph: Vec<f64>,
    pub mean: Vec<f64>,
    pub std: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
struct Saved {
    cfg: CirGnnConfig,
    rnn_params: Vec<f64>,
    p1: Vec<f64>,
    p3: Vec<f64>,
    ph: Vec<f64>,
    mean: Vec<f64>,
    std: Vec<f64>,
}

fn build_rnn(cfg: &CirGnnConfig) -> Result<ConnectomeRnn, String> {
    let conn: Connectome = match &cfg.tvb_dir {
        Some(d) => Connectome::from_tvb_dir(Path::new(d))?.log_compressed(),
        None => Connectome::synthetic_human(cfg.per_hemi, cfg.density, cfg.conn_seed)?.log_compressed(),
    };
    let (inp, out) = default_roles(&conn)?;
    let mode = MaskMode::parse(&cfg.wiring).ok_or_else(|| format!("unknown wiring '{}'", cfg.wiring))?;
    let tau = 10.0;
    let rcfg = RnnConfig {
        mode,
        dt_ms: cfg.alpha * tau,
        tau_ms: tau,
        use_delays: false,
        calibration: Calibration::Power,
        gain: cfg.gain,
        n_in: cfg.u_dim,
        n_out: cfg.o_dim,
        input_regions: inp,
        output_regions: out,
        seed: cfg.seed,
        ..Default::default()
    };
    let mut rnn = ConnectomeRnn::new(&conn, rcfg)?;
    if cfg.max_delay > 0 {
        let r = conn.n();
        let mut maxlen = 0.0f64;
        for e in &rnn.edges {
            if matches!(e.kind, EdgeKind::InterEE | EdgeKind::InterEI) {
                maxlen = maxlen.max(conn.length(e.post % r, e.pre % r));
            }
        }
        for e in rnn.edges.iter_mut() {
            if matches!(e.kind, EdgeKind::InterEE | EdgeKind::InterEI) && maxlen > 0.0 {
                let d = (conn.length(e.post % r, e.pre % r) / maxlen * cfg.max_delay as f64).round() as usize;
                e.delay = d.clamp(1, cfg.max_delay);
            }
        }
        rnn.cfg.use_delays = true;
    }
    if cfg.radius_cap > 0.0 {
        rnn.enforce_radius_cap(cfg.radius_cap);
    }
    rnn.check_constraints()?;
    Ok(rnn)
}

#[inline]
fn dense_fwd(w: &[f64], b: &[f64], x: &[f64], n: usize, din: usize, dout: usize, f: impl Fn(f64) -> f64, out: &mut [f64]) {
    for v in 0..n {
        let xv = &x[v * din..(v + 1) * din];
        for j in 0..dout {
            let wj = &w[j * din..(j + 1) * din];
            let mut s = b[j];
            for i in 0..din {
                s += wj[i] * xv[i];
            }
            out[v * dout + j] = f(s);
        }
    }
}

fn stable_bce(z: f64, y: f64) -> f64 {
    z.max(0.0) - z * y + (-z.abs()).exp().ln_1p()
}
fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

fn mix(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// Deterministic per-transaction training weight (0 = not sampled).
pub fn sample_weight(l: &LossCfg, global_idx: usize, label: bool) -> f64 {
    if label {
        l.pos_weight
    } else if (mix(global_idx as u64 ^ l.seed) % 1_000_000) as f64 / 1e6 < l.neg_keep {
        1.0
    } else {
        0.0
    }
}

struct WinCache {
    n: usize,
    x4: Vec<f64>,
    u: Vec<f64>,
    gy: Vec<f64>,
    gu: Vec<f64>,
}

impl CirGnn {
    pub fn new(cfg: CirGnnConfig) -> Result<Self, String> {
        let rnn = build_rnn(&cfg)?;
        let mut rng = Rng::new(cfg.seed ^ 0xC1_26_77);
        let (u, o, g, hh) = (cfg.u_dim, cfg.o_dim, cfg.g_dim, cfg.h_dim);
        let init = |len: usize, fan_in: usize, rng: &mut Rng| -> Vec<f64> { (0..len).map(|_| rng.normal() / (fan_in as f64).sqrt()).collect() };
        let mut p1 = init(u * 4 * NODE_FEATS, 4 * NODE_FEATS, &mut rng);
        p1.extend(std::iter::repeat(0.0).take(u));
        let mut p3 = init(g * 4 * o, 4 * o, &mut rng);
        p3.extend(std::iter::repeat(0.0).take(g));
        let mut ph = init(hh * (2 * g + EDGE_FEATS), 2 * g + EDGE_FEATS, &mut rng);
        ph.extend(std::iter::repeat(0.0).take(hh)); // bh
        ph.extend(init(hh, hh, &mut rng)); // vo
        ph.push(-3.0); // co: rare positives
        Ok(CirGnn { cfg, rnn, p1, p3, ph, mean: vec![0.0; NODE_FEATS], std: vec![1.0; NODE_FEATS] })
    }

    pub fn num_params(&self) -> usize {
        self.rnn.params.len() + self.p1.len() + self.p3.len() + self.ph.len()
    }

    // ---- head parameter offsets ----
    fn ph_offsets(&self) -> (usize, usize, usize, usize, usize, usize) {
        let (g, hh) = (self.cfg.g_dim, self.cfg.h_dim);
        let ws = 0;
        let wr = ws + hh * g;
        let we = wr + hh * g;
        let bh = we + hh * EDGE_FEATS;
        let vo = bh + hh;
        let co = vo + hh;
        (ws, wr, we, bh, vo, co)
    }

    /// Fit the node-feature standardisation on raw windows.
    pub fn fit_norm(&mut self, wins: impl Iterator<Item = WindowData>) {
        let (mut s, mut s2, mut cnt) = (vec![0.0; NODE_FEATS], vec![0.0; NODE_FEATS], 0.0f64);
        for w in wins {
            if !w.counted {
                continue;
            }
            for v in 0..w.graph.n() {
                for k in 0..NODE_FEATS {
                    let x = w.x_raw[v * NODE_FEATS + k];
                    s[k] += x;
                    s2[k] += x * x;
                }
                cnt += 1.0;
            }
        }
        if cnt > 0.0 {
            for k in 0..NODE_FEATS {
                let m = s[k] / cnt;
                self.mean[k] = m;
                self.std[k] = (s2[k] / cnt - m * m).max(1e-12).sqrt().max(1e-3);
            }
        }
    }

    /// `[x ‖ Ĥx ‖ A_in x ‖ A_out x]` for node features or memory read-outs `x` (n x d).
    fn assemble(&self, g: &WindowGraph, x: &[f64], d: usize) -> Vec<f64> {
        let n = g.n();
        let mut out = vec![0.0; n * 4 * d];
        let mut h = vec![0.0; n * d];
        let mut ai = vec![0.0; n * d];
        let mut ao = vec![0.0; n * d];
        if self.cfg.use_hypergraph {
            g.hyper(x, d, &mut h);
        }
        if self.cfg.use_graph {
            g.mean_in(x, d, &mut ai);
            g.mean_out(x, d, &mut ao);
        }
        for v in 0..n {
            let o = &mut out[v * 4 * d..(v + 1) * 4 * d];
            o[..d].copy_from_slice(&x[v * d..(v + 1) * d]);
            o[d..2 * d].copy_from_slice(&h[v * d..(v + 1) * d]);
            o[2 * d..3 * d].copy_from_slice(&ai[v * d..(v + 1) * d]);
            o[3 * d..].copy_from_slice(&ao[v * d..(v + 1) * d]);
        }
        out
    }

    /// Adjoint of [`assemble`]: gradient w.r.t. `x` from the gradient w.r.t. the 4d-block input.
    fn assemble_t(&self, g: &WindowGraph, dz: &[f64], d: usize) -> Vec<f64> {
        let n = g.n();
        let mut gx = vec![0.0; n * d];
        let (mut bh, mut bi, mut bo) = (vec![0.0; n * d], vec![0.0; n * d], vec![0.0; n * d]);
        for v in 0..n {
            let z = &dz[v * 4 * d..(v + 1) * 4 * d];
            gx[v * d..(v + 1) * d].copy_from_slice(&z[..d]);
            bh[v * d..(v + 1) * d].copy_from_slice(&z[d..2 * d]);
            bi[v * d..(v + 1) * d].copy_from_slice(&z[2 * d..3 * d]);
            bo[v * d..(v + 1) * d].copy_from_slice(&z[3 * d..]);
        }
        if self.cfg.use_hypergraph {
            g.hyper_t(&bh, d, &mut gx);
        }
        if self.cfg.use_graph {
            g.mean_in_t(&bi, d, &mut gx);
            g.mean_out_t(&bo, d, &mut gx);
        }
        gx
    }

    /// Run the model over consecutive windows, carrying `store` forward. In train mode also
    /// back-propagates (truncated to this chunk) and accumulates gradients.
    pub fn run_chunk(&self, wins: &[WindowData], store: &mut MemStore, mut mode: Mode, threads: usize) -> ChunkOut {
        let cell = Cell::from_rnn(&self.rnn);
        let (nu, no, gd, hh) = (self.cfg.u_dim, self.cfg.o_dim, self.cfg.g_dim, self.cfg.h_dim);
        let nf = NODE_FEATS;
        let (ws_o, wr_o, we_o, bh_o, vo_o, co_o) = self.ph_offsets();
        let train = matches!(mode, Mode::Train { .. });

        // total sampling weight (loss normaliser) over the chunk
        let mut norm = 0.0;
        if let Mode::Train { loss, .. } = &mode {
            for w in wins.iter().filter(|w| w.counted) {
                for (k, &ti) in w.txn_idx.iter().enumerate() {
                    norm += sample_weight(loss, ti, w.label[k]);
                }
            }
        }
        let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };

        let mut out = ChunkOut::default();
        let mut caches: Vec<WinCache> = Vec::with_capacity(wins.len());
        let mut seqs: Vec<NodeSeq> = Vec::new();
        let mut seq_map: HashMap<i64, usize> = HashMap::new();
        let mut gnn_grad = if train { Some((vec![0.0; self.p3.len()], vec![0.0; self.ph.len()])) } else { None };
        let mut loss_acc = 0.0;

        for (wi, wd) in wins.iter().enumerate() {
            let g = &wd.graph;
            let n = g.n();
            if !self.cfg.use_memory {
                seq_map.clear();
            }
            // ---- stage 1: hypergraph + graph convolution on the standardised features ----
            let mut xs = vec![0.0; n * nf];
            for v in 0..n {
                for k in 0..nf {
                    xs[v * nf + k] = (wd.x_raw[v * nf + k] - self.mean[k]) / self.std[k];
                }
            }
            let x4 = self.assemble(g, &xs, nf);
            let (w1, b1) = self.p1.split_at(nu * 4 * nf);
            let mut u = vec![0.0; n * nu];
            dense_fwd(w1, b1, &x4, n, 4 * nf, nu, f64::tanh, &mut u);

            // ---- stage 2: connectome memory, one step per active account ----
            let mut local_of_seq: Vec<usize> = vec![usize::MAX; seqs.len()];
            for v in 0..n {
                let acct = g.accounts[v];
                let si = match seq_map.get(&acct) {
                    Some(&si) => si,
                    None => {
                        let prior = if self.cfg.use_memory { store.map.get(&acct) } else { None };
                        seqs.push(NodeSeq::new(&cell, acct, prior));
                        local_of_seq.push(usize::MAX);
                        seq_map.insert(acct, seqs.len() - 1);
                        seqs.len() - 1
                    }
                };
                local_of_seq[si] = v;
            }
            let mut y = vec![0.0; n * no];
            {
                let mut refs: Vec<(&mut NodeSeq, usize)> = seqs
                    .iter_mut()
                    .enumerate()
                    .filter_map(|(si, s)| if local_of_seq[si] != usize::MAX { Some((s, local_of_seq[si])) } else { None })
                    .collect();
                let t = threads.min((refs.len() / 192).max(1));
                if t <= 1 {
                    for (s, l) in refs.iter_mut() {
                        s.at.push((wi as u32, *l as u32));
                        cell.step(s, &u[*l * nu..(*l + 1) * nu], &mut y[*l * no..(*l + 1) * no]);
                    }
                } else {
                    let per = (refs.len() + t - 1) / t;
                    let (cell_r, u_r) = (&cell, &u);
                    let results: Vec<Vec<(usize, Vec<f64>)>> = std::thread::scope(|sc| {
                        let hs: Vec<_> = refs
                            .chunks_mut(per)
                            .map(|chunk| {
                                sc.spawn(move || {
                                    chunk
                                        .iter_mut()
                                        .map(|(s, l)| {
                                            s.at.push((wi as u32, *l as u32));
                                            let mut yy = vec![0.0; no];
                                            cell_r.step(s, &u_r[*l * nu..(*l + 1) * nu], &mut yy);
                                            (*l, yy)
                                        })
                                        .collect::<Vec<_>>()
                                })
                            })
                            .collect();
                        hs.into_iter().map(|h| h.join().expect("memory worker panicked")).collect()
                    });
                    for r in results {
                        for (l, yy) in r {
                            y[l * no..(l + 1) * no].copy_from_slice(&yy);
                        }
                    }
                }
            }

            // ---- stage 3: GNN over the memory read-outs ----
            let z4 = self.assemble(g, &y, no);
            let (w3, b3) = self.p3.split_at(gd * 4 * no);
            let mut gact = vec![0.0; n * gd];
            dense_fwd(w3, b3, &z4, n, 4 * no, gd, f64::tanh, &mut gact);

            // ---- head: one risk per transaction ----
            let m = g.s.len();
            let mut dg = if train { vec![0.0; n * gd] } else { Vec::new() };
            for k in 0..m {
                let (s, r) = (g.s[k] as usize, g.r[k] as usize);
                let gs = &gact[s * gd..(s + 1) * gd];
                let gr = &gact[r * gd..(r + 1) * gd];
                let ef = &wd.ef[k * EDGE_FEATS..(k + 1) * EDGE_FEATS];
                let mut h = vec![0.0; hh];
                let mut logit = self.ph[co_o];
                for j in 0..hh {
                    let mut p = self.ph[bh_o + j];
                    let (a, b) = (&self.ph[ws_o + j * gd..ws_o + (j + 1) * gd], &self.ph[wr_o + j * gd..wr_o + (j + 1) * gd]);
                    for i in 0..gd {
                        p += a[i] * gs[i] + b[i] * gr[i];
                    }
                    let c = &self.ph[we_o + j * EDGE_FEATS..we_o + (j + 1) * EDGE_FEATS];
                    for e in 0..EDGE_FEATS {
                        p += c[e] * ef[e];
                    }
                    h[j] = p.tanh();
                    logit += self.ph[vo_o + j] * h[j];
                }
                if wd.counted {
                    out.preds.push((wd.txn_idx[k], sigmoid(logit) as f32));
                }
                if let (Mode::Train { loss, .. }, Some((_, gph))) = (&mode, gnn_grad.as_mut()) {
                    if !wd.counted {
                        continue;
                    }
                    let wgt = sample_weight(loss, wd.txn_idx[k], wd.label[k]);
                    if wgt == 0.0 {
                        continue;
                    }
                    let yv = if wd.label[k] { 1.0 } else { 0.0 };
                    loss_acc += wgt * stable_bce(logit, yv) * inv_norm;
                    let dz = wgt * (sigmoid(logit) - yv) * inv_norm;
                    gph[co_o] += dz;
                    for j in 0..hh {
                        gph[vo_o + j] += dz * h[j];
                        let dp = dz * self.ph[vo_o + j] * (1.0 - h[j] * h[j]);
                        gph[bh_o + j] += dp;
                        for i in 0..gd {
                            gph[ws_o + j * gd + i] += dp * gs[i];
                            gph[wr_o + j * gd + i] += dp * gr[i];
                            dg[s * gd + i] += dp * self.ph[ws_o + j * gd + i];
                            dg[r * gd + i] += dp * self.ph[wr_o + j * gd + i];
                        }
                        for e in 0..EDGE_FEATS {
                            gph[we_o + j * EDGE_FEATS + e] += dp * ef[e];
                        }
                    }
                }
            }

            // ---- back through stage 3 to the memory read-outs ----
            let mut gy = vec![0.0; n * no];
            if train {
                let (g3, _) = gnn_grad.as_mut().unwrap();
                let mut dz4 = vec![0.0; n * 4 * no];
                for v in 0..n {
                    for j in 0..gd {
                        let a = gact[v * gd + j];
                        let dp = dg[v * gd + j] * (1.0 - a * a);
                        if dp == 0.0 {
                            continue;
                        }
                        g3[gd * 4 * no + j] += dp;
                        for i in 0..4 * no {
                            g3[j * 4 * no + i] += dp * z4[v * 4 * no + i];
                            dz4[v * 4 * no + i] += dp * w3[j * 4 * no + i];
                        }
                    }
                }
                gy = self.assemble_t(g, &dz4, no);
            }
            caches.push(WinCache { n, x4, u, gy, gu: Vec::new() });
        }
        out.loss = loss_acc;
        out.weight = norm;

        // ---- backward: per-account BPTT, then stage 1 ----
        if let Mode::Train { grad, .. } = &mut mode {
            let (g3, gph) = gnn_grad.take().unwrap();
            for (a, b) in grad.p3.iter_mut().zip(&g3) {
                *a += b;
            }
            for (a, b) in grad.ph.iter_mut().zip(&gph) {
                *a += b;
            }
            let np = self.rnn.params.len();
            let t = threads.min((seqs.len() / 64).max(1));
            let per = (seqs.len() + t - 1) / t.max(1);
            let caches_r = &caches;
            let cell_r = &cell;
            let work = |chunk: &[NodeSeq]| -> (Vec<f64>, Vec<Vec<f64>>) {
                let mut gr = vec![0.0; np];
                let mut gus = Vec::with_capacity(chunk.len());
                for seq in chunk {
                    let gu = cell_r.bptt(
                        seq,
                        |s| {
                            let (w, l) = seq.at[s];
                            let c = &caches_r[w as usize];
                            &c.u[l as usize * nu..(l as usize + 1) * nu]
                        },
                        |s| {
                            let (w, l) = seq.at[s];
                            let c = &caches_r[w as usize];
                            &c.gy[l as usize * no..(l as usize + 1) * no]
                        },
                        &mut gr,
                    );
                    gus.push(gu);
                }
                (gr, gus)
            };
            let parts: Vec<(Vec<f64>, Vec<Vec<f64>>)> = if t <= 1 || seqs.is_empty() {
                vec![work(&seqs)]
            } else {
                std::thread::scope(|sc| {
                    let hs: Vec<_> = seqs.chunks(per).map(|chunk| sc.spawn(|| work(chunk))).collect();
                    hs.into_iter().map(|h| h.join().expect("bptt worker panicked")).collect()
                })
            };
            for c in caches.iter_mut() {
                c.gu = vec![0.0; c.n * nu];
            }
            let mut si = 0;
            for (gr, gus) in parts {
                for (a, b) in grad.rnn.iter_mut().zip(&gr) {
                    *a += b;
                }
                for gu in gus {
                    let seq = &seqs[si];
                    for (s, &(w, l)) in seq.at.iter().enumerate() {
                        let dst = &mut caches[w as usize].gu[l as usize * nu..(l as usize + 1) * nu];
                        for c in 0..nu {
                            dst[c] += gu[s * nu + c];
                        }
                    }
                    si += 1;
                }
            }
            // stage 1: u = tanh(W1 x4 + b1)
            for c in &caches {
                for v in 0..c.n {
                    for j in 0..nu {
                        let uv = c.u[v * nu + j];
                        let dp = c.gu[v * nu + j] * (1.0 - uv * uv);
                        if dp == 0.0 {
                            continue;
                        }
                        grad.p1[nu * 4 * nf + j] += dp;
                        for i in 0..4 * nf {
                            grad.p1[j * 4 * nf + i] += dp * c.x4[v * 4 * nf + i];
                        }
                    }
                }
            }
        }

        // ---- persist memory ----
        if self.cfg.use_memory {
            for seq in &seqs {
                let last = seq.at.last().map(|&(w, _)| wins[w as usize].graph.win).unwrap_or(0);
                store.map.insert(seq.id, seq.to_state(&cell, last));
            }
            if let Some(w) = wins.last() {
                store.evict(w.graph.win, self.cfg.evict_windows);
            }
        }
        out
    }

    // ---------------------------------------------------------------- persistence / flat views

    pub fn save(&self, path: &Path) -> Result<(), String> {
        let s = Saved {
            cfg: self.cfg.clone(),
            rnn_params: self.rnn.params.clone(),
            p1: self.p1.clone(),
            p3: self.p3.clone(),
            ph: self.ph.clone(),
            mean: self.mean.clone(),
            std: self.std.clone(),
        };
        std::fs::write(path, serde_json::to_vec(&s).map_err(|e| e.to_string())?).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let s: Saved = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
        let mut m = CirGnn::new(s.cfg)?;
        if s.rnn_params.len() != m.rnn.params.len() || s.p1.len() != m.p1.len() || s.p3.len() != m.p3.len() || s.ph.len() != m.ph.len() {
            return Err("model file does not match its configuration".into());
        }
        m.rnn.params = s.rnn_params;
        m.p1 = s.p1;
        m.p3 = s.p3;
        m.ph = s.ph;
        m.mean = s.mean;
        m.std = s.std;
        m.rnn.check_constraints()?;
        Ok(m)
    }

    pub fn snapshot(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        (self.rnn.params.clone(), self.p1.clone(), self.p3.clone(), self.ph.clone())
    }
    pub fn restore(&mut self, s: &(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)) {
        self.rnn.params = s.0.clone();
        self.p1 = s.1.clone();
        self.p3 = s.2.clone();
        self.ph = s.3.clone();
    }
}
