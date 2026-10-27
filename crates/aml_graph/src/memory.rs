//! Per-account recurrent memory built on a connectome-constrained network.
//!
//! Every account owns one copy of the *state* of a [`hcrnn::ConnectomeRnn`] (membrane potentials and
//! a short ring of past firing rates for the conduction delays); the *weights* are shared by all
//! accounts. The wiring, Dale's-law signs, conduction delays and input/output regions are exactly those of
//! the connectome RNN; only the way it is driven changed:
//!
//! * the drive is the output of the hypergraph/graph layer (`u`, `n_in` channels) instead of a task
//!   input, entering through the E units of the input regions only;
//! * the network advances one step each time the account is *active in a window* (event-driven), so
//!   an idle account costs nothing and its memory simply persists (the idle time is an input feature);
//! * the readout (`n_out` channels from the E units of the output regions) feeds the second graph layer.
//!
//! Dynamics per step (identical to `ConnectomeRnn::forward_with`):
//! ```text
//!   a      = b + W_in u + Σ_e w_e · r_{t-delay_e}[pre_e]   (scattered to post_e)
//!   x_new  = (1-α) x + α a ,   r_new = tanh(x_new) ,   y = W_out r_new[out_units] + b_out
//! ```
//! Back-propagation is exact (truncated to the chunk), and is checked against finite differences in the
//! model tests.

use hcrnn::network::{ConnectomeRnn, Layout};

/// Immutable view of the shared weights used by the hot loops.
pub struct Cell {
    pub n: usize,
    /// rows of firing-rate history kept per account (= max delay + 1)
    pub h: usize,
    pub alpha: f64,
    pub post: Vec<usize>,
    pub pre: Vec<usize>,
    pub delay: Vec<usize>,
    /// effective synaptic weights  sign(pre) * softplus(theta)
    pub w: Vec<f64>,
    /// d w / d theta = sign(pre) * sigmoid(theta)
    pub dw: Vec<f64>,
    pub b: Vec<f64>,
    pub bin: Vec<f64>,
    pub wout: Vec<f64>,
    pub bout: Vec<f64>,
    pub in_units: Vec<usize>,
    pub out_units: Vec<usize>,
    pub nu: usize,
    pub no: usize,
    pub layout: Layout,
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

impl Cell {
    pub fn from_rnn(rnn: &ConnectomeRnn) -> Self {
        let l = rnn.layout.clone();
        let w = rnn.effective_weights();
        let dw = rnn
            .edges
            .iter()
            .enumerate()
            .map(|(k, e)| if rnn.is_excitatory(e.pre) { 1.0 } else { -1.0 } * sigmoid(rnn.params[l.theta + k]))
            .collect();
        Cell {
            n: rnn.n_units,
            h: rnn.max_delay() + 1,
            alpha: rnn.alpha,
            post: rnn.edges.iter().map(|e| e.post).collect(),
            pre: rnn.edges.iter().map(|e| e.pre).collect(),
            delay: rnn.edges.iter().map(|e| e.delay).collect(),
            w,
            dw,
            b: rnn.params[l.b..l.w_out].to_vec(),
            bin: rnn.params[l.w_in..l.b].to_vec(),
            wout: rnn.params[l.w_out..l.b_out].to_vec(),
            bout: rnn.params[l.b_out..l.total].to_vec(),
            in_units: rnn.in_units.clone(),
            out_units: rnn.out_units.clone(),
            nu: rnn.cfg.n_in,
            no: rnn.cfg.n_out,
            layout: l,
        }
    }

    pub fn state_len(&self) -> usize {
        self.n * (1 + self.h)
    }
}

/// Persistent per-account state between chunks (single precision to keep millions of accounts affordable).
#[derive(Clone)]
pub struct NodeState {
    pub x: Vec<f32>,
    /// `h` rows of firing rates, oldest first
    pub hist: Vec<f32>,
    pub last_win: i64,
}

/// An account's trajectory inside one chunk: the history prefix followed by one row per step taken.
pub struct NodeSeq {
    pub id: i64,
    pub x: Vec<f64>,
    /// (h + steps) rows of n firing rates
    pub rows: Vec<f64>,
    pub steps: usize,
    /// (position of the window inside the chunk, local node index in that window)
    pub at: Vec<(u32, u32)>,
}

impl NodeSeq {
    pub fn new(cell: &Cell, id: i64, prior: Option<&NodeState>) -> Self {
        let n = cell.n;
        let mut rows = Vec::with_capacity(n * (cell.h + 4));
        let x;
        match prior {
            Some(s) => {
                x = s.x.iter().map(|&v| v as f64).collect();
                rows.extend(s.hist.iter().map(|&v| v as f64));
            }
            None => {
                x = vec![0.0; n];
                rows.resize(n * cell.h, 0.0);
            }
        }
        NodeSeq { id, x, rows, steps: 0, at: Vec::new() }
    }

    pub fn to_state(&self, cell: &Cell, last_win: i64) -> NodeState {
        let n = cell.n;
        let total_rows = cell.h + self.steps;
        let start = (total_rows - cell.h) * n;
        NodeState {
            x: self.x.iter().map(|&v| v as f32).collect(),
            hist: self.rows[start..].iter().map(|&v| v as f32).collect(),
            last_win,
        }
    }

    /// Output row (firing rates after the last step).
    pub fn last_rates<'a>(&'a self, cell: &Cell) -> &'a [f64] {
        let r = cell.h + self.steps - 1;
        &self.rows[r * cell.n..(r + 1) * cell.n]
    }
}

impl Cell {
    /// Advance one step with drive `u` (length `nu`), writing the readout into `y` (length `no`).
    pub fn step(&self, seq: &mut NodeSeq, u: &[f64], y: &mut [f64]) {
        let n = self.n;
        let base_row = self.h - 1 + seq.steps;
        let mut a = self.b.clone();
        for (kk, &unit) in self.in_units.iter().enumerate() {
            let w = &self.bin[kk * self.nu..(kk + 1) * self.nu];
            a[unit] += w.iter().zip(u).map(|(p, q)| p * q).sum::<f64>();
        }
        for e in 0..self.post.len() {
            let row = base_row - self.delay[e];
            a[self.post[e]] += self.w[e] * seq.rows[row * n + self.pre[e]];
        }
        let al = self.alpha;
        let start = seq.rows.len();
        seq.rows.resize(start + n, 0.0);
        for i in 0..n {
            let xn = (1.0 - al) * seq.x[i] + al * a[i];
            seq.x[i] = xn;
            seq.rows[start + i] = xn.tanh();
        }
        let nq = self.out_units.len();
        for k in 0..self.no {
            let mut s = self.bout[k];
            for (q, &unit) in self.out_units.iter().enumerate() {
                s += self.wout[k * nq + q] * seq.rows[start + unit];
            }
            y[k] = s;
        }
        seq.steps += 1;
    }

    /// Exact BPTT through one account's chunk trajectory.
    ///
    /// * `u_of(s)` / `gy_of(s)` give the drive and the loss gradient w.r.t. the readout at step `s`.
    /// * Parameter gradients are accumulated into `grad` (layout of the connectome RNN parameter vector).
    /// * Returns the gradient w.r.t. the drive, `steps x nu`.
    pub fn bptt<'a>(
        &self,
        seq: &NodeSeq,
        u_of: impl Fn(usize) -> &'a [f64],
        gy_of: impl Fn(usize) -> &'a [f64],
        grad: &mut [f64],
    ) -> Vec<f64> {
        let (n, h, s_len, al) = (self.n, self.h, seq.steps, self.alpha);
        let l = &self.layout;
        let nq = self.out_units.len();
        let mut gu = vec![0.0; s_len * self.nu];
        let mut gr = vec![0.0; (h + s_len) * n];
        let mut gx_next = vec![0.0; n];
        let mut gx = vec![0.0; n];
        let mut ga = vec![0.0; n];
        for s in (0..s_len).rev() {
            let row = h + s;
            let gy = gy_of(s);
            for k in 0..self.no {
                let dy = gy[k];
                if dy == 0.0 {
                    continue;
                }
                grad[l.b_out + k] += dy;
                for (q, &unit) in self.out_units.iter().enumerate() {
                    grad[l.w_out + k * nq + q] += dy * seq.rows[row * n + unit];
                    gr[row * n + unit] += dy * self.wout[k * nq + q];
                }
            }
            for i in 0..n {
                let rv = seq.rows[row * n + i];
                gx[i] = gr[row * n + i] * (1.0 - rv * rv) + (1.0 - al) * gx_next[i];
                ga[i] = al * gx[i];
                grad[l.b + i] += ga[i];
            }
            let u = u_of(s);
            for (kk, &unit) in self.in_units.iter().enumerate() {
                let g = ga[unit];
                if g == 0.0 {
                    continue;
                }
                for c in 0..self.nu {
                    grad[l.w_in + kk * self.nu + c] += g * u[c];
                    gu[s * self.nu + c] += g * self.bin[kk * self.nu + c];
                }
            }
            let base_row = h - 1 + s;
            for e in 0..self.post.len() {
                let g = ga[self.post[e]];
                if g == 0.0 {
                    continue;
                }
                let sp = base_row - self.delay[e];
                grad[l.theta + e] += g * seq.rows[sp * n + self.pre[e]] * self.dw[e];
                if sp >= h {
                    gr[sp * n + self.pre[e]] += self.w[e] * g;
                }
            }
            std::mem::swap(&mut gx, &mut gx_next);
        }
        gu
    }
}
