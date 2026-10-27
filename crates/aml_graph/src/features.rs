//! Causal feature extraction and the window pipeline.
//!
//! Features for window `t` are computed from the window itself and from per-account history of windows
//! `< t` only, then the history is updated. Running the pipeline over a time-sorted stream therefore
//! yields exactly the features an online system would have seen.

use crate::ownership::{OwnershipMap, Relation};
use crate::types::GTxn;
use crate::window::WindowGraph;
use aml_engine::{Country, EngineConfig};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Number of node features.
pub const NODE_FEATS: usize = 32;
/// Number of per-transaction features fed to the risk head.
pub const EDGE_FEATS: usize = 10;

pub const NODE_FEATURE_NAMES: [&str; NODE_FEATS] = [
    "ln_n_in", "ln_n_out", "ln_sum_in", "ln_sum_out", "ln_distinct_in", "ln_distinct_out", "cash_share",
    "cross_border_share", "high_risk_jur_share", "retention", "amount_vs_history", "ln_gap_windows",
    "ln_max_hyperedge", "nb_walk_mass", "below_threshold_share", "ln_horizon_distinct_in",
    "ln_horizon_distinct_out", "rule_max", "rule_mean", "first_seen",
    // beneficial ownership (zero unless an OwnershipMap is supplied and `use_ownership` is on)
    "own_known", "own_ln_n_owners", "own_layers", "own_unresolved", "own_opacity", "own_sanctioned_pct",
    "own_network_exposure", "own_pep", "own_high_risk_jur", "own_group_flow_share", "own_group_active_members",
    "own_group_out_volume",
];

#[derive(Clone, Debug)]
pub struct FeatureConfig {
    pub window_secs: i64,
    pub min_star: usize,
    /// trailing windows used for the multi-horizon hyperedge statistics
    pub horizon: usize,
    pub nb_steps: usize,
    /// reporting threshold used by the "just below the threshold" share
    pub band_threshold: f64,
    pub use_rules: bool,
    pub use_ownership: bool,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        FeatureConfig { window_secs: 86_400, min_star: 3, horizon: 7, nb_steps: 3, band_threshold: 10_000.0, use_rules: true, use_ownership: true }
    }
}

/// Country risk tiers copied from the engine configuration (so the graph layer never drifts from it).
#[derive(Clone)]
pub struct RiskTable {
    risky: Vec<bool>,
    /// beneficial-ownership context (optional); static lookup tables travel together
    own: Option<Arc<OwnershipMap>>,
}

impl RiskTable {
    pub fn from_engine(cfg: &EngineConfig) -> Self {
        let mut risky = vec![false; 26 * 26];
        for list in [&cfg.sanctioned_countries, &cfg.very_high_risk_countries, &cfg.high_risk_countries] {
            for c in list {
                if let Some(cc) = Country::parse(c) {
                    risky[cc.index_pub()] = true;
                }
            }
        }
        RiskTable { risky, own: None }
    }
    /// Attach beneficial-ownership context (see `aml_bo`).
    pub fn with_ownership(mut self, own: Arc<OwnershipMap>) -> Self {
        self.own = Some(own);
        self
    }
    pub fn ownership(&self) -> Option<&Arc<OwnershipMap>> {
        self.own.as_ref()
    }
    pub fn is_risky(&self, c: Country) -> bool {
        self.risky[c.index_pub()]
    }
}

trait CountryIdx {
    fn index_pub(&self) -> usize;
}
impl CountryIdx for Country {
    fn index_pub(&self) -> usize {
        let a = self.0[0].wrapping_sub(b'A') as usize;
        let b = self.0[1].wrapping_sub(b'A') as usize;
        if a < 26 && b < 26 { a * 26 + b } else { 0 }
    }
}

#[derive(Default)]
struct AcctHist {
    last_win: i64,
    n_obs: u32,
    ewma_log_amt: f64,
    /// (window, counterparty, inbound) — bounded
    cps: VecDeque<(i64, i64, bool)>,
}

const HIST_CAP: usize = 192;

pub struct FeatureState {
    accts: HashMap<i64, AcctHist>,
    cfg: FeatureConfig,
    risk: RiskTable,
}

/// Everything the model needs for one window.
pub struct WindowData {
    pub graph: WindowGraph,
    /// n x NODE_FEATS raw (unstandardised) node features
    pub x_raw: Vec<f64>,
    /// m x EDGE_FEATS
    pub ef: Vec<f64>,
    /// global indices (into the stream slice) of this window's transactions
    pub txn_idx: Vec<usize>,
    pub label: Vec<bool>,
    pub typology: Vec<u8>,
    pub rf: Vec<u32>,
    pub ts: i64,
    /// false during warm-up (history still empty): the window steps the memory but is not scored/trained
    pub counted: bool,
}

impl FeatureState {
    pub fn new(cfg: FeatureConfig, risk: RiskTable) -> Self {
        FeatureState { accts: HashMap::new(), cfg, risk }
    }

    pub fn config(&self) -> &FeatureConfig {
        &self.cfg
    }

    pub fn tracked_accounts(&self) -> usize {
        self.accts.len()
    }

    /// Drop history of accounts idle for more than `windows` windows.
    pub fn evict(&mut self, now_win: i64, windows: i64) {
        self.accts.retain(|_, h| now_win - h.last_win <= windows);
    }

    pub fn build_window(&mut self, win: i64, txns: &[GTxn], base_idx: usize, counted: bool) -> WindowData {
        let own: Option<Arc<OwnershipMap>> = if self.cfg.use_ownership { self.risk.own.clone() } else { None };
        let group_fn = own.as_ref().map(|o| {
            let o = o.clone();
            move |a: i64| o.get(a).map(|i| i.group).unwrap_or(0)
        });
        let g = WindowGraph::build_with(win, txns, self.cfg.min_star, group_fn.as_ref().map(|f| f as &dyn Fn(i64) -> u32));
        let n = g.n();
        let nb = g.nb_walk_mass(self.cfg.nb_steps);
        let hemax = g.max_hyperedge_size();

        #[derive(Default, Clone)]
        struct Acc {
            n_in: f64, n_out: f64, s_in: f64, s_out: f64, cash: f64, xb: f64, hrj: f64, band: f64,
            cnt: f64, rule_max: f64, rule_sum: f64, log_amt_sum: f64,
        }
        let mut a = vec![Acc::default(); n];
        let band_lo = 0.9 * self.cfg.band_threshold;
        let mut ef = Vec::with_capacity(txns.len() * EDGE_FEATS);
        for (k, t) in txns.iter().enumerate() {
            let (s, r) = (g.s[k] as usize, g.r[k] as usize);
            let rule = if self.cfg.use_rules { t.rule as f64 } else { 0.0 };
            let risky = self.risk.is_risky(t.sc) || self.risk.is_risky(t.rc);
            let xb = t.sc != t.rc;
            let band = t.amount >= band_lo && t.amount < self.cfg.band_threshold;
            for (side, i) in [(0, s), (1, r)] {
                let x = &mut a[i];
                if side == 0 { x.n_out += 1.0; x.s_out += t.amount } else { x.n_in += 1.0; x.s_in += t.amount }
                x.cnt += 1.0;
                x.log_amt_sum += t.amount.max(1.0).ln();
                if t.ptype.is_cash() { x.cash += 1.0 }
                if xb { x.xb += 1.0 }
                if risky { x.hrj += 1.0 }
                if band { x.band += 1.0 }
                x.rule_max = x.rule_max.max(rule);
                x.rule_sum += rule;
            }
            let (mut e_rel, mut e_grp, mut e_opq, mut e_sanc) = (0.0, 0.0, 0.0, 0.0);
            if let Some(o) = &own {
                match o.relation(t.sender, t.receiver) {
                    Relation::SameParty | Relation::SharedOwner => e_rel = 1.0,
                    Relation::SameGroup => e_grp = 1.0,
                    Relation::None => {}
                }
                let (a, b) = (o.get(t.sender), o.get(t.receiver));
                e_opq = a.map_or(0.0, |i| i.opacity as f64).max(b.map_or(0.0, |i| i.opacity as f64));
                e_sanc = (a.map_or(false, |i| i.sanctioned_pct >= 0.5) || b.map_or(false, |i| i.sanctioned_pct >= 0.5)) as u8 as f64;
            }
            ef.extend_from_slice(&[
                t.amount.max(0.0).ln_1p() / 10.0,
                t.ptype.is_cash() as u8 as f64,
                xb as u8 as f64,
                risky as u8 as f64,
                t.cross_ccy as u8 as f64,
                rule,
                e_rel,
                e_grp,
                e_opq,
                e_sanc,
            ]);
        }

        let mut x_raw = vec![0.0; n * NODE_FEATS];
        let h = self.cfg.horizon as i64;
        let mut tmp: Vec<i64> = Vec::new();
        for v in 0..n {
            let acct = g.accounts[v];
            let hist = self.accts.get(&acct);
            let f = &mut x_raw[v * NODE_FEATS..(v + 1) * NODE_FEATS];
            let ac = &a[v];
            f[0] = ac.n_in.ln_1p();
            f[1] = ac.n_out.ln_1p();
            f[2] = ac.s_in.ln_1p() / 5.0;
            f[3] = ac.s_out.ln_1p() / 5.0;
            f[4] = (g.in_nbrs(v).len() as f64).ln_1p();
            f[5] = (g.out_nbrs(v).len() as f64).ln_1p();
            let c = ac.cnt.max(1.0);
            f[6] = ac.cash / c;
            f[7] = ac.xb / c;
            f[8] = ac.hrj / c;
            f[9] = (ac.s_in - ac.s_out) / (ac.s_in + ac.s_out + 1.0);
            let mean_log = ac.log_amt_sum / c;
            f[10] = match hist {
                Some(hh) if hh.n_obs >= 3 => (mean_log - hh.ewma_log_amt).clamp(-3.0, 3.0),
                _ => 0.0,
            };
            f[11] = match hist {
                Some(hh) => ((win - hh.last_win - 1).max(0) as f64).ln_1p() / 3.0,
                None => 0.0,
            };
            f[12] = (hemax[v] as f64).ln_1p() / 3.0;
            f[13] = nb[v].ln_1p() / 3.0;
            f[14] = ac.band / c;
            // multi-horizon hyperedge statistics: distinct counterparties over the trailing `horizon` windows
            for (slot, inbound) in [(15usize, true), (16usize, false)] {
                tmp.clear();
                if let Some(hh) = hist {
                    for &(w, cp, inb) in hh.cps.iter().rev() {
                        if w <= win - h {
                            break;
                        }
                        if inb == inbound {
                            tmp.push(cp);
                        }
                    }
                }
                let cur = if inbound { g.in_nbrs(v) } else { g.out_nbrs(v) };
                tmp.extend(cur.iter().map(|&u| g.accounts[u as usize]));
                tmp.sort_unstable();
                tmp.dedup();
                f[slot] = (tmp.len() as f64).ln_1p() / 3.0;
            }
            f[17] = ac.rule_max;
            f[18] = ac.rule_sum / c;
            f[19] = if hist.is_none() { 1.0 } else { 0.0 };
        }

        if let Some(o) = &own {
            let mut gmembers: HashMap<u32, usize> = HashMap::new();
            let mut gvol: HashMap<u32, f64> = HashMap::new();
            for v in 0..n {
                if let Some(i) = o.get(g.accounts[v]) {
                    if i.group != 0 {
                        *gmembers.entry(i.group).or_default() += 1;
                    }
                }
            }
            let mut related = vec![0.0f64; n];
            for (k, t) in txns.iter().enumerate() {
                if let Some(i) = o.get(t.sender) {
                    if i.group != 0 {
                        *gvol.entry(i.group).or_default() += t.amount;
                    }
                }
                if o.relation(t.sender, t.receiver) != Relation::None {
                    related[g.s[k] as usize] += 1.0;
                    related[g.r[k] as usize] += 1.0;
                }
            }
            for v in 0..n {
                let f = &mut x_raw[v * NODE_FEATS + 20..(v + 1) * NODE_FEATS];
                if let Some(i) = o.get(g.accounts[v]) {
                    f[0] = 1.0;
                    f[1] = (i.owners.len() as f64).ln_1p() / 2.0;
                    f[2] = i.layers as f64 / 5.0;
                    f[3] = i.unresolved as f64;
                    f[4] = i.opacity as f64;
                    f[5] = i.sanctioned_pct as f64;
                    f[6] = (i.exposure as f64 * 50.0).min(1.0);
                    f[7] = i.pep as u8 as f64;
                    f[8] = i.high_risk_jurisdiction as u8 as f64;
                    f[9] = related[v] / a[v].cnt.max(1.0);
                    if i.group != 0 {
                        f[10] = (*gmembers.get(&i.group).unwrap_or(&0) as f64).ln_1p() / 2.0;
                        f[11] = gvol.get(&i.group).copied().unwrap_or(0.0).ln_1p() / 10.0;
                    }
                }
            }
        }

        // causal update of the history *after* the features were computed
        for (k, t) in txns.iter().enumerate() {
            let _ = k;
            for (acct, cp, inb) in [(t.sender, t.receiver, false), (t.receiver, t.sender, true)] {
                let hh = self.accts.entry(acct).or_default();
                if hh.cps.len() >= HIST_CAP {
                    hh.cps.pop_front();
                }
                hh.cps.push_back((win, cp, inb));
            }
        }
        for v in 0..n {
            let ac = &a[v];
            let hh = self.accts.entry(g.accounts[v]).or_default();
            let mean_log = ac.log_amt_sum / ac.cnt.max(1.0);
            hh.ewma_log_amt = if hh.n_obs == 0 { mean_log } else { 0.8 * hh.ewma_log_amt + 0.2 * mean_log };
            hh.n_obs += 1;
            hh.last_win = win;
        }

        WindowData {
            x_raw,
            ef,
            txn_idx: (0..txns.len()).map(|k| base_idx + k).collect(),
            label: txns.iter().map(|t| t.label).collect(),
            typology: txns.iter().map(|t| t.typology).collect(),
            rf: txns.iter().map(|t| t.rf).collect(),
            ts: txns.first().map(|t| t.ts).unwrap_or(0),
            counted,
            graph: g,
        }
    }
}

/// Iterates a time-sorted slice window by window.
pub struct Pipeline<'a> {
    txns: &'a [GTxn],
    pos: usize,
    first_win: Option<i64>,
    warmup_windows: i64,
    pub state: FeatureState,
}

impl<'a> Pipeline<'a> {
    pub fn new(txns: &'a [GTxn], cfg: FeatureConfig, risk: RiskTable, warmup_windows: usize) -> Self {
        Pipeline { txns, pos: 0, first_win: None, warmup_windows: warmup_windows as i64, state: FeatureState::new(cfg, risk) }
    }

    pub fn window_of(&self, ts: i64) -> i64 {
        ts.div_euclid(self.state.cfg.window_secs)
    }
}

impl<'a> Iterator for Pipeline<'a> {
    type Item = WindowData;
    fn next(&mut self) -> Option<WindowData> {
        if self.pos >= self.txns.len() {
            return None;
        }
        let w0 = self.window_of(self.txns[self.pos].ts);
        let mut end = self.pos;
        while end < self.txns.len() && self.window_of(self.txns[end].ts) == w0 {
            end += 1;
        }
        let first = *self.first_win.get_or_insert(w0);
        let counted = w0 - first >= self.warmup_windows;
        let wd = self.state.build_window(w0, &self.txns[self.pos..end], self.pos, counted);
        self.pos = end;
        Some(wd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aml_engine::{default_config, PaymentType};

    fn tx(ts: i64, s: i64, r: i64, amt: f64) -> GTxn {
        GTxn {
            ts, sender: s, receiver: r, amount: amt, ptype: PaymentType::Ach,
            sc: Country::parse("GB").unwrap(), rc: Country::parse("GB").unwrap(),
            cross_ccy: false, rf: 0, label: false, typology: 0, rule: 0.0, rule_retro: 0.0,
        }
    }

    #[test]
    fn features_are_causal() {
        // changing a later window must not change earlier windows' features
        let day = 86_400;
        let mut a: Vec<GTxn> = (0..40).map(|i| tx(i * 20_000, i % 7, 100 + i % 5, 50.0 + i as f64)).collect();
        let mut b = a.clone();
        let last = b.len() - 1;
        b[last].amount = 9_999_999.0;
        b[last].sender = 12345;
        a.sort_by_key(|t| t.ts);
        b.sort_by_key(|t| t.ts);
        let risk = RiskTable::from_engine(&default_config());
        let pa: Vec<WindowData> = Pipeline::new(&a, FeatureConfig::default(), risk.clone(), 0).collect();
        let pb: Vec<WindowData> = Pipeline::new(&b, FeatureConfig::default(), risk, 0).collect();
        assert_eq!(pa.len(), pb.len());
        let n = pa.len();
        assert!(a.last().unwrap().ts / day > a[0].ts / day);
        for i in 0..n - 1 {
            assert_eq!(pa[i].x_raw, pb[i].x_raw, "window {i} leaked future information");
        }
    }
}
