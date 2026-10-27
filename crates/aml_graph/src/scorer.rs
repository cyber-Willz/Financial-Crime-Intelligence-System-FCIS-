//! Streaming inference: push transactions as they arrive, get a graph risk for every transaction of a window
//! when the window closes.
//!
//! The rule engine scores each transaction instantly; the graph model works at window granularity because
//! its input is a whole-window hypergraph (a fan-in is only visible once the senders have all appeared).
//! The two are complementary: use the engine for real-time blocking/alerting and this scorer for end-of-window
//! (e.g. hourly / daily) re-scoring and case prioritisation.

use crate::features::{FeatureState, RiskTable};
use crate::model::{CirGnn, MemStore, Mode};
use crate::train::feature_config;
use crate::types::GTxn;
use aml_engine::{Engine, EngineConfig, TxnIn};

#[derive(Debug, Clone)]
pub struct WindowScores {
    pub window: i64,
    /// (caller reference `rf`, risk in `[0, 1]`) for every transaction of the window
    pub scores: Vec<(u32, f32)>,
    /// highest transaction risk touching each account in the window (descending)
    pub accounts: Vec<(i64, f32)>,
}

pub struct GraphScorer {
    model: CirGnn,
    feats: FeatureState,
    store: MemStore,
    engine: Option<Engine>,
    buf: Vec<GTxn>,
    cur_win: Option<i64>,
    threads: usize,
    closed: u64,
}

impl GraphScorer {
    /// `engine_cfg`: when the model uses rule features the scorer runs its own engine to fill them in.
    pub fn new(model: CirGnn, engine_cfg: &EngineConfig) -> Self {
        Self::with_context(model, engine_cfg, RiskTable::from_engine(engine_cfg))
    }

    /// As [`new`](Self::new) but with an explicit context, e.g. `RiskTable::from_engine(cfg).with_ownership(map)`.
    pub fn with_context(model: CirGnn, engine_cfg: &EngineConfig, risk: RiskTable) -> Self {
        let feats = FeatureState::new(feature_config(&model), risk);
        let engine = if model.cfg.use_rules {
            let mut e = Engine::new(engine_cfg.clone());
            e.explain = false;
            Some(e)
        } else {
            None
        };
        GraphScorer { model, feats, store: MemStore::default(), engine, buf: Vec::new(), cur_win: None, threads: hcrnn::network::default_threads(), closed: 0 }
    }

    pub fn model(&self) -> &CirGnn {
        &self.model
    }
    pub fn tracked_accounts(&self) -> usize {
        self.store.len()
    }
    pub fn windows_closed(&self) -> u64 {
        self.closed
    }

    /// Add a transaction. Returns the scores of the *previous* window when this transaction opens a new one.
    /// Transactions older than the open window are rejected (the memory is causal).
    pub fn push(&mut self, mut t: GTxn) -> Result<Option<WindowScores>, String> {
        let w = t.ts.div_euclid(self.model.cfg.window_secs);
        let mut done = None;
        match self.cur_win {
            Some(c) if w < c => return Err(format!("transaction is {} window(s) older than the open window", c - w)),
            Some(c) if w > c => done = self.close(),
            _ => {}
        }
        self.cur_win = Some(w);
        if let Some(e) = self.engine.as_mut() {
            let tin = TxnIn {
                ts: t.ts, sender: t.sender, receiver: t.receiver, amount_gbp: t.amount, payment_type: t.ptype,
                sender_country: t.sc, receiver_country: t.rc, cross_currency: t.cross_ccy,
            };
            t.rule = e.process(&tin, t.rf).score as f32 / 100.0;
        }
        self.buf.push(t);
        Ok(done)
    }

    /// Close the open window (e.g. at end of day or on shutdown) and score it.
    pub fn flush(&mut self) -> Option<WindowScores> {
        let r = self.close();
        self.cur_win = None;
        r
    }

    fn close(&mut self) -> Option<WindowScores> {
        if self.buf.is_empty() {
            return None;
        }
        let txns = std::mem::take(&mut self.buf);
        let win = self.cur_win.unwrap_or(0);
        let wd = self.feats.build_window(win, &txns, 0, true);
        let out = self.model.run_chunk(std::slice::from_ref(&wd), &mut self.store, Mode::Eval, self.threads);
        self.closed += 1;
        if self.closed % 30 == 0 {
            self.feats.evict(win, self.model.cfg.evict_windows);
        }
        let mut per_acct: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
        let mut scores = Vec::with_capacity(out.preds.len());
        for (k, p) in out.preds {
            let t = &txns[k];
            scores.push((t.rf, p));
            for a in [t.sender, t.receiver] {
                let e = per_acct.entry(a).or_insert(0.0);
                *e = e.max(p);
            }
        }
        let mut accounts: Vec<(i64, f32)> = per_acct.into_iter().collect();
        accounts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(WindowScores { window: win, scores, accounts })
    }
}
