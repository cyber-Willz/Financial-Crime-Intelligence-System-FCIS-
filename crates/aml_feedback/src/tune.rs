//! "Update Graph Attention": guarded fine-tuning of the graph model's *head* on analyst outcomes.
//!
//! The head `p = σ(v·tanh(Ws g_s + Wr g_r + We f + b) + c)` is the pairwise interaction scorer that decides
//! how much a sender's and a receiver's graph embedding, together with the edge features, count toward a
//! transaction's risk (the nearest thing this model has to attention; it contains no softmax attention).
//! Only these parameters move. The connectome RNN, both graph layers and the feature standardisation are
//! untouched, so every stored per-account memory state stays valid and no history has to be replayed
//! after an update.
//!
//! Safeguards, all enforced here and covered by tests:
//! * L2 anchor to the original head (`anchor`), small learning rate, few epochs;
//! * the most recent `holdout_frac` of windows is never trained on; the update is **kept only if** average
//!   precision on that holdout improves by at least `min_gain`, otherwise the original weights are restored;
//! * refused outright when there are too few analyst-confirmed positives in either part;
//! * non-finite values anywhere abort and restore.
//!
//! Label caveat: outcomes exist only for alerts that were reviewed. `apply_outcomes` marks reviewed-confirmed
//! transactions positive and everything else negative, which is the standard "positive-unlabelled" shortcut;
//! unreviewed laundering is silently treated as clean, so keep `neg_keep` low and treat gains as indicative.

use aml_graph::features::{Pipeline, RiskTable};
use aml_graph::metrics::average_precision;
use aml_graph::model::{CirGnn, Grad, LossCfg, MemStore, Mode};
use aml_graph::train::{feature_config, forward_pass, window_starts};
use aml_graph::types::GTxn;
use hcrnn::optim::Adam;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneConfig {
    pub epochs: usize,
    pub lr: f64,
    /// strength of the pull back to the original head parameters
    pub anchor: f64,
    pub holdout_frac: f64,
    /// minimum analyst-confirmed positive transactions in the training part
    pub min_positives: usize,
    /// minimum in the holdout part
    pub min_holdout_positives: usize,
    /// required improvement in holdout average precision
    pub min_gain: f64,
    pub neg_keep: f64,
    pub chunk: usize,
    pub clip: f64,
    pub threads: usize,
    pub seed: u64,
}

impl Default for TuneConfig {
    fn default() -> Self {
        TuneConfig {
            epochs: 4,
            lr: 0.002,
            anchor: 0.5,
            holdout_frac: 0.3,
            min_positives: 20,
            min_holdout_positives: 8,
            min_gain: 0.005,
            neg_keep: 0.15,
            chunk: 8,
            clip: 5.0,
            threads: hcrnn::network::default_threads(),
            seed: 17,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneReport {
    pub accepted: bool,
    pub reason: String,
    pub ap_before: f64,
    pub ap_after: f64,
    pub train_positives: usize,
    pub holdout_positives: usize,
    pub windows: usize,
    /// ‖Δhead‖ / ‖head‖ of the *kept* result (0 when rejected)
    pub head_change: f64,
}

impl TuneReport {
    fn refused(reason: impl Into<String>, windows: usize, trp: usize, hop: usize) -> Self {
        TuneReport { accepted: false, reason: reason.into(), ap_before: 0.0, ap_after: 0.0, train_positives: trp, holdout_positives: hop, windows, head_change: 0.0 }
    }
}

/// Set labels from analyst outcomes keyed by the caller reference `rf`: reviewed-positive -> `true`,
/// everything else (reviewed-negative and unreviewed) -> `false`. Returns the number of positives.
pub fn apply_outcomes(txns: &mut [GTxn], outcomes: &HashMap<u32, bool>) -> usize {
    let mut n = 0;
    for t in txns.iter_mut() {
        t.label = outcomes.get(&t.rf).copied().unwrap_or(false);
        n += t.label as usize;
    }
    n
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Fine-tune `model.ph` on `txns` (time-ordered, labels already set from outcomes). Returns a report;
/// `model` is left unchanged unless `accepted`.
pub fn tune_head(model: &mut CirGnn, txns: &[GTxn], risk: &RiskTable, cfg: &TuneConfig) -> Result<TuneReport, String> {
    let starts = window_starts(txns, model.cfg.window_secs);
    let nw = starts.len();
    let warm = model.cfg.warmup_windows;
    if nw < warm + 8 {
        return Ok(TuneReport::refused(format!("only {nw} windows; need at least {} (warm-up {warm} + 8)", warm + 8), nw, 0, 0));
    }
    let hold = ((nw as f64) * cfg.holdout_frac).ceil() as usize;
    let h = nw.saturating_sub(hold).max(warm + 3);
    if h >= nw {
        return Ok(TuneReport::refused("holdout leaves no training windows", nw, 0, 0));
    }
    let cut = starts[h];
    let first_counted = starts.get(warm).copied().unwrap_or(0);
    let trp = txns[first_counted..cut].iter().filter(|t| t.label).count();
    let hop = txns[cut..].iter().filter(|t| t.label).count();
    if trp < cfg.min_positives {
        return Ok(TuneReport::refused(format!("{trp} analyst-confirmed transactions in the training windows; need {}", cfg.min_positives), nw, trp, hop));
    }
    if hop < cfg.min_holdout_positives {
        return Ok(TuneReport::refused(format!("{hop} analyst-confirmed transactions in the holdout windows; need {}", cfg.min_holdout_positives), nw, trp, hop));
    }

    let ap_of = |m: &CirGnn| -> f64 {
        let preds = forward_pass(m, txns, risk, cfg.threads, cfg.chunk, h, nw);
        let scored: Vec<(f32, bool)> = preds.iter().map(|&(i, p)| (p, txns[i].label)).collect();
        average_precision(&scored)
    };
    let ap_before = ap_of(model);
    let snap = model.snapshot();
    let ph0 = model.ph.clone();

    let tot = txns[first_counted..cut].len();
    let pw = (((tot - trp) as f64) * cfg.neg_keep / trp as f64 * 0.3).clamp(1.0, 50.0);
    let loss_cfg = LossCfg { pos_weight: pw, neg_keep: cfg.neg_keep, seed: cfg.seed };
    let mut adam = Adam::new(model.ph.len(), cfg.lr, 0.0);
    let fcfg = feature_config(model);

    for _epoch in 0..cfg.epochs {
        let mut store = MemStore::default();
        let mut pipe = Pipeline::new(txns, fcfg.clone(), risk.clone(), warm);
        let mut buf = Vec::with_capacity(cfg.chunk);
        let mut seen = 0usize;
        while seen < h {
            match pipe.next() {
                Some(w) => {
                    buf.push(w);
                    seen += 1;
                }
                None => break,
            }
            if buf.len() == cfg.chunk || seen == h {
                let mut grad = Grad::zeros_like(model);
                let out = model.run_chunk(&buf, &mut store, Mode::Train { grad: &mut grad, loss: &loss_cfg }, cfg.threads);
                buf.clear();
                if out.weight <= 0.0 {
                    continue;
                }
                let mut gh = grad.ph;
                for (g, (p, p0)) in gh.iter_mut().zip(model.ph.iter().zip(&ph0)) {
                    *g += cfg.anchor * (p - p0);
                }
                if !gh.iter().all(|g| g.is_finite()) {
                    model.restore(&snap);
                    return Ok(TuneReport::refused("non-finite gradient; original model restored", nw, trp, hop));
                }
                let n = norm(&gh);
                if cfg.clip > 0.0 && n > cfg.clip {
                    for g in gh.iter_mut() {
                        *g *= cfg.clip / n;
                    }
                }
                adam.step(&mut model.ph, &gh);
            }
        }
    }

    if !model.ph.iter().all(|x| x.is_finite()) {
        model.restore(&snap);
        return Ok(TuneReport::refused("non-finite head after tuning; original model restored", nw, trp, hop));
    }
    let ap_after = ap_of(model);
    let change = {
        let d: Vec<f64> = model.ph.iter().zip(&ph0).map(|(a, b)| a - b).collect();
        norm(&d) / norm(&ph0).max(1e-12)
    };
    if ap_after.is_finite() && ap_after >= ap_before + cfg.min_gain {
        Ok(TuneReport {
            accepted: true,
            reason: format!("holdout AP {:.3} -> {:.3}", ap_before, ap_after),
            ap_before,
            ap_after,
            train_positives: trp,
            holdout_positives: hop,
            windows: nw,
            head_change: change,
        })
    } else {
        model.restore(&snap);
        Ok(TuneReport {
            accepted: false,
            reason: format!("holdout AP {:.3} -> {:.3} does not clear the required gain of {:.3}; original head kept", ap_before, ap_after, cfg.min_gain),
            ap_before,
            ap_after,
            train_positives: trp,
            holdout_positives: hop,
            windows: nw,
            head_change: 0.0,
        })
    }
}
