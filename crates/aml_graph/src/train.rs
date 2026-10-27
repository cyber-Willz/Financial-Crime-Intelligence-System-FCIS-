//! Training and evaluation on a time-ordered transaction stream.
//!
//! * The stream is cut into windows; the first `train_frac` of the windows train the model, the next
//!   `val_frac` select the epoch (early stopping on validation average precision), the rest are the
//!   untouched test period. The split is by time, never random, so no future information is used.
//! * Training is truncated BPTT: `chunk` consecutive windows are run forward and backward together, the
//!   per-account memory state is carried across chunks (detached), and one Adam step is taken per chunk.

use crate::features::{FeatureConfig, Pipeline, RiskTable, WindowData};
use crate::metrics::{self, Operating};
use crate::model::{CirGnn, Grad, LossCfg, MemStore, Mode};
use crate::types::GTxn;
use hcrnn::optim::Adam;
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub chunk: usize,
    pub lr: f64,
    pub lr_memory: f64,
    pub clip: f64,
    /// <= 0 means automatic
    pub pos_weight: f64,
    pub neg_keep: f64,
    pub train_frac: f64,
    pub val_frac: f64,
    pub threads: usize,
    pub patience: usize,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            epochs: 12,
            chunk: 8,
            lr: 0.004,
            lr_memory: 0.004,
            clip: 5.0,
            pos_weight: 0.0,
            neg_keep: 0.15,
            train_frac: 0.6,
            val_frac: 0.15,
            threads: hcrnn::network::default_threads(),
            patience: 4,
            seed: 7,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EpochLog {
    pub epoch: usize,
    pub loss: f64,
    pub val_ap: f64,
    pub grad_norm: f64,
    pub radius: f64,
    pub secs: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Split {
    pub windows: usize,
    pub train_end: usize,
    pub val_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ScoreSummary {
    pub ap: f64,
    pub auc: f64,
    pub best_f1: Operating,
    /// operating point when exactly as many alerts are raised as the *online* rule engine raises
    pub at_online_volume: Operating,
    /// ... as the rule engine raises after retroactive linking
    pub at_retro_volume: Operating,
}

#[derive(Debug, Clone, Serialize)]
pub struct TypologyRecall {
    pub typology: u8,
    pub n: usize,
    pub rules_online: f64,
    pub rules_retro: f64,
    /// model recall when it alerts the same number of transactions as the online rule engine
    pub model_at_online_volume: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct TestReport {
    pub txns: usize,
    pub positives: usize,
    pub base_rate: f64,
    pub online_alerts: usize,
    pub retro_alerts: usize,
    pub rules_online: ScoreSummary,
    pub rules_retro: ScoreSummary,
    pub model: ScoreSummary,
    pub typology: Vec<TypologyRecall>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainReport {
    pub split: Split,
    pub epochs: Vec<EpochLog>,
    pub best_epoch: usize,
    pub best_val_ap: f64,
    pub test: TestReport,
}

pub fn window_starts(txns: &[GTxn], window_secs: i64) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut last = None;
    for (i, t) in txns.iter().enumerate() {
        let w = t.ts.div_euclid(window_secs);
        if last != Some(w) {
            starts.push(i);
            last = Some(w);
        }
    }
    starts
}

pub fn feature_config(m: &CirGnn) -> FeatureConfig {
    FeatureConfig {
        window_secs: m.cfg.window_secs,
        min_star: m.cfg.min_star,
        horizon: m.cfg.horizon,
        nb_steps: m.cfg.nb_steps,
        use_rules: m.cfg.use_rules,
        use_ownership: m.cfg.use_ownership,
        ..Default::default()
    }
}

/// Run the model forward over windows `[0, upto)` of the stream (carrying memory), recording predictions
/// of counted windows with index >= `record_from`.
pub fn forward_pass(m: &CirGnn, txns: &[GTxn], risk: &RiskTable, threads: usize, chunk: usize, record_from: usize, upto: usize) -> Vec<(usize, f32)> {
    let mut store = MemStore::default();
    let mut pipe = Pipeline::new(txns, feature_config(m), risk.clone(), m.cfg.warmup_windows);
    let mut preds = Vec::new();
    let mut buf: Vec<WindowData> = Vec::with_capacity(chunk);
    let mut wi = 0usize;
    loop {
        let next = if wi < upto { pipe.next() } else { None };
        let done = next.is_none();
        if let Some(w) = next {
            buf.push(w);
            wi += 1;
        }
        if buf.len() == chunk || (done && !buf.is_empty()) {
            let first = wi - buf.len();
            let out = m.run_chunk(&buf, &mut store, Mode::Eval, threads);
            // keep predictions only for windows >= record_from
            let keep_from = record_from.saturating_sub(first);
            if keep_from == 0 {
                preds.extend(out.preds);
            } else if keep_from < buf.len() {
                let cut: usize = buf[keep_from].txn_idx.first().copied().unwrap_or(usize::MAX);
                preds.extend(out.preds.into_iter().filter(|p| p.0 >= cut));
            }
            buf.clear();
        }
        if done {
            break;
        }
    }
    preds
}

pub fn summarize(scored: &[(f32, bool)], k_online: usize, k_retro: usize) -> ScoreSummary {
    ScoreSummary {
        ap: metrics::average_precision(scored),
        auc: metrics::roc_auc(scored),
        best_f1: metrics::best_f1(scored),
        at_online_volume: metrics::at_top_k(scored, k_online),
        at_retro_volume: metrics::at_top_k(scored, k_retro),
    }
}

/// Train `model` on `txns` (time-sorted, with `rule` scores filled in if `use_rules`) and evaluate on the
/// held-out final part of the stream.
pub fn train_model(model: &mut CirGnn, txns: &[GTxn], risk: &RiskTable, tc: &TrainConfig, mut log: impl FnMut(&EpochLog)) -> Result<TrainReport, String> {
    let starts = window_starts(txns, model.cfg.window_secs);
    let nw = starts.len();
    if nw < model.cfg.warmup_windows + 12 {
        return Err(format!("only {nw} windows: need at least warm-up ({}) + 12", model.cfg.warmup_windows));
    }
    let train_end = ((nw as f64) * tc.train_frac) as usize;
    let val_end = ((nw as f64) * (tc.train_frac + tc.val_frac)) as usize;
    if train_end <= model.cfg.warmup_windows + 2 || val_end <= train_end || val_end >= nw {
        return Err("train/val/test split leaves an empty part".into());
    }
    let split = Split { windows: nw, train_end, val_end };
    let fcfg = feature_config(model);

    // feature standardisation from the training period only
    let pipe = Pipeline::new(txns, fcfg.clone(), risk.clone(), model.cfg.warmup_windows);
    model.fit_norm(pipe.take(train_end));

    // class weight
    let (mut pos, mut tot) = (0usize, 0usize);
    let cut = starts.get(train_end).copied().unwrap_or(txns.len());
    let first_counted = starts.get(model.cfg.warmup_windows).copied().unwrap_or(0);
    for t in &txns[first_counted..cut] {
        tot += 1;
        pos += t.label as usize;
    }
    if pos == 0 {
        return Err("no laundering transactions in the training period".into());
    }
    let pw = if tc.pos_weight > 0.0 { tc.pos_weight } else { ((tot - pos) as f64 * tc.neg_keep / pos as f64 * 0.3).clamp(1.0, 50.0) };
    let loss_cfg = LossCfg { pos_weight: pw, neg_keep: tc.neg_keep, seed: tc.seed };

    let mut adam_rnn = Adam::new(model.rnn.params.len(), tc.lr_memory, 0.0);
    let mut adam_p1 = Adam::new(model.p1.len(), tc.lr, 0.0);
    let mut adam_p3 = Adam::new(model.p3.len(), tc.lr, 0.0);
    let mut adam_ph = Adam::new(model.ph.len(), tc.lr, 0.0);

    let mut logs: Vec<EpochLog> = Vec::new();
    let mut best = (model.snapshot(), f64::NEG_INFINITY, 0usize);
    let mut bad = 0usize;
    for epoch in 1..=tc.epochs {
        let t0 = Instant::now();
        let mut store = MemStore::default();
        let mut pipe = Pipeline::new(txns, fcfg.clone(), risk.clone(), model.cfg.warmup_windows);
        let mut buf: Vec<WindowData> = Vec::with_capacity(tc.chunk);
        let (mut loss_sum, mut loss_w, mut gnorm_sum, mut steps) = (0.0, 0.0, 0.0, 0usize);
        let mut seen = 0usize;
        while seen < train_end {
            match pipe.next() {
                Some(w) => {
                    buf.push(w);
                    seen += 1;
                }
                None => break,
            }
            if buf.len() == tc.chunk || seen == train_end {
                let mut grad = Grad::zeros_like(model);
                let out = model.run_chunk(&buf, &mut store, Mode::Train { grad: &mut grad, loss: &loss_cfg }, tc.threads);
                buf.clear();
                if out.weight <= 0.0 {
                    continue;
                }
                let gn = grad.norm();
                if !gn.is_finite() {
                    return Err("non-finite gradient".into());
                }
                let scale = if tc.clip > 0.0 && gn > tc.clip { tc.clip / gn } else { 1.0 };
                for g in grad.rnn.iter_mut().chain(grad.p1.iter_mut()).chain(grad.p3.iter_mut()).chain(grad.ph.iter_mut()) {
                    *g *= scale;
                }
                adam_rnn.step(&mut model.rnn.params, &grad.rnn);
                adam_p1.step(&mut model.p1, &grad.p1);
                adam_p3.step(&mut model.p3, &grad.p3);
                adam_ph.step(&mut model.ph, &grad.ph);
                loss_sum += out.loss * out.weight;
                loss_w += out.weight;
                gnorm_sum += gn;
                steps += 1;
            }
        }
        let radius = if model.cfg.radius_cap > 0.0 { model.rnn.enforce_radius_cap(model.cfg.radius_cap) } else { model.rnn.spectral_radius() };

        // validation: replay the stream with the updated weights and score the validation windows
        let preds = forward_pass(model, txns, risk, tc.threads, tc.chunk, train_end, val_end);
        let scored: Vec<(f32, bool)> = preds.iter().map(|&(i, p)| (p, txns[i].label)).collect();
        let val_ap = metrics::average_precision(&scored);
        let rec = EpochLog { epoch, loss: loss_sum / loss_w.max(1e-12), val_ap, grad_norm: gnorm_sum / steps.max(1) as f64, radius, secs: t0.elapsed().as_secs_f64() };
        log(&rec);
        logs.push(rec);
        if val_ap > best.1 {
            best = (model.snapshot(), val_ap, epoch);
            bad = 0;
        } else {
            bad += 1;
            if bad >= tc.patience {
                break;
            }
        }
    }
    model.restore(&best.0);
    model.rnn.check_constraints()?;

    let test = evaluate(model, txns, risk, tc.threads, tc.chunk, val_end, nw, &starts)?;
    Ok(TrainReport { split, epochs: logs, best_epoch: best.2, best_val_ap: best.1, test })
}

/// Score windows `[from, upto)` (after replaying everything before them to build memory) and compare
/// with the rule engine at equal alert volume.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(m: &CirGnn, txns: &[GTxn], risk: &RiskTable, threads: usize, chunk: usize, from: usize, upto: usize, _starts: &[usize]) -> Result<TestReport, String> {
    let preds = forward_pass(m, txns, risk, threads, chunk, from, upto);
    if preds.is_empty() {
        return Err("no transactions in the evaluation period".into());
    }
    let model_scored: Vec<(f32, bool)> = preds.iter().map(|&(i, p)| (p, txns[i].label)).collect();
    let online: Vec<(f32, bool)> = preds.iter().map(|&(i, _)| (txns[i].rule, txns[i].label)).collect();
    let retro: Vec<(f32, bool)> = preds.iter().map(|&(i, _)| (txns[i].rule_retro, txns[i].label)).collect();
    let k_online = online.iter().filter(|s| s.0 >= RULE_ALERT).count();
    let k_retro = retro.iter().filter(|s| s.0 >= RULE_ALERT).count();
    let positives = model_scored.iter().filter(|s| s.1).count();

    let mut order_m: Vec<usize> = (0..preds.len()).collect();
    order_m.sort_by(|&a, &b| preds[b].1.partial_cmp(&preds[a].1).unwrap_or(std::cmp::Ordering::Equal));
    let model_flag: std::collections::HashSet<usize> = order_m.iter().take(k_online).map(|&j| preds[j].0).collect();
    let mut by_typ: std::collections::BTreeMap<u8, (usize, usize, usize, usize)> = Default::default();
    for &(i, _) in &preds {
        let t = &txns[i];
        if t.typology > 0 {
            let e = by_typ.entry(t.typology).or_default();
            e.0 += 1;
            e.1 += (t.rule >= RULE_ALERT) as usize;
            e.2 += (t.rule_retro >= RULE_ALERT) as usize;
            e.3 += model_flag.contains(&i) as usize;
        }
    }
    Ok(TestReport {
        txns: preds.len(),
        positives,
        base_rate: positives as f64 / preds.len() as f64,
        online_alerts: k_online,
        retro_alerts: k_retro,
        rules_online: summarize(&online, k_online, k_retro),
        rules_retro: summarize(&retro, k_online, k_retro),
        model: summarize(&model_scored, k_online, k_retro),
        typology: by_typ
            .into_iter()
            .map(|(typology, (n, a, b, c))| TypologyRecall {
                typology,
                n,
                rules_online: a as f64 / n as f64,
                rules_retro: b as f64 / n as f64,
                model_at_online_volume: c as f64 / n as f64,
            })
            .collect(),
    })
}

/// Default engine alert threshold (60) on the 0..1 scale of `GTxn::rule` (scores are rounded to integers).
pub const RULE_ALERT: f32 = 0.595;
