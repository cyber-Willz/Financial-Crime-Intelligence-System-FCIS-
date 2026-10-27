//! Run the rule engine over a stream and attach its scores to the transactions.

use crate::types::GTxn;
use aml_engine::{config::ALL_RULES, Engine, EngineConfig, TxnIn};
use std::collections::BTreeMap;

/// Fills `rule` (causal, score at processing time) and `rule_retro` (after retroactive linking of the
/// earlier legs of a completed pattern, exactly as `aml-eval` computes it).
///
/// `txns` must be time-sorted. Profiles / watchlists can be set on the engine beforehand by passing a
/// prepared engine to [`score_with_engine`].
pub fn score_rules(txns: &mut [GTxn], cfg: &EngineConfig) {
    let mut engine = Engine::new(cfg.clone());
    engine.explain = false;
    score_with_engine(txns, &mut engine);
}

pub fn score_with_engine(txns: &mut [GTxn], engine: &mut Engine) {
    let bit: BTreeMap<&str, u32> = ALL_RULES.iter().enumerate().map(|(i, r)| (*r, 1u32 << i)).collect();
    let n = txns.len();
    let mut keep_causal = vec![1.0f64; n];
    let mut keep_retro = vec![1.0f64; n];
    let mut mask = vec![0u32; n];
    for row in 0..n {
        let t = &txns[row];
        let tin = TxnIn {
            ts: t.ts,
            sender: t.sender,
            receiver: t.receiver,
            amount_gbp: t.amount,
            payment_type: t.ptype,
            sender_country: t.sc,
            receiver_country: t.rc,
            cross_currency: t.cross_ccy,
        };
        let ev = engine.process(&tin, row as u32);
        for h in &ev.hits {
            keep_causal[row] *= 1.0 - h.weight;
            keep_retro[row] *= 1.0 - h.weight;
            mask[row] |= bit[h.rule];
        }
        for l in &ev.linked {
            let b = bit[l.rule];
            let i = l.rf as usize;
            if mask[i] & b == 0 {
                mask[i] |= b;
                keep_retro[i] *= 1.0 - engine.rule_weight(l.rule);
            }
        }
    }
    for i in 0..n {
        txns[i].rule = (1.0 - keep_causal[i]) as f32;
        txns[i].rule_retro = (1.0 - keep_retro[i]) as f32;
    }
}
