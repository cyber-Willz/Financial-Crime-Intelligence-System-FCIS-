//! The head fine-tune must (a) move only the head, (b) keep only changes that help on the holdout, (c) refuse thin data.

use aml_engine::default_config;
use aml_feedback::tune::{apply_outcomes, tune_head, TuneConfig};
use aml_graph::rules::score_rules;
use aml_graph::synth::{generate, SynthConfig};
use aml_graph::{CirGnn, CirGnnConfig, Pipeline, RiskTable};
use aml_graph::train::feature_config;
use std::collections::HashMap;

fn small_cfg() -> CirGnnConfig {
    CirGnnConfig { per_hemi: 14, u_dim: 8, o_dim: 6, g_dim: 8, h_dim: 8, warmup_windows: 7, ..Default::default() }
}

fn world() -> (CirGnn, Vec<aml_graph::GTxn>, RiskTable) {
    let mut txns = generate(&SynthConfig { days: 70, individuals: 220, merchants: 12, schemes_per_type: 5, seed: 21 });
    for (i, t) in txns.iter_mut().enumerate() {
        t.rf = i as u32;
    }
    score_rules(&mut txns, &default_config());
    let risk = RiskTable::from_engine(&default_config());
    let mut m = CirGnn::new(small_cfg()).unwrap();
    let fc = feature_config(&m);
    m.fit_norm(Pipeline::new(&txns, fc, risk.clone(), m.cfg.warmup_windows).take(40));
    (m, txns, risk)
}

fn quick() -> TuneConfig {
    TuneConfig { epochs: 3, threads: 1, min_positives: 10, min_holdout_positives: 4, ..Default::default() }
}

#[test]
fn only_the_head_moves_and_holdout_decides() {
    let (mut m, mut txns, risk) = world();
    // analyst outcomes = ground truth for every reviewed transaction (best case for the loop)
    let outcomes: HashMap<u32, bool> = txns.iter().map(|t| (t.rf, t.label)).collect();
    let npos = apply_outcomes(&mut txns, &outcomes);
    assert!(npos > 50, "world has too few positives: {npos}");
    let (rnn0, p10, p30, ph0, mean0, std0) = (m.rnn.params.clone(), m.p1.clone(), m.p3.clone(), m.ph.clone(), m.mean.clone(), m.std.clone());

    let rep = tune_head(&mut m, &txns, &risk, &quick()).unwrap();
    eprintln!("{rep:?}");
    assert!(rep.ap_before.is_finite() && rep.ap_after.is_finite());
    // everything but the head is bit-identical
    assert_eq!(m.rnn.params, rnn0);
    assert_eq!(m.p1, p10);
    assert_eq!(m.p3, p30);
    assert_eq!(m.mean, mean0);
    assert_eq!(m.std, std0);
    if rep.accepted {
        assert!(rep.ap_after >= rep.ap_before + 0.005 - 1e-12);
        assert_ne!(m.ph, ph0);
        assert!(rep.head_change > 0.0);
    } else {
        assert_eq!(m.ph, ph0, "a rejected update must restore the original head exactly");
    }
}

#[test]
fn an_unreachable_gain_rolls_the_head_back_exactly() {
    let (mut m, mut txns, risk) = world();
    let outcomes: HashMap<u32, bool> = txns.iter().map(|t| (t.rf, t.label)).collect();
    apply_outcomes(&mut txns, &outcomes);
    let ph0 = m.ph.clone();
    let cfg = TuneConfig { min_gain: 5.0, ..quick() }; // AP is at most 1
    let rep = tune_head(&mut m, &txns, &risk, &cfg).unwrap();
    assert!(!rep.accepted);
    assert_eq!(m.ph, ph0);
    assert_eq!(rep.head_change, 0.0);
}

#[test]
fn a_helpful_update_is_kept() {
    // random head + true labels: the head alone can learn a lot from fixed features
    let (mut m, mut txns, risk) = world();
    let outcomes: HashMap<u32, bool> = txns.iter().map(|t| (t.rf, t.label)).collect();
    apply_outcomes(&mut txns, &outcomes);
    let cfg = TuneConfig { epochs: 8, lr: 0.01, anchor: 0.0, min_gain: 0.0, ..quick() };
    let rep = tune_head(&mut m, &txns, &risk, &cfg).unwrap();
    eprintln!("{rep:?}");
    assert!(rep.accepted, "{rep:?}");
    assert!(rep.ap_after > rep.ap_before);
}

#[test]
fn thin_feedback_is_refused_without_touching_the_model() {
    let (mut m, mut txns, risk) = world();
    // only three reviewed positives
    let mut outcomes = HashMap::new();
    let mut k = 0;
    for t in &txns {
        if t.label && k < 3 {
            outcomes.insert(t.rf, true);
            k += 1;
        }
    }
    apply_outcomes(&mut txns, &outcomes);
    let ph0 = m.ph.clone();
    let rep = tune_head(&mut m, &txns, &risk, &quick()).unwrap();
    assert!(!rep.accepted && rep.reason.contains("need"), "{rep:?}");
    assert_eq!(m.ph, ph0);
}
