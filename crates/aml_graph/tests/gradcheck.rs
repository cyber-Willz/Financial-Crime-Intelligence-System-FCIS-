//! Finite-difference check of every analytic gradient, over all ablation switches.

use aml_engine::default_config;
use aml_graph::rules::score_rules;
use aml_graph::synth::{generate, SynthConfig};
use aml_graph::train::feature_config;
use aml_graph::{CirGnn, CirGnnConfig, Grad, LossCfg, MemStore, Mode, Pipeline, RiskTable, WindowData};
use hcrnn::rng::Rng;

fn setup(cfg: CirGnnConfig) -> (CirGnn, Vec<WindowData>, LossCfg) {
    let mut txns = generate(&SynthConfig { days: 45, individuals: 90, merchants: 8, schemes_per_type: 2, seed: 4 });
    score_rules(&mut txns, &default_config());
    let risk = RiskTable::from_engine(&default_config());
    let mut m = CirGnn::new(cfg).unwrap();
    let fc = feature_config(&m);
    m.fit_norm(Pipeline::new(&txns, fc.clone(), risk.clone(), 0).take(30));
    let wins: Vec<WindowData> = Pipeline::new(&txns, fc, risk, 0).skip(14).take(6).collect();
    // make the weights large enough that nothing sits in the linear regime of tanh
    let mut rng = Rng::new(99);
    for v in m.p1.iter_mut().chain(m.p3.iter_mut()).chain(m.ph.iter_mut()) {
        *v += 0.4 * rng.normal();
    }
    let l = m.rnn.layout.clone();
    for k in l.w_in..l.b {
        m.rnn.params[k] *= 4.0;
    }
    for k in l.b..l.w_out {
        m.rnn.params[k] = 0.3 * rng.normal();
    }
    for k in l.w_out..l.total {
        m.rnn.params[k] += 0.5 * rng.normal();
    }
    (m, wins, LossCfg { pos_weight: 3.0, neg_keep: 0.6, seed: 5 })
}

fn loss(m: &CirGnn, wins: &[WindowData], lc: &LossCfg, threads: usize) -> (f64, Grad) {
    let mut g = Grad::zeros_like(m);
    let mut store = MemStore::default();
    let out = m.run_chunk(wins, &mut store, Mode::Train { grad: &mut g, loss: lc }, threads);
    (out.loss, g)
}

fn check(cfg: CirGnnConfig, label: &str) {
    let (mut m, wins, lc) = setup(cfg);
    let (l0, g) = loss(&m, &wins, &lc, 1);
    assert!(l0.is_finite() && l0 > 0.0, "{label}: loss {l0}");
    let mut rng = Rng::new(7);
    let eps = 1e-5;
    let mut worst: f64 = 0.0;
    let mut checked = 0;
    // (group, length)
    let groups = [(0usize, m.rnn.params.len()), (1, m.p1.len()), (2, m.p3.len()), (3, m.ph.len())];
    for (gi, len) in groups {
        let mut idx: Vec<usize> = (0..45).map(|_| rng.below(len)).collect();
        if gi == 0 {
            // always include the interesting blocks of the connectome RNN: synapses, input, bias, readout
            let l = m.rnn.layout.clone();
            idx.extend([l.theta, l.theta + 1, l.w_in, l.w_in + 2, l.b, l.b + 3, l.w_out, l.w_out + 1, l.b_out]);
        }
        for i in idx {
            let get = |m: &mut CirGnn| -> *mut f64 {
                match gi {
                    0 => &mut m.rnn.params[i],
                    1 => &mut m.p1[i],
                    2 => &mut m.p3[i],
                    _ => &mut m.ph[i],
                }
            };
            let p = get(&mut m);
            let orig = unsafe { *p };
            unsafe { *p = orig + eps };
            let (lp, _) = loss(&m, &wins, &lc, 1);
            unsafe { *get(&mut m) = orig - eps };
            let (lm, _) = loss(&m, &wins, &lc, 1);
            unsafe { *get(&mut m) = orig };
            let num = (lp - lm) / (2.0 * eps);
            let ana = [&g.rnn, &g.p1, &g.p3, &g.ph][gi][i];
            let err = (num - ana).abs() / (num.abs() + ana.abs()).max(1e-7);
            if (num - ana).abs() > 1e-8 {
                worst = worst.max(err);
            }
            assert!(err < 2e-4 || (num - ana).abs() < 1e-8, "{label}: group {gi} param {i}: analytic {ana:.8e} vs numeric {num:.8e}");
            checked += 1;
        }
    }
    println!("{label}: {checked} parameters checked, worst relative error {worst:.2e}");
}

fn base() -> CirGnnConfig {
    CirGnnConfig { per_hemi: 7, density: 0.35, u_dim: 4, o_dim: 3, g_dim: 5, h_dim: 4, warmup_windows: 0, ..Default::default() }
}

#[test]
fn full_model_gradients_match_finite_differences() {
    check(base(), "full");
}

#[test]
fn ablated_models_gradients_match_finite_differences() {
    check(CirGnnConfig { use_memory: false, ..base() }, "no-memory");
    check(CirGnnConfig { use_hypergraph: false, ..base() }, "no-hypergraph");
    check(CirGnnConfig { use_graph: false, ..base() }, "no-graph");
    check(CirGnnConfig { use_hypergraph: false, use_graph: false, ..base() }, "no-message-passing");
    check(CirGnnConfig { max_delay: 0, ..base() }, "no-delays");
}

#[test]
fn threaded_run_matches_single_thread() {
    // large enough windows that the parallel paths are really taken
    let (m, wins, lc) = setup(base());
    let (l1, g1) = loss(&m, &wins, &lc, 1);
    let (l4, g4) = loss(&m, &wins, &lc, 4);
    assert!((l1 - l4).abs() < 1e-12 * l1.abs().max(1.0));
    for (a, b) in g1.rnn.iter().zip(&g4.rnn).chain(g1.p1.iter().zip(&g4.p1)).chain(g1.ph.iter().zip(&g4.ph)) {
        assert!((a - b).abs() < 1e-9 * (1.0 + a.abs()), "{a} vs {b}");
    }
}

#[test]
fn memory_is_causal_and_persists_between_chunks() {
    // Running windows [0..6) in one chunk must equal running [0..3) then [3..6) with the state carried over
    // (up to the f32 rounding of the persisted state).
    let (m, wins, _) = setup(base());
    let mut s1 = MemStore::default();
    let all = m.run_chunk(&wins, &mut s1, Mode::Eval, 1);
    let mut s2 = MemStore::default();
    let a = m.run_chunk(&wins[..3], &mut s2, Mode::Eval, 1);
    let b = m.run_chunk(&wins[3..], &mut s2, Mode::Eval, 1);
    let split: Vec<(usize, f32)> = a.preds.into_iter().chain(b.preds).collect();
    assert_eq!(all.preds.len(), split.len());
    for (x, y) in all.preds.iter().zip(&split) {
        assert_eq!(x.0, y.0);
        assert!((x.1 - y.1).abs() < 1e-4, "{} vs {}", x.1, y.1);
    }
}
