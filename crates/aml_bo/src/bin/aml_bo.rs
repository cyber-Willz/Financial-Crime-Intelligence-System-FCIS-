//! `aml-bo` — beneficial-ownership analysis and its use in the graph model.
//!
//!   aml-bo synth   --out DIR                 # synthetic world: transactions.csv, compliance.json, accounts.csv
//!   aml-bo analyze --compliance F --accounts F [--party ID] [--top N]
//!   aml-bo eval    [--seeds N] [--json-out F]  # graph vs graph+ownership vs graph+ownership+BO rules

use aml_bo::synth::{generate_world, BoSynthConfig};
use aml_bo::{fuse, AccountRegistry, BoConfig, BoMonitor, MonitorConfig, OwnershipAnalysis};
use aml_engine::{default_config, Engine};
use aml_graph::metrics::{at_top_k, average_precision, roc_auc};
use aml_graph::rules::score_with_engine;
use aml_graph::train::{train_model, TrainConfig, TrainReport, RULE_ALERT};
use aml_graph::{CirGnn, CirGnnConfig, GTxn, OwnerInfo, OwnershipMap, RiskTable};
use anyhow::{anyhow, Result};
use cdd_sanctions::engine::Compliance;
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "aml-bo", version, about = "Beneficial ownership analysis for the AML system")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Synth {
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        world: World,
    },
    Analyze {
        #[arg(long)]
        compliance: PathBuf,
        #[arg(long)]
        accounts: PathBuf,
        /// print the full ownership profile of one party
        #[arg(long)]
        party: Option<u64>,
        #[arg(long, default_value_t = 10)]
        top: usize,
    },
    Eval(EvalArgs),
}

#[derive(ClapArgs, Clone)]
struct World {
    #[arg(long, default_value_t = 150)]
    days: i64,
    #[arg(long, default_value_t = 1500)]
    individuals: usize,
    /// schemes per ownership typology
    #[arg(long, default_value_t = 12)]
    schemes: usize,
    #[arg(long, default_value_t = 11)]
    data_seed: u64,
}

#[derive(ClapArgs)]
struct EvalArgs {
    #[command(flatten)]
    world: World,
    #[arg(long, default_value_t = 1)]
    seeds: u64,
    #[arg(long, default_value_t = 10)]
    epochs: usize,
    #[arg(long, default_value_t = 4)]
    patience: usize,
    #[arg(long)]
    threads: Option<usize>,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

fn cfg_of(w: &World) -> BoSynthConfig {
    let mut c = BoSynthConfig::default();
    c.base.days = w.days;
    c.base.individuals = w.individuals;
    c.base.seed = w.data_seed;
    c.schemes = w.schemes;
    c.confounders = w.schemes;
    c
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let sd = if v.len() > 1 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt() } else { 0.0 };
    (m, sd)
}
fn fmt(v: &[f64]) -> String {
    let (m, s) = mean_sd(v);
    if v.len() > 1 { format!("{:>6.2} ±{:<5.2}", 100.0 * m, 100.0 * s) } else { format!("{:>6.2}       ", 100.0 * m) }
}

/// Same accounts, ownership assigned to the wrong accounts: a placebo for "does the model use real ownership".
fn shuffled(m: &OwnershipMap, seed: u64) -> OwnershipMap {
    let mut keys: Vec<i64> = m.iter().map(|(a, _)| *a).collect();
    keys.sort_unstable();
    let infos: Vec<OwnerInfo> = keys.iter().map(|a| m.get(*a).unwrap().clone()).collect();
    let mut idx: Vec<usize> = (0..keys.len()).collect();
    hcrnn::rng::Rng::new(seed).shuffle(&mut idx);
    let mut out = OwnershipMap::new();
    for (k, i) in keys.iter().zip(idx) {
        out.insert(*k, infos[i].clone());
    }
    out
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Synth { out, world } => {
            std::fs::create_dir_all(&out)?;
            let w = generate_world(&cfg_of(&world));
            aml_graph::synth::write_saml_csv_named(&w.txns, &out.join("transactions.csv"), &w.typologies)?;
            w.compliance.save(&out.join("compliance.json")).map_err(|e| anyhow!("{e}"))?;
            w.registry.write_csv(&out.join("accounts.csv"))?;
            println!("wrote {} transactions ({} laundering), {} parties, {} accounts to {}", w.txns.len(), w.txns.iter().filter(|t| t.label).count(), w.compliance.parties().len(), w.registry.len(), out.display());
        }
        Cmd::Analyze { compliance, accounts, party, top } => {
            let c = Compliance::load(&compliance).map_err(|e| anyhow!("{e}"))?;
            let reg = AccountRegistry::from_csv(&accounts)?;
            let an = OwnershipAnalysis::build(&c, reg, BoConfig::default())?;
            println!("{}", serde_json::to_string_pretty(&an.summary)?);
            if let Some(p) = party {
                match an.parties.get(&p) {
                    Some(b) => print!("\n{}", b.render()),
                    None => println!("party {p} holds no account in the registry"),
                }
            } else {
                println!("\nmost opaque account holders:");
                for b in an.top_opaque(top) {
                    print!("{}", b.render());
                }
            }
        }
        Cmd::Eval(a) => eval(a)?,
    }
    Ok(())
}

fn eval(a: EvalArgs) -> Result<()> {
    let w = generate_world(&cfg_of(&a.world));
    let mut txns = w.txns;
    let mut engine = Engine::new(default_config());
    engine.explain = false;
    score_with_engine(&mut txns, &mut engine);
    let an = OwnershipAnalysis::build(&w.compliance, w.registry.clone(), BoConfig::default())?;
    let bo = BoMonitor::score_all(&an, &txns, MonitorConfig::default());
    let mut fused_txns = txns.clone();
    for (t, b) in fused_txns.iter_mut().zip(&bo) {
        t.rule = fuse(t.rule, *b);
    }
    let pos = txns.iter().filter(|t| t.label).count();
    println!("SYNTHETIC world (own generator, not real data): {} txns, {} laundering ({:.2}%), {} parties", txns.len(), pos, 100.0 * pos as f64 / txns.len() as f64, w.compliance.parties().len());
    println!("ownership summary: {}", serde_json::to_string(&an.summary)?);

    let map = an.graph_map();
    let ecfg = default_config();
    let risk_plain = RiskTable::from_engine(&ecfg);
    let risk_own = RiskTable::from_engine(&ecfg).with_ownership(Arc::new(map.clone()));
    let threads = a.threads.unwrap_or_else(hcrnn::network::default_threads);

    type Run<'x> = (&'static str, bool, &'x [GTxn], bool);
    let variants: Vec<Run> = vec![
        ("graph (no ownership)", false, &txns, false),
        ("graph + ownership", true, &txns, false),
        ("graph + ownership + BO rules", true, &fused_txns, false),
        ("placebo: shuffled ownership", true, &txns, true),
    ];
    let mut results: Vec<(&str, Vec<TrainReport>)> = vec![];
    for (name, own, data, placebo) in &variants {
        let mut reps = vec![];
        for s in 0..a.seeds {
            let cfg = CirGnnConfig { use_ownership: *own, seed: 1 + s, ..Default::default() };
            let mut model = CirGnn::new(cfg).map_err(|e| anyhow!(e))?;
            let risk = if !*own { risk_plain.clone() } else if *placebo { RiskTable::from_engine(&ecfg).with_ownership(Arc::new(shuffled(&map, 99 + s))) } else { risk_own.clone() };
            let tc = TrainConfig { epochs: a.epochs, patience: a.patience, threads, seed: 7 + s, ..Default::default() };
            let rep = train_model(&mut model, data, &risk, &tc, |e| eprintln!("  [{name} seed {s}] epoch {:>2} loss {:.4} val AP {:.4} {:.1}s", e.epoch, e.loss, e.val_ap, e.secs)).map_err(|e| anyhow!(e))?;
            eprintln!("  [{name} seed {s}] best epoch {} test AP {:.4}", rep.best_epoch, rep.test.model.ap);
            reps.push(rep);
        }
        results.push((name, reps));
    }

    // baselines on the same test period
    let sp = &results[0].1[0].split;
    let ws = 24 * 3600;
    let mut starts = vec![];
    let mut last = None;
    for (i, t) in txns.iter().enumerate() {
        let d = t.ts.div_euclid(ws);
        if last != Some(d) {
            starts.push(i);
            last = Some(d);
        }
    }
    let from = starts[sp.val_end];
    let r0 = &results[0].1[0].test;
    let k = r0.online_alerts;
    let base = |f: &dyn Fn(usize) -> f32| -> (f64, f64, f64, f64) {
        let sc: Vec<(f32, bool)> = (from..txns.len()).map(|i| (f(i), txns[i].label)).collect();
        let o = at_top_k(&sc, k);
        (average_precision(&sc), roc_auc(&sc), o.precision, o.recall)
    };
    println!("\ntest period: last {} windows, {} txns, {} laundering; k = {} (the online rule engine's alert count)\n", sp.windows - sp.val_end, txns.len() - from, txns[from..].iter().filter(|t| t.label).count(), k);
    println!("{:<34} {:>14} {:>14} {:>14} {:>14}", "TEST", "AP", "ROC-AUC", "precision@k", "recall@k");
    for (name, f) in [
        ("rule engine, online", Box::new(|i: usize| txns[i].rule) as Box<dyn Fn(usize) -> f32>),
        ("BO rules only", Box::new(|i: usize| bo[i])),
        ("rule engine + BO rules (noisy-OR)", Box::new(|i: usize| fuse(txns[i].rule, bo[i]))),
    ] {
        let (ap, auc, p, r) = base(&*f);
        println!("{:<34} {:>13.2}% {:>14.3} {:>13.2}% {:>13.2}%", name, 100.0 * ap, auc, 100.0 * p, 100.0 * r);
    }
    for (name, reps) in &results {
        let g = |f: &dyn Fn(&TrainReport) -> f64| fmt(&reps.iter().map(f).collect::<Vec<_>>());
        println!(
            "{:<34} {:>14} {:>14.3} {:>14} {:>14}",
            format!("model: {name}"),
            g(&|r| r.test.model.ap),
            mean_sd(&reps.iter().map(|r| r.test.model.auc).collect::<Vec<_>>()).0,
            g(&|r| r.test.model.at_online_volume.precision),
            g(&|r| r.test.model.at_online_volume.recall)
        );
    }
    println!("(percent; AUC as a fraction; {} seed(s), mean ± sd; for the +BO row the engine row above is the plain engine)", a.seeds);

    println!("\nrecall by typology: rule engine / BO rules / engine+BO at their alert threshold ({RULE_ALERT:.3}); models at k = {k}");
    print!("{:<26} {:>6} {:>8} {:>8} {:>8}", "typology", "n", "engine", "BO", "both");
    for (n, _) in &results {
        print!(" {:>12}", &n.chars().take(12).collect::<String>());
    }
    println!();
    for ty in 1..w.typologies.len() as u8 {
        let idx: Vec<usize> = (from..txns.len()).filter(|&i| txns[i].typology == ty).collect();
        if idx.is_empty() {
            continue;
        }
        let rec = |f: &dyn Fn(usize) -> f32| idx.iter().filter(|&&i| f(i) >= RULE_ALERT).count() as f64 / idx.len() as f64;
        print!("{:<26} {:>6} {:>7.1}% {:>7.1}% {:>7.1}%", w.typologies[ty as usize], idx.len(), 100.0 * rec(&|i| txns[i].rule), 100.0 * rec(&|i| bo[i]), 100.0 * rec(&|i| fuse(txns[i].rule, bo[i])));
        for (_, reps) in &results {
            let v: Vec<f64> = reps.iter().filter_map(|r| r.test.typology.iter().find(|x| x.typology == ty)).map(|x| x.model_at_online_volume).collect();
            print!(" {:>11.1}%", 100.0 * mean_sd(&v).0);
        }
        println!();
    }
    if let Some(p) = &a.json_out {
        let j: Vec<_> = results.iter().map(|(n, r)| serde_json::json!({"variant": n, "runs": r})).collect();
        std::fs::write(p, serde_json::to_vec_pretty(&serde_json::json!({"summary": an.summary, "results": j}))?)?;
    }
    Ok(())
}
