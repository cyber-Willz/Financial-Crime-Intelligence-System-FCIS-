//! `aml-gnn` — train, ablate and run the connectome-informed recurrent graph neural network.
//!
//!   aml-gnn synth   --out synth.csv                       # synthetic SAML-D-format data
//!   aml-gnn eval    --synth                               # train + ablations on the synthetic world
//!   aml-gnn eval    --csv SAML-D.csv --format saml-d --limit 3000000 --variants full,no-memory,rewired
//!   aml-gnn score   --model m.json --csv new.csv          # stream a CSV through a saved model
//!   aml-gnn wiring  [--tvb DIR]                           # topology of the memory wiring and its nulls

use aml_datasets::{Format, Item};
use aml_engine::{default_config as builtin_config, Engine, EngineConfig, Profile};
use std::sync::OnceLock;

static ENGINE_CFG: OnceLock<EngineConfig> = OnceLock::new();

/// Engine configuration used for rule features: `--engine-config` (e.g. profiles/amlsim_fitted.json) or the built-in defaults.
fn default_config() -> EngineConfig {
    ENGINE_CFG.get().cloned().unwrap_or_else(builtin_config)
}

fn set_engine_config(path: &Option<PathBuf>) -> Result<()> {
    if let Some(p) = path {
        let cfg: EngineConfig = serde_json::from_slice(&std::fs::read(p)?).map_err(|e| anyhow!("parsing {}: {e}", p.display()))?;
        let _ = ENGINE_CFG.set(cfg);
    }
    Ok(())
}
use aml_graph::rules::score_with_engine;
use aml_graph::synth::{self, SynthConfig, TYPOLOGIES};
use aml_graph::train::{train_model, TestReport, TrainConfig, TrainReport};
use aml_graph::{CirGnn, CirGnnConfig, GTxn, GraphScorer, RiskTable};
use anyhow::{anyhow, bail, Context, Result};
use clap::{Args as ClapArgs, Parser, Subcommand};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "aml-gnn", version, about = "Connectome-informed recurrent graph neural network for AML risk")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a synthetic stream in SAML-D CSV layout.
    Synth {
        #[arg(long)]
        out: PathBuf,
        #[command(flatten)]
        world: World,
    },
    /// Train on a time-ordered stream, evaluate on its final part and compare variants.
    Eval(EvalArgs),
    /// Stream a CSV through a saved model and report the riskiest accounts (and metrics if labelled).
    Score(ScoreArgs),
    /// Print topology diagnostics of the connectome wiring and its null models.
    Wiring {
        #[arg(long)]
        tvb: Option<String>,
        #[arg(long, default_value_t = 33)]
        per_hemi: usize,
        #[arg(long, default_value_t = 0.2)]
        density: f64,
    },
}

#[derive(ClapArgs, Clone)]
struct World {
    #[arg(long, default_value_t = 150)]
    days: i64,
    #[arg(long, default_value_t = 2500)]
    individuals: usize,
    #[arg(long, default_value_t = 14)]
    schemes: usize,
    #[arg(long, default_value_t = 11)]
    data_seed: u64,
}

#[derive(ClapArgs)]
struct EvalArgs {
    /// Labelled CSV (see --format). Without it the synthetic world is used.
    #[arg(long)]
    csv: Option<PathBuf>,
    #[arg(long, default_value = "saml-d")]
    format: String,
    #[arg(long)]
    limit: Option<usize>,
    #[command(flatten)]
    world: World,
    /// full,tabular,no-memory,no-hypergraph,no-graph,no-rules,rewired,random,dense
    #[arg(long, value_delimiter = ',', default_value = "full,no-memory,no-hypergraph,no-graph,no-rules")]
    variants: Vec<String>,
    #[arg(long, default_value_t = 1)]
    seeds: u64,
    #[arg(long, default_value_t = 12)]
    epochs: usize,
    #[arg(long, default_value_t = 8)]
    chunk: usize,
    #[arg(long, default_value_t = 0.004)]
    lr: f64,
    #[arg(long, default_value_t = 4)]
    patience: usize,
    #[arg(long)]
    threads: Option<usize>,
    /// Window length in hours (24 = daily snapshots).
    #[arg(long, default_value_t = 24)]
    window_hours: i64,
    #[arg(long, default_value_t = 14)]
    warmup_windows: usize,
    /// Regions per hemisphere of the synthetic connectome (33 = 66 regions, like TVB-66).
    #[arg(long, default_value_t = 33)]
    per_hemi: usize,
    /// Directory with a TVB-style connectome (weights.txt, tract_lengths.txt, centres.txt); overrides --per-hemi.
    #[arg(long)]
    tvb: Option<String>,
    #[arg(long, default_value_t = 3)]
    max_delay: usize,
    /// Save the first variant's first-seed model here.
    #[arg(long)]
    save: Option<PathBuf>,
    /// EngineConfig JSON for the rule features (e.g. profiles/amlsim_fitted.json).
    #[arg(long)]
    engine_config: Option<PathBuf>,
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(ClapArgs)]
struct ScoreArgs {
    #[arg(long)]
    engine_config: Option<PathBuf>,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    csv: PathBuf,
    #[arg(long, default_value = "saml-d")]
    format: String,
    #[arg(long)]
    limit: Option<usize>,
    #[arg(long, default_value_t = 20)]
    top: usize,
    /// Write per-transaction risk (row, risk, label) here.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn synth_stream(w: &World) -> Vec<GTxn> {
    synth::generate(&SynthConfig { days: w.days, individuals: w.individuals, merchants: 60, schemes_per_type: w.schemes, seed: w.data_seed })
}

/// Read a labelled CSV through the dataset adapters; returns time-sorted transactions and typology names.
fn load_csv(path: &PathBuf, format: &str, limit: Option<usize>) -> Result<(Vec<GTxn>, Vec<String>, Engine)> {
    let fmt = Format::parse(format)?;
    let mut engine = Engine::new(default_config());
    engine.explain = false;
    let mut names = vec!["(normal)".to_string()];
    let mut ids = std::collections::BTreeMap::new();
    let (mut out, mut bad) = (Vec::new(), 0usize);
    for item in aml_datasets::open(path, fmt, limit).with_context(|| format!("reading {}", path.display()))? {
        match item {
            Item::Customer(c) => {
                engine.set_profile(c.account, &Profile { risk: c.risk, annual_income_gbp: c.annual_income_gbp, pep: c.pep, trade_exposed: c.trade_exposed() });
                if c.sanctioned {
                    engine.set_watch(c.account, true);
                }
            }
            Item::Bad => bad += 1,
            Item::Row(r) => {
                let mut t = GTxn::from_row(&r, out.len() as u32);
                if r.label == Some(true) {
                    let name = r.typology.clone().unwrap_or_else(|| "(positive)".into());
                    let next = (ids.len() + 1) as u8;
                    let id = *ids.entry(name.clone()).or_insert_with(|| {
                        names.push(name);
                        next
                    });
                    t.typology = id;
                }
                out.push(t);
            }
        }
    }
    if bad > 0 {
        eprintln!("note: {bad} rows could not be mapped and were skipped");
    }
    if out.is_empty() {
        bail!("no usable rows in {}", path.display());
    }
    out.sort_by_key(|t| t.ts);
    for (i, t) in out.iter_mut().enumerate() {
        t.rf = i as u32;
    }
    Ok((out, names, engine))
}

fn variant_cfg(name: &str, base: &CirGnnConfig) -> Result<CirGnnConfig> {
    let mut c = base.clone();
    match name {
        "full" => {}
        "tabular" => {
            c.use_memory = false;
            c.use_hypergraph = false;
            c.use_graph = false;
        }
        "no-memory" => c.use_memory = false,
        "no-hypergraph" => c.use_hypergraph = false,
        "no-graph" => c.use_graph = false,
        "no-rules" => c.use_rules = false,
        "rewired" => c.wiring = "degree-preserving".into(),
        "random" => c.wiring = "density-matched".into(),
        "dense" => c.wiring = "dense".into(),
        other => bail!("unknown variant '{other}'"),
    }
    Ok(c)
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
    let sd = if v.len() > 1 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt() } else { 0.0 };
    (m, sd)
}

fn fmt_ms(v: &[f64], pct: bool) -> String {
    let (m, s) = mean_sd(v);
    let k = if pct { 100.0 } else { 1.0 };
    if v.len() > 1 { format!("{:>6.2} ±{:<5.2}", m * k, s * k) } else { format!("{:>6.2}        ", m * k) }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Synth { out, world } => {
            let t = synth_stream(&world);
            synth::write_saml_csv(&t, &out)?;
            println!("wrote {} transactions ({} laundering) to {}", t.len(), t.iter().filter(|x| x.label).count(), out.display());
        }
        Cmd::Wiring { tvb, per_hemi, density } => wiring(tvb, per_hemi, density)?,
        Cmd::Eval(a) => eval(a)?,
        Cmd::Score(a) => score(a)?,
    }
    Ok(())
}

fn wiring(tvb: Option<String>, per_hemi: usize, density: f64) -> Result<()> {
    use hcrnn::topology::topology_report;
    println!("{:<18} {:>7} {:>6} {:>9} {:>8} {:>10} {:>9} {:>8} {:>10}", "wiring", "regions", "edges", "mean_deg", "rho_B", "||A||/rho_B", "fiedler", "synapses", "radius");
    for w in ["connectome", "degree-preserving", "density-matched", "dense"] {
        let cfg = CirGnnConfig { wiring: w.into(), tvb_dir: tvb.clone(), per_hemi, density, ..Default::default() };
        let m = CirGnn::new(cfg.clone()).map_err(|e| anyhow!(e))?;
        // topology of the *source* wiring graph used by this mode
        let conn = match &tvb {
            Some(d) => hcrnn::Connectome::from_tvb_dir(std::path::Path::new(d)).map_err(|e| anyhow!(e))?.log_compressed(),
            None => hcrnn::Connectome::synthetic_human(per_hemi, density, cfg.conn_seed).map_err(|e| anyhow!(e))?.log_compressed(),
        };
        let mut rng = hcrnn::rng::Rng::new(cfg.seed ^ 0x5EED_C0DE);
        let base = conn.scaled_to_mean_strength(1.0);
        let g = match w {
            "degree-preserving" => base.degree_preserving_rewire(&mut rng, 20),
            "density-matched" => base.density_matched_random(&mut rng),
            "dense" => base.dense_uniform(),
            _ => base,
        };
        let t = topology_report(&g);
        println!(
            "{:<18} {:>7} {:>6} {:>9.2} {:>8.3} {:>10.3} {:>9.4} {:>8} {:>10.3}",
            w, t.n, t.edges, t.mean_degree, t.rho_b, t.expansion_ratio, t.fiedler, m.rnn.edges.len(), m.rnn.spectral_radius()
        );
    }
    Ok(())
}

fn eval(a: EvalArgs) -> Result<()> {
    set_engine_config(&a.engine_config)?;
    let t0 = Instant::now();
    let (mut txns, names, engine) = match &a.csv {
        Some(p) => {
            let (t, n, e) = load_csv(p, &a.format, a.limit)?;
            (t, n, Some(e))
        }
        None => (synth_stream(&a.world), TYPOLOGIES.iter().map(|s| s.to_string()).collect(), None),
    };
    let mut engine = engine.unwrap_or_else(|| {
        let mut e = Engine::new(default_config());
        e.explain = false;
        e
    });
    score_with_engine(&mut txns, &mut engine);
    let ecfg = default_config();
    let risk = RiskTable::from_engine(&ecfg);
    let pos = txns.iter().filter(|t| t.label).count();
    println!(
        "data: {} transactions, {} laundering ({:.3}%), {} days, {} accounts   [{}]",
        txns.len(), pos, 100.0 * pos as f64 / txns.len() as f64,
        (txns.last().unwrap().ts - txns[0].ts) / 86_400 + 1,
        txns.iter().flat_map(|t| [t.sender, t.receiver]).collect::<std::collections::HashSet<_>>().len(),
        if a.csv.is_some() { a.format.as_str() } else { "SYNTHETIC world, not real data" }
    );

    let base = CirGnnConfig {
        window_secs: a.window_hours * 3600,
        warmup_windows: a.warmup_windows,
        per_hemi: a.per_hemi,
        tvb_dir: a.tvb.clone(),
        max_delay: a.max_delay,
        ..Default::default()
    };
    let tcfg_base = TrainConfig {
        epochs: a.epochs,
        chunk: a.chunk,
        lr: a.lr,
        lr_memory: a.lr,
        patience: a.patience,
        threads: a.threads.unwrap_or_else(hcrnn::network::default_threads),
        ..Default::default()
    };

    let mut results: Vec<(String, Vec<TrainReport>)> = Vec::new();
    for (vi, v) in a.variants.iter().enumerate() {
        let mut reps = Vec::new();
        for s in 0..a.seeds {
            let mut cfg = variant_cfg(v, &base)?;
            cfg.seed = base.seed + s;
            let mut model = CirGnn::new(cfg).map_err(|e| anyhow!(e))?;
            if vi == 0 && s == 0 {
                println!(
                    "memory network: {} units, {} synapses, delays 0..{}, spectral radius {:.3}; {} trainable parameters in total",
                    model.rnn.n_units, model.rnn.edges.len(), model.rnn.max_delay(), model.rnn.spectral_radius(), model.num_params()
                );
            }
            let tc = TrainConfig { seed: tcfg_base.seed + s, ..tcfg_base.clone() };
            let t1 = Instant::now();
            let rep = train_model(&mut model, &txns, &risk, &tc, |e| {
                eprintln!("  [{v} seed {s}] epoch {:>2}  loss {:.4}  val AP {:.4}  |g| {:.2}  radius {:.3}  {:.1}s", e.epoch, e.loss, e.val_ap, e.grad_norm, e.radius, e.secs)
            })
            .map_err(|e| anyhow!(e))?;
            eprintln!("  [{v} seed {s}] best epoch {} (val AP {:.4}), test AP {:.4}, {:.0}s", rep.best_epoch, rep.best_val_ap, rep.test.model.ap, t1.elapsed().as_secs_f64());
            if vi == 0 && s == 0 {
                if let Some(p) = &a.save {
                    model.save(p).map_err(|e| anyhow!(e))?;
                    println!("saved model to {}", p.display());
                }
            }
            reps.push(rep);
        }
        results.push((v.clone(), reps));
    }

    let r0: &TestReport = &results[0].1[0].test;
    let sp = &results[0].1[0].split;
    println!(
        "\nsplit by time: {} windows, train < {}, validation < {}, test = last {} windows ({} txns, {} laundering, base rate {:.3}%)",
        sp.windows, sp.train_end, sp.val_end, sp.windows - sp.val_end, r0.txns, r0.positives, 100.0 * r0.base_rate
    );
    println!("alerts at the rule threshold (60): online {} · after retro-linking {}\n", r0.online_alerts, r0.retro_alerts);
    println!("{:<26} {:>14} {:>14} {:>14} {:>14} {:>14}", "TEST (higher is better)", "AP", "ROC-AUC", "best F1", "prec@online-k", "recall@online-k");
    let row = |name: &str, f: &dyn Fn(&TestReport) -> [f64; 5], reps: &[&TestReport]| {
        let cols: Vec<String> = (0..5)
            .map(|c| fmt_ms(&reps.iter().map(|r| f(r)[c]).collect::<Vec<_>>(), c != 1))
            .collect();
        println!("{:<26} {:>14} {:>14} {:>14} {:>14} {:>14}", name, cols[0], cols[1], cols[2], cols[3], cols[4]);
    };
    let pick = |s: &aml_graph::train::ScoreSummary| [s.ap, s.auc, s.best_f1.f1, s.at_online_volume.precision, s.at_online_volume.recall];
    row("rules, online (causal)", &|r| pick(&r.rules_online), &[r0]);
    row("rules, retro-linked", &|r| pick(&r.rules_retro), &[r0]);
    for (name, reps) in &results {
        let rr: Vec<&TestReport> = reps.iter().map(|r| &r.test).collect();
        row(&format!("model: {name}"), &|r| pick(&r.model), &rr);
    }
    println!("(AP, AUC in % ... AUC shown as a fraction; precision/recall in %; {} seed(s) per model, mean ± sd)", a.seeds);

    println!("\nrecall by typology at the online rule engine's alert volume ({} alerts)", r0.online_alerts);
    println!("{:<26} {:>8} {:>14} {:>14} {:>14}", "typology", "n", "rules online", "rules retro", "model: full");
    let full = &results[0].1;
    for t in &r0.typology {
        let m: Vec<f64> = full.iter().filter_map(|r| r.test.typology.iter().find(|x| x.typology == t.typology)).map(|x| x.model_at_online_volume).collect();
        println!(
            "{:<26} {:>8} {:>13.1}% {:>13.1}% {:>14}",
            names.get(t.typology as usize).map(|s| s.as_str()).unwrap_or("?"), t.n, 100.0 * t.rules_online, 100.0 * t.rules_retro, fmt_ms(&m, true)
        );
    }
    if let Some(p) = &a.json_out {
        let j: Vec<_> = results.iter().map(|(n, r)| serde_json::json!({"variant": n, "runs": r})).collect();
        std::fs::write(p, serde_json::to_vec_pretty(&serde_json::json!({"typologies": names, "results": j}))?)?;
        println!("\nfull report written to {}", p.display());
    }
    println!("\ntotal {:.0}s", t0.elapsed().as_secs_f64());
    Ok(())
}

fn score(a: ScoreArgs) -> Result<()> {
    set_engine_config(&a.engine_config)?;
    let model = CirGnn::load(&a.model).map_err(|e| anyhow!(e))?;
    let (txns, _names, _e) = load_csv(&a.csv, &a.format, a.limit)?;
    let mut sc = GraphScorer::new(model, &default_config());
    let mut all: Vec<(u32, f32)> = Vec::new();
    let mut accounts: std::collections::HashMap<i64, f32> = std::collections::HashMap::new();
    let mut take = |w: aml_graph::WindowScores| {
        all.extend(w.scores);
        for (acct, p) in w.accounts {
            let e = accounts.entry(acct).or_insert(0.0);
            *e = e.max(p);
        }
    };
    for t in &txns {
        if let Some(w) = sc.push(t.clone()).map_err(|e| anyhow!(e))? {
            take(w);
        }
    }
    if let Some(w) = sc.flush() {
        take(w);
    }
    let labelled = txns.iter().any(|t| t.label);
    if labelled {
        let scored: Vec<(f32, bool)> = all.iter().map(|&(rf, p)| (p, txns[rf as usize].label)).collect();
        println!("AP {:.4}  ROC-AUC {:.4}  best F1 {:.4}  (includes the model's own training period if the CSV overlaps it)", aml_graph::metrics::average_precision(&scored), aml_graph::metrics::roc_auc(&scored), aml_graph::metrics::best_f1(&scored).f1);
    }
    let mut top: Vec<(i64, f32)> = accounts.into_iter().collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    println!("top {} accounts by graph risk:", a.top);
    for (acct, p) in top.iter().take(a.top) {
        println!("  {acct:>12}  {:.3}", p);
    }
    if let Some(p) = &a.out {
        let mut w = csv::Writer::from_path(p)?;
        w.write_record(["row", "graph_risk", "label"])?;
        all.sort_by_key(|x| x.0);
        for (rf, pr) in &all {
            w.write_record([rf.to_string(), format!("{pr:.5}"), (txns[*rf as usize].label as u8).to_string()])?;
        }
        w.flush()?;
        println!("wrote {} scores to {}", all.len(), p.display());
    }
    Ok(())
}
