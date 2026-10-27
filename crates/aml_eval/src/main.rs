//! Back-test the AML engine against the labelled SAML-D dataset.
//!
//!   aml-eval --csv SAML-D.csv [--limit 3000000] [--threshold 50]
//!            [--set FAN_IN.min_counterparties=8] [--weight FAN_IN=0.5] [--disable DORMANT_REACTIVATION]

use aml_datasets::{Format, Item};
use aml_engine::{config::ALL_RULES, default_config, Engine, EngineConfig, Profile, TxnIn};
use anyhow::{anyhow, Context, Result};
use clap::Parser;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    csv: PathBuf,
    /// Only the first N rows (the file is time-ordered).
    #[arg(long)]
    limit: Option<usize>,
    /// JSON file holding a full EngineConfig (as served by GET /api/v1/rules).
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    threshold: Option<u8>,
    /// Treat the first N days as warm-up: rows build account history but are excluded from metrics.
    #[arg(long)]
    warmup_days: Option<i64>,
    /// saml-d | aml-suspicious | risk-prediction | fraudshield
    #[arg(long, default_value = "saml-d")]
    format: String,
    /// RULE.param=value (repeatable)
    #[arg(long = "set")]
    sets: Vec<String>,
    /// RULE=weight (repeatable)
    #[arg(long = "weight")]
    weights: Vec<String>,
    #[arg(long, value_delimiter = ',')]
    disable: Vec<String>,
    /// Switch on rules that are off by default (e.g. AMOUNT_DROP).
    #[arg(long, value_delimiter = ',')]
    enable: Vec<String>,
    /// Calibrate: estimate each rule's lift on the first `--fit-fraction` of the data and write a tuned EngineConfig here.
    #[arg(long)]
    fit_out: Option<PathBuf>,
    #[arg(long, default_value_t = 0.7)]
    fit_fraction: f64,
    /// Report metrics only on rows after this fraction of the data (time-based holdout).
    #[arg(long, default_value_t = 0.0)]
    eval_from: f64,
    /// Exclude rows of these typologies from metrics and calibration (e.g. a flag shown to be pure noise).
    #[arg(long, value_delimiter = ',')]
    drop_typology: Vec<String>,
    /// Write the full report as JSON here.
    #[arg(long)]
    json_out: Option<PathBuf>,
    /// Write every false positive and false negative (in the evaluation window, at the alert threshold) to this CSV, with the rules that fired
    #[arg(long)]
    dump_errors: Option<PathBuf>,
}

/// Compliance screens are not tuned for precision: every hit must be reviewed.
const PROTECTED: [&str; 2] = ["WATCHLIST_HIT", "SANCTIONED_JURISDICTION"];

fn f(rec: &csv::ByteRecord, i: usize) -> &str {
    std::str::from_utf8(rec.get(i).unwrap_or(b"")).unwrap_or("")
}

fn main() -> Result<()> {
    let a = Args::parse();
    let mut cfg: EngineConfig = match &a.config {
        Some(p) => serde_json::from_slice(&std::fs::read(p)?).context("parsing config json")?,
        None => default_config(),
    };
    for s in &a.sets {
        let (lhs, v) = s.split_once('=').ok_or_else(|| anyhow!("--set expects RULE.param=value"))?;
        let (rule, param) = lhs.split_once('.').ok_or_else(|| anyhow!("--set expects RULE.param=value"))?;
        cfg.rules.get_mut(rule).ok_or_else(|| anyhow!("unknown rule {rule}"))?.params.insert(param.into(), v.parse()?);
    }
    for s in &a.weights {
        let (rule, v) = s.split_once('=').ok_or_else(|| anyhow!("--weight expects RULE=w"))?;
        cfg.rules.get_mut(rule).ok_or_else(|| anyhow!("unknown rule {rule}"))?.weight = v.parse()?;
    }
    for r in &a.enable {
        cfg.rules.get_mut(r.as_str()).ok_or_else(|| anyhow!("unknown rule {r}"))?.enabled = true;
    }
    for r in &a.disable {
        cfg.rules.get_mut(r.as_str()).ok_or_else(|| anyhow!("unknown rule {r}"))?.enabled = false;
    }
    if let Some(t) = a.threshold {
        cfg.alert_threshold = t;
    }
    let threshold = cfg.alert_threshold;
    let cfg_base = cfg.clone();
    let mut engine = Engine::new(cfg);
    engine.explain = false;

    let fmt = Format::parse(&a.format)?;
    let warmup = a.warmup_days.unwrap_or(fmt.default_warmup_days());
    let rule_bit: BTreeMap<&str, u32> = ALL_RULES.iter().enumerate().map(|(i, r)| (*r, 1u32 << i)).collect();
    let (mut keep, mut mask, mut typ): (Vec<f32>, Vec<u32>, Vec<u8>) = (Vec::new(), Vec::new(), Vec::new());
    let mut typ_names: Vec<String> = vec!["(normal)".into()];
    let mut typ_ids: BTreeMap<String, u8> = BTreeMap::new();
    let (t0, mut n, mut bad, mut customers) = (Instant::now(), 0usize, 0usize, 0usize);
    let mut start_ts: Option<i64> = None;
    let mut ok: Vec<bool> = Vec::new();
    let mut dropped: Vec<bool> = Vec::new();
    let mut raw: Vec<(i64, u64, u64, f64)> = Vec::new();

    for item in aml_datasets::open(&a.csv, fmt, a.limit)? {
        let r = match item {
            Item::Customer(c) => {
                engine.set_profile(c.account, &Profile { risk: c.risk, annual_income_gbp: c.annual_income_gbp, pep: c.pep, trade_exposed: c.trade_exposed() });
                if c.sanctioned {
                    engine.set_watch(c.account, true);
                }
                customers += 1;
                continue;
            }
            Item::Bad => {
                bad += 1;
                continue;
            }
            Item::Row(r) => r,
        };
        let is_dropped = r.label == Some(true) && r.typology.as_deref().is_some_and(|t| a.drop_typology.iter().any(|d| d == t));
        let tid = if r.label == Some(true) && !is_dropped {
            let name = r.typology.clone().unwrap_or_else(|| "(positive)".into());
            let next = (typ_ids.len() + 1) as u8;
            *typ_ids.entry(name).or_insert(next)
        } else {
            0
        };
        if tid as usize >= typ_names.len() {
            let mut names = vec!["(normal)".to_string(); typ_ids.len() + 1];
            for (k, v) in &typ_ids {
                names[*v as usize] = k.clone();
            }
            typ_names = names;
        }
        let t = TxnIn {
            ts: r.ts,
            sender: r.sender,
            receiver: r.receiver,
            amount_gbp: r.amount_gbp,
            payment_type: r.ptype,
            sender_country: r.scountry,
            receiver_country: r.rcountry,
            cross_currency: r.pcur != r.rcur,
        };
        let row = n;
        n += 1;
        let st = *start_ts.get_or_insert(t.ts);
        ok.push(t.ts - st >= warmup * 86_400);
        dropped.push(is_dropped);
        raw.push((t.ts, t.sender as u64, t.receiver as u64, t.amount_gbp as f64));
        keep.push(1.0);
        mask.push(0);
        typ.push(tid);
        let ev = engine.process(&t, row as u32);
        for h in &ev.hits {
            keep[row] *= (1.0 - h.weight) as f32;
            mask[row] |= rule_bit[h.rule];
        }
        for l in &ev.linked {
            let b = rule_bit[l.rule];
            if mask[l.rf as usize] & b == 0 {
                mask[l.rf as usize] |= b;
                keep[l.rf as usize] *= (1.0 - engine.rule_weight(l.rule)) as f32;
            }
        }
        if n % 1_000_000 == 0 {
            eprintln!("  .. {n} rows, {:.0}s", t0.elapsed().as_secs_f64());
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    for i in 0..n {
        if dropped[i] {
            ok[i] = false;
        }
    }
    let ok_train = ok.clone();
    let eval_cut = (a.eval_from * n as f64) as usize;
    for i in 0..eval_cut.min(n) {
        ok[i] = false;
    }
    let score: Vec<u8> = keep.iter().map(|k| ((1.0 - *k as f64) * 100.0).round() as u8).collect();
    let pos: usize = (0..n).filter(|i| ok[*i] && typ[*i] > 0).count();
    let n_eff = ok.iter().filter(|b| **b).count();
    println!("format={} customers={customers}\nrows={n} scored={n_eff} after {}d warm-up (unmappable {bad}) laundering={pos} base_rate={:.4}%  engine={:.0}s ({:.0} rows/s)  accounts_tracked={}",
        a.format, warmup, 100.0 * pos as f64 / n_eff.max(1) as f64, secs, n as f64 / secs, engine.account_count());

    let at = |thr: u8| {
        let (mut tp, mut fp) = (0usize, 0usize);
        for i in (0..n).filter(|i| ok[*i]) {
            if score[i] >= thr {
                if typ[i] > 0 { tp += 1 } else { fp += 1 }
            }
        }
        let p = tp as f64 / (tp + fp).max(1) as f64;
        let r = tp as f64 / pos.max(1) as f64;
        (tp, fp, p, r, 2.0 * p * r / (p + r).max(1e-12))
    };
    println!("\nthreshold sweep (transaction level)");
    println!("{:>5} {:>9} {:>9} {:>9} {:>8} {:>8} {:>8}", "thr", "TP", "FP", "alerts", "prec", "recall", "F1");
    let mut sweep = vec![];
    for thr in [20u8, 30, 40, 45, 50, 55, 60, 70, 80, 90] {
        let (tp, fp, p, r, f1) = at(thr);
        println!("{:>5} {:>9} {:>9} {:>9} {:>7.2}% {:>7.2}% {:>8.3}", thr, tp, fp, tp + fp, 100.0 * p, 100.0 * r, f1);
        sweep.push(serde_json::json!({"threshold":thr,"tp":tp,"fp":fp,"precision":p,"recall":r,"f1":f1}));
    }
    let (tp, fp, p, r, f1) = at(threshold);
    println!("\n== at configured alert threshold {threshold}: TP={tp} FP={fp} precision={:.2}% recall={:.2}% F1={:.3}", 100.0 * p, 100.0 * r, f1);

    println!("\nrecall by laundering typology (flagged at threshold {threshold})");
    let mut tcount = vec![(0usize, 0usize); typ_names.len()];
    for i in (0..n).filter(|i| ok[*i]) {
        if typ[i] > 0 {
            tcount[typ[i] as usize].0 += 1;
            if score[i] >= threshold {
                tcount[typ[i] as usize].1 += 1;
            }
        }
    }
    let mut typ_json = serde_json::Map::new();
    let mut rows: Vec<_> = (1..typ_names.len()).map(|i| (typ_names[i].clone(), tcount[i])).collect();
    rows.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    for (name, (tot, hit)) in &rows {
        println!("  {:<22} {:>6} txns  recall {:>6.1}%", name, tot, 100.0 * *hit as f64 / (*tot).max(1) as f64);
        typ_json.insert(name.clone(), serde_json::json!({"n":tot,"flagged":hit,"recall":*hit as f64/(*tot).max(1) as f64}));
    }

    println!("\nper-rule behaviour (direct or linked hits)   [lift = precision / base rate]");
    println!("  {:<24} {:>9} {:>9} {:>8} {:>8} {:>7}", "rule", "hits", "laund.", "prec", "recall", "lift");
    let mut rule_json = serde_json::Map::new();
    for (code, bit) in &rule_bit {
        let (mut h, mut l) = (0usize, 0usize);
        for i in (0..n).filter(|i| ok[*i]) {
            if mask[i] & bit != 0 {
                h += 1;
                if typ[i] > 0 {
                    l += 1;
                }
            }
        }
        if h == 0 {
            println!("  {:<24} {:>9}", code, 0);
            continue;
        }
        let prec = l as f64 / h as f64;
        let base = pos as f64 / n_eff as f64;
        println!("  {:<24} {:>9} {:>9} {:>7.2}% {:>7.2}% {:>6.1}x", code, h, l, 100.0 * prec, 100.0 * l as f64 / pos.max(1) as f64, prec / base);
        rule_json.insert(code.to_string(), serde_json::json!({"hits":h,"laundering":l,"precision":prec,"lift":prec/base}));
    }

    if let Some(path) = &a.dump_errors {
        let mut w = String::from("row,ts,sender,receiver,amount,kind,typology,score,rules\n");
        for i in (0..n).filter(|i| ok[*i]) {
            let flagged = score[i] >= threshold;
            let kind = match (flagged, typ[i] > 0) {
                (true, false) => "FP",
                (false, true) => "FN",
                (true, true) => "TP",
                _ => continue,
            };
            let names: Vec<&str> = rule_bit.iter().filter(|(_, b)| mask[i] & **b != 0).map(|(c, _)| *c).collect();
            let (ts, s, r, amt) = raw[i];
            w.push_str(&format!("{i},{ts},{s},{r},{amt},{kind},{},{},{}\n", typ_names[typ[i] as usize], score[i], names.join("|")));
        }
        std::fs::write(path, w)?;
    }

    if let Some(path) = &a.fit_out {
        let cut = (a.fit_fraction * n as f64) as usize;
        let pos_t = (0..cut).filter(|i| ok_train[*i] && typ[*i] > 0).count();
        let n_t = (0..cut).filter(|i| ok_train[*i]).count();
        let base = pos_t as f64 / n_t.max(1) as f64;
        let mut fitted = cfg_base.clone();
        println!("\ncalibration on first {:.0}% ({} rows, {} positives, base rate {:.3}%)", a.fit_fraction * 100.0, n_t, pos_t, 100.0 * base);
        println!("  {:<24} {:>8} {:>8} {:>7}  {:>6} -> {:<6}", "rule", "hits", "prec", "lift", "w_old", "w_new");
        for (code, bit) in &rule_bit {
            let (mut h, mut l) = (0usize, 0usize);
            for i in (0..cut).filter(|i| ok_train[*i]) {
                if mask[i] & bit != 0 {
                    h += 1;
                    if typ[i] > 0 {
                        l += 1;
                    }
                }
            }
            let r = fitted.rules.get_mut(*code).unwrap();
            let old = r.weight;
            if PROTECTED.contains(code) {
                println!("  {:<24} {:>8} {:>8} {:>7}  {:>6.2} -> {:<6.2} (regulatory screen: never tuned down)", code, h, "-", "-", old, old);
                continue;
            }
            if h >= 30 && base > 0.0 {
                let lift = (l as f64 / h as f64) / base;
                if lift < 1.3 {
                    r.weight = 0.02; // fires, but carries no information in this data
                } else {
                    r.weight = (0.10 + 0.20 * lift.log10()).clamp(0.05, 0.90);
                }
                // secondary weights (LARGE_TXN extreme_weight, HIGH_RISK_JURISDICTION very_high_weight, ...) are variants of the same
                // rule and move with it; otherwise a demoted rule would still alert through its strongest variant
                if old > 0.0 {
                    let k = r.weight / old;
                    for (name, v) in r.params.iter_mut() {
                        if name.ends_with("_weight") {
                            *v = (*v * k).clamp(0.02, 0.95);
                        }
                    }
                }
                println!("  {:<24} {:>8} {:>7.2}% {:>6.1}x  {:>6.2} -> {:<6.2}", code, h, 100.0 * l as f64 / h as f64, lift, old, r.weight);
            } else {
                println!("  {:<24} {:>8} {:>8} {:>7}  {:>6.2} -> {:<6.2} (insufficient evidence, unchanged)", code, h, "-", "-", old, old);
            }
        }
        std::fs::write(path, serde_json::to_vec_pretty(&fitted)?)?;
        println!("tuned config written to {}", path.display());
    }
    if let Some(p) = a.json_out {
        std::fs::write(p, serde_json::to_vec_pretty(&serde_json::json!({
            "rows": n, "laundering": pos, "threshold": threshold, "sweep": sweep,
            "typology": typ_json, "rules": rule_json,
        }))?)?;
    }
    Ok(())
}
