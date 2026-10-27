//! Command-line front end: `cddctl --state state.json <command> ...`

use cdd_sanctions::calibrate::{CalibrationOptions, Label};
use cdd_sanctions::cdd::{Party, PartyKind, RelKind};
use cdd_sanctions::links::{Link, LinkKind};
use cdd_sanctions::engine::{Actor, Compliance, HistoryRecord, Outcome, Role};
use cdd_sanctions::ml::{ActivationGates, MlMode, TrainOptions};
use cdd_sanctions::error::{Error, Result};
use cdd_sanctions::lists::{belize, eu, jsonl, ofac, un};
use cdd_sanctions::model::*;
use cdd_sanctions::screening::Subject;
use chrono::Utc;
use std::path::PathBuf;

const USAGE: &str = "\
cddctl --state FILE [--actor NAME] [--role analyst|reviewer|mlro] <command>

  load-list NAME ofac SDN.csv [ALT.csv ADD.csv] [--label L] [--force]
  load-list NAME un|eu|belize|jsonl FILE     [--label L] [--force]   (belize = Belize FIU consolidated list XML)
  screen --name N [--kind individual|entity|vessel|aircraft] [--dob D] [--nationality C]
         [--country C] [--id kind:value[:country]]      ad-hoc check, nothing is stored
  party-add PARTY.json            add a party (JSON, see README)
  rel-add FROM TO KIND [PCT]      record that FROM holds KIND over TO; KIND = owner (needs PCT) | director | signatory |
                                  controller | trustee | settlor | beneficiary
  link-add ID KIND VALUE [--country C]
                                  record an address | phone | email | bank_account | registered_agent for party ID; parties sharing
                                  a value become associated in the network analysis (never screened)
  calibrate LABELS.json [--recall R] [--precision P] [--dampings a,b,c]
                                  measure the network score against labelled parties ([{\"party\":12,\"linked\":true},...]):
                                  AUC, precision/recall, suggested thresholds, what each link family contributes. Advisory only.
  party-screen ID                 screen the party and its owners/controllers
  risk ID                         risk rating, due-diligence level, blockers
  alerts [--open] [--queue]       list alerts; --queue = unresolved alerts in reading order with band and priority
  ml-status                       prioritiser mode, active model, label counts
  ml-train [--history FILE] [--activate] [--folds K] [--label L] [--report FILE]
           [--min-examples N] [--min-positives N] [--min-negatives N]
                                  (MLRO) train a match-probability model on resolved alerts (+ optional historical decisions
                                  [{\"subject\":{..},\"list\":\"OFAC-SDN\",\"uid\":\"123\",\"confirmed\":false}]), cross-validate, run the
                                  activation gates. --activate installs it only if every gate passes. Orders alerts; never decides them.
  ml-mode off|shadow|active [--cutoff P]
                                  (MLRO) active needs a validated model; --cutoff sets the Low-band priority limit (0 = no Low band)
  ml-shadow [--version V]         prospective check: priority stored at alert time vs rule score, on alerts resolved since
  ml-explain ALERT                per-feature contributions behind an alert's priority
  ml-export FILE                  write resolved alerts as JSONL (features + label) for offline analysis
  network [--min X] [--top N]     spectral hypergraph report: connected groups (cohesion, weakest cut, connector parties)
                                  and indirect sanctions exposure of parties outside the ownership rules' reach
  network ID                      why party ID is (or is not) indirectly exposed, step by step
  resolve ALERT false-positive|confirm|escalate --note TEXT
  due [DAYS]                      periodic reviews due within DAYS (default 30)
  audit-verify                    verify the audit hash chain
  stale [DAYS]                    lists not refreshed within DAYS (default 35); exit 3 if any
  stats                           list versions and counts";

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}
fn has(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let state = PathBuf::from(flag(&args, "--state").ok_or_else(|| Error::Invalid(USAGE.into()))?);
    let role = match flag(&args, "--role").as_deref() {
        Some("mlro") => Role::Mlro,
        Some("reviewer") => Role::Reviewer,
        _ => Role::Analyst,
    };
    let actor = Actor::new(&flag(&args, "--actor").or_else(|| std::env::var("USER").ok()).unwrap_or_else(|| "cli".into()), role);
    // positional args: everything not a flag or a flag value
    let value_flags = ["--state", "--actor", "--role", "--label", "--name", "--kind", "--dob", "--nationality", "--country", "--id", "--note", "--min", "--top", "--recall", "--precision", "--dampings", "--history", "--folds", "--report", "--min-examples", "--min-positives", "--min-negatives", "--cutoff", "--version"];
    let mut pos: Vec<&String> = Vec::new();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
        } else if value_flags.contains(&a.as_str()) {
            skip = true;
        } else if !a.starts_with("--") {
            pos.push(a);
        }
    }
    let now = Utc::now();
    let mut c = if state.exists() { Compliance::load(&state)? } else { Compliance::default() };
    let mut dirty = true;

    match pos.first().map(|s| s.as_str()) {
        Some("load-list") => {
            let (name, fmt) = (pos.get(1).ok_or_else(|| Error::Invalid(USAGE.into()))?, pos.get(2).ok_or_else(|| Error::Invalid(USAGE.into()))?);
            let read = |i: usize| -> Result<String> {
                let p = pos.get(i).ok_or_else(|| Error::Invalid("missing file argument".into()))?;
                Ok(String::from_utf8_lossy(&std::fs::read(p)?).into_owned())
            };
            let entries = match fmt.as_str() {
                "ofac" => {
                    let alt = if pos.len() > 4 { Some(read(4)?) } else { None };
                    let add = if pos.len() > 5 { Some(read(5)?) } else { None };
                    ofac::parse(&read(3)?, alt.as_deref(), add.as_deref(), name)?
                }
                "un" => un::parse(&read(3)?, name)?,
                "belize" => {
                    let (e, rep) = belize::parse(&read(3)?, name)?;
                    if rep.repaired_entries > 0 {
                        eprintln!("note: {} Belize entries had corrupted spellings (stray 'BZ'); repaired variants added alongside the published names", rep.repaired_entries);
                    }
                    e
                }
                "eu" => eu::parse(&read(3)?, name)?,
                "jsonl" => jsonl::parse(&read(3)?, name)?,
                other => return Err(Error::Invalid(format!("unknown format '{other}'"))),
            };
            let label = flag(&args, "--label").unwrap_or_else(|| now.format("%Y-%m-%d").to_string());
            let rep = c.load_list(name, entries, &label, has(&args, "--force"), &actor, now)?;
            println!(
                "{name}: +{} added, ~{} changed, -{} removed; screened {} parties, {} new alerts, {} alerts flagged as delisted",
                rep.diff.added.len(), rep.diff.changed.len(), rep.diff.removed.len(), rep.parties_screened, rep.new_alerts, rep.delisted_alerts
            );
        }
        Some("screen") => {
            dirty = false;
            let name = flag(&args, "--name").ok_or_else(|| Error::Invalid("--name required".into()))?;
            let mut s = Subject::new(SubjectKind::parse(&flag(&args, "--kind").unwrap_or_default()), &name);
            if let Some(d) = flag(&args, "--dob") {
                s.dob.push(PartialDate::parse(&d).ok_or_else(|| Error::Invalid(format!("bad date '{d}'")))?);
            }
            s.nationalities.extend(flag(&args, "--nationality"));
            s.countries.extend(flag(&args, "--country"));
            if let Some(id) = flag(&args, "--id") {
                let p: Vec<&str> = id.splitn(3, ':').collect();
                let kind = match p[0] {
                    "passport" => IdKind::Passport,
                    "national_id" => IdKind::NationalId,
                    "tax_id" => IdKind::TaxId,
                    "registration" => IdKind::Registration,
                    "imo" => IdKind::Imo,
                    "swift" => IdKind::Swift,
                    "crypto" => IdKind::Crypto,
                    _ => IdKind::Other,
                };
                s.identifiers.push(Identifier::new(kind, p.get(1).copied().unwrap_or(""), p.get(2).copied()));
            }
            let r = c.index().screen(&s, &c.screening_cfg);
            if r.hits.is_empty() {
                println!("no potential matches ({} lists entries indexed)", c.index().len());
            }
            for h in r.hits {
                println!("{:?} {:.2}  [{}:{}] {}  (matched '{}')  programs: {}", h.level, h.score, h.list, h.uid, h.primary_name, h.matched_name, h.programs.join(","));
                for x in h.reasons {
                    println!("    - {x}");
                }
            }
        }
        Some("party-add") => {
            let p: Party = serde_json::from_slice(&std::fs::read(pos.get(1).ok_or_else(|| Error::Invalid(USAGE.into()))?)?)?;
            let id = c.add_party(p, &actor, now)?;
            println!("party {id} added");
        }
        Some("rel-add") => {
            let num = |i: usize| -> Result<u64> { pos.get(i).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Invalid(USAGE.into())) };
            let (from, to) = (num(1)?, num(2)?);
            let kind = match pos.get(3).map(|s| s.as_str()) {
                Some("owner") => RelKind::Owner,
                Some("director") => RelKind::Director,
                Some("signatory") => RelKind::Signatory,
                Some("controller") => RelKind::Controller,
                Some("trustee") => RelKind::Trustee,
                Some("settlor") => RelKind::Settlor,
                Some("beneficiary") => RelKind::Beneficiary,
                _ => return Err(Error::Invalid(USAGE.into())),
            };
            let pct = match pos.get(4) {
                Some(p) => Some(p.parse::<f64>().map_err(|_| Error::Invalid(format!("bad percentage '{p}'")))?),
                None => None,
            };
            let id = c.add_relationship(from, to, kind, pct, &actor, now)?;
            println!("relationship {id} added");
        }
        Some("network") => {
            dirty = false;
            let name = |id: u64| c.party(id).map(|p| p.name.clone()).unwrap_or_else(|| "?".into());
            if let Some(id) = pos.get(1).and_then(|s| s.parse::<u64>().ok()) {
                let e = c.network_exposure(id)?;
                println!("{} (#{id}): indirect exposure {:.4} (moderate >= {}, strong >= {})", name(id), e.exposure, c.risk_cfg.network.moderate_exposure, c.risk_cfg.network.strong_exposure);
                match (e.source, e.hops) {
                    (Some(s), Some(h)) => {
                        println!("strongest source: {} (#{s}), {h} link(s) away", name(s));
                        for line in &e.path {
                            println!("  {line}");
                        }
                    }
                    _ => println!("no sanctioned party outside this party's own ownership/control network is connected to it"),
                }
            } else {
                let min: f64 = flag(&args, "--min").and_then(|v| v.parse().ok()).unwrap_or(c.risk_cfg.network.moderate_exposure);
                let top: usize = flag(&args, "--top").and_then(|v| v.parse().ok()).unwrap_or(15);
                let rep = c.network_analysis()?;
                println!("{} connected group(s) with 3+ parties or a sanctioned member", rep.components.len());
                for comp in &rep.components {
                    let s = &comp.structure;
                    let mut line = format!("  group {}: {} parties, {} hyperedges, {} sanctioned seed(s)", comp.id, comp.parties.len(), comp.hyperedges, comp.seeds.len());
                    if let Some(l2) = s.algebraic_connectivity {
                        line += &format!(", cohesion (lambda2) {l2:.3}");
                    }
                    if let Some(sp) = &s.split {
                        line += &format!(", weakest cut {:.3} ({}|{} parties)", sp.conductance, sp.side_a.len(), sp.side_b.len());
                        if !sp.connectors.is_empty() {
                            let names: Vec<String> = sp.connectors.iter().take(5).map(|p| format!("{} (#{p})", name(*p))).collect();
                            line += &format!(", connector(s): {}", names.join(", "));
                        }
                    }
                    if let Some(n) = &s.note {
                        line += &format!(" [{n}]");
                    }
                    println!("{line}");
                }
                if rep.seeds_truncated {
                    println!("note: more sanctioned seeds than network.max_seeds; only the strongest were used");
                }
                if rep.skipped_shared_identifier_groups > 0 {
                    println!("note: {} shared identifier/link group(s) skipped as implausibly large (junk values, or a service provider shared by many clients)", rep.skipped_shared_identifier_groups);
                }
                let shown: Vec<_> = rep.exposures.iter().filter(|e| e.exposure >= min).take(top).collect();
                println!("\nindirect exposure >= {min} (highest first, up to {top}):");
                if shown.is_empty() {
                    println!("  none");
                }
                for e in shown {
                    println!(
                        "  {:.4}  {} (#{})  source {} hops {:?}{}",
                        e.exposure,
                        name(e.party),
                        e.party,
                        e.source.map(|s| format!("{} (#{s})", name(s))).unwrap_or_default(),
                        e.hops,
                        if e.covered_by_ownership_rules { "  [covered by ownership rules]" } else { "" }
                    );
                }
            }
        }
        Some("link-add") => {
            let id: u64 = pos.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let kind = pos.get(2).and_then(|k| LinkKind::parse(k)).ok_or_else(|| Error::Invalid(format!("link kind must be one of: address, phone, email, bank_account, registered_agent\n{USAGE}")))?;
            let value = pos.get(3).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let link = Link::new(kind, value, flag(&args, "--country").as_deref());
            if link.key(c.risk_cfg.network.home_dial_code.as_deref()).is_none() {
                return Err(Error::Invalid(format!("'{value}' is not a usable {} (empty, a placeholder, or too short to tell customers apart)", kind.label())));
            }
            c.update_party(id, |p| {
                if !p.links.contains(&link) {
                    p.links.push(link.clone());
                }
            }, &actor, now)?;
            println!("{} recorded for party {id}", kind.label());
        }
        Some("calibrate") => {
            dirty = false;
            let path = pos.get(1).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let labels: Vec<Label> = serde_json::from_slice(&std::fs::read(path)?)?;
            let mut opts = CalibrationOptions::default();
            if let Some(r) = flag(&args, "--recall").and_then(|v| v.parse().ok()) {
                opts.recall_target = r;
            }
            if let Some(p) = flag(&args, "--precision").and_then(|v| v.parse().ok()) {
                opts.precision_target = p;
            }
            let dampings: Vec<f64> = match flag(&args, "--dampings") {
                Some(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
                None => vec![0.3, 0.5, 0.6, 0.7, 0.8, 0.9],
            };
            let rep = c.calibrate_network(&labels, &opts, &dampings)?;
            let fmt = |a: Option<f64>| a.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into());
            let b = &rep.baseline;
            println!("labelled parties used: {} linked, {} not linked", b.positives, b.negatives);
            println!("AUC at damping {}: {}   (0.5 = chance, 1.0 = perfect)", rep.baseline_damping, fmt(b.auc));
            if let Some(p) = b.best_f1 {
                println!("best F1 {:.3} at exposure >= {:.5} (precision {:.2}, recall {:.2})", p.f1, p.threshold, p.precision, p.recall);
            }
            println!("current thresholds: moderate {} strong {}", c.risk_cfg.network.moderate_exposure, c.risk_cfg.network.strong_exposure);
            match (b.suggested_moderate, b.suggested_strong) {
                (None, None) => println!("suggested thresholds: none (see warnings)"),
                (m, s) => println!(
                    "suggested (recall >= {:.0}%, precision >= {:.0}%): moderate_exposure {} strong_exposure {}",
                    opts.recall_target * 100.0,
                    opts.precision_target * 100.0,
                    m.map(|v| format!("{v:.5}")).unwrap_or_else(|| "n/a".into()),
                    s.map(|v| format!("{v:.5}")).unwrap_or_else(|| "n/a".into())
                ),
            }
            if !rep.damping_sweep.is_empty() {
                println!("\ndamping sweep (AUC):");
                for (d, a) in &rep.damping_sweep {
                    println!("  {d:.2}  {}", fmt(*a));
                }
            }
            if !rep.families.is_empty() {
                println!("\nedge families in the data: AUC with the family switched off (change vs {}):", fmt(b.auc));
                for f in &rep.families {
                    let d = f.delta_auc.map(|d| format!("{d:+.3}")).unwrap_or_else(|| "n/a".into());
                    println!("  {:<18} {:>5} edge(s)  AUC without {}  ({d})", f.family, f.edges, fmt(f.auc_without));
                }
                println!("  a negative change means the family helps; positive means it hurts; ~0 means it adds nothing here");
            }
            for w in &b.warnings {
                println!("warning: {w}");
            }
            println!("\nThese figures are advisory. Review them with your MLCO; nothing has been changed.");
        }
        Some("party-screen") => {
            let id: u64 = pos.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            println!("{}", serde_json::to_string_pretty(&c.screen_party(id, &actor, now)?)?);
        }
        Some("risk") => {
            dirty = false;
            let id: u64 = pos.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let (a, blockers) = c.readiness(id, now)?;
            println!("{}", serde_json::to_string_pretty(&serde_json::json!({"assessment": a, "blockers": blockers}))?);
        }
        Some("alerts") => {
            dirty = false;
            if has(&args, "--queue") {
                for (i, q) in c.alert_queue().iter().enumerate() {
                    let a = q.alert;
                    println!(
                        "{:>3}. #{} {:?} {:?} score {:.2} priority {} party {} ~ [{}:{}] {}{}",
                        i + 1, a.id, q.band, a.level, a.score,
                        a.priority.map(|p| format!("{p:.4}")).unwrap_or_else(|| "n/a".into()),
                        a.party, a.list, a.uid, a.primary_name,
                        if a.delisted { "  [DELISTED]" } else { "" }
                    );
                }
            } else {
                for a in c.alerts().values().filter(|a| !has(&args, "--open") || a.status.is_unresolved()) {
                    println!(
                        "#{} {:?} {:?} {:.2} party {} (root {}) ~ [{}:{}] {}{}",
                        a.id, a.status, a.level, a.score, a.party, a.root_party, a.list, a.uid, a.primary_name,
                        if a.delisted { "  [DELISTED]" } else { "" }
                    );
                }
            }
        }
        Some("ml-status") => {
            dirty = false;
            println!("{}", serde_json::to_string_pretty(&c.ml_status())?);
        }
        Some("ml-mode") => {
            let mode = match pos.get(1).map(|s| s.as_str()) {
                Some("off") => MlMode::Off,
                Some("shadow") => MlMode::Shadow,
                Some("active") => MlMode::Active,
                _ => return Err(Error::Invalid(USAGE.into())),
            };
            let cutoff = flag(&args, "--cutoff").map(|v| v.parse::<f64>().map_err(|_| Error::Invalid("--cutoff must be a number".into()))).transpose()?;
            c.set_ml_mode(mode, cutoff, &actor, now)?;
            println!("prioritiser mode: {mode:?}");
        }
        Some("ml-train") => {
            let mut opts = TrainOptions { label: flag(&args, "--label").unwrap_or_else(|| now.format("%Y%m%d").to_string()), ..Default::default() };
            if let Some(k) = flag(&args, "--folds").and_then(|v| v.parse().ok()) {
                opts.folds = k;
            }
            let mut g = ActivationGates::default();
            if let Some(v) = flag(&args, "--min-examples").and_then(|v| v.parse().ok()) {
                g.min_examples = v;
            }
            if let Some(v) = flag(&args, "--min-positives").and_then(|v| v.parse().ok()) {
                g.min_positives = v;
            }
            if let Some(v) = flag(&args, "--min-negatives").and_then(|v| v.parse().ok()) {
                g.min_negatives = v;
            }
            opts.gates = g;
            let extra = match flag(&args, "--history") {
                Some(path) => {
                    let recs: Vec<HistoryRecord> = serde_json::from_slice(&std::fs::read(&path)?)?;
                    let imp = c.import_history(&recs);
                    println!(
                        "history: {} records -> {} usable ({} list entry not loaded, {} below the review threshold today)",
                        recs.len(), imp.examples.len(), imp.skipped_entry_missing, imp.skipped_below_review_threshold
                    );
                    imp.examples.iter().map(|e| e.to_example()).collect()
                }
                None => vec![],
            };
            let (own, without) = c.labelled_examples();
            println!("engine labels: {} ({} confirmed); {} resolved alerts predate feature capture and are unusable", own.len(), own.iter().filter(|e| e.y).count(), without);
            let (r, activated) = c.train_match_model(&extra, &opts, has(&args, "--activate"), &actor, now)?;
            let fmt = |a: Option<f64>| a.map(|v| format!("{v:.3}")).unwrap_or_else(|| "n/a".into());
            let pct = |a: Option<f64>| a.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "n/a".into());
            println!("examples {} ({} confirmed, {} false positive) from {} customers; {}-fold grouped CV; ridge toward scorecard = {}", r.examples, r.positives, r.negatives, r.customers, r.folds, r.lambda);
            println!("{:<26}{:>8}{:>8}{:>10}{:>12}{:>12}", "out-of-fold", "AUC", "AP", "Brier", "queue@95%", "queue@100%");
            for (n, m) in [("learned model", &r.model), ("hand-set scorecard", &r.prior), ("current rule score", &r.rule)] {
                println!(
                    "{:<26}{:>8}{:>8}{:>10}{:>12}{:>12}",
                    n, fmt(m.auc), fmt(m.average_precision),
                    if m.brier.is_nan() { "-".into() } else { format!("{:.4}", m.brier) },
                    pct(m.queue_at_95_recall), pct(m.queue_at_100_recall)
                );
            }
            if let Some((lo, hi)) = r.auc_gain_vs_rule {
                println!("AUC gain over the rule score: 95% bootstrap interval [{lo:+.3}, {hi:+.3}]");
            }
            println!("largest learned weights (learned / scorecard):");
            for (n, w, p) in &r.top_weights {
                println!("  {n:<22}{w:>8.2}{p:>8.2}");
            }
            for gt in &r.gates {
                println!("gate [{}] {} - {}", if gt.passed { "pass" } else { "FAIL" }, gt.name, gt.detail);
            }
            for w in &r.warnings {
                println!("warning: {w}");
            }
            if let Some(cut) = r.suggested_low_band_max_priority {
                println!("suggested Low-band cut-off: {cut:.5} (no out-of-fold confirmed match scored below twice this). Set with: ml-mode shadow --cutoff {cut:.5}");
            }
            if let Some(path) = flag(&args, "--report") {
                std::fs::write(&path, serde_json::to_vec_pretty(&r)?)?;
                println!("full report written to {path}");
            }
            match (has(&args, "--activate"), activated) {
                (true, true) => println!("ACTIVATED model {} (mode is {:?}); open alerts re-prioritised. Use ml-mode active to order the queue by it.", r.candidate.version, c.ml_cfg.mode),
                (true, false) => println!("NOT activated: not every gate passed. The engine keeps its previous model."),
                (false, _) => println!("dry run: nothing was activated (add --activate to install a model that passes every gate)"),
            }
        }
        Some("ml-shadow") => {
            dirty = false;
            let r = c.shadow_report(flag(&args, "--version").as_deref());
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Some("ml-explain") => {
            dirty = false;
            let id: u64 = pos.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let (parts, bias, ver) = c.explain_alert(id)?;
            println!("alert {id} scored by {ver}: logit = bias {bias:+.2}");
            for (n, v) in parts {
                println!("  {n:<22}{v:>+8.2}");
            }
        }
        Some("ml-export") => {
            dirty = false;
            let path = pos.get(1).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let mut out = String::new();
            for a in c.alerts().values() {
                let y = match a.status {
                    cdd_sanctions::engine::AlertStatus::Confirmed => true,
                    cdd_sanctions::engine::AlertStatus::FalsePositive => false,
                    _ => continue,
                };
                if let Some(f) = &a.features {
                    out.push_str(&serde_json::to_string(&serde_json::json!({"alert": a.id, "customer": a.root_party, "confirmed": y, "features": f}))?);
                    out.push('\n');
                }
            }
            std::fs::write(path, out)?;
            println!("written {path}");
        }
        Some("resolve") => {
            let id: u64 = pos.get(1).and_then(|s| s.parse().ok()).ok_or_else(|| Error::Invalid(USAGE.into()))?;
            let outcome = match pos.get(2).map(|s| s.as_str()) {
                Some("false-positive") => Outcome::FalsePositive,
                Some("confirm") => Outcome::ConfirmedMatch,
                Some("escalate") => Outcome::Escalate,
                _ => return Err(Error::Invalid(USAGE.into())),
            };
            c.resolve_alert(id, outcome, &flag(&args, "--note").unwrap_or_default(), &actor, now)?;
            println!("alert {id} updated");
        }
        Some("due") => {
            dirty = false;
            let days: i64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);
            for (id, d) in c.reviews_due(now.date_naive(), days) {
                println!("party {id} ({}) review due {d}", c.party(id).map(|p| p.name.as_str()).unwrap_or("?"));
            }
        }
        Some("audit-verify") => {
            dirty = false;
            match c.audit().verify() {
                Ok(()) => println!("audit chain OK ({} events)", c.audit().events().len()),
                Err(seq) => {
                    println!("audit chain BROKEN at event {seq}");
                    std::process::exit(2);
                }
            }
        }
        Some("stale") => {
            dirty = false;
            let days: i64 = pos.get(1).and_then(|s| s.parse().ok()).unwrap_or(35);
            let stale = c.stale_lists(now.date_naive(), days);
            if c.list_versions().is_empty() {
                println!("no sanctions lists loaded");
                std::process::exit(3);
            }
            for (l, d) in &stale {
                println!("STALE: {l} last loaded {d} (> {days} days)");
            }
            if stale.is_empty() {
                println!("all lists refreshed within {days} days");
            } else {
                std::process::exit(3);
            }
        }
        Some("stats") => {
            dirty = false;
            for v in c.list_versions() {
                println!("{} '{}' loaded {} ({} entries)", v.list, v.label, v.loaded_on, v.entries);
            }
            println!("{} parties, {} alerts, {} cases, {} indexed entries", c.parties().len(), c.alerts().len(), c.cases().len(), c.index().len());
        }
        _ => return Err(Error::Invalid(USAGE.into())),
    }
    if dirty {
        c.save(&state)?;
    }
    let _ = PartyKind::Individual;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
