//! The learned prioritiser inside the real engine: capture, ordering, gates, safety, persistence.

use cdd_sanctions::cdd::*;
use cdd_sanctions::engine::*;
use cdd_sanctions::error::Error;
use cdd_sanctions::ml::{ActivationGates, MlMode, PriorityBand, TrainOptions, PRIOR_VERSION};
use cdd_sanctions::model::*;
use cdd_sanctions::screening::HitLevel;
use chrono::{DateTime, TimeZone, Utc};
use std::collections::HashMap;

fn t(d: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, d, 9, 0, 0).unwrap()
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn f(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn entry(uid: usize, name: &str, dob: Option<PartialDate>) -> ListEntry {
    ListEntry {
        list: "SYN".into(),
        uid: uid.to_string(),
        kind: SubjectKind::Individual,
        names: vec![NameVariant { name: name.into(), strength: AliasStrength::Primary }],
        dobs: dob.into_iter().collect(),
        nationalities: vec![],
        countries: vec![],
        identifiers: vec![],
        programs: vec!["SYN".into()],
        remarks: None,
        listed_on: None,
        content_hash: String::new(),
    }
    .finalize()
}

const SYL: [&str; 16] = ["ka", "lo", "mi", "ra", "su", "te", "vo", "ni", "da", "be", "zu", "fa", "go", "hi", "je", "pu"];
fn word(r: &mut Rng, n: usize) -> String {
    (0..n).map(|_| SYL[r.below(16)]).collect()
}

struct World {
    c: Compliance,
    mia: Actor,
    ana: Actor,
    /// party id -> list uid it truly is
    truth: HashMap<u64, String>,
}

/// A list where a handful of common surnames collide a lot, plus customers that are either a perturbed listed
/// person (true match) or an unrelated person with a common name.
fn world(customers: usize, seed: u64) -> World {
    let mut r = Rng(seed);
    let common: Vec<String> = (0..12).map(|_| word(&mut r, 3)).collect();
    let given: Vec<String> = (0..40).map(|_| word(&mut r, 3)).collect();
    let mut people = Vec::new();
    for i in 0..600 {
        let last = if r.f() < 0.5 { common[r.below(common.len())].clone() } else { word(&mut r, 5) };
        let first = given[r.below(given.len())].clone();
        let dob = PartialDate::ymd(1950 + r.below(40) as i32, 1 + r.below(12) as u8, 1 + r.below(28) as u8);
        people.push((i, first, last, dob));
    }
    let entries: Vec<ListEntry> = people
        .iter()
        .map(|(i, f, l, d)| entry(*i, &format!("{}, {}", l.to_uppercase(), f), if r.f() < 0.85 { Some(*d) } else { None }))
        .collect();
    let (mia, ana) = (Actor::new("mia", Role::Mlro), Actor::new("ana", Role::Analyst));
    let mut c = Compliance::default();
    c.load_list("SYN", entries, "v1", false, &mia, t(1)).unwrap();
    let mut truth = HashMap::new();
    for k in 0..customers {
        let is_match = r.f() < 0.10;
        let (name, dob) = if is_match {
            let (i, f, l, d) = people[r.below(people.len())].clone();
            let dob = if r.f() < 0.7 { Some(d) } else { None };
            (format!("{f} {l}"), (dob, Some(i)))
        } else {
            let last = if r.f() < 0.6 { common[r.below(common.len())].clone() } else { word(&mut r, 5) };
            let first = given[r.below(given.len())].clone();
            let dob = if r.f() < 0.3 { Some(PartialDate::ymd(1950 + r.below(40) as i32, 1 + r.below(12) as u8, 1 + r.below(28) as u8)) } else { None };
            (format!("{first} {last}"), (dob, None))
        };
        let mut p = Party::new(PartyKind::Individual, &name);
        p.dob = dob.0.into_iter().collect();
        let id = c.add_party(p, &ana, t(2)).unwrap();
        if let Some(i) = dob.1 {
            truth.insert(id, i.to_string());
        }
        c.screen_party(id, &ana, t(2)).unwrap();
        let _ = k;
    }
    World { c, mia, ana, truth }
}

fn resolve_all(w: &mut World) {
    let open: Vec<(u64, u64, String)> =
        w.c.alerts().values().filter(|a| a.status.is_unresolved()).map(|a| (a.id, a.party, a.uid.clone())).collect();
    for (id, party, uid) in open {
        let is_true = w.truth.get(&party).map_or(false, |u| *u == uid);
        let (o, note) = if is_true { (Outcome::ConfirmedMatch, "confirmed by documents") } else { (Outcome::FalsePositive, "different person") };
        w.c.resolve_alert(id, o, note, &w.mia, t(3)).unwrap();
    }
}

fn small_gates() -> TrainOptions {
    TrainOptions {
        bootstrap_rounds: 300,
        gates: ActivationGates { min_examples: 60, min_positives: 10, min_negatives: 20, ..Default::default() },
        label: "test".into(),
        ..Default::default()
    }
}

#[test]
fn alerts_capture_features_and_a_stored_priority() {
    let w = world(60, 1);
    let alerts = w.c.alerts();
    assert!(alerts.len() > 5, "need some alerts, got {}", alerts.len());
    for a in alerts.values() {
        let f = a.features.as_ref().expect("features captured");
        assert!((f.rule_score - a.score).abs() < 1e-12, "feature rule_score must equal the alert score");
        let p = a.priority.expect("shadow mode stores a priority");
        assert!((0.0..=1.0).contains(&p));
        assert_eq!(a.model_version.as_deref(), Some(PRIOR_VERSION));
    }
}

#[test]
fn off_mode_stores_nothing_and_changes_nothing() {
    let mut c = Compliance::default();
    let mia = Actor::new("mia", Role::Mlro);
    c.set_ml_mode(MlMode::Off, None, &mia, t(1)).unwrap();
    c.load_list("SYN", vec![entry(1, "PETROV, Ivan", Some(PartialDate::ymd(1965, 3, 12)))], "v1", false, &mia, t(1)).unwrap();
    let mut p = Party::new(PartyKind::Individual, "Ivan Petrov");
    p.dob = vec![PartialDate::ymd(1965, 3, 12)];
    let id = c.add_party(p, &mia, t(2)).unwrap();
    c.screen_party(id, &mia, t(2)).unwrap();
    let a = c.alerts().values().next().unwrap();
    assert!(a.priority.is_none() && a.model_version.is_none());
    assert!(a.features.is_some(), "features are still captured so labels accumulate");
    assert_eq!(c.alert_queue().len(), 1);
}

#[test]
fn queue_puts_critical_first_and_never_hides_anything() {
    let w = world(150, 2);
    let q = w.c.alert_queue();
    let open = w.c.alerts().values().filter(|a| a.status.is_unresolved()).count();
    assert_eq!(q.len(), open, "every unresolved alert is in the queue");
    // bands are non-decreasing: Critical, Standard, Low
    assert!(q.windows(2).all(|p| p[0].band <= p[1].band));
    for i in &q {
        if i.alert.level == HitLevel::High {
            assert_eq!(i.band, PriorityBand::Critical);
        }
    }
    // Shadow mode orders by rule score within a band
    for band in [PriorityBand::Critical, PriorityBand::Standard] {
        let s: Vec<f64> = q.iter().filter(|i| i.band == band).map(|i| i.alert.score).collect();
        assert!(s.windows(2).all(|p| p[0] >= p[1]), "rule-score order inside {band:?}");
    }
}

#[test]
fn too_few_labels_cannot_activate_and_active_mode_is_refused() {
    let mut w = world(80, 3);
    // resolve only a few alerts
    let ids: Vec<(u64, u64, String)> = w.c.alerts().values().take(6).map(|a| (a.id, a.party, a.uid.clone())).collect();
    for (id, party, uid) in ids {
        let o = if w.truth.get(&party).map_or(false, |u| *u == uid) { Outcome::ConfirmedMatch } else { Outcome::FalsePositive };
        w.c.resolve_alert(id, o, "reviewed", &w.mia, t(3)).unwrap();
    }
    let r = w.c.train_match_model(&[], &small_gates(), true, &w.mia, t(4));
    match r {
        Err(Error::Invalid(m)) => assert!(m.contains("need both"), "{m}"),
        Ok((rep, activated)) => {
            assert!(!activated && !rep.passed);
            assert!(w.c.ml_model().is_none());
        }
        Err(e) => panic!("{e}"),
    }
    let e = w.c.set_ml_mode(MlMode::Active, None, &w.mia, t(4)).unwrap_err();
    assert!(matches!(e, Error::Invalid(ref m) if m.contains("validation gates")), "{e}");
}

#[test]
fn only_an_mlro_trains_or_changes_mode() {
    let mut w = world(40, 4);
    let e = w.c.train_match_model(&[], &small_gates(), false, &w.ana, t(4)).unwrap_err();
    assert!(matches!(e, Error::Forbidden(_)), "{e}");
    let e = w.c.set_ml_mode(MlMode::Off, None, &w.ana, t(4)).unwrap_err();
    assert!(matches!(e, Error::Forbidden(_)), "{e}");
}

#[test]
fn full_cycle_train_activate_reorder_and_nothing_is_decided_by_the_model() {
    let mut w = world(1500, 5);
    // first batch is resolved by humans; the second batch stays open to be re-prioritised
    let all: Vec<u64> = w.c.alerts().keys().copied().collect();
    let half = all.len() / 2;
    let resolved_ids: Vec<u64> = all[..half].to_vec();
    for id in &resolved_ids {
        let a = &w.c.alerts()[id];
        let is_true = w.truth.get(&a.party).map_or(false, |u| *u == a.uid);
        let (party, uid) = (a.party, a.uid.clone());
        let _ = (party, uid);
        w.c.resolve_alert(*id, if is_true { Outcome::ConfirmedMatch } else { Outcome::FalsePositive }, "reviewed by MLRO", &w.mia, t(3)).unwrap();
    }
    let status_before: HashMap<u64, (AlertStatus, HitLevel)> = w.c.alerts().iter().map(|(k, a)| (*k, (a.status, a.level))).collect();
    let open_before = w.c.alerts().values().filter(|a| a.status.is_unresolved()).count();
    assert!(open_before > 50);

    let (rep, activated) = w.c.train_match_model(&[], &small_gates(), true, &w.mia, t(4)).unwrap();
    assert!(rep.passed, "gates: {:?}", rep.gates);
    assert!(activated);
    let m = w.c.ml_model().expect("installed");
    assert!(m.training.is_some());

    // the model decided nothing: every status and level is exactly as the humans/rules left them
    for (k, a) in w.c.alerts() {
        assert_eq!((a.status, a.level), status_before[k], "alert {k} changed");
    }
    assert_eq!(w.c.alerts().values().filter(|a| a.status.is_unresolved()).count(), open_before);

    // open alerts were re-prioritised by the new model; resolved ones keep the priority they had at the time
    let version = w.c.ml_model().unwrap().version.clone();
    for a in w.c.alerts().values() {
        if a.status.is_unresolved() {
            assert_eq!(a.model_version.as_deref(), Some(version.as_str()));
        } else {
            assert_eq!(a.model_version.as_deref(), Some(PRIOR_VERSION));
        }
    }

    // Active mode is now allowed; queue is ordered by priority inside bands
    w.c.set_ml_mode(MlMode::Active, None, &w.mia, t(4)).unwrap();
    let q = w.c.alert_queue();
    assert_eq!(q.len(), open_before);
    for band in [PriorityBand::Critical, PriorityBand::Standard] {
        let p: Vec<f64> = q.iter().filter(|i| i.band == band).filter_map(|i| i.alert.priority).collect();
        assert!(p.windows(2).all(|x| x[0] >= x[1]), "priority order inside {band:?}");
    }

    // the open true matches should now read earlier than they would by rule score alone
    let truth = &w.truth;
    let is_true = |a: &Alert| truth.get(&a.party).map_or(false, |u| *u == a.uid);
    let pos_rank = |order: Vec<&Alert>| -> f64 {
        let ranks: Vec<usize> = order.iter().enumerate().filter(|(_, a)| is_true(a)).map(|(i, _)| i).collect();
        assert!(!ranks.is_empty());
        ranks.iter().sum::<usize>() as f64 / ranks.len() as f64
    };
    let model_order: Vec<&Alert> = q.iter().map(|i| i.alert).collect();
    let mut rule_order: Vec<&Alert> = w.c.alerts().values().filter(|a| a.status.is_unresolved()).collect();
    rule_order.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then(a.id.cmp(&b.id)));
    assert!(pos_rank(model_order) < pos_rank(rule_order), "model order should place true matches earlier on average");

    // audit trail records training, activation and the mode change, and the chain still verifies
    assert!(w.c.audit().verify().is_ok());
    let acts: Vec<&str> = w.c.audit().events().iter().map(|e| e.action.as_str()).collect();
    assert!(acts.contains(&"ml.train") && acts.contains(&"ml.mode"));

    // every alert is explainable
    let any_open = w.c.alerts().values().find(|a| a.status.is_unresolved()).unwrap().id;
    let (parts, _bias, ver) = w.c.explain_alert(any_open).unwrap();
    assert_eq!(ver, version);
    assert!(!parts.is_empty());

    // state round-trips with the model and the stored priorities
    let dir = std::env::temp_dir().join(format!("cdd_ml_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.json");
    w.c.save(&path).unwrap();
    let back = Compliance::load(&path).unwrap();
    assert_eq!(back.ml_model().unwrap().fingerprint(), w.c.ml_model().unwrap().fingerprint());
    assert_eq!(back.ml_cfg.mode, MlMode::Active);
    assert_eq!(back.alert_queue().len(), open_before);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn shadow_report_compares_stored_priority_with_the_rule_score() {
    let mut w = world(800, 6);
    resolve_all(&mut w);
    let r = w.c.shadow_report(None);
    assert_eq!(r.alerts, w.c.alerts().len());
    assert!(r.positives > 0);
    assert!(r.model_auc.is_some() && r.rule_auc.is_some());
    assert_eq!(w.c.shadow_report(Some("nonexistent-model")).alerts, 0);
}

#[test]
fn state_written_before_this_feature_still_loads() {
    let mut w = world(30, 7);
    let dir = std::env::temp_dir().join(format!("cdd_ml_old_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("state.json");
    w.c.save(&path).unwrap();
    // strip every field this feature added
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let obj = v.as_object_mut().unwrap();
    obj.remove("ml_cfg");
    obj.remove("ml_model");
    for a in obj.get_mut("alerts").unwrap().as_object_mut().unwrap().values_mut() {
        let a = a.as_object_mut().unwrap();
        a.remove("features");
        a.remove("priority");
        a.remove("model_version");
    }
    std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
    let back = Compliance::load(&path).unwrap();
    assert_eq!(back.ml_cfg.mode, MlMode::Shadow);
    assert!(back.ml_model().is_none());
    // alerts without features are still queued (by level / rule score) and simply unusable as labels
    let open = back.alerts().values().filter(|a| a.status.is_unresolved()).count();
    assert_eq!(back.alert_queue().len(), open);
    assert_eq!(back.ml_status().resolved_without_features, 0);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = w.mia;
}

#[test]
fn history_import_recomputes_features_and_skips_what_it_cannot_use() {
    let w = world(10, 8);
    let any = w.c.alerts().values().next().expect("alert");
    let (uid, name) = (any.uid.clone(), w.c.party(any.party).unwrap().name.clone());
    let subj = cdd_sanctions::screening::Subject::new(SubjectKind::Individual, &name);
    let recs = vec![
        HistoryRecord { subject: subj.clone(), list: "SYN".into(), uid: uid.clone(), confirmed: false, customer: Some(77) },
        HistoryRecord { subject: subj.clone(), list: "SYN".into(), uid: "no-such-entry".into(), confirmed: false, customer: None },
        HistoryRecord { subject: cdd_sanctions::screening::Subject::new(SubjectKind::Individual, "Completely Unrelated Name"), list: "SYN".into(), uid, confirmed: true, customer: None },
    ];
    let imp = w.c.import_history(&recs);
    assert_eq!(imp.examples.len(), 1);
    assert_eq!(imp.examples[0].group, 77);
    assert_eq!(imp.skipped_entry_missing, 1);
    assert_eq!(imp.skipped_below_review_threshold, 1);
}
