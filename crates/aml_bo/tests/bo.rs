use aml_bo::synth::{generate_world, BoSynthConfig};
use aml_bo::*;
use aml_graph::GTxn;
use cdd_sanctions::cdd::*;
use cdd_sanctions::engine::*;
use chrono::{TimeZone, Utc};

fn small() -> BoSynthConfig {
    let mut c = BoSynthConfig::default();
    c.base.days = 110;
    c.base.individuals = 300;
    c.base.merchants = 12;
    c.base.schemes_per_type = 2;
    c.schemes = 3;
    c.confounders = 3;
    c
}

#[test]
fn walk_agrees_with_cdd_engine_on_effective_ownership_and_unresolved() {
    let w = generate_world(&small());
    let an = Analyzer::new(&w.compliance, BoConfig::default()).unwrap();
    let mut checked = 0;
    for (p, party) in w.compliance.parties() {
        if party.kind != PartyKind::Entity {
            continue;
        }
        let rep = w.compliance.ownership_report(*p);
        let bo = an.analyse(*p).unwrap();
        for (u, f) in &rep.ultimate {
            let mine = bo.related_owners.contains(u) || *f < 0.10;
            assert!(mine || bo.ubos.iter().any(|x| x.party == *u), "owner {u} of {p} lost");
            if let Some(x) = bo.ubos.iter().find(|x| x.party == *u) {
                assert!((x.effective - f).abs() < 1e-9, "party {p} owner {u}: {} vs {}", x.effective, f);
            }
        }
        assert!((bo.unresolved - rep.unresolved_fraction().min(1.0)).abs() < 1e-9, "unresolved differs for {p}");
        assert_eq!(bo.layers, rep.max_layers, "layers differ for {p}");
        checked += 1;
    }
    assert!(checked > 50);
}

#[test]
fn circular_holdings_are_reported_not_counted() {
    let mut c = Compliance::default();
    let ana = Actor::new("ana", Role::Analyst);
    let now = Utc.with_ymd_and_hms(2026, 10, 1, 9, 0, 0).unwrap();
    let a = c.add_party(Party::new(PartyKind::Entity, "A Ltd"), &ana, now).unwrap();
    let b = c.add_party(Party::new(PartyKind::Entity, "B Ltd"), &ana, now).unwrap();
    let p = c.add_party(Party::new(PartyKind::Individual, "Pat Person"), &ana, now).unwrap();
    c.add_relationship(b, a, RelKind::Owner, Some(60.0), &ana, now).unwrap();
    c.add_relationship(a, b, RelKind::Owner, Some(30.0), &ana, now).unwrap();
    c.add_relationship(p, b, RelKind::Owner, Some(70.0), &ana, now).unwrap();
    let an = Analyzer::new(&c, BoConfig::default()).unwrap();
    let bo = an.analyse(a).unwrap();
    assert!(!bo.circular.is_empty());
    assert!((bo.ubos[0].effective - 0.6 * 0.7).abs() < 1e-9);
    assert!(bo.opacity_reasons.iter().any(|r| r.contains("circular")));
}

#[test]
fn sanctioned_ownership_is_found_and_near_misses_are_not_blocked() {
    let w = generate_world(&small());
    let an = OwnershipAnalysis::build(&w.compliance, w.registry.clone(), BoConfig::default()).unwrap();
    let blocked: Vec<_> = an.parties.values().filter(|b| b.blocked_ownership >= 0.5 && b.kind == PartyKind::Entity).collect();
    assert_eq!(blocked.len(), 3, "one per evasion scheme");
    let near: Vec<_> = an.parties.values().filter(|b| b.name.starts_with("Minority")).collect();
    assert_eq!(near.len(), 3);
    assert!(near.iter().all(|b| b.blocked_ownership > 0.25 && b.blocked_ownership < 0.5), "{:?}", near.iter().map(|b| b.blocked_ownership).collect::<Vec<_>>());
    // namesake false positives never become blocked
    assert!(an.parties.values().filter(|b| b.name.starts_with("Namesake")).all(|b| b.blocked_ownership == 0.0));
}

#[test]
fn groups_join_accounts_of_one_ownership_structure() {
    let w = generate_world(&small());
    let an = OwnershipAnalysis::build(&w.compliance, w.registry.clone(), BoConfig::default()).unwrap();
    let m = an.graph_map();
    assert_eq!(m.len(), w.registry.len());
    let group_of = |name: &str| -> Vec<u32> { an.parties.values().filter(|b| b.name.starts_with(name)).map(|b| b.group).collect() };
    let subs = group_of("Group 0 ");
    assert!(subs.len() >= 4 && subs.iter().all(|g| *g != 0 && *g == subs[0]), "{subs:?}");
    let a = an.parties.values().find(|b| b.name == "Group 0 Sub 0 Ltd").unwrap();
    let c = an.parties.values().find(|b| b.name == "Group 1 Sub 0 Ltd").unwrap();
    assert_ne!(a.group, c.group);
    // fragmented family business: four 24% holders, none a beneficial owner
    let fam = an.parties.values().find(|b| b.name == "Family 0 Ltd").unwrap();
    assert!(fam.fragmented && fam.no_owner_identified);
}

#[test]
fn monitor_fires_on_the_planted_structures_and_is_causal() {
    let w = generate_world(&small());
    let an = OwnershipAnalysis::build(&w.compliance, w.registry.clone(), BoConfig::default()).unwrap();
    let s = BoMonitor::score_all(&an, &w.txns, MonitorConfig::default());
    let rate = |typ: u8| {
        let v: Vec<f32> = w.txns.iter().zip(&s).filter(|(t, _)| t.typology == typ).map(|(_, s)| *s).collect();
        v.iter().filter(|x| **x >= 0.595).count() as f64 / v.len().max(1) as f64
    };
    assert!(rate(9) > 0.9, "sanctioned ownership recall {}", rate(9));
    assert!(rate(8) > 0.2, "shell group recall {}", rate(8));
    let normal: Vec<f32> = w.txns.iter().zip(&s).filter(|(t, _)| !t.label).map(|(_, s)| *s).collect();
    let fp = normal.iter().filter(|x| **x >= 0.595).count() as f64 / normal.len() as f64;
    assert!(fp < 0.05, "false-positive rate {fp}");
    // causal: scoring a prefix gives the same scores as scoring the whole stream
    let half = w.txns.len() / 2;
    let s2 = BoMonitor::score_all(&an, &w.txns[..half], MonitorConfig::default());
    assert_eq!(&s[..half], &s2[..]);
}

#[test]
fn world_is_deterministic_and_csv_roundtrips() {
    let a = generate_world(&small());
    let b = generate_world(&small());
    assert_eq!(a.txns.len(), b.txns.len());
    assert!(a.txns.iter().zip(&b.txns).all(|(x, y)| x.sender == y.sender && x.receiver == y.receiver && x.ts == y.ts && x.amount == y.amount));
    for ty in 1..12u8 {
        assert!(a.txns.iter().any(|t| t.typology == ty), "typology {ty} missing");
    }
    let dir = std::env::temp_dir().join(format!("aml_bo_rt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    a.registry.write_csv(&dir.join("a.csv")).unwrap();
    let r = AccountRegistry::from_csv(&dir.join("a.csv")).unwrap();
    assert_eq!(r.len(), a.registry.len());
    a.compliance.save(&dir.join("c.json")).unwrap();
    let c2 = Compliance::load(&dir.join("c.json")).unwrap();
    assert_eq!(c2.parties().len(), a.compliance.parties().len());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn ownership_features_reach_the_graph_model_inputs() {
    use aml_graph::{Pipeline, RiskTable};
    let w = generate_world(&small());
    let an = OwnershipAnalysis::build(&w.compliance, w.registry.clone(), BoConfig::default()).unwrap();
    let ecfg = aml_engine::default_config();
    let plain = RiskTable::from_engine(&ecfg);
    let own = RiskTable::from_engine(&ecfg).with_ownership(std::sync::Arc::new(an.graph_map()));
    let m = aml_graph::CirGnn::new(aml_graph::CirGnnConfig::default()).unwrap();
    let fc = aml_graph::train::feature_config(&m);
    let a: Vec<aml_graph::WindowData> = Pipeline::new(&w.txns, fc.clone(), plain, 14).take(40).collect();
    let b: Vec<aml_graph::WindowData> = Pipeline::new(&w.txns, fc, own, 14).take(40).collect();
    let diff = a.iter().zip(&b).any(|(x, y)| x.x_raw != y.x_raw);
    assert!(diff, "ownership did not change any node features");
    let _: Option<&GTxn> = None;
}
