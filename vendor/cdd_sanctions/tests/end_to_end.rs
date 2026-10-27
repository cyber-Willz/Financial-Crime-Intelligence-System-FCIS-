use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use cdd_sanctions::cdd::*;
use cdd_sanctions::engine::*;
use cdd_sanctions::error::Error;
use cdd_sanctions::hypernet::NetworkConfig;
use cdd_sanctions::model::*;
use cdd_sanctions::screening::HitLevel;

fn t(y: i32, m: u32, d: u32) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(y, m, d, 9, 0, 0).unwrap()
}
fn date(y: i32, m: u32, d: u32) -> NaiveDate {
    NaiveDate::from_ymd_opt(y, m, d).unwrap()
}

fn entry(uid: &str, kind: SubjectKind, name: &str, dob: Option<PartialDate>, nat: &str) -> ListEntry {
    ListEntry {
        list: "TEST".into(),
        uid: uid.into(),
        kind,
        names: vec![NameVariant { name: name.into(), strength: AliasStrength::Primary }],
        dobs: dob.into_iter().collect(),
        nationalities: if nat.is_empty() { vec![] } else { vec![nat.into()] },
        countries: vec![],
        identifiers: vec![],
        programs: vec!["TESTPROG".into()],
        remarks: None,
        listed_on: None,
        content_hash: String::new(),
    }
    .finalize()
}

fn petrov() -> ListEntry {
    entry("1", SubjectKind::Individual, "PETROV, Ivan Sergeyevich", Some(PartialDate::ymd(1965, 3, 12)), "RU")
}

fn unrelated() -> ListEntry {
    entry("99", SubjectKind::Individual, "UNRELATED, Zed", None, "")
}

struct Ctx {
    c: Compliance,
    ana: Actor,
    rex: Actor,
    mia: Actor,
}

fn setup(entries: Vec<ListEntry>) -> Ctx {
    let mut c = Compliance::default();
    let (ana, rex, mia) = (Actor::new("ana", Role::Analyst), Actor::new("rex", Role::Reviewer), Actor::new("mia", Role::Mlro));
    if !entries.is_empty() {
        c.load_list("TEST", entries, "v1", false, &mia, t(2026, 10, 1)).unwrap();
    }
    Ctx { c, ana, rex, mia }
}

fn verified(kind: DocKind) -> Document {
    let mut d = Document::new(kind, "ref");
    d.verified_by = Some("ana".into());
    d.expires = Some(date(2035, 1, 1));
    d
}

fn person(name: &str, dob: Option<PartialDate>, nat: &str) -> Party {
    let mut p = Party::new(PartyKind::Individual, name);
    p.dob = dob.into_iter().collect();
    p.nationalities = vec![nat.into()];
    p
}

#[test]
fn low_risk_onboarding_happy_path_and_four_eyes() {
    let Ctx { mut c, ana, rex, mia } = setup(vec![petrov()]);
    let now = t(2026, 10, 13);
    let mut p = person("Maria Gonzalez", Some(PartialDate::ymd(1990, 5, 5)), "Belize");
    p.documents = vec![verified(DocKind::IdDocument), verified(DocKind::ProofOfAddress)];
    let id = c.add_party(p, &ana, now).unwrap();
    assert_eq!(c.screen_party(id, &ana, now).unwrap().new_alerts, 0);

    let case = c.open_case(id, CaseKind::Onboarding, &ana, now).unwrap();
    assert!(c.submit_case(case, &ana, now).unwrap().is_empty());
    // analysts cannot approve
    assert!(matches!(c.approve_case(case, &ana, now), Err(Error::Forbidden(_))));
    c.approve_case(case, &rex, now).unwrap();
    assert_eq!(c.party(id).unwrap().status, PartyStatus::Active);
    let cs = &c.cases()[&case];
    assert_eq!(cs.status, CaseStatus::Approved);
    assert_eq!(cs.next_review, Some(date(2029, 10, 13))); // low tier: 36 months

    // a reviewer cannot approve a case they prepared
    let case2 = c.open_case(id, CaseKind::PeriodicReview, &rex, now).unwrap();
    c.submit_case(case2, &rex, now).unwrap();
    let e = c.approve_case(case2, &rex, now).unwrap_err();
    assert!(matches!(e, Error::Forbidden(ref m) if m.contains("four-eyes")), "{e}");
    c.approve_case(case2, &mia, now).unwrap();

    // review scheduling
    assert!(c.reviews_due(date(2027, 1, 1), 30).is_empty());
    assert_eq!(c.reviews_due(date(2029, 10, 1), 30).len(), 1);
    assert!(c.audit().verify().is_ok());
}

#[test]
fn approval_lists_every_unmet_requirement() {
    let Ctx { mut c, ana, rex, .. } = setup(vec![petrov()]);
    let now = t(2026, 10, 13);
    let mut expired = verified(DocKind::IdDocument);
    expired.expires = Some(date(2026, 1, 1));
    let mut p = person("Maria Gonzalez", None, "Belize");
    p.pep = PepStatus::No;
    p.documents = vec![expired, Document::new(DocKind::ProofOfAddress, "unverified bill")];
    p.channel = Channel::NonFaceToFace;
    p.industry = Some("casino operator".into()); // 10 + 20 => Medium => standard DD
    let id = c.add_party(p, &ana, now).unwrap();
    let case = c.open_case(id, CaseKind::Onboarding, &ana, now).unwrap();
    let blockers = c.submit_case(case, &ana, now).unwrap();
    let codes: Vec<&str> = blockers.iter().map(|b| b.code.as_str()).collect();
    assert!(codes.contains(&"NOT_SCREENED"), "{codes:?}");
    assert!(codes.contains(&"DOC_EXPIRED"), "{codes:?}");
    assert!(codes.contains(&"DOC_UNVERIFIED"), "{codes:?}");
    match c.approve_case(case, &rex, now) {
        Err(Error::Blocked(v)) => assert!(v.len() >= 3),
        other => panic!("expected Blocked, got {other:?}"),
    }
    assert_eq!(c.party(id).unwrap().status, PartyStatus::Prospect);
}

#[test]
fn high_risk_needs_mlro_and_enhanced_documents() {
    let Ctx { mut c, ana, rex, mia } = setup(vec![unrelated()]);
    let now = t(2026, 10, 13);
    let mut p = person("Carlos Mena", Some(PartyKind::Individual).map(|_| PartialDate::year(1970)), "Belize");
    p.pep = PepStatus::Pep;
    p.documents = vec![verified(DocKind::IdDocument), verified(DocKind::ProofOfAddress)];
    let id = c.add_party(p, &ana, now).unwrap();
    c.screen_party(id, &ana, now).unwrap();
    let case = c.open_case(id, CaseKind::Onboarding, &ana, now).unwrap();
    let blockers = c.submit_case(case, &ana, now).unwrap();
    let missing: Vec<String> = blockers.iter().filter(|b| b.code == "DOC_MISSING").map(|b| b.message.clone()).collect();
    assert_eq!(missing.len(), 2, "{blockers:?}"); // source of wealth + source of funds
    c.add_document(id, verified(DocKind::SourceOfWealth), &ana, now).unwrap();
    c.add_document(id, verified(DocKind::SourceOfFunds), &ana, now).unwrap();
    assert!(matches!(c.approve_case(case, &rex, now), Err(Error::Forbidden(_)))); // EDD sign-off is MLRO-only
    c.approve_case(case, &mia, now).unwrap();
    assert_eq!(c.cases()[&case].next_review, Some(date(2027, 10, 13))); // high: 12 months
}

#[test]
fn confirmed_match_blocks_party_and_holds_owned_entities() {
    let Ctx { mut c, ana, rex, mia } = setup(vec![petrov()]);
    let now = t(2026, 10, 13);
    let a = c.add_party(person("Ivan Petrov", Some(PartialDate::ymd(1965, 3, 12)), "Russia"), &ana, now).unwrap();
    let e = c.add_party(Party::new(PartyKind::Entity, "Acme Holdings Ltd"), &ana, now).unwrap();
    c.add_relationship(a, e, RelKind::Owner, Some(60.0), &ana, now).unwrap();
    let case = c.open_case(e, CaseKind::Onboarding, &ana, now).unwrap();

    let sum = c.screen_party(e, &ana, now).unwrap();
    assert_eq!((sum.parties_screened, sum.new_alerts), (2, 1)); // the owner was screened too
    let alert = c.alerts().values().next().unwrap().clone();
    assert_eq!((alert.party, alert.root_party, alert.level), (a, e, HitLevel::High));
    assert!(alert.reasons.iter().any(|r| r.contains("date of birth matches")));

    // role gates and rationale requirement
    assert!(matches!(c.resolve_alert(alert.id, Outcome::FalsePositive, "looks different", &ana, now), Err(Error::Forbidden(_))));
    assert!(matches!(c.resolve_alert(alert.id, Outcome::ConfirmedMatch, "same person", &rex, now), Err(Error::Forbidden(_))));
    assert!(matches!(c.resolve_alert(alert.id, Outcome::ConfirmedMatch, "  ", &mia, now), Err(Error::Invalid(_))));
    c.resolve_alert(alert.id, Outcome::Escalate, "needs MLRO", &ana, now).unwrap();
    assert!(c.alerts()[&alert.id].status.is_unresolved());

    c.resolve_alert(alert.id, Outcome::ConfirmedMatch, "passport and DOB match listing", &mia, now).unwrap();
    assert_eq!(c.party(a).unwrap().status, PartyStatus::Blocked);
    assert_eq!(c.party(e).unwrap().status, PartyStatus::OnHold);
    let cs = &c.cases()[&case];
    assert!(cs.report_required);
    assert_eq!(cs.status, CaseStatus::Rejected);

    let ra = c.assess_risk(e, now).unwrap();
    assert_eq!(ra.tier, RiskTier::Prohibited); // 60% owned by a blocked person
    let (_, blockers) = c.readiness(e, now).unwrap();
    assert_eq!(blockers[0].code, "PROHIBITED");
    assert!(c.resolve_alert(alert.id, Outcome::FalsePositive, "oops", &mia, now).is_err()); // final
}

#[test]
fn false_positive_dispositions_survive_rescreens_but_not_changes() {
    let Ctx { mut c, ana, mia, .. } = setup(vec![petrov()]);
    let now = t(2026, 10, 13);
    // name-only match, no DOB => Review level
    let id = c.add_party(person("Ivan Sergeyevich Petrov", None, "Belize"), &ana, now).unwrap();
    assert_eq!(c.screen_party(id, &ana, now).unwrap().new_alerts, 1);
    let alert = c.alerts().values().next().unwrap().clone();
    assert_eq!(alert.level, HitLevel::Review);
    c.resolve_alert(alert.id, Outcome::FalsePositive, "Belizean national, different person", &ana, now).unwrap();

    let again = c.screen_party(id, &ana, now).unwrap();
    assert_eq!((again.new_alerts, again.suppressed), (0, 1));

    // customer details change => the old disposition no longer applies
    c.update_party(id, |p| p.aliases.push("Vanya Petrov".into()), &ana, now).unwrap();
    assert_eq!(c.screen_party(id, &ana, now).unwrap().new_alerts, 1);
    let second = c.alerts().values().filter(|a| a.status == AlertStatus::Open).next().unwrap().id;
    c.resolve_alert(second, Outcome::FalsePositive, "re-reviewed with alias; still different person", &ana, now).unwrap();
    assert_eq!(c.screen_party(id, &ana, now).unwrap().suppressed, 1);

    // listing changes => re-alert, even though customer data did not change
    let mut changed = petrov();
    changed.remarks = Some("new information".into());
    let changed = changed.finalize();
    let rep = c.load_list("TEST", vec![changed], "v2", false, &mia, t(2026, 10, 20)).unwrap();
    assert_eq!(rep.diff.changed, vec!["1"]);
    assert_eq!(rep.new_alerts, 1);
}

#[test]
fn list_updates_rescreen_existing_customers_and_flag_delistings() {
    let Ctx { mut c, ana, mia, .. } = setup(vec![]);
    let now = t(2026, 10, 13);
    let id = c.add_party(person("Ivan Petrov", Some(PartialDate::ymd(1965, 3, 12)), "Russia"), &ana, now).unwrap();
    c.screen_party(id, &ana, now).unwrap();
    let case = c.open_case(id, CaseKind::PeriodicReview, &ana, now).unwrap();
    assert!(c.alerts().is_empty());

    // a new designation appears after the customer was onboarded
    let other = entry("2", SubjectKind::Individual, "GONZALEZ, Maria", None, "");
    let rep = c.load_list("TEST", vec![petrov(), other.clone()], "v1", false, &mia, t(2026, 10, 14)).unwrap();
    assert_eq!((rep.diff.added.len(), rep.new_alerts, rep.parties_screened), (2, 1, 1));
    let aid = *c.alerts().keys().next().unwrap();
    assert_eq!(c.alerts()[&aid].level, HitLevel::High);

    // unchanged re-load: nothing to screen
    let rep = c.load_list("TEST", vec![petrov(), other.clone()], "v1b", false, &mia, t(2026, 10, 15)).unwrap();
    assert_eq!((rep.parties_screened, rep.new_alerts), (0, 0));

    // delisting flags the open alert for review
    let third = entry("3", SubjectKind::Individual, "SMITH, Alan", None, "");
    let rep = c.load_list("TEST", vec![other.clone(), third.clone()], "v2", false, &mia, t(2026, 10, 16)).unwrap();
    assert_eq!(rep.diff.removed, vec!["1"]);
    assert_eq!(rep.delisted_alerts, 1);
    assert!(c.alerts()[&aid].delisted);
    let _ = case;

    // guards against truncated downloads
    assert!(c.load_list("TEST", vec![], "bad", false, &mia, now).is_err());
    let many: Vec<ListEntry> = (0..10).map(|i| entry(&format!("x{i}"), SubjectKind::Entity, &format!("Entity {i}"), None, "")).collect();
    c.load_list("TEST", many, "v3", true, &mia, now).unwrap();
    let e = c.load_list("TEST", vec![third], "v4", false, &mia, now).unwrap_err();
    assert!(e.to_string().contains("shrink"), "{e}");
    assert!(c.audit().verify().is_ok());
}

#[test]
fn fifty_percent_rule_via_two_listed_owners() {
    let list = vec![
        entry("1", SubjectKind::Individual, "PETROV, Ivan Sergeyevich", Some(PartialDate::ymd(1965, 3, 12)), "RU"),
        entry("2", SubjectKind::Individual, "KOVALENKO, Dmitri Olegovich", Some(PartialDate::ymd(1971, 8, 2)), "UA"),
    ];
    let Ctx { mut c, ana, mia, .. } = setup(list);
    let now = t(2026, 10, 13);
    let p1 = c.add_party(person("Ivan Petrov", Some(PartialDate::ymd(1965, 3, 12)), "Russia"), &ana, now).unwrap();
    let p2 = c.add_party(person("Dmitri Kovalenko", Some(PartialDate::ymd(1971, 8, 2)), "Ukraine"), &ana, now).unwrap();
    let p3 = c.add_party(person("Maria Gonzalez", Some(PartialDate::ymd(1990, 5, 5)), "Belize"), &ana, now).unwrap();
    let e = c.add_party(Party::new(PartyKind::Entity, "Reef Developments Ltd"), &ana, now).unwrap();
    for (p, pct) in [(p1, 30.0), (p2, 25.0), (p3, 45.0)] {
        c.add_relationship(p, e, RelKind::Owner, Some(pct), &ana, now).unwrap();
    }
    c.screen_party(e, &ana, now).unwrap();
    assert_ne!(c.assess_risk(e, now).unwrap().tier, RiskTier::Prohibited); // alerts open, nothing confirmed yet
    let ids: Vec<u64> = c.alerts().keys().copied().collect();
    assert_eq!(ids.len(), 2);
    c.resolve_alert(ids[0], Outcome::ConfirmedMatch, "confirmed", &mia, now).unwrap();
    // only one owner confirmed: 30% or 25% < 50%, but the entity is on hold
    let after_one = c.assess_risk(e, now).unwrap();
    assert_eq!(after_one.tier, RiskTier::High, "{:?}", after_one.overrides);
    assert_eq!(c.party(e).unwrap().status, PartyStatus::OnHold);
    assert!(matches!(c.release_hold(e, "x", &Actor::new("rex", Role::Reviewer), now), Err(Error::Forbidden(_))));
    assert!(matches!(c.release_hold(c.alerts()[&ids[0]].party, "x", &mia, now), Err(Error::Invalid(_)))); // blocked, not on hold
    c.resolve_alert(ids[1], Outcome::ConfirmedMatch, "confirmed", &mia, now).unwrap();
    let r = c.assess_risk(e, now).unwrap();
    assert_eq!(r.tier, RiskTier::Prohibited);
    assert!(r.overrides.iter().any(|o| o.contains("55%")), "{:?}", r.overrides);
}

#[test]
fn ownership_gaps_block_entity_approval() {
    let Ctx { mut c, ana, rex, .. } = setup(vec![unrelated()]);
    let now = t(2026, 10, 13);
    let mut e = Party::new(PartyKind::Entity, "Opaque Holdings Ltd");
    e.documents = vec![verified(DocKind::Registration), verified(DocKind::OwnershipChart)];
    let e = c.add_party(e, &ana, now).unwrap();
    let parent = c.add_party(Party::new(PartyKind::Entity, "Parent Corp"), &ana, now).unwrap();
    c.add_relationship(parent, e, RelKind::Owner, Some(100.0), &ana, now).unwrap();
    c.screen_party(e, &ana, now).unwrap();
    let case = c.open_case(e, CaseKind::Onboarding, &ana, now).unwrap();
    let codes: Vec<String> = c.submit_case(case, &ana, now).unwrap().into_iter().map(|b| b.code).collect();
    assert!(codes.contains(&"OWNERSHIP_UNRESOLVED".to_string()), "{codes:?}");
    assert!(codes.contains(&"NO_UBO_OR_SMO".to_string()), "{codes:?}");
    assert!(matches!(c.approve_case(case, &rex, now), Err(Error::Blocked(_))));

    // trace to a natural person with a verified ID: unblocked
    let mut owner = person("Maria Gonzalez", Some(PartialDate::ymd(1990, 5, 5)), "Belize");
    owner.documents = vec![verified(DocKind::IdDocument)];
    let o = c.add_party(owner, &ana, now).unwrap();
    c.add_relationship(o, parent, RelKind::Owner, Some(100.0), &ana, now).unwrap();
    c.screen_party(e, &ana, now).unwrap();
    assert_eq!(c.beneficial_owners(e), vec![o]);
    let (_, blockers) = c.readiness(e, now).unwrap();
    assert!(blockers.is_empty(), "{blockers:?}");
    c.submit_case(case, &ana, now).unwrap();
    c.approve_case(case, &rex, now).unwrap();
}

#[test]
fn relationship_validation() {
    let Ctx { mut c, ana, .. } = setup(vec![]);
    let now = t(2026, 10, 13);
    let a = c.add_party(person("A Person", None, "Belize"), &ana, now).unwrap();
    let b = c.add_party(person("B Person", None, "Belize"), &ana, now).unwrap();
    let e = c.add_party(Party::new(PartyKind::Entity, "E Ltd"), &ana, now).unwrap();
    assert!(c.add_relationship(a, a, RelKind::Director, None, &ana, now).is_err());
    assert!(c.add_relationship(a, b, RelKind::Owner, Some(50.0), &ana, now).is_err()); // people aren't owned
    assert!(c.add_relationship(a, e, RelKind::Owner, None, &ana, now).is_err());
    assert!(c.add_relationship(a, e, RelKind::Owner, Some(120.0), &ana, now).is_err());
    assert!(c.add_relationship(a, 999, RelKind::Owner, Some(10.0), &ana, now).is_err());
    assert!(c.add_relationship(a, e, RelKind::Director, None, &ana, now).is_ok());
}

#[test]
fn state_roundtrips_through_disk_and_rebuilds_the_index() {
    let Ctx { mut c, ana, mia, .. } = setup(vec![petrov()]);
    let now = t(2026, 10, 13);
    let id = c.add_party(person("Ivan Petrov", Some(PartialDate::ymd(1965, 3, 12)), "Russia"), &ana, now).unwrap();
    c.screen_party(id, &ana, now).unwrap();
    let path = std::env::temp_dir().join(format!("cdd_state_{}.json", std::process::id()));
    c.save(&path).unwrap();
    let mut loaded = Compliance::load(&path).unwrap();
    std::fs::remove_file(&path).ok();
    assert_eq!(loaded.parties().len(), 1);
    assert_eq!(loaded.alerts().len(), 1);
    assert_eq!(loaded.index().len(), 1);
    assert!(loaded.audit().verify().is_ok());
    // still operational: ids continue, screening works, alert is deduplicated
    let s = loaded.screen_party(id, &ana, t(2026, 10, 14)).unwrap();
    assert_eq!((s.new_alerts, s.hits), (0, 1));
    let aid = *loaded.alerts().keys().next().unwrap();
    loaded.resolve_alert(aid, Outcome::ConfirmedMatch, "confirmed", &mia, t(2026, 10, 14)).unwrap();
    let nid = loaded.add_party(person("Next Person", None, "Belize"), &ana, now).unwrap();
    assert!(nid > aid);
    assert_eq!(loaded.list_versions().len(), 1);
}

#[test]
fn approval_needs_loaded_and_fresh_lists() {
    // no list at all: screening is vacuous, so approval must not go through
    let Ctx { mut c, ana, .. } = setup(vec![]);
    let now = t(2026, 10, 13);
    let mut p = person("Maria Gonzalez", Some(PartialDate::ymd(1990, 5, 5)), "Belize");
    p.documents = vec![verified(DocKind::IdDocument), verified(DocKind::ProofOfAddress)];
    let id = c.add_party(p, &ana, now).unwrap();
    c.screen_party(id, &ana, now).unwrap();
    let codes = |c: &Compliance, now| c.readiness(id, now).unwrap().1.into_iter().map(|b| b.code).collect::<Vec<_>>();
    assert!(codes(&c, now).contains(&"NO_LISTS_LOADED".to_string()));

    // a loaded list clears it; staleness only matters once a maximum age is configured
    let mia = Actor::new("mia", Role::Mlro);
    c.load_list("BZ-FIU", vec![unrelated()], "2026-07", false, &mia, t(2026, 10, 1)).unwrap();
    c.screen_party(id, &ana, now).unwrap();
    assert!(codes(&c, t(2026, 12, 1)).is_empty());
    c.screening_cfg.max_list_age_days = Some(35);
    assert!(codes(&c, t(2026, 10, 20)).is_empty());
    assert_eq!(codes(&c, t(2026, 12, 1)), vec!["LIST_STALE"]);
    assert_eq!(c.stale_lists(date(2026, 12, 1), 35), vec![("BZ-FIU".to_string(), date(2026, 10, 1))]);
    // a fresh load of that list clears the warning
    c.load_list("BZ-FIU", vec![unrelated(), entry("98", SubjectKind::Entity, "OTHER CO", None, "")], "2026-12", false, &mia, t(2026, 11, 30)).unwrap();
    assert!(c.stale_lists(date(2026, 12, 1), 35).is_empty());
}

// ---------------------------------------------------------------------------------------
// Spectral hypergraph network analysis
// ---------------------------------------------------------------------------------------

struct World {
    c: Compliance,
    ana: Actor,
    mia: Actor,
    petrov: PartyId,
    volga: PartyId,
    orlov: PartyId,
    direct: PartyId, // customer sharing Orlov (a Volga director) as its own director
    quiet: PartyId,  // customer with no connection
}

fn ent(c: &mut Compliance, name: &str, ana: &Actor) -> PartyId {
    c.add_party(Party::new(PartyKind::Entity, name), ana, t(2026, 10, 1)).unwrap()
}

/// Petrov (listed) owns 80% of Volga; Orlov directs Volga and also directs the customer `direct`.
/// Petrov is not in `direct`'s ownership/control network, so only the spectral analysis can see the link.
fn world(confirm: bool) -> World {
    build_world(true, confirm)
}

/// `screen = false`: nothing screened yet, so no alerts and no seeds.
fn build_world(screen: bool, confirm: bool) -> World {
    let Ctx { mut c, ana, mia, .. } = setup(vec![petrov()]);
    let now = t(2026, 10, 1);
    let petrov_id = c.add_party(person("Ivan Petrov", Some(PartialDate::ymd(1965, 3, 12)), "Russia"), &ana, now).unwrap();
    let volga = ent(&mut c, "Volga Trading Ltd", &ana);
    let orlov = c.add_party(person("Dmitri Orlov", None, "Belize"), &ana, now).unwrap();
    let direct = ent(&mut c, "Shared Director Ltd", &ana);
    let quiet = ent(&mut c, "Quiet Customer Ltd", &ana);
    let quiet_dir = c.add_party(person("Unrelated Director", None, "Belize"), &ana, now).unwrap();
    c.add_relationship(petrov_id, volga, RelKind::Owner, Some(80.0), &ana, now).unwrap();
    c.add_relationship(orlov, volga, RelKind::Director, None, &ana, now).unwrap();
    c.add_relationship(orlov, direct, RelKind::Director, None, &ana, now).unwrap();
    // Orlov sits on other boards too, which dilutes how exclusively `direct` is tied to Volga
    for name in ["Side Business One Ltd", "Side Business Two Ltd"] {
        let side = ent(&mut c, name, &ana);
        c.add_relationship(orlov, side, RelKind::Director, None, &ana, now).unwrap();
    }
    c.add_relationship(quiet_dir, quiet, RelKind::Director, None, &ana, now).unwrap();
    if screen {
        c.screen_party(petrov_id, &ana, now).unwrap();
    }
    if confirm {
        let alert = c.alerts().values().find(|a| a.party == petrov_id).expect("alert on Petrov").id;
        c.resolve_alert(alert, Outcome::ConfirmedMatch, "passport and DOB match listing", &mia, now).unwrap();
    }
    World { c, ana, mia, petrov: petrov_id, volga, orlov, direct, quiet }
}

#[test]
fn indirect_link_to_a_sanctioned_structure_raises_risk_but_never_blocks() {
    let now = t(2026, 10, 13);
    let before = build_world(false, false);
    assert_eq!(before.c.assess_risk(before.direct, now).unwrap().score, 5, "entity points only: nothing screened, no seeds");

    let w = world(true);
    let e = w.c.network_exposure(w.direct).unwrap();
    assert_eq!(e.source, Some(w.petrov));
    assert_eq!(e.hops, Some(3));
    assert_eq!(e.path.len(), 3, "{:?}", e.path);
    assert!(e.path[0].contains("Ivan Petrov") && e.path[2].contains("Shared Director Ltd"), "{:?}", e.path);
    assert!(e.exposure >= w.c.risk_cfg.network.moderate_exposure && e.exposure < w.c.risk_cfg.network.strong_exposure, "{}", e.exposure);

    let a = w.c.assess_risk(w.direct, now).unwrap();
    assert_eq!(a.tier, RiskTier::Medium, "{a:?}");
    assert_eq!(a.dd_level, DdLevel::Standard);
    assert!(a.factors.iter().any(|f| f.name.starts_with("indirect sanctions exposure")));
    // advisory only: nothing was blocked, held or prohibited
    assert_eq!(w.c.party(w.direct).unwrap().status, PartyStatus::Prospect);
    assert!(a.tier < RiskTier::Prohibited);

    // an unconnected customer is untouched
    let q = w.c.assess_risk(w.quiet, now).unwrap();
    assert_eq!((q.tier, q.score), (RiskTier::Low, 5));
    assert_eq!(w.c.network_exposure(w.quiet).unwrap().exposure, 0.0);
}

#[test]
fn directors_of_the_sanctioned_owned_company_get_a_high_floor_needing_the_mlro() {
    let w = world(true);
    let now = t(2026, 10, 13);
    let a = w.c.assess_risk(w.orlov, now).unwrap();
    assert_eq!(a.tier, RiskTier::High, "{a:?}");
    assert_eq!(a.dd_level, DdLevel::Enhanced);
    assert!(a.overrides.iter().any(|o| o.contains("indirect sanctions exposure") && o.contains("MLRO")), "{:?}", a.overrides);
    assert!(a.factors.iter().any(|f| f.name.starts_with("strong indirect")));
}

#[test]
fn sanctioned_parties_inside_the_ownership_network_are_not_double_counted() {
    // Volga is owned 80% by Petrov: the ownership rules already handle that (>= 50% => prohibited).
    // The spectral factor must not pile on for a seed that sits in the party's own network.
    let w = world(true);
    let now = t(2026, 10, 13);
    let e = w.c.network_exposure(w.volga).unwrap();
    assert_eq!(e.exposure, 0.0);
    let a = w.c.assess_risk(w.volga, now).unwrap();
    assert_eq!(a.tier, RiskTier::Prohibited);
    assert!(!a.factors.iter().any(|f| f.name.contains("indirect")), "{:?}", a.factors);
    // ...but the portfolio report still shows it, flagged as covered by the ownership rules
    let rep = w.c.network_analysis().unwrap();
    let volga = rep.exposures.iter().find(|x| x.party == w.volga).expect("volga listed");
    assert!(volga.covered_by_ownership_rules);
    let orlov = rep.exposures.iter().find(|x| x.party == w.orlov).expect("orlov listed");
    assert!(!orlov.covered_by_ownership_rules);
    // the quiet customer's 2-party component has no seed and is below the reporting size
    assert_eq!(rep.components.len(), 1);
    assert_eq!(rep.components[0].seeds, vec![(w.petrov, 1.0)]);
}

#[test]
fn unresolved_high_alert_is_a_weaker_seed_than_a_confirmed_match() {
    let confirmed = world(true);
    let pending = world(false);
    assert!(pending.c.network_seeds().iter().any(|(p, w)| *p == pending.petrov && (*w - 0.3).abs() < 1e-12), "{:?}", pending.c.network_seeds());
    let a = confirmed.c.network_exposure(confirmed.direct).unwrap().exposure;
    let b = pending.c.network_exposure(pending.direct).unwrap().exposure;
    assert!(b > 0.0 && (b / a - 0.3).abs() < 1e-9, "ratio {}", b / a);
    // review-level alerts seed nothing by default
    assert_eq!(pending.c.risk_cfg.network.review_alert_seed, 0.0);
}

#[test]
fn false_positive_dispositions_do_not_seed_the_network() {
    let Ctx { mut c, ana, mia, .. } = setup(vec![petrov()]);
    let now = t(2026, 10, 1);
    let p = c.add_party(person("Ivan Petrov", Some(PartialDate::ymd(1971, 1, 1)), "Belize"), &ana, now).unwrap();
    let co = ent(&mut c, "Neighbour Ltd", &ana);
    c.add_relationship(p, co, RelKind::Director, None, &ana, now).unwrap();
    c.screen_party(co, &ana, now).unwrap();
    for a in c.alerts().values().map(|a| a.id).collect::<Vec<_>>() {
        c.resolve_alert(a, Outcome::FalsePositive, "DOB conflict: different person", &mia, now).unwrap();
    }
    assert!(c.network_seeds().is_empty());
}

#[test]
fn exposure_factor_can_be_switched_off_and_old_configs_still_load() {
    let mut w = world(true);
    let now = t(2026, 10, 13);
    w.c.risk_cfg.network.enabled = false;
    assert_eq!(w.c.network_exposure(w.direct).unwrap().exposure, 0.0);
    assert_eq!(w.c.assess_risk(w.direct, now).unwrap().tier, RiskTier::Low);

    // a risk config written before this feature (and the shipped example) has no `network` block
    let old: RiskConfig = serde_json::from_str(include_str!("../config/risk.example.json")).unwrap();
    assert!(old.network.enabled);
    assert_eq!(old.tier_high, 60);
    let partial: RiskConfig = serde_json::from_str(r#"{"network": {"strong_exposure": 0.05}}"#).unwrap();
    assert_eq!(partial.network.strong_exposure, 0.05);
    assert_eq!(partial.network.moderate_exposure, NetworkConfig::default().moderate_exposure);
}

#[test]
fn analysis_survives_persistence_and_is_deterministic() {
    let w = world(true);
    let path = std::env::temp_dir().join(format!("cdd_hypernet_{}.json", std::process::id()));
    w.c.save(&path).unwrap();
    let loaded = Compliance::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    let (a, b) = (w.c.network_analysis().unwrap(), loaded.network_analysis().unwrap());
    assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&b).unwrap());
    assert_eq!(serde_json::to_string(&a).unwrap(), serde_json::to_string(&w.c.network_analysis().unwrap()).unwrap());
    assert!(w.ana.name == "ana" && w.mia.name == "mia");
}

#[test]
fn shared_identifier_links_otherwise_unrelated_customers() {
    // Two customers with no recorded relationship quote the same passport number; one of them
    // directs a company owned by a confirmed sanctioned person.
    let w = world(true);
    let now = t(2026, 10, 13);
    let mut c = w.c;
    let twin = c.add_party(person("Passport Twin", None, "Belize"), &w.ana, now).unwrap();
    for p in [twin, w.orlov] {
        c.update_party(p, |x| x.identifiers.push(Identifier::new(IdKind::Passport, "P9988776", Some("BZ"))), &w.ana, now).unwrap();
    }
    let e = c.network_exposure(twin).unwrap();
    assert!(e.exposure > c.risk_cfg.network.moderate_exposure, "{}", e.exposure);
    assert!(e.path.iter().any(|l| l.contains("shared Passport identifier ending 776")), "{:?}", e.path);
    assert!(c.assess_risk(twin, now).unwrap().tier >= RiskTier::Medium);
}

#[test]
fn exclusivity_of_the_tie_matters() {
    // A customer whose only link is a director whose *only other role* is the sanctioned-owned
    // company is more tightly bound to it than one sharing a director who sits on many boards.
    let w = world(true);
    let now = t(2026, 10, 13);
    let mut c = w.c;
    let sole = c.add_party(person("Sole Director", None, "Belize"), &w.ana, now).unwrap();
    let exclusive = ent(&mut c, "Exclusive Ltd", &w.ana);
    c.add_relationship(sole, w.volga, RelKind::Director, None, &w.ana, now).unwrap();
    c.add_relationship(sole, exclusive, RelKind::Director, None, &w.ana, now).unwrap();
    let (diluted, tight) = (c.network_exposure(w.direct).unwrap().exposure, c.network_exposure(exclusive).unwrap().exposure);
    assert!(tight > 1.5 * diluted, "tight {tight} vs diluted {diluted}");
    let moderate = c.risk_cfg.network.moderate_exposure;
    assert!(diluted >= moderate, "diluted {diluted}");
    // place the strong threshold between the two: only the exclusive tie crosses it
    c.risk_cfg.network.strong_exposure = (tight + diluted) / 2.0;
    let a = c.assess_risk(exclusive, now).unwrap();
    assert_eq!((a.tier, a.dd_level), (RiskTier::High, DdLevel::Enhanced), "{a:?}");
    assert_eq!(c.assess_risk(w.direct, now).unwrap().tier, RiskTier::Medium);
}

// ----------------------------------------------------------------------------------------
// Network links (address / phone / email / bank account / registered agent) and calibration
// ----------------------------------------------------------------------------------------

use cdd_sanctions::calibrate::{CalibrationOptions, Label};
use cdd_sanctions::links::{Link, LinkKind};

/// A fresh customer who shares one link value with Petrov (written differently on each side).
fn customer_sharing(c: &mut Compliance, w_ana: &Actor, petrov: PartyId, kind: LinkKind, petrov_value: &str, customer_value: &str) -> PartyId {
    let now = t(2026, 10, 13);
    let x = c.add_party(person(&format!("Linked by {}", kind.label()), None, "Belize"), w_ana, now).unwrap();
    c.update_party(petrov, |p| p.links.push(Link::new(kind, petrov_value, Some("BZ"))), w_ana, now).unwrap();
    c.update_party(x, |p| p.links.push(Link::new(kind, customer_value, Some("BZ"))), w_ana, now).unwrap();
    x
}

#[test]
fn every_link_kind_connects_a_customer_to_a_sanctioned_party_and_stronger_kinds_expose_more() {
    let cases = [
        (LinkKind::BankAccount, "0012 3456 78", "00-12345678"),
        (LinkKind::Phone, "+501 622-1234", "00501 6221234"),
        (LinkKind::Address, "12 Main St., Belize City", "No. 12 MAIN STREET, belize city"),
        (LinkKind::RegisteredAgent, "Caribbean Corporate Services Ltd", "CARIBBEAN CORPORATE SERVICES LIMITED"),
    ];
    let mut exposure = Vec::new();
    for (kind, a, b) in cases {
        let w = world(true);
        let mut c = w.c;
        let x = customer_sharing(&mut c, &w.ana, w.petrov, kind, a, b);
        let e = c.network_exposure(x).unwrap();
        assert!(e.exposure > 0.0, "{kind:?}: no exposure");
        assert_eq!(e.source, Some(w.petrov), "{kind:?}");
        assert!(e.path.iter().any(|l| l.contains(&format!("shared {}", kind.label()))), "{kind:?}: {:?}", e.path);
        exposure.push((kind, e.exposure));
    }
    // same topology, only the configured edge weight differs (1.0 > 0.6 > 0.5 > 0.3)
    let v: Vec<f64> = exposure.iter().map(|(_, e)| *e).collect();
    assert!(v[0] > v[1] && v[1] > v[2] && v[2] > v[3], "{exposure:?}");
}

#[test]
fn email_formats_and_local_phone_numbers_match_when_configured() {
    let w = world(true);
    let mut c = w.c;
    let x = customer_sharing(&mut c, &w.ana, w.petrov, LinkKind::Email, "Ivan.P+kyc@Corp.bz", "ivan.p@corp.bz");
    assert!(c.network_exposure(x).unwrap().exposure > 0.0);
    // a local number does not match its international form until the home dial code is set
    let y = customer_sharing(&mut c, &w.ana, w.petrov, LinkKind::Phone, "+501 622 7777", "622-7777");
    assert_eq!(c.network_exposure(y).unwrap().exposure, 0.0);
    c.risk_cfg.network.home_dial_code = Some("501".into());
    assert!(c.network_exposure(y).unwrap().exposure > 0.0);
}

#[test]
fn links_never_touch_screening() {
    let now = t(2026, 10, 13);
    // a list entry whose identifier is a bare number of kind Other (what the UN parser emits for unknown id types)
    let mut e = entry("7", SubjectKind::Individual, "KOVALENKO, Stepan", None, "");
    e.identifiers = vec![Identifier::new(IdKind::Other, "5016221234", None)];
    let e = e.finalize();
    let Ctx { mut c, ana, .. } = setup(vec![e]);
    let id = c.add_party(person("Completely Different Name", None, "Belize"), &ana, now).unwrap();

    // control: the same number as a screened identifier is an identifier hit
    c.update_party(id, |p| p.identifiers.push(Identifier::new(IdKind::Other, "5016221234", None)), &ana, now).unwrap();
    let control = c.screen_party(id, &ana, now).unwrap();
    assert!(control.hits > 0 && control.new_alerts > 0, "control: an identifier equal to a listed one must alert: {control:?}");
    c.update_party(id, |p| p.identifiers.clear(), &ana, now).unwrap();

    // the same number recorded as a *link* is invisible to screening and to the profile hash
    let before = c.party(id).unwrap().profile_hash();
    c.update_party(id, |p| p.links.push(Link::new(LinkKind::Phone, "501-622-1234", None)), &ana, now).unwrap();
    assert_eq!(c.party(id).unwrap().profile_hash(), before);
    let hits = c.screen_party(id, &ana, now).unwrap();
    assert_eq!((hits.hits, hits.new_alerts), (0, 0), "{hits:?}");
}

#[test]
fn a_registered_agent_serving_many_clients_is_not_an_association() {
    let build = |clients: usize| {
        let w = world(true);
        let now = t(2026, 10, 13);
        let mut c = w.c;
        // Volga (owned by the sanctioned Petrov) and its sibling clients share a registered agent
        let mut ids = vec![w.volga];
        for i in 0..clients {
            ids.push(c.add_party(Party::new(PartyKind::Entity, &format!("Agent Client {i} Ltd")), &w.ana, now).unwrap());
        }
        for &p in &ids {
            c.update_party(p, |x| x.links.push(Link::new(LinkKind::RegisteredAgent, "Island Corporate Services Ltd", Some("BZ"))), &w.ana, now).unwrap();
        }
        (c, ids)
    };
    // boutique agent: Volga + 5 clients, under the cap of 15
    let (c, ids) = build(5);
    assert!(c.network_exposure(ids[1]).unwrap().exposure > 0.0);
    // a service provider with 20 clients: skipped, so the clients are not tied to Volga
    let (c, ids) = build(20);
    assert_eq!(c.network_exposure(ids[1]).unwrap().exposure, 0.0);
    assert!(c.network_analysis().unwrap().skipped_shared_identifier_groups >= 1);
}

#[test]
fn old_data_and_partial_configs_still_load() {
    let p: Party = serde_json::from_str(r#"{"kind":"individual","name":"Old Record"}"#).unwrap();
    assert!(p.links.is_empty());
    let cfg: NetworkConfig = serde_json::from_str(r#"{"damping":0.5,"links":{"phone":{"weight":0.9,"max_group":3}}}"#).unwrap();
    assert_eq!(cfg.damping, 0.5);
    assert_eq!((cfg.links.phone.weight, cfg.links.phone.max_group), (0.9, 3));
    assert_eq!(cfg.links.bank_account.max_group, NetworkConfig::default().links.bank_account.max_group, "unspecified kinds keep their defaults");
    let none: NetworkConfig = serde_json::from_str("{}").unwrap();
    assert_eq!(none.links, NetworkConfig::default().links);
    // links survive the persistence round trip
    let mut party = Party::new(PartyKind::Entity, "Roundtrip Ltd");
    party.links.push(Link::new(LinkKind::BankAccount, "GB82 WEST 1234 5698 7654 32", Some("GB")));
    let back: Party = serde_json::from_str(&serde_json::to_string(&party).unwrap()).unwrap();
    assert_eq!(back.links, party.links);
}

/// Petrov holds three bank accounts, each shared with 4 customers who really are associated with him
/// (12 linked). Ten other customers share only a registered agent with Petrov's company and were judged not
/// associated. The right answer is known, so calibration must find it.
fn calibration_world() -> (Compliance, Vec<Label>, PartyId) {
    let w = world(true);
    let now = t(2026, 10, 13);
    let mut c = w.c;
    let mut labels = Vec::new();
    for acct in ["0011 0011 11", "0022 0022 22", "0033 0033 33"] {
        c.update_party(w.petrov, |p| p.links.push(Link::new(LinkKind::BankAccount, acct, Some("BZ"))), &w.ana, now).unwrap();
        for i in 0..4 {
            let x = c.add_party(person(&format!("Account Sharer {acct} #{i}"), None, "Belize"), &w.ana, now).unwrap();
            c.update_party(x, |p| p.links.push(Link::new(LinkKind::BankAccount, acct, Some("BZ"))), &w.ana, now).unwrap();
            labels.push(Label { party: x, linked: true });
        }
    }
    c.update_party(w.volga, |p| p.links.push(Link::new(LinkKind::RegisteredAgent, "Island Corporate Services Ltd", Some("BZ"))), &w.ana, now).unwrap();
    for i in 0..10 {
        let x = c.add_party(Party::new(PartyKind::Entity, &format!("Agent Client {i} Ltd")), &w.ana, now).unwrap();
        c.update_party(x, |p| p.links.push(Link::new(LinkKind::RegisteredAgent, "Island Corporate Services Ltd", Some("BZ"))), &w.ana, now).unwrap();
        labels.push(Label { party: x, linked: false });
    }
    (c, labels, w.petrov)
}

#[test]
fn calibration_finds_the_signal_that_is_actually_there() {
    let (c, mut labels, petrov) = calibration_world();
    // a seed labelled by mistake is skipped, as is a party that does not exist
    labels.push(Label { party: petrov, linked: true });
    labels.push(Label { party: 99_999, linked: false });
    let rep = c.calibrate_network(&labels, &CalibrationOptions::default(), &[0.3, 0.6, 0.9]).unwrap();

    assert_eq!((rep.skipped_seed_labels, rep.skipped_unknown_labels), (1, 1));
    assert_eq!((rep.baseline.positives, rep.baseline.negatives), (12, 10));
    assert_eq!(rep.baseline.detected_positives, 12);
    assert_eq!(rep.baseline.auc, Some(1.0), "strong bank-account ties must outrank the weak agent ties");
    assert!(rep.baseline.suggested_moderate.is_some() && rep.baseline.suggested_strong.is_some(), "{:?}", rep.baseline.warnings);

    // switching off bank accounts removes all the signal (positives drop to zero exposure, negatives keep theirs)
    let fam = |name: &str| rep.families.iter().find(|f| f.family == name).unwrap_or_else(|| panic!("no family {name}: {:?}", rep.families));
    assert_eq!(fam("bank_account").edges, 3);
    assert_eq!(fam("bank_account").auc_without, Some(0.0));
    assert_eq!(fam("bank_account").delta_auc, Some(-1.0));
    // the weak agent edge carries no discriminating signal: removing it does not change the ranking
    assert_eq!(fam("registered_agent").edges, 1);
    assert_eq!(fam("registered_agent").delta_auc, Some(0.0));
    // families with no edges in this data are not listed as if they mattered
    assert!(rep.families.iter().all(|f| f.family != "phone" && f.family != "email" && f.family != "address"));
    assert_eq!(rep.damping_sweep.len(), 3);
    assert!(rep.damping_sweep.iter().all(|(_, a)| *a == Some(1.0)), "{:?}", rep.damping_sweep);
}

#[test]
fn calibration_is_read_only_and_rejects_contradictory_labels() {
    let (c, labels, _) = calibration_world();
    let before = serde_json::to_string(&c.parties().values().map(|p| p.profile_hash()).collect::<Vec<_>>()).unwrap();
    let cfg_before = serde_json::to_string(&c.risk_cfg.network).unwrap();
    c.calibrate_network(&labels, &CalibrationOptions::default(), &[0.5]).unwrap();
    assert_eq!(serde_json::to_string(&c.parties().values().map(|p| p.profile_hash()).collect::<Vec<_>>()).unwrap(), before);
    assert_eq!(serde_json::to_string(&c.risk_cfg.network).unwrap(), cfg_before, "nothing is applied automatically");

    let dup = vec![Label { party: labels[0].party, linked: true }, Label { party: labels[0].party, linked: false }];
    assert!(matches!(c.calibrate_network(&dup, &CalibrationOptions::default(), &[]), Err(Error::Invalid(_))));
}

#[test]
fn the_shipped_example_config_loads_and_matches_the_defaults_it_documents() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/config/risk.example.json")).unwrap();
    let cfg: RiskConfig = serde_json::from_str(&raw).expect("config/risk.example.json must load");
    let d = NetworkConfig::default();
    assert_eq!(cfg.network.links, d.links, "the example documents the default link policies");
    assert_eq!(cfg.network.home_dial_code.as_deref(), Some("501"));
    assert_eq!(cfg.network.damping, d.damping);
}
