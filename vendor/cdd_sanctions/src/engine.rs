//! The compliance engine: state, workflows (screening, alerts, cases), list
//! updates with delta re-screening, and persistence.

use crate::audit::AuditLog;
use crate::cdd::*;
use crate::error::{Error, Result};
use crate::calibrate::{self, CalibrationOptions, CalibrationReport, FamilyEffect, Label};
use crate::hypernet::{self, NetworkAnalysis, RootExposure};
use crate::ml::{LogisticModel, MatchFeatures, MlConfig};
use crate::model::*;
use crate::screening::{Hit, HitLevel, SanctionsIndex, ScreeningConfig};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

// ---------------------------------------------------------------------------------------
// Actors
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    Analyst,
    Reviewer,
    /// Money-laundering reporting officer / senior compliance officer.
    Mlro,
}

#[derive(Clone, Debug)]
pub struct Actor {
    pub name: String,
    pub role: Role,
}

impl Actor {
    pub fn new(name: &str, role: Role) -> Self {
        Self { name: name.into(), role }
    }
    fn require(&self, min: Role, what: &str) -> Result<()> {
        if self.role >= min {
            Ok(())
        } else {
            Err(Error::Forbidden(format!("{what} requires role {min:?} or higher (actor {} is {:?})", self.name, self.role)))
        }
    }
}

// ---------------------------------------------------------------------------------------
// Alerts and cases
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertStatus {
    Open,
    Escalated,
    FalsePositive,
    Confirmed,
}

impl AlertStatus {
    pub fn is_unresolved(self) -> bool {
        matches!(self, Self::Open | Self::Escalated)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    FalsePositive,
    ConfirmedMatch,
    Escalate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Alert {
    pub id: u64,
    /// The party whose details matched.
    pub party: PartyId,
    /// The party whose network screening raised it (differs for owners/controllers).
    pub root_party: PartyId,
    pub list: String,
    pub uid: String,
    pub entry_hash: String,
    pub profile_hash: String,
    pub primary_name: String,
    pub matched_name: String,
    pub score: f64,
    pub level: HitLevel,
    pub reasons: Vec<String>,
    pub programs: Vec<String>,
    pub status: AlertStatus,
    pub created_on: NaiveDate,
    pub resolved_by: Option<String>,
    pub resolved_on: Option<NaiveDate>,
    pub note: Option<String>,
    /// The list entry was removed in a later list version.
    pub delisted: bool,
    pub history: Vec<String>,
    /// Structured evidence captured when the alert was raised (input to the learned prioritiser).
    #[serde(default)]
    pub features: Option<MatchFeatures>,
    /// Match-likelihood score in 0..1 that orders the queue; never decides the alert. A calibrated probability only when
    /// `model_version` names a trained model that passed validation (see the reliability bins in `ml-train`); for the
    /// built-in scorecard (`prior-v1`) it is a ranking score only.
    #[serde(default)]
    pub priority: Option<f64>,
    /// Which model produced `priority`.
    #[serde(default)]
    pub model_version: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseKind {
    Onboarding,
    PeriodicReview,
    EventDriven,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseStatus {
    Draft,
    InReview,
    Approved,
    Rejected,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CddCase {
    pub id: u64,
    pub party: PartyId,
    pub kind: CaseKind,
    pub status: CaseStatus,
    pub opened_on: NaiveDate,
    pub prepared_by: String,
    pub assessment: Option<RiskAssessment>,
    pub approved_by: Option<String>,
    pub decided_on: Option<NaiveDate>,
    pub next_review: Option<NaiveDate>,
    /// A confirmed sanctions match touches this case; a regulatory report/freeze
    /// decision is needed. The engine flags it; it never files anything itself.
    pub report_required: bool,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ListVersion {
    pub list: String,
    pub label: String,
    pub loaded_on: NaiveDate,
    pub entries: usize,
    pub sha256: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ListDiff {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DeltaReport {
    pub diff: ListDiff,
    pub parties_screened: usize,
    pub new_alerts: usize,
    pub delisted_alerts: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScreenSummary {
    pub parties_screened: usize,
    pub hits: usize,
    pub new_alerts: usize,
    pub suppressed: usize,
}

// ---------------------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------------------

#[derive(Default, Serialize, Deserialize)]
pub struct Compliance {
    pub screening_cfg: ScreeningConfig,
    pub risk_cfg: RiskConfig,
    parties: BTreeMap<PartyId, Party>,
    relationships: Vec<Relationship>,
    cases: BTreeMap<u64, CddCase>,
    alerts: BTreeMap<u64, Alert>,
    entries: Vec<ListEntry>,
    versions: Vec<ListVersion>,
    audit: AuditLog,
    next_id: u64,
    #[serde(skip)]
    index: SanctionsIndex,
    /// Learned alert prioritisation settings (mode, low-band cut-off).
    #[serde(default)]
    pub ml_cfg: MlConfig,
    /// The activated, validated model. `None` = the hand-set scorecard prior is used for the stored priority.
    #[serde(default)]
    ml_model: Option<LogisticModel>,
}

mod ml_ops;
pub use ml_ops::{ExampleOwned, HistoryImport, HistoryRecord, MlStatus, QueueItem};

const MAX_NETWORK_DEPTH: usize = 8;

impl Compliance {
    pub fn new(screening_cfg: ScreeningConfig, risk_cfg: RiskConfig) -> Self {
        Self { screening_cfg, risk_cfg, ..Default::default() }
    }

    fn next(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    fn log(&mut self, now: DateTime<Utc>, actor: &str, action: &str, target: &str, detail: serde_json::Value) {
        self.audit.append(now, actor, action, target, detail);
    }

    // ---- accessors -------------------------------------------------------------------
    pub fn party(&self, id: PartyId) -> Option<&Party> {
        self.parties.get(&id)
    }
    pub fn parties(&self) -> &BTreeMap<PartyId, Party> {
        &self.parties
    }
    pub fn relationships(&self) -> &[Relationship] {
        &self.relationships
    }
    pub fn alerts(&self) -> &BTreeMap<u64, Alert> {
        &self.alerts
    }
    pub fn cases(&self) -> &BTreeMap<u64, CddCase> {
        &self.cases
    }
    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }
    pub fn list_versions(&self) -> &[ListVersion] {
        &self.versions
    }
    /// Lists whose latest load is more than `max_age_days` old: (list, last loaded).
    pub fn stale_lists(&self, today: NaiveDate, max_age_days: i64) -> Vec<(String, NaiveDate)> {
        let mut latest: BTreeMap<&str, NaiveDate> = BTreeMap::new();
        for v in &self.versions {
            let e = latest.entry(v.list.as_str()).or_insert(v.loaded_on);
            if v.loaded_on > *e {
                *e = v.loaded_on;
            }
        }
        latest
            .into_iter()
            .filter(|(_, d)| (today - *d).num_days() > max_age_days)
            .map(|(l, d)| (l.to_string(), d))
            .collect()
    }
    pub fn index(&self) -> &SanctionsIndex {
        &self.index
    }

    fn require_party(&self, id: PartyId) -> Result<&Party> {
        self.parties.get(&id).ok_or_else(|| Error::NotFound(format!("party {id}")))
    }

    // ---- parties ---------------------------------------------------------------------
    pub fn add_party(&mut self, mut p: Party, actor: &Actor, now: DateTime<Utc>) -> Result<PartyId> {
        if p.name.trim().is_empty() {
            return Err(Error::Invalid("party name is empty".into()));
        }
        p.id = self.next();
        p.created_on = Some(now.date_naive());
        p.status = PartyStatus::Prospect;
        p.last_screened_on = None;
        p.screened_hash = None;
        let id = p.id;
        let kind = p.kind;
        let name = p.name.clone();
        self.parties.insert(id, p);
        self.log(now, &actor.name, "party.add", &format!("party:{id}"), json!({"kind": kind, "name": name}));
        Ok(id)
    }

    /// Update screening-relevant details. Invalidates prior screening for the party.
    pub fn update_party(&mut self, id: PartyId, f: impl FnOnce(&mut Party), actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        let p = self.parties.get_mut(&id).ok_or_else(|| Error::NotFound(format!("party {id}")))?;
        let before = p.profile_hash();
        let status = p.status;
        f(p);
        p.id = id;
        p.status = status; // status changes only through workflows
        let changed = before != p.profile_hash();
        self.log(now, &actor.name, "party.update", &format!("party:{id}"), json!({"screening_relevant_change": changed}));
        Ok(())
    }

    pub fn add_relationship(&mut self, from: PartyId, to: PartyId, kind: RelKind, percent: Option<f64>, actor: &Actor, now: DateTime<Utc>) -> Result<u64> {
        self.require_party(from)?;
        self.require_party(to)?;
        if from == to {
            return Err(Error::Invalid("a party cannot be related to itself".into()));
        }
        if self.require_party(to)?.kind != PartyKind::Entity && kind == RelKind::Owner {
            return Err(Error::Invalid("only entities can be owned".into()));
        }
        if let Some(p) = percent {
            if !(p > 0.0 && p <= 100.0) {
                return Err(Error::Invalid(format!("percent {p} out of range (0,100]")));
            }
        }
        if kind == RelKind::Owner && percent.is_none() {
            return Err(Error::Invalid("ownership needs a percentage".into()));
        }
        let id = self.next();
        self.relationships.push(Relationship { id, from, to, kind, percent });
        self.log(now, &actor.name, "relationship.add", &format!("party:{to}"), json!({"from": from, "kind": kind, "percent": percent}));
        Ok(id)
    }

    pub fn add_document(&mut self, party: PartyId, doc: Document, actor: &Actor, now: DateTime<Utc>) -> Result<usize> {
        let p = self.parties.get_mut(&party).ok_or_else(|| Error::NotFound(format!("party {party}")))?;
        let kind = doc.kind;
        p.documents.push(doc);
        let n = p.documents.len() - 1;
        self.log(now, &actor.name, "document.add", &format!("party:{party}"), json!({"kind": kind, "index": n}));
        Ok(n)
    }

    pub fn verify_document(&mut self, party: PartyId, index: usize, actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        actor.require(Role::Analyst, "verifying documents")?;
        let p = self.parties.get_mut(&party).ok_or_else(|| Error::NotFound(format!("party {party}")))?;
        let d = p.documents.get_mut(index).ok_or_else(|| Error::NotFound(format!("document {index}")))?;
        d.verified_by = Some(actor.name.clone());
        d.verified_on = Some(now.date_naive());
        self.log(now, &actor.name, "document.verify", &format!("party:{party}"), json!({"index": index}));
        Ok(())
    }

    /// The party plus everything that owns or controls it, transitively.
    pub fn network(&self, root: PartyId) -> Vec<PartyId> {
        let mut seen: HashSet<PartyId> = HashSet::from([root]);
        let mut order = vec![root];
        let mut frontier = vec![root];
        for _ in 0..MAX_NETWORK_DEPTH {
            let mut next = Vec::new();
            for cur in &frontier {
                for r in self.relationships.iter().filter(|r| r.to == *cur) {
                    if self.parties.contains_key(&r.from) && seen.insert(r.from) {
                        order.push(r.from);
                        next.push(r.from);
                    }
                }
            }
            if next.is_empty() {
                break;
            }
            frontier = next;
        }
        order
    }

    // ---- sanctions lists ---------------------------------------------------------------
    /// Replace one list with a fresh download. Refuses empty or sharply shrunken
    /// lists unless `force` (guards against truncated downloads). New and changed
    /// entries are screened against every live party straight away.
    pub fn load_list(&mut self, list: &str, mut entries: Vec<ListEntry>, label: &str, force: bool, actor: &Actor, now: DateTime<Utc>) -> Result<DeltaReport> {
        // last occurrence of a uid wins
        let mut seen: HashSet<String> = HashSet::new();
        entries.reverse();
        entries.retain(|e| seen.insert(e.uid.clone()));
        entries.reverse();
        for e in &mut entries {
            e.list = list.to_string();
        }
        let old: HashMap<String, String> =
            self.entries.iter().filter(|e| e.list == list).map(|e| (e.uid.clone(), e.content_hash.clone())).collect();
        if entries.is_empty() {
            return Err(Error::Invalid("refusing to load an empty list".into()));
        }
        if !force && !old.is_empty() && entries.len() * 2 < old.len() {
            return Err(Error::Invalid(format!(
                "list '{list}' would shrink from {} to {} entries; re-run with force if intended",
                old.len(),
                entries.len()
            )));
        }
        let mut diff = ListDiff::default();
        let mut delta: Vec<ListEntry> = Vec::new();
        for e in &entries {
            match old.get(&e.uid) {
                None => {
                    diff.added.push(e.uid.clone());
                    delta.push(e.clone());
                }
                Some(h) if *h != e.content_hash => {
                    diff.changed.push(e.uid.clone());
                    delta.push(e.clone());
                }
                _ => {}
            }
        }
        let new_uids: HashSet<&String> = entries.iter().map(|e| &e.uid).collect();
        diff.removed = old.keys().filter(|u| !new_uids.contains(u)).cloned().collect();
        diff.removed.sort();

        let sha = {
            let mut hs: Vec<&str> = entries.iter().map(|e| e.content_hash.as_str()).collect();
            hs.sort();
            sha256_hex(hs.concat().as_bytes())
        };
        let count = entries.len();
        self.entries.retain(|e| e.list != list);
        self.entries.extend(entries);
        self.index = SanctionsIndex::build(self.entries.clone());
        self.versions.push(ListVersion { list: list.into(), label: label.into(), loaded_on: now.date_naive(), entries: count, sha256: sha });

        let mut report = DeltaReport { diff: diff.clone(), ..Default::default() };
        if !delta.is_empty() {
            let delta_index = SanctionsIndex::build(delta);
            let live: Vec<PartyId> = self
                .parties
                .values()
                .filter(|p| !matches!(p.status, PartyStatus::Exited))
                .map(|p| p.id)
                .collect();
            for pid in live {
                let (subject, hash) = {
                    let p = &self.parties[&pid];
                    (p.to_subject(), p.profile_hash())
                };
                let res = delta_index.screen(&subject, &self.screening_cfg);
                let (n, _) = self.raise_alerts(pid, pid, &res.hits, &hash, now);
                report.parties_screened += 1;
                report.new_alerts += n;
                if let Some(p) = self.parties.get_mut(&pid) {
                    // screened against previous list state earlier (or is brand new and unscreened)
                    if p.screened_hash.as_deref() == Some(hash.as_str()) {
                        p.last_screened_on = Some(now.date_naive());
                    }
                }
            }
        }
        if !diff.removed.is_empty() {
            let removed: HashSet<&String> = diff.removed.iter().collect();
            for a in self.alerts.values_mut() {
                if a.list == list && a.status.is_unresolved() && removed.contains(&a.uid) && !a.delisted {
                    a.delisted = true;
                    a.history.push(format!("{}: entry removed from list version '{label}' — review for closure", now.date_naive()));
                    report.delisted_alerts += 1;
                }
            }
        }
        self.log(
            now,
            &actor.name,
            "list.load",
            &format!("list:{list}"),
            json!({"label": label, "entries": count, "added": diff.added.len(), "changed": diff.changed.len(),
                   "removed": diff.removed.len(), "new_alerts": report.new_alerts}),
        );
        Ok(report)
    }

    /// Rebuild the in-memory index from stored entries (after loading state).
    pub fn rebuild_index(&mut self) {
        self.index = SanctionsIndex::build(self.entries.clone());
    }

    // ---- screening ---------------------------------------------------------------------
    fn raise_alerts(&mut self, party: PartyId, root: PartyId, hits: &[Hit], profile_hash: &str, now: DateTime<Utc>) -> (usize, usize) {
        let today = now.date_naive();
        let (mut created, mut suppressed) = (0, 0);
        for h in hits {
            let (pr, mv) = self.priority_for(&h.features);
            let existing = self
                .alerts
                .values()
                .filter(|a| a.party == party && a.list == h.list && a.uid == h.uid)
                .map(|a| a.id)
                .max();
            if let Some(eid) = existing {
                let a = self.alerts.get_mut(&eid).expect("exists");
                match a.status {
                    AlertStatus::Open | AlertStatus::Escalated => {
                        if a.entry_hash != h.entry_hash {
                            a.history.push(format!("{today}: listing changed while alert open"));
                            a.entry_hash = h.entry_hash.clone();
                        }
                        a.score = h.score;
                        a.level = h.level;
                        a.reasons = h.reasons.clone();
                        a.features = Some(h.features.clone());
                        a.priority = pr;
                        a.model_version = mv.clone();
                        continue;
                    }
                    AlertStatus::Confirmed => continue,
                    AlertStatus::FalsePositive => {
                        if a.entry_hash == h.entry_hash && a.profile_hash == profile_hash {
                            suppressed += 1;
                            continue; // same listing, same customer data: disposition still valid
                        }
                    }
                }
            }
            let id = self.next();
            self.alerts.insert(
                id,
                Alert {
                    id,
                    party,
                    root_party: root,
                    list: h.list.clone(),
                    uid: h.uid.clone(),
                    entry_hash: h.entry_hash.clone(),
                    profile_hash: profile_hash.to_string(),
                    primary_name: h.primary_name.clone(),
                    matched_name: h.matched_name.clone(),
                    score: h.score,
                    level: h.level,
                    reasons: h.reasons.clone(),
                    programs: h.programs.clone(),
                    status: AlertStatus::Open,
                    created_on: today,
                    resolved_by: None,
                    resolved_on: None,
                    note: None,
                    delisted: false,
                    history: vec![],
                    features: Some(h.features.clone()),
                    priority: pr,
                    model_version: mv,
                },
            );
            created += 1;
        }
        (created, suppressed)
    }

    /// Screen a party and everyone who owns or controls it.
    pub fn screen_party(&mut self, root: PartyId, actor: &Actor, now: DateTime<Utc>) -> Result<ScreenSummary> {
        self.require_party(root)?;
        let today = now.date_naive();
        let mut sum = ScreenSummary::default();
        for pid in self.network(root) {
            let (subject, hash) = {
                let p = &self.parties[&pid];
                (p.to_subject(), p.profile_hash())
            };
            let res = self.index.screen(&subject, &self.screening_cfg);
            let (n, s) = self.raise_alerts(pid, root, &res.hits, &hash, now);
            sum.parties_screened += 1;
            sum.hits += res.hits.len();
            sum.new_alerts += n;
            sum.suppressed += s;
            let p = self.parties.get_mut(&pid).expect("exists");
            p.last_screened_on = Some(today);
            p.screened_hash = Some(hash);
        }
        self.log(now, &actor.name, "screen.party", &format!("party:{root}"), json!(sum));
        Ok(sum)
    }

    pub fn resolve_alert(&mut self, id: u64, outcome: Outcome, note: &str, actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        if note.trim().is_empty() {
            return Err(Error::Invalid("a written rationale is required".into()));
        }
        let (status, level, party) = {
            let a = self.alerts.get(&id).ok_or_else(|| Error::NotFound(format!("alert {id}")))?;
            (a.status, a.level, a.party)
        };
        if !status.is_unresolved() {
            return Err(Error::Invalid(format!("alert {id} is already {status:?}")));
        }
        match (outcome, level) {
            (Outcome::Escalate, _) | (Outcome::FalsePositive, HitLevel::Review) => actor.require(Role::Analyst, "this disposition")?,
            (Outcome::FalsePositive, HitLevel::High) => actor.require(Role::Mlro, "closing a high-confidence alert as a false positive")?,
            (Outcome::ConfirmedMatch, _) => actor.require(Role::Mlro, "confirming a sanctions match")?,
        }
        let today = now.date_naive();
        {
            let a = self.alerts.get_mut(&id).expect("exists");
            a.status = match outcome {
                Outcome::FalsePositive => AlertStatus::FalsePositive,
                Outcome::ConfirmedMatch => AlertStatus::Confirmed,
                Outcome::Escalate => AlertStatus::Escalated,
            };
            a.note = Some(note.to_string());
            if outcome != Outcome::Escalate {
                a.resolved_by = Some(actor.name.clone());
                a.resolved_on = Some(today);
            }
            a.history.push(format!("{today}: {:?} by {} — {note}", outcome, actor.name));
        }
        if outcome == Outcome::ConfirmedMatch {
            self.apply_confirmed_match(party, now);
        }
        self.log(now, &actor.name, "alert.resolve", &format!("alert:{id}"), json!({"outcome": outcome, "party": party, "note": note}));
        Ok(())
    }

    fn apply_confirmed_match(&mut self, matched: PartyId, now: DateTime<Utc>) {
        if let Some(p) = self.parties.get_mut(&matched) {
            p.status = PartyStatus::Blocked;
        }
        // everyone whose ownership/control network includes the matched party
        let affected: Vec<PartyId> =
            self.parties.keys().copied().filter(|pid| *pid != matched && self.network(*pid).contains(&matched)).collect();
        for pid in &affected {
            if let Some(p) = self.parties.get_mut(pid) {
                if !matches!(p.status, PartyStatus::Blocked | PartyStatus::Exited) {
                    p.status = PartyStatus::OnHold;
                }
            }
        }
        let touched: HashSet<PartyId> = affected.into_iter().chain(std::iter::once(matched)).collect();
        let today = now.date_naive();
        for c in self.cases.values_mut().filter(|c| touched.contains(&c.party)) {
            c.report_required = true;
            c.notes.push(format!("{today}: sanctions match confirmed on party {matched}; reporting/freezing decision required"));
            if matches!(c.status, CaseStatus::Draft | CaseStatus::InReview) {
                c.status = CaseStatus::Rejected;
            }
        }
    }

    /// MLRO releases an on-hold party (e.g. after a related-party match was assessed as
    /// below the blocking threshold). Blocked parties cannot be released.
    pub fn release_hold(&mut self, party: PartyId, note: &str, actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        actor.require(Role::Mlro, "releasing a hold")?;
        if note.trim().is_empty() {
            return Err(Error::Invalid("a written rationale is required".into()));
        }
        let p = self.parties.get_mut(&party).ok_or_else(|| Error::NotFound(format!("party {party}")))?;
        if p.status != PartyStatus::OnHold {
            return Err(Error::Invalid(format!("party {party} is {:?}, not OnHold", p.status)));
        }
        p.status = PartyStatus::Prospect;
        self.log(now, &actor.name, "party.release_hold", &format!("party:{party}"), json!({"note": note}));
        Ok(())
    }

    // ---- ownership & risk --------------------------------------------------------------
    pub fn ownership_report(&self, root: PartyId) -> OwnershipReport {
        ownership(&self.relationships, &self.parties, root)
    }

    fn blocked_set(&self) -> HashSet<PartyId> {
        let mut s: HashSet<PartyId> = self.parties.values().filter(|p| p.status == PartyStatus::Blocked).map(|p| p.id).collect();
        s.extend(self.alerts.values().filter(|a| a.status == AlertStatus::Confirmed).map(|a| a.party));
        s
    }

    /// Beneficial owners: natural persons at or above the ownership threshold, plus
    /// anyone holding a control relationship (controller, trustee, settlor).
    pub fn beneficial_owners(&self, root: PartyId) -> Vec<PartyId> {
        let rep = self.ownership_report(root);
        let mut set: Vec<PartyId> =
            rep.ultimate.iter().filter(|(_, f)| **f >= self.risk_cfg.ubo_threshold).map(|(p, _)| *p).collect();
        for r in self.relationships.iter().filter(|r| r.to == root && matches!(r.kind, RelKind::Controller | RelKind::Trustee | RelKind::Settlor)) {
            if !set.contains(&r.from) && self.parties.get(&r.from).map(|p| p.kind) == Some(PartyKind::Individual) {
                set.push(r.from);
            }
        }
        set.sort_unstable();
        set
    }

    fn risk_inputs(&self, root: PartyId) -> Result<RiskInputs> {
        let net: HashSet<PartyId> = self.network(root).into_iter().collect();
        let rep = self.ownership_report(root);
        let (mut rev, mut high, mut confirmed, mut related_confirmed) = (0, 0, false, false);
        for a in self.alerts.values().filter(|a| net.contains(&a.party)) {
            match (a.status, a.level) {
                (AlertStatus::Confirmed, _) if a.party == root => confirmed = true,
                (AlertStatus::Confirmed, _) => related_confirmed = true,
                (s, HitLevel::High) if s.is_unresolved() => high += 1,
                (s, HitLevel::Review) if s.is_unresolved() => rev += 1,
                _ => {}
            }
        }
        let indirect = self.network_exposure(root)?;
        Ok(RiskInputs {
            open_review_alerts: rev,
            open_high_alerts: high,
            confirmed_match: confirmed,
            related_confirmed,
            blocked_ownership: blocked_ownership(&self.relationships, &self.parties, root, &self.blocked_set()),
            layers: rep.max_layers,
            unresolved_fraction: rep.unresolved_fraction(),
            related_pep: net.iter().any(|p| *p != root && self.parties[p].pep != PepStatus::No),
            network_exposure: indirect.exposure,
            network_source: indirect.source,
            network_hops: indirect.hops,
        })
    }

    pub fn assess_risk(&self, root: PartyId, now: DateTime<Utc>) -> Result<RiskAssessment> {
        let p = self.require_party(root)?;
        Ok(assess(p, &self.risk_inputs(root)?, &self.risk_cfg, now.date_naive()))
    }

    // ---- spectral network analysis ---------------------------------------------------------
    /// Sanctioned "seeds" for the exposure analysis: confirmed matches and blocked parties at
    /// full strength, unresolved high/review alerts at the configured (lower) strengths.
    pub fn network_seeds(&self) -> Vec<(PartyId, f64)> {
        let c = &self.risk_cfg.network;
        let mut w: BTreeMap<PartyId, f64> = BTreeMap::new();
        let mut put = |p: PartyId, v: f64| {
            if v > 0.0 {
                let e = w.entry(p).or_insert(0.0);
                *e = e.max(v);
            }
        };
        for p in self.parties.values().filter(|p| p.status == PartyStatus::Blocked) {
            put(p.id, c.confirmed_seed);
        }
        for a in self.alerts.values() {
            match (a.status, a.level) {
                (AlertStatus::Confirmed, _) => put(a.party, c.confirmed_seed),
                (s, HitLevel::High) if s.is_unresolved() => put(a.party, c.high_alert_seed),
                (s, HitLevel::Review) if s.is_unresolved() => put(a.party, c.review_alert_seed),
                _ => {}
            }
        }
        w.into_iter().collect()
    }

    /// Indirect sanctions exposure of one party: the share of random-walk mass from sanctioned
    /// parties that reaches it through shared directors, co-owners, stakes and shared
    /// identifiers (`hypernet`). Sanctioned parties already in the party's own ownership/control
    /// network are excluded: the ownership rules (50% rule, related-confirmed hold) cover those,
    /// and counting them again would double-count. Includes a step-by-step explanation.
    pub fn network_exposure(&self, root: PartyId) -> Result<RootExposure> {
        self.require_party(root)?;
        if !self.risk_cfg.network.enabled {
            return Ok(RootExposure::default());
        }
        let own: HashSet<PartyId> = self.network(root).into_iter().collect();
        let seeds: Vec<(PartyId, f64)> = self.network_seeds().into_iter().filter(|(s, _)| !own.contains(s)).collect();
        hypernet::exposure_for(&self.parties, &self.relationships, &seeds, root, &self.risk_cfg.network)
    }

    /// Portfolio-wide spectral analysis: connected components (with algebraic connectivity,
    /// weakest cut and connector parties) and every party's exposure, highest first.
    /// Read-only; each exposure is flagged when the ownership rules already cover its source.
    pub fn network_analysis(&self) -> Result<NetworkAnalysis> {
        let mut rep = hypernet::analyze(&self.parties, &self.relationships, &self.network_seeds(), &self.risk_cfg.network)?;
        for e in &mut rep.exposures {
            e.covered_by_ownership_rules = e.source.map_or(false, |s| self.network(e.party).contains(&s));
        }
        Ok(rep)
    }

    /// Exposure of each party as risk rating would compute it under `cfg`: one portfolio pass,
    /// except for parties whose own ownership/control network contains a seed, which are
    /// recomputed with those seeds removed (the ownership rules cover them; see `network_exposure`).
    fn exposures_as_rated(&self, ids: &[PartyId], seeds: &[(PartyId, f64)], cfg: &crate::hypernet::NetworkConfig) -> Result<Vec<f64>> {
        let mut cfg = cfg.clone();
        cfg.max_seeds = usize::MAX; // rating uses every seed
        cfg.structure_max_size = 0; // structure analysis is not needed here (and is the slow part)
        let rep = hypernet::analyze(&self.parties, &self.relationships, seeds, &cfg)?;
        let by_party: std::collections::HashMap<PartyId, f64> = rep.exposures.iter().map(|e| (e.party, e.exposure)).collect();
        let seed_ids: HashSet<PartyId> = seeds.iter().map(|(s, _)| *s).collect();
        let mut out = Vec::with_capacity(ids.len());
        for &pid in ids {
            let own: HashSet<PartyId> = self.network(pid).into_iter().collect();
            if own.is_disjoint(&seed_ids) {
                out.push(by_party.get(&pid).copied().unwrap_or(0.0));
            } else {
                let filtered: Vec<(PartyId, f64)> = seeds.iter().copied().filter(|(s, _)| !own.contains(s)).collect();
                out.push(hypernet::exposure_for(&self.parties, &self.relationships, &filtered, pid, &cfg)?.exposure);
            }
        }
        Ok(out)
    }

    /// Measures the current network settings against labelled cases (`calibrate`): AUC and a
    /// precision/recall curve with suggested thresholds, an AUC sweep over `dampings`, and what
    /// each edge family (ownership, control, stakes, shared identifiers, and each link kind)
    /// contributes. Read-only and advisory: nothing is applied.
    pub fn calibrate_network(&self, labels: &[Label], opts: &CalibrationOptions, dampings: &[f64]) -> Result<CalibrationReport> {
        let mut by_party: BTreeMap<PartyId, bool> = BTreeMap::new();
        for l in labels {
            if let Some(prev) = by_party.insert(l.party, l.linked) {
                if prev != l.linked {
                    return Err(Error::Invalid(format!("party {} is labelled both linked and not linked", l.party)));
                }
            }
        }
        let seeds = self.network_seeds();
        let seed_ids: HashSet<PartyId> = seeds.iter().map(|(s, _)| *s).collect();
        let mut rep = CalibrationReport { baseline_damping: self.risk_cfg.network.damping, ..Default::default() };
        let mut ids = Vec::new();
        let mut truth = Vec::new();
        for (pid, linked) in by_party {
            if !self.parties.contains_key(&pid) {
                rep.skipped_unknown_labels += 1;
            } else if seed_ids.contains(&pid) {
                rep.skipped_seed_labels += 1;
            } else {
                ids.push(pid);
                truth.push(linked);
            }
        }
        let score = |cfg: &crate::hypernet::NetworkConfig| -> Result<Vec<(bool, f64)>> {
            Ok(truth.iter().copied().zip(self.exposures_as_rated(&ids, &seeds, cfg)?).collect())
        };
        let base_cfg = self.risk_cfg.network.clone();
        let base_scores = score(&base_cfg)?;
        rep.baseline = calibrate::evaluate(&base_scores, opts);
        let base_auc = rep.baseline.auc;
        if rep.skipped_seed_labels > 0 {
            rep.baseline.warnings.push(format!("{} labelled part(y/ies) skipped because they are seeds themselves", rep.skipped_seed_labels));
        }
        if rep.skipped_unknown_labels > 0 {
            rep.baseline.warnings.push(format!("{} label(s) skipped: no such party", rep.skipped_unknown_labels));
        }
        if seeds.is_empty() {
            rep.baseline.warnings.push("there are no sanctioned seeds (no confirmed matches, blocked parties or unresolved high alerts), so every exposure is zero".into());
        }

        for &d in dampings {
            let mut c = base_cfg.clone();
            c.damping = d;
            rep.damping_sweep.push((d, calibrate::auc(&score(&c)?)));
        }

        let (specs, _) = hypernet::edge_specs(&self.parties, &self.relationships, &base_cfg);
        let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
        for e in &specs {
            *counts.entry(e.kind.family()).or_default() += 1;
        }
        for (family, edges) in counts {
            let without = match base_cfg.without_family(family) {
                Some(c) => calibrate::auc(&score(&c)?),
                None => None,
            };
            rep.families.push(FamilyEffect { family: family.to_string(), edges, auc_without: without, delta_auc: base_auc.zip(without).map(|(b, w)| w - b) });
        }
        Ok(rep)
    }

    // ---- cases ---------------------------------------------------------------------------
    pub fn open_case(&mut self, party: PartyId, kind: CaseKind, actor: &Actor, now: DateTime<Utc>) -> Result<u64> {
        self.require_party(party)?;
        let id = self.next();
        self.cases.insert(
            id,
            CddCase {
                id,
                party,
                kind,
                status: CaseStatus::Draft,
                opened_on: now.date_naive(),
                prepared_by: actor.name.clone(),
                assessment: None,
                approved_by: None,
                decided_on: None,
                next_review: None,
                report_required: false,
                notes: vec![],
            },
        );
        self.log(now, &actor.name, "case.open", &format!("case:{id}"), json!({"party": party, "kind": kind}));
        Ok(id)
    }

    /// Everything that currently prevents approval, with the risk assessment used.
    pub fn readiness(&self, party: PartyId, now: DateTime<Utc>) -> Result<(RiskAssessment, Vec<Blocker>)> {
        let p = self.require_party(party)?;
        let today = now.date_naive();
        let assessment = self.assess_risk(party, now)?;
        let mut b: Vec<Blocker> = Vec::new();

        if assessment.tier == RiskTier::Prohibited {
            b.push(Blocker::new("PROHIBITED", format!("relationship prohibited: {}", assessment.overrides.join("; "))));
            return Ok((assessment, b));
        }
        if matches!(p.status, PartyStatus::OnHold | PartyStatus::Blocked) {
            b.push(Blocker::new("PARTY_ON_HOLD", format!("party status is {:?}", p.status)));
        }
        if self.index.is_empty() {
            b.push(Blocker::new("NO_LISTS_LOADED", "no sanctions list is loaded, so screening results are meaningless"));
        }
        if let Some(max) = self.screening_cfg.max_list_age_days {
            for (l, d) in self.stale_lists(today, max as i64) {
                b.push(Blocker::new("LIST_STALE", format!("sanctions list '{l}' was last loaded {d}, more than {max} days ago")));
            }
        }
        let net = self.network(party);
        for pid in &net {
            let q = &self.parties[pid];
            let label = format!("{} (#{pid})", q.name);
            if q.screened_hash.as_deref() != Some(q.profile_hash().as_str()) {
                b.push(Blocker::new("NOT_SCREENED", format!("{label}: not screened since details last changed")));
            }
            let open = self.alerts.values().filter(|a| a.party == *pid && a.status.is_unresolved()).count();
            if open > 0 {
                b.push(Blocker::new("OPEN_ALERTS", format!("{label}: {open} unresolved sanctions alert(s)")));
            }
        }
        let req = required_documents(p.kind, assessment.dd_level);
        b.extend(check_documents(p, &req, today, &format!("{} (#{party})", p.name)));

        if p.kind == PartyKind::Entity {
            let rep = self.ownership_report(party);
            let ubos = self.beneficial_owners(party);
            if rep.unresolved_fraction() >= self.risk_cfg.ubo_threshold {
                b.push(Blocker::new(
                    "OWNERSHIP_UNRESOLVED",
                    format!("{:.0}% of ownership sits with corporate owners that are not traced to natural persons", rep.unresolved_fraction() * 100.0),
                ));
            }
            if ubos.is_empty() {
                let smo = self.relationships.iter().any(|r| {
                    r.to == party && matches!(r.kind, RelKind::Director | RelKind::Controller) && self.parties.contains_key(&r.from)
                });
                if !smo {
                    b.push(Blocker::new("NO_UBO_OR_SMO", "no beneficial owner identified and no senior managing official recorded"));
                }
            }
            let id_req = [DocKind::IdDocument];
            for u in ubos {
                let q = &self.parties[&u];
                b.extend(check_documents(q, &id_req, today, &format!("beneficial owner {} (#{u})", q.name)));
            }
        }
        Ok((assessment, b))
    }

    pub fn submit_case(&mut self, case: u64, actor: &Actor, now: DateTime<Utc>) -> Result<Vec<Blocker>> {
        let (party, status) = {
            let c = self.cases.get(&case).ok_or_else(|| Error::NotFound(format!("case {case}")))?;
            (c.party, c.status)
        };
        if !matches!(status, CaseStatus::Draft | CaseStatus::InReview) {
            return Err(Error::Invalid(format!("case {case} is {status:?}")));
        }
        let (assessment, blockers) = self.readiness(party, now)?;
        let c = self.cases.get_mut(&case).expect("exists");
        c.assessment = Some(assessment);
        c.status = CaseStatus::InReview;
        self.log(now, &actor.name, "case.submit", &format!("case:{case}"), json!({"blockers": blockers.len()}));
        Ok(blockers)
    }

    /// Four-eyes approval. Fails with `Error::Blocked` listing every unmet requirement.
    pub fn approve_case(&mut self, case: u64, actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        actor.require(Role::Reviewer, "approving a CDD case")?;
        let (party, status, prepared_by) = {
            let c = self.cases.get(&case).ok_or_else(|| Error::NotFound(format!("case {case}")))?;
            (c.party, c.status, c.prepared_by.clone())
        };
        if status != CaseStatus::InReview {
            return Err(Error::Invalid(format!("case {case} is {status:?}; submit it first")));
        }
        if prepared_by == actor.name {
            return Err(Error::Forbidden("the preparer cannot approve their own case (four-eyes)".into()));
        }
        let (assessment, blockers) = self.readiness(party, now)?;
        if !blockers.is_empty() {
            return Err(Error::Blocked(blockers.into_iter().map(|b| format!("{}: {}", b.code, b.message)).collect()));
        }
        if matches!(assessment.tier, RiskTier::High) {
            actor.require(Role::Mlro, "approving a high-risk relationship (EDD sign-off)")?;
        }
        let today = now.date_naive();
        let next_review = self.risk_cfg.next_review(today, assessment.tier);
        let tier = assessment.tier;
        let c = self.cases.get_mut(&case).expect("exists");
        c.assessment = Some(assessment);
        c.status = CaseStatus::Approved;
        c.approved_by = Some(actor.name.clone());
        c.decided_on = Some(today);
        c.next_review = Some(next_review);
        if let Some(p) = self.parties.get_mut(&party) {
            p.status = PartyStatus::Active;
        }
        self.log(now, &actor.name, "case.approve", &format!("case:{case}"), json!({"party": party, "tier": tier, "next_review": next_review}));
        Ok(())
    }

    pub fn reject_case(&mut self, case: u64, reason: &str, actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        actor.require(Role::Reviewer, "rejecting a CDD case")?;
        if reason.trim().is_empty() {
            return Err(Error::Invalid("a reason is required".into()));
        }
        let c = self.cases.get_mut(&case).ok_or_else(|| Error::NotFound(format!("case {case}")))?;
        if matches!(c.status, CaseStatus::Approved | CaseStatus::Rejected) {
            return Err(Error::Invalid(format!("case {case} is already {:?}", c.status)));
        }
        c.status = CaseStatus::Rejected;
        c.decided_on = Some(now.date_naive());
        c.notes.push(format!("rejected by {}: {reason}", actor.name));
        self.log(now, &actor.name, "case.reject", &format!("case:{case}"), json!({"reason": reason}));
        Ok(())
    }

    /// Active parties whose periodic review falls due within `within_days` (or is overdue),
    /// including active parties with no approved case on record.
    pub fn reviews_due(&self, today: NaiveDate, within_days: i64) -> Vec<(PartyId, NaiveDate)> {
        let horizon = today + chrono::Duration::days(within_days);
        let mut out = Vec::new();
        for p in self.parties.values().filter(|p| p.status == PartyStatus::Active) {
            let due = self
                .cases
                .values()
                .filter(|c| c.party == p.id && c.status == CaseStatus::Approved)
                .filter_map(|c| c.next_review)
                .max()
                .unwrap_or(today);
            if due <= horizon {
                out.push((p.id, due));
            }
        }
        out.sort_by_key(|(id, d)| (*d, *id));
        out
    }

    // ---- persistence -----------------------------------------------------------------------
    pub fn save(&self, path: &Path) -> Result<()> {
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let mut c: Compliance = serde_json::from_slice(&std::fs::read(path)?)?;
        c.rebuild_index();
        Ok(c)
    }
}
