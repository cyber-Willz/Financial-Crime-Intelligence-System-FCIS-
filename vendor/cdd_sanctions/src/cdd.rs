//! Customer Due Diligence domain: parties, ownership, risk rating, document requirements.

use crate::country;
use crate::hypernet::NetworkConfig;
use crate::model::*;
use crate::screening::Subject;
use chrono::{Months, NaiveDate};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

pub type PartyId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartyKind {
    Individual,
    Entity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartyStatus {
    #[default]
    Prospect,
    Active,
    OnHold,
    Blocked,
    Exited,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PepStatus {
    #[default]
    No,
    Pep,
    FamilyOrAssociate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Channel {
    #[default]
    FaceToFace,
    NonFaceToFace,
    Intermediary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocKind {
    IdDocument,
    ProofOfAddress,
    Registration,
    OwnershipChart,
    SourceOfWealth,
    SourceOfFunds,
    Other,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Document {
    pub kind: DocKind,
    pub reference: String,
    #[serde(default)]
    pub issued: Option<NaiveDate>,
    #[serde(default)]
    pub expires: Option<NaiveDate>,
    #[serde(default)]
    pub verified_by: Option<String>,
    #[serde(default)]
    pub verified_on: Option<NaiveDate>,
}

impl Document {
    pub fn new(kind: DocKind, reference: &str) -> Self {
        Self { kind, reference: reference.into(), issued: None, expires: None, verified_by: None, verified_on: None }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Party {
    #[serde(default)]
    pub id: PartyId,
    pub kind: PartyKind,
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub dob: Vec<PartialDate>,
    #[serde(default)]
    pub nationalities: Vec<String>,
    #[serde(default)]
    pub countries: Vec<String>,
    #[serde(default)]
    pub identifiers: Vec<Identifier>,
    /// Contact points and service providers used only to build the customer network
    /// (see [`crate::links`]); never screened, never part of the screening profile hash.
    #[serde(default)]
    pub links: Vec<crate::links::Link>,
    #[serde(default)]
    pub pep: PepStatus,
    #[serde(default)]
    pub industry: Option<String>,
    #[serde(default)]
    pub channel: Channel,
    #[serde(default)]
    pub adverse_media: bool,
    #[serde(default)]
    pub nominee_or_bearer_shares: bool,
    #[serde(default)]
    pub status: PartyStatus,
    #[serde(default)]
    pub created_on: Option<NaiveDate>,
    #[serde(default)]
    pub documents: Vec<Document>,
    #[serde(default)]
    pub last_screened_on: Option<NaiveDate>,
    #[serde(default)]
    pub screened_hash: Option<String>,
}

impl Party {
    pub fn new(kind: PartyKind, name: &str) -> Self {
        Self {
            id: 0,
            kind,
            name: name.into(),
            aliases: vec![],
            dob: vec![],
            nationalities: vec![],
            countries: vec![],
            identifiers: vec![],
            links: vec![],
            pep: PepStatus::No,
            industry: None,
            channel: Channel::FaceToFace,
            adverse_media: false,
            nominee_or_bearer_shares: false,
            status: PartyStatus::Prospect,
            created_on: None,
            documents: vec![],
            last_screened_on: None,
            screened_hash: None,
        }
    }

    pub fn to_subject(&self) -> Subject {
        Subject {
            kind: match self.kind {
                PartyKind::Individual => SubjectKind::Individual,
                PartyKind::Entity => SubjectKind::Entity,
            },
            name: self.name.clone(),
            aliases: self.aliases.clone(),
            dob: self.dob.clone(),
            nationalities: self.nationalities.clone(),
            countries: self.countries.clone(),
            identifiers: self.identifiers.clone(),
        }
    }

    /// Hash of every field that affects screening; a change invalidates earlier
    /// screening results and any false-positive dispositions made on them.
    pub fn profile_hash(&self) -> String {
        let v = serde_json::json!([self.kind, self.name, self.aliases, self.dob, self.nationalities, self.countries, self.identifiers]);
        sha256_hex(v.to_string().as_bytes())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelKind {
    Owner,
    Director,
    Signatory,
    Controller,
    Trustee,
    Settlor,
    Beneficiary,
}

/// `from` holds `kind` over `to` (e.g. Owner with `percent` of the entity `to`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Relationship {
    pub id: u64,
    pub from: PartyId,
    pub to: PartyId,
    pub kind: RelKind,
    #[serde(default)]
    pub percent: Option<f64>,
}

// ----------------------------------------------------------------------------------------
// Ownership
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OwnershipReport {
    /// Effective ownership (0..1) held by each natural person, aggregated over all paths.
    pub ultimate: BTreeMap<PartyId, f64>,
    /// Corporate owners with no further owners recorded: (owner, effective fraction).
    pub unresolved: Vec<(PartyId, f64)>,
    pub max_layers: usize,
}

impl OwnershipReport {
    pub fn unresolved_fraction(&self) -> f64 {
        self.unresolved.iter().map(|(_, f)| f).sum()
    }
}

fn has_owners(rels: &[Relationship], p: PartyId) -> bool {
    rels.iter().any(|r| r.to == p && r.kind == RelKind::Owner && r.percent.is_some())
}

pub fn ownership(rels: &[Relationship], parties: &BTreeMap<PartyId, Party>, root: PartyId) -> OwnershipReport {
    fn walk(
        rels: &[Relationship],
        parties: &BTreeMap<PartyId, Party>,
        cur: PartyId,
        frac: f64,
        depth: usize,
        path: &mut Vec<PartyId>,
        rep: &mut OwnershipReport,
    ) {
        for r in rels.iter().filter(|r| r.to == cur && r.kind == RelKind::Owner) {
            let Some(pct) = r.percent else { continue };
            if path.contains(&r.from) {
                continue; // circular ownership: do not count the loop
            }
            let f = frac * pct / 100.0;
            rep.max_layers = rep.max_layers.max(depth + 1);
            match parties.get(&r.from).map(|p| p.kind) {
                Some(PartyKind::Individual) => *rep.ultimate.entry(r.from).or_insert(0.0) += f,
                Some(PartyKind::Entity) if has_owners(rels, r.from) => {
                    path.push(r.from);
                    walk(rels, parties, r.from, f, depth + 1, path, rep);
                    path.pop();
                }
                _ => rep.unresolved.push((r.from, f)),
            }
        }
    }
    let mut rep = OwnershipReport::default();
    walk(rels, parties, root, 1.0, 0, &mut vec![root], &mut rep);
    rep
}

/// Fraction of `root` owned, directly or indirectly, by parties in `blocked`
/// (the "50% rule" input). A blocked owner's stake counts in full and is not
/// descended into, so nothing is double counted.
pub fn blocked_ownership(
    rels: &[Relationship],
    parties: &BTreeMap<PartyId, Party>,
    root: PartyId,
    blocked: &HashSet<PartyId>,
) -> f64 {
    fn walk(
        rels: &[Relationship],
        parties: &BTreeMap<PartyId, Party>,
        cur: PartyId,
        frac: f64,
        blocked: &HashSet<PartyId>,
        path: &mut Vec<PartyId>,
    ) -> f64 {
        let mut total = 0.0;
        for r in rels.iter().filter(|r| r.to == cur && r.kind == RelKind::Owner) {
            let Some(pct) = r.percent else { continue };
            if path.contains(&r.from) {
                continue;
            }
            let f = frac * pct / 100.0;
            if blocked.contains(&r.from) {
                total += f;
            } else if parties.get(&r.from).map(|p| p.kind) == Some(PartyKind::Entity) {
                path.push(r.from);
                total += walk(rels, parties, r.from, f, blocked, path);
                path.pop();
            }
        }
        total
    }
    walk(rels, parties, root, 1.0, blocked, &mut vec![root])
}

// ----------------------------------------------------------------------------------------
// Risk rating
// ----------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskTier {
    Low,
    Medium,
    High,
    Prohibited,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DdLevel {
    Simplified,
    Standard,
    Enhanced,
    /// Relationship may not proceed.
    Prohibited,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskFactor {
    pub name: String,
    pub points: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub score: u32,
    pub tier: RiskTier,
    pub dd_level: DdLevel,
    pub factors: Vec<RiskFactor>,
    pub overrides: Vec<String>,
    pub assessed_on: NaiveDate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct RiskConfig {
    /// ISO alpha-2 -> 0 (low) .. 3 (very high / prohibited-level). Maintain this from your
    /// regulator's and FATF's current lists; nothing is hard-coded.
    pub country_risk: HashMap<String, u8>,
    pub default_country_risk: u8,
    pub high_risk_industries: Vec<String>,
    pub medium_risk_industries: Vec<String>,
    pub complex_layers: usize,
    pub ubo_threshold: f64,
    pub sanctions_ownership_threshold: f64,
    pub tier_medium: u32,
    pub tier_high: u32,
    pub review_months_low: u32,
    pub review_months_medium: u32,
    pub review_months_high: u32,
    pub allow_simplified: bool,
    /// Spectral hypergraph exposure analysis (indirect links to sanctioned parties).
    pub network: NetworkConfig,
}

impl Default for RiskConfig {
    fn default() -> Self {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect();
        Self {
            country_risk: HashMap::new(),
            default_country_risk: 0,
            high_risk_industries: s(&[
                "casino", "gaming", "gambling", "money service", "remittance", "money transfer",
                "virtual asset", "crypto", "precious metal", "precious stone", "jewel", "arms",
                "weapons", "shell",
            ]),
            medium_risk_industries: s(&[
                "real estate", "construction", "import", "export", "trading", "cash", "car dealer",
                "art ", "antique", "non-profit", "charity",
            ]),
            complex_layers: 3,
            ubo_threshold: 0.25,
            sanctions_ownership_threshold: 0.5,
            tier_medium: 30,
            tier_high: 60,
            review_months_low: 36,
            review_months_medium: 24,
            review_months_high: 12,
            allow_simplified: true,
            network: NetworkConfig::default(),
        }
    }
}

impl RiskConfig {
    pub fn country_level(&self, c: &str) -> u8 {
        self.country_risk.get(&country::normalize(c)).copied().unwrap_or(self.default_country_risk)
    }
    pub fn review_months(&self, t: RiskTier) -> u32 {
        match t {
            RiskTier::Low => self.review_months_low,
            RiskTier::Medium => self.review_months_medium,
            RiskTier::High | RiskTier::Prohibited => self.review_months_high,
        }
    }
    pub fn next_review(&self, from: NaiveDate, t: RiskTier) -> NaiveDate {
        from.checked_add_months(Months::new(self.review_months(t))).unwrap_or(from)
    }
}

/// Facts about the party's network that the engine computes before rating.
#[derive(Clone, Debug, Default)]
pub struct RiskInputs {
    pub open_review_alerts: usize,
    pub open_high_alerts: usize,
    pub confirmed_match: bool,
    pub blocked_ownership: f64,
    pub layers: usize,
    pub unresolved_fraction: f64,
    pub related_pep: bool,
    /// A confirmed match sits among owners/controllers (not the party itself).
    pub related_confirmed: bool,
    /// Share of random-walk mass from sanctioned parties *outside* the party's
    /// ownership/control network that lands on it (see `hypernet`). 0 = no indirect link.
    pub network_exposure: f64,
    pub network_source: Option<PartyId>,
    pub network_hops: Option<usize>,
}

pub fn assess(p: &Party, inp: &RiskInputs, cfg: &RiskConfig, today: NaiveDate) -> RiskAssessment {
    let mut factors: Vec<RiskFactor> = Vec::new();
    let mut overrides: Vec<String> = Vec::new();
    let add = |name: &str, pts: u32, f: &mut Vec<RiskFactor>| {
        if pts > 0 {
            f.push(RiskFactor { name: name.to_string(), points: pts });
        }
    };
    let mut floor = RiskTier::Low;

    if p.kind == PartyKind::Entity {
        add("entity customer", 5, &mut factors);
    }
    let worst = p.nationalities.iter().chain(p.countries.iter()).map(|c| cfg.country_level(c)).max().unwrap_or(cfg.default_country_risk);
    add("country risk", [0, 10, 25, 50][worst.min(3) as usize], &mut factors);
    if worst >= 3 {
        floor = floor.max(RiskTier::High);
        overrides.push("very-high-risk jurisdiction: minimum tier High".into());
    }
    match p.pep {
        PepStatus::Pep => {
            add("politically exposed person", 35, &mut factors);
            floor = floor.max(RiskTier::High);
            overrides.push("PEP: minimum tier High (EDD)".into());
        }
        PepStatus::FamilyOrAssociate => {
            add("family member / close associate of PEP", 20, &mut factors);
            floor = floor.max(RiskTier::High);
            overrides.push("PEP associate: minimum tier High (EDD)".into());
        }
        PepStatus::No => {}
    }
    if let Some(ind) = p.industry.as_ref().map(|s| s.to_lowercase()) {
        if cfg.high_risk_industries.iter().any(|k| ind.contains(k.as_str())) {
            add("high-risk industry", 20, &mut factors);
        } else if cfg.medium_risk_industries.iter().any(|k| ind.contains(k.as_str())) {
            add("medium-risk industry", 10, &mut factors);
        }
    }
    if p.channel != Channel::FaceToFace {
        add("non-face-to-face or intermediated channel", 10, &mut factors);
    }
    if p.adverse_media {
        add("adverse media", 20, &mut factors);
    }
    if p.nominee_or_bearer_shares {
        add("nominee arrangements or bearer shares", 20, &mut factors);
    }
    if inp.layers >= cfg.complex_layers {
        add("complex ownership structure", 10, &mut factors);
    }
    if inp.unresolved_fraction >= cfg.ubo_threshold {
        add("ownership not resolved to natural persons", 15, &mut factors);
    }
    if inp.related_pep {
        add("PEP among owners/controllers", 20, &mut factors);
        floor = floor.max(RiskTier::High);
    }
    if inp.related_confirmed {
        floor = floor.max(RiskTier::High);
        overrides.push("confirmed sanctioned party among owners/controllers (below aggregate threshold): MLRO decision required".into());
    }
    if cfg.network.enabled && inp.network_exposure >= cfg.network.moderate_exposure && inp.network_exposure > 0.0 {
        let via = match (inp.network_source, inp.network_hops) {
            (Some(s), Some(h)) => format!(" (sanctioned party #{s}, {h} link(s) away)"),
            _ => String::new(),
        };
        if inp.network_exposure >= cfg.network.strong_exposure {
            add("strong indirect sanctions exposure via shared structure", 30, &mut factors);
            floor = floor.max(RiskTier::High);
            overrides.push(format!(
                "indirect sanctions exposure {:.3}{via} through shared directors/owners/identifiers: minimum tier High, MLRO decision required",
                inp.network_exposure
            ));
        } else {
            add("indirect sanctions exposure via shared structure", 15, &mut factors);
            floor = floor.max(RiskTier::Medium);
            overrides.push(format!(
                "indirect sanctions exposure {:.4}{via} through shared directors/owners/identifiers: minimum tier Medium (standard due diligence)",
                inp.network_exposure
            ));
        }
    }
    if inp.open_high_alerts > 0 {
        add("open high-confidence sanctions alert", 40, &mut factors);
    } else if inp.open_review_alerts > 0 {
        add("open sanctions alert under review", 15, &mut factors);
    }

    let score = factors.iter().map(|f| f.points).sum::<u32>().min(100);
    let mut tier = if score < cfg.tier_medium {
        RiskTier::Low
    } else if score < cfg.tier_high {
        RiskTier::Medium
    } else {
        RiskTier::High
    };
    tier = tier.max(floor);
    if inp.confirmed_match {
        tier = RiskTier::Prohibited;
        overrides.push("confirmed sanctions match".into());
    }
    if inp.blocked_ownership >= cfg.sanctions_ownership_threshold {
        tier = RiskTier::Prohibited;
        overrides.push(format!(
            "owned {:.0}% in aggregate by blocked parties (threshold {:.0}%)",
            inp.blocked_ownership * 100.0,
            cfg.sanctions_ownership_threshold * 100.0
        ));
    }
    let dd_level = match tier {
        RiskTier::Prohibited => DdLevel::Prohibited,
        RiskTier::High => DdLevel::Enhanced,
        RiskTier::Medium => DdLevel::Standard,
        RiskTier::Low if cfg.allow_simplified => DdLevel::Simplified,
        RiskTier::Low => DdLevel::Standard,
    };
    RiskAssessment { score, tier, dd_level, factors, overrides, assessed_on: today }
}

// ----------------------------------------------------------------------------------------
// Document requirements
// ----------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blocker {
    pub code: String,
    pub message: String,
}

impl Blocker {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self { code: code.into(), message: message.into() }
    }
}

pub fn required_documents(kind: PartyKind, dd: DdLevel) -> Vec<DocKind> {
    use DocKind::*;
    match (kind, dd) {
        (_, DdLevel::Prohibited) => vec![],
        (PartyKind::Individual, DdLevel::Simplified) => vec![IdDocument],
        (PartyKind::Individual, DdLevel::Standard) => vec![IdDocument, ProofOfAddress],
        (PartyKind::Individual, DdLevel::Enhanced) => vec![IdDocument, ProofOfAddress, SourceOfWealth, SourceOfFunds],
        (PartyKind::Entity, DdLevel::Enhanced) => vec![Registration, OwnershipChart, SourceOfWealth, SourceOfFunds],
        (PartyKind::Entity, _) => vec![Registration, OwnershipChart],
    }
}

/// Missing, unverified or expired documents, as blockers.
pub fn check_documents(p: &Party, required: &[DocKind], today: NaiveDate, label: &str) -> Vec<Blocker> {
    let mut out = Vec::new();
    for k in required {
        let docs: Vec<&Document> = p.documents.iter().filter(|d| d.kind == *k).collect();
        if docs.is_empty() {
            out.push(Blocker::new("DOC_MISSING", format!("{label}: {k:?} not on file")));
            continue;
        }
        let valid = |d: &&Document| d.verified_by.is_some() && d.expires.map_or(true, |e| e >= today);
        if docs.iter().any(valid) {
            continue;
        }
        if docs.iter().any(|d| d.verified_by.is_some()) {
            out.push(Blocker::new("DOC_EXPIRED", format!("{label}: {k:?} has expired")));
        } else {
            out.push(Blocker::new("DOC_UNVERIFIED", format!("{label}: {k:?} not yet verified")));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(y: i32, m: u32, dd: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, dd).unwrap()
    }

    fn ind(id: PartyId) -> Party {
        let mut p = Party::new(PartyKind::Individual, &format!("Person {id}"));
        p.id = id;
        p
    }
    fn ent(id: PartyId) -> Party {
        let mut p = Party::new(PartyKind::Entity, &format!("Company {id}"));
        p.id = id;
        p
    }
    fn own(id: u64, from: PartyId, to: PartyId, pct: f64) -> Relationship {
        Relationship { id, from, to, kind: RelKind::Owner, percent: Some(pct) }
    }
    fn parties(v: Vec<Party>) -> BTreeMap<PartyId, Party> {
        v.into_iter().map(|p| (p.id, p)).collect()
    }

    #[test]
    fn indirect_and_aggregated_ownership() {
        // 1 = target. Holding 2 owns 60%; person 10 owns 50% of 2 and 20% of 1 directly.
        // Effective for 10 = 0.2 + 0.6*0.5 = 0.5. Person 11 owns the other 50% of 2 => 0.3.
        let ps = parties(vec![ent(1), ent(2), ind(10), ind(11), ind(12)]);
        let rels = vec![own(1, 2, 1, 60.0), own(2, 10, 2, 50.0), own(3, 11, 2, 50.0), own(4, 10, 1, 20.0), own(5, 12, 1, 20.0)];
        let r = ownership(&rels, &ps, 1);
        assert!((r.ultimate[&10] - 0.5).abs() < 1e-9);
        assert!((r.ultimate[&11] - 0.3).abs() < 1e-9);
        assert!((r.ultimate[&12] - 0.2).abs() < 1e-9);
        assert_eq!(r.max_layers, 2);
        assert!(r.unresolved.is_empty());
    }

    #[test]
    fn opaque_corporate_owner_is_unresolved_and_cycles_terminate() {
        let ps = parties(vec![ent(1), ent(2), ent(3), ind(10)]);
        let rels = vec![own(1, 2, 1, 40.0), own(2, 3, 1, 10.0), own(3, 1, 3, 100.0), own(4, 10, 1, 50.0)];
        let r = ownership(&rels, &ps, 1); // 3 is owned by 1 (cycle) and owns 10% of 1
        assert!((r.unresolved_fraction() - 0.4).abs() < 1e-9); // entity 2 has no recorded owners
        assert!((r.ultimate[&10] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn fifty_percent_rule_aggregates_without_double_counting() {
        let ps = parties(vec![ent(1), ent(2), ind(10), ind(11), ind(12)]);
        // 10 owns 30% directly; 2 owns 40% and is itself 50% owned by 11 (blocked) => 20%; 12 not blocked.
        let rels = vec![own(1, 10, 1, 30.0), own(2, 2, 1, 40.0), own(3, 11, 2, 50.0), own(4, 12, 2, 50.0), own(5, 12, 1, 30.0)];
        let blocked: HashSet<PartyId> = [10, 11].into_iter().collect();
        assert!((blocked_ownership(&rels, &ps, 1, &blocked) - 0.5).abs() < 1e-9);
        // if the intermediary itself is blocked, its full 40% counts once (not 40% + 20%)
        let blocked2: HashSet<PartyId> = [2].into_iter().collect();
        assert!((blocked_ownership(&rels, &ps, 1, &blocked2) - 0.4).abs() < 1e-9);
    }

    #[test]
    fn risk_tiers_and_overrides() {
        let cfg = RiskConfig { country_risk: [("IR".to_string(), 3), ("PA".to_string(), 1)].into(), ..Default::default() };
        let today = d(2026, 10, 13);
        let plain = ind(1);
        let a = assess(&plain, &RiskInputs::default(), &cfg, today);
        assert_eq!((a.tier, a.score, a.dd_level), (RiskTier::Low, 0, DdLevel::Simplified));

        let mut medium = ent(2);
        medium.industry = Some("Import/Export trading".into());
        medium.countries = vec!["Panama".into()];
        medium.channel = Channel::NonFaceToFace;
        let a = assess(&medium, &RiskInputs::default(), &cfg, today);
        assert_eq!(a.score, 5 + 10 + 10 + 10);
        assert_eq!(a.tier, RiskTier::Medium);

        let mut pep = ind(3);
        pep.pep = PepStatus::Pep;
        let a = assess(&pep, &RiskInputs::default(), &cfg, today);
        assert_eq!(a.tier, RiskTier::High); // 35 points alone would be Medium; PEP floor lifts it
        assert_eq!(a.dd_level, DdLevel::Enhanced);

        let mut iran = ind(4);
        iran.nationalities = vec!["Iran".into()];
        assert_eq!(assess(&iran, &RiskInputs::default(), &cfg, today).tier, RiskTier::High);

        let confirmed = RiskInputs { confirmed_match: true, ..Default::default() };
        let a = assess(&plain, &confirmed, &cfg, today);
        assert_eq!((a.tier, a.dd_level), (RiskTier::Prohibited, DdLevel::Prohibited));
        let owned = RiskInputs { blocked_ownership: 0.5, ..Default::default() };
        assert_eq!(assess(&ent(5), &owned, &cfg, today).tier, RiskTier::Prohibited);
        let under = RiskInputs { blocked_ownership: 0.49, ..Default::default() };
        assert_ne!(assess(&ent(5), &under, &cfg, today).tier, RiskTier::Prohibited);
    }

    #[test]
    fn review_dates_follow_tier() {
        let cfg = RiskConfig::default();
        assert_eq!(cfg.next_review(d(2026, 1, 31), RiskTier::High), d(2027, 1, 31));
        assert_eq!(cfg.next_review(d(2026, 1, 31), RiskTier::Medium), d(2028, 1, 31));
        assert_eq!(cfg.next_review(d(2026, 1, 31), RiskTier::Low), d(2029, 1, 31));
    }

    #[test]
    fn document_checks() {
        let today = d(2026, 10, 13);
        let mut p = ind(1);
        let req = required_documents(PartyKind::Individual, DdLevel::Standard);
        let codes = |p: &Party| check_documents(p, &req, today, "x").into_iter().map(|b| b.code).collect::<Vec<_>>();
        assert_eq!(codes(&p), vec!["DOC_MISSING", "DOC_MISSING"]);

        let mut id = Document::new(DocKind::IdDocument, "P123");
        p.documents.push(id.clone());
        p.documents.push(Document::new(DocKind::ProofOfAddress, "bill"));
        assert_eq!(codes(&p), vec!["DOC_UNVERIFIED", "DOC_UNVERIFIED"]);

        id.verified_by = Some("alice".into());
        id.expires = Some(d(2026, 1, 1));
        p.documents[0] = id.clone();
        p.documents[1].verified_by = Some("alice".into());
        assert_eq!(codes(&p), vec!["DOC_EXPIRED"]);

        id.expires = Some(d(2030, 1, 1));
        p.documents[0] = id;
        assert!(codes(&p).is_empty());
    }
}
