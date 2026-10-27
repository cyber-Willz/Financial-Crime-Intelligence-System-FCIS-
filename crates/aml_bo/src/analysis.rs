//! Beneficial ownership analysis on top of the CDD engine (`cdd_sanctions`).
//!
//! What the CDD engine already does (and this module reuses, never re-implements): ownership percentages,
//! the beneficial-owner threshold, the 50% rule inputs (who is blocked), alerts, and the spectral
//! hypergraph network exposure. What this module adds:
//!
//! * an **indexed** ownership walk (the engine's walk scans every relationship at every step, which is fine for a
//!   case but slow for a portfolio) that also returns the number of paths per owner, the intermediate entities
//!   and *circular holdings*, which the engine's walk silently skips. It is tested to agree with
//!   `Compliance::ownership_report` on effective percentages;
//! * **opacity** indicators for each holder, with reasons;
//! * **ownership groups**: parties connected through ownership or control, the unit at which several accounts
//!   turn out to be one customer;
//! * an export of all of this per *account* for the graph model ([`OwnershipAnalysis::graph_map`]).
//!
//! The opacity weights are judgement-based starting points, not validated against real cases.

use crate::registry::AccountRegistry;
use aml_graph::{OwnerInfo, OwnershipMap};
use anyhow::Result;
use cdd_sanctions::cdd::{PartyId, PartyKind, PartyStatus, PepStatus, RelKind, Relationship};
use cdd_sanctions::engine::{AlertStatus, Compliance};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct BoConfig {
    /// Natural persons holding at least this effective fraction are *related owners* (used to decide whether two
    /// holders share an owner). Below the CDD beneficial-owner threshold on purpose: splitting stakes to stay
    /// under it is a known evasion.
    pub related_owner_min: f64,
    /// Ownership stakes below this percentage do not join parties into one ownership group.
    pub group_min_percent: f64,
    /// Groups larger than this are dropped (a nominee shareholder of hundreds of shells would otherwise chain
    /// unrelated customers together). Dropped groups are reported.
    pub max_group_parties: usize,
    /// No beneficial owner reaches the CDD threshold but at least two sit in `[fragment_low, threshold)` and
    /// together hold at least `fragment_total`.
    pub fragment_low: f64,
    pub fragment_total: f64,
    /// Country risk level (RiskConfig::country_risk) at or above which a jurisdiction in the chain counts as opaque/high risk.
    pub high_risk_level: u8,
    /// Share of a holder owned by blocked parties at which the 50% rule applies.
    pub sanctioned_ownership: f64,
    /// opacity points (sum, capped at 1)
    pub w_layers: f64,
    pub w_unresolved: f64,
    pub w_no_owner: f64,
    pub w_nominee: f64,
    pub w_jurisdiction: f64,
    pub w_cycle: f64,
    pub w_fragmented: f64,
}

impl Default for BoConfig {
    fn default() -> Self {
        BoConfig {
            related_owner_min: 0.10,
            group_min_percent: 10.0,
            max_group_parties: 200,
            fragment_low: 0.15,
            fragment_total: 0.30,
            high_risk_level: 2,
            sanctioned_ownership: 0.5,
            w_layers: 0.20,
            w_unresolved: 0.30,
            w_no_owner: 0.25,
            w_nominee: 0.20,
            w_jurisdiction: 0.15,
            w_cycle: 0.15,
            w_fragmented: 0.10,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ubo {
    pub party: PartyId,
    pub name: String,
    /// effective ownership aggregated over all paths (0..1)
    pub effective: f64,
    pub paths: usize,
    /// holds the stake at or above the CDD beneficial-owner threshold
    pub meets_threshold: bool,
    /// holds control (controller / trustee / settlor) rather than, or as well as, shares
    pub controls: bool,
    pub pep: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartyBo {
    pub party: PartyId,
    pub name: String,
    pub kind: PartyKind,
    pub ubos: Vec<Ubo>,
    /// natural persons used to decide shared ownership (beneficial owners plus smaller stakes)
    pub related_owners: Vec<PartyId>,
    pub layers: usize,
    pub unresolved: f64,
    pub unresolved_parties: Vec<(PartyId, f64)>,
    pub no_owner_identified: bool,
    pub nominee_in_chain: bool,
    pub high_risk_jurisdiction: bool,
    pub circular: Vec<Vec<PartyId>>,
    pub fragmented: bool,
    pub blocked_ownership: f64,
    pub holder_blocked: bool,
    pub exposure: f64,
    pub exposure_source: Option<PartyId>,
    pub pep: bool,
    pub group: u32,
    pub opacity: f64,
    pub opacity_reasons: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Summary {
    pub parties: usize,
    pub entities: usize,
    pub with_accounts: usize,
    pub opaque: usize,
    pub no_owner_identified: usize,
    pub circular: usize,
    pub fragmented: usize,
    pub sanctioned_owned: usize,
    pub exposed: usize,
    pub groups: usize,
    pub multi_account_groups: usize,
    pub groups_dropped_too_large: usize,
}

struct Resolved {
    ultimate: BTreeMap<PartyId, (f64, usize)>,
    unresolved: Vec<(PartyId, f64)>,
    layers: usize,
    intermediates: BTreeSet<PartyId>,
    cycles: Vec<Vec<PartyId>>,
}

pub struct OwnershipAnalysis {
    pub cfg: BoConfig,
    pub parties: BTreeMap<PartyId, PartyBo>,
    pub registry: AccountRegistry,
    pub summary: Summary,
}

/// Indexed view of the CDD engine's relationships and sanctions state.
pub struct Analyzer<'a> {
    c: &'a Compliance,
    cfg: BoConfig,
    owners_of: HashMap<PartyId, Vec<&'a Relationship>>,
    controllers_of: HashMap<PartyId, Vec<PartyId>>,
    blocked: HashSet<PartyId>,
    exposure: HashMap<PartyId, (f64, Option<PartyId>)>,
    group_of: HashMap<PartyId, u32>,
    pub groups_dropped: usize,
}

impl<'a> Analyzer<'a> {
    pub fn new(c: &'a Compliance, cfg: BoConfig) -> Result<Self> {
        let mut owners_of: HashMap<PartyId, Vec<&Relationship>> = HashMap::new();
        let mut controllers_of: HashMap<PartyId, Vec<PartyId>> = HashMap::new();
        for r in c.relationships() {
            match r.kind {
                RelKind::Owner if r.percent.is_some() => owners_of.entry(r.to).or_default().push(r),
                RelKind::Controller | RelKind::Trustee | RelKind::Settlor => controllers_of.entry(r.to).or_default().push(r.from),
                _ => {}
            }
        }
        let mut blocked: HashSet<PartyId> = c.parties().values().filter(|p| p.status == PartyStatus::Blocked).map(|p| p.id).collect();
        blocked.extend(c.alerts().values().filter(|a| a.status == AlertStatus::Confirmed).map(|a| a.party));

        // indirect sanctions exposure (spectral hypergraph analysis in the CDD engine), skipping exposures whose
        // source is already inside the party's own ownership/control network (the ownership rules cover those)
        let mut exposure = HashMap::new();
        if !blocked.is_empty() || !c.network_seeds().is_empty() {
            for e in c.network_analysis()?.exposures {
                if !e.covered_by_ownership_rules && e.exposure > 0.0 {
                    exposure.insert(e.party, (e.exposure, e.source));
                }
            }
        }

        // ownership groups: union-find over ownership (>= group_min_percent) and control relationships
        let ids: Vec<PartyId> = c.parties().keys().copied().collect();
        let index: HashMap<PartyId, usize> = ids.iter().enumerate().map(|(i, p)| (*p, i)).collect();
        let mut uf: Vec<usize> = (0..ids.len()).collect();
        fn find(uf: &mut Vec<usize>, mut x: usize) -> usize {
            while uf[x] != x {
                uf[x] = uf[uf[x]];
                x = uf[x];
            }
            x
        }
        for r in c.relationships() {
            let linked = match r.kind {
                RelKind::Owner => r.percent.map_or(false, |p| p >= cfg.group_min_percent),
                RelKind::Controller | RelKind::Trustee | RelKind::Settlor | RelKind::Beneficiary => true,
                _ => false,
            };
            if !linked {
                continue;
            }
            if let (Some(&a), Some(&b)) = (index.get(&r.from), index.get(&r.to)) {
                let (ra, rb) = (find(&mut uf, a), find(&mut uf, b));
                if ra != rb {
                    uf[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let mut comps: BTreeMap<usize, Vec<PartyId>> = BTreeMap::new();
        for (i, p) in ids.iter().enumerate() {
            let r = find(&mut uf, i);
            comps.entry(r).or_default().push(*p);
        }
        let (mut group_of, mut next, mut dropped) = (HashMap::new(), 1u32, 0usize);
        for members in comps.values() {
            if members.len() < 2 {
                continue;
            }
            if members.len() > cfg.max_group_parties {
                dropped += 1;
                continue;
            }
            for p in members {
                group_of.insert(*p, next);
            }
            next += 1;
        }
        Ok(Analyzer { c, cfg, owners_of, controllers_of, blocked, exposure, group_of, groups_dropped: dropped })
    }

    pub fn is_blocked(&self, p: PartyId) -> bool {
        self.blocked.contains(&p)
    }

    fn resolve(&self, root: PartyId) -> Resolved {
        struct St<'x, 'a> {
            a: &'x Analyzer<'a>,
            out: Resolved,
        }
        fn walk(st: &mut St, cur: PartyId, frac: f64, depth: usize, path: &mut Vec<PartyId>) {
            let Some(rels) = st.a.owners_of.get(&cur) else { return };
            for r in rels {
                let pct = r.percent.unwrap_or(0.0);
                if let Some(pos) = path.iter().position(|p| *p == r.from) {
                    let mut cyc: Vec<PartyId> = path[pos..].to_vec();
                    cyc.push(r.from);
                    if !st.out.cycles.iter().any(|c| c.iter().collect::<BTreeSet<_>>() == cyc.iter().collect::<BTreeSet<_>>()) {
                        st.out.cycles.push(cyc);
                    }
                    continue;
                }
                let f = frac * pct / 100.0;
                st.out.layers = st.out.layers.max(depth + 1);
                match st.a.c.party(r.from).map(|p| p.kind) {
                    Some(PartyKind::Individual) => {
                        let e = st.out.ultimate.entry(r.from).or_insert((0.0, 0));
                        e.0 += f;
                        e.1 += 1;
                    }
                    Some(PartyKind::Entity) if st.a.owners_of.contains_key(&r.from) => {
                        st.out.intermediates.insert(r.from);
                        path.push(r.from);
                        walk(st, r.from, f, depth + 1, path);
                        path.pop();
                    }
                    _ => st.out.unresolved.push((r.from, f)),
                }
            }
        }
        let mut st = St { a: self, out: Resolved { ultimate: BTreeMap::new(), unresolved: vec![], layers: 0, intermediates: BTreeSet::new(), cycles: vec![] } };
        walk(&mut st, root, 1.0, 0, &mut vec![root]);
        st.out
    }

    /// Fraction of `root` owned by blocked parties (a blocked owner counts in full and is not descended into).
    fn blocked_fraction(&self, root: PartyId) -> f64 {
        fn walk(a: &Analyzer, cur: PartyId, frac: f64, path: &mut Vec<PartyId>) -> f64 {
            let Some(rels) = a.owners_of.get(&cur) else { return 0.0 };
            let mut t = 0.0;
            for r in rels {
                if path.contains(&r.from) {
                    continue;
                }
                let f = frac * r.percent.unwrap_or(0.0) / 100.0;
                if a.blocked.contains(&r.from) {
                    t += f;
                } else if a.c.party(r.from).map(|p| p.kind) == Some(PartyKind::Entity) {
                    path.push(r.from);
                    t += walk(a, r.from, f, path);
                    path.pop();
                }
            }
            t
        }
        walk(self, root, 1.0, &mut vec![root])
    }

    pub fn analyse(&self, id: PartyId) -> Option<PartyBo> {
        let p = self.c.party(id)?;
        let risk = &self.c.risk_cfg;
        let cfg = &self.cfg;
        let level_hi = |countries: &[String]| countries.iter().any(|c| risk.country_level(c) >= cfg.high_risk_level);
        let pep_of = |pid: PartyId| self.c.party(pid).map_or(false, |x| x.pep != PepStatus::No);

        let mut bo = PartyBo {
            party: id,
            name: p.name.clone(),
            kind: p.kind,
            ubos: vec![],
            related_owners: vec![],
            layers: 0,
            unresolved: 0.0,
            unresolved_parties: vec![],
            no_owner_identified: false,
            nominee_in_chain: p.nominee_or_bearer_shares,
            high_risk_jurisdiction: level_hi(&p.countries) || level_hi(&p.nationalities),
            circular: vec![],
            fragmented: false,
            blocked_ownership: if self.blocked.contains(&id) { 1.0 } else { 0.0 },
            holder_blocked: self.blocked.contains(&id),
            exposure: self.exposure.get(&id).map_or(0.0, |e| e.0),
            exposure_source: self.exposure.get(&id).and_then(|e| e.1),
            pep: p.pep != PepStatus::No,
            group: self.group_of.get(&id).copied().unwrap_or(0),
            opacity: 0.0,
            opacity_reasons: vec![],
        };

        if p.kind == PartyKind::Individual {
            bo.related_owners = vec![id];
            bo.ubos = vec![Ubo { party: id, name: p.name.clone(), effective: 1.0, paths: 1, meets_threshold: true, controls: false, pep: bo.pep }];
            return Some(bo);
        }

        let r = self.resolve(id);
        bo.layers = r.layers;
        bo.unresolved = r.unresolved.iter().map(|(_, f)| f).sum::<f64>().min(1.0);
        bo.unresolved_parties = r.unresolved.clone();
        bo.circular = r.cycles.clone();
        for im in &r.intermediates {
            if let Some(x) = self.c.party(*im) {
                bo.nominee_in_chain |= x.nominee_or_bearer_shares;
                bo.high_risk_jurisdiction |= level_hi(&x.countries);
            }
        }
        for (u, _) in &r.unresolved {
            if let Some(x) = self.c.party(*u) {
                bo.nominee_in_chain |= x.nominee_or_bearer_shares;
                bo.high_risk_jurisdiction |= level_hi(&x.countries);
            }
        }
        let controllers: BTreeSet<PartyId> = self
            .controllers_of
            .get(&id)
            .into_iter()
            .flatten()
            .copied()
            .filter(|c| self.c.party(*c).map(|x| x.kind) == Some(PartyKind::Individual))
            .collect();
        let thr = risk.ubo_threshold;
        let mut ubo_ids: BTreeSet<PartyId> = BTreeSet::new();
        for (u, (f, paths)) in &r.ultimate {
            let meets = *f >= thr;
            let ctl = controllers.contains(u);
            if meets || ctl {
                ubo_ids.insert(*u);
                bo.ubos.push(Ubo { party: *u, name: self.c.party(*u).map(|x| x.name.clone()).unwrap_or_default(), effective: *f, paths: *paths, meets_threshold: meets, controls: ctl, pep: pep_of(*u) });
            }
            if *f >= cfg.related_owner_min || ctl {
                bo.related_owners.push(*u);
            }
        }
        for c in &controllers {
            if !r.ultimate.contains_key(c) {
                ubo_ids.insert(*c);
                bo.related_owners.push(*c);
                bo.ubos.push(Ubo { party: *c, name: self.c.party(*c).map(|x| x.name.clone()).unwrap_or_default(), effective: 0.0, paths: 0, meets_threshold: false, controls: true, pep: pep_of(*c) });
            }
        }
        bo.related_owners.sort_unstable();
        bo.related_owners.dedup();
        bo.ubos.sort_by(|a, b| b.effective.partial_cmp(&a.effective).unwrap_or(std::cmp::Ordering::Equal).then(a.party.cmp(&b.party)));
        bo.no_owner_identified = ubo_ids.is_empty();
        let mid: Vec<f64> = r.ultimate.values().map(|(f, _)| *f).filter(|f| *f >= cfg.fragment_low && *f < thr).collect();
        bo.fragmented = !bo.ubos.iter().any(|u| u.meets_threshold) && mid.len() >= 2 && mid.iter().sum::<f64>() >= cfg.fragment_total;
        bo.blocked_ownership = if bo.holder_blocked { 1.0 } else { self.blocked_fraction(id).min(1.0) };
        bo.pep |= bo.ubos.iter().any(|u| u.pep);

        // opacity
        let mut pts = 0.0;
        if bo.layers >= risk.complex_layers {
            let v = cfg.w_layers + 0.05 * ((bo.layers - risk.complex_layers).min(4) as f64);
            pts += v;
            bo.opacity_reasons.push(format!("{} ownership layers", bo.layers));
        }
        if bo.unresolved > 0.0 {
            pts += cfg.w_unresolved * bo.unresolved;
            bo.opacity_reasons.push(format!("{:.0}% of the ownership ends at a corporate owner with no recorded owners", 100.0 * bo.unresolved));
        }
        if bo.no_owner_identified {
            pts += cfg.w_no_owner;
            bo.opacity_reasons.push("no beneficial owner identified".into());
        }
        if bo.nominee_in_chain {
            pts += cfg.w_nominee;
            bo.opacity_reasons.push("nominee / bearer shares in the chain".into());
        }
        if bo.high_risk_jurisdiction {
            pts += cfg.w_jurisdiction;
            bo.opacity_reasons.push("high-risk jurisdiction in the chain".into());
        }
        if !bo.circular.is_empty() {
            pts += cfg.w_cycle;
            bo.opacity_reasons.push(format!("{} circular holding(s)", bo.circular.len()));
        }
        if bo.fragmented {
            pts += cfg.w_fragmented;
            bo.opacity_reasons.push("stakes split just below the beneficial-owner threshold".into());
        }
        bo.opacity = pts.min(1.0);
        Some(bo)
    }
}

impl OwnershipAnalysis {
    /// Analyse every party that holds an account in `registry`.
    pub fn build(c: &Compliance, registry: AccountRegistry, cfg: BoConfig) -> Result<Self> {
        let an = Analyzer::new(c, cfg.clone())?;
        let by_party = registry.accounts_by_party();
        let mut parties = BTreeMap::new();
        for pid in by_party.keys() {
            if let Some(bo) = an.analyse(*pid) {
                parties.insert(*pid, bo);
            }
        }
        let mut s = Summary { parties: parties.len(), with_accounts: by_party.len(), groups_dropped_too_large: an.groups_dropped, ..Default::default() };
        let mut group_accounts: HashMap<u32, usize> = HashMap::new();
        for (pid, bo) in &parties {
            if bo.kind == PartyKind::Entity {
                s.entities += 1;
                s.opaque += (bo.opacity >= 0.6) as usize;
                s.no_owner_identified += bo.no_owner_identified as usize;
                s.circular += !bo.circular.is_empty() as usize;
                s.fragmented += bo.fragmented as usize;
            }
            s.sanctioned_owned += (bo.blocked_ownership >= cfg.sanctioned_ownership) as usize;
            s.exposed += (bo.exposure > 0.0) as usize;
            if bo.group != 0 {
                *group_accounts.entry(bo.group).or_default() += by_party[pid].len();
            }
        }
        s.groups = group_accounts.len();
        s.multi_account_groups = group_accounts.values().filter(|n| **n >= 2).count();
        Ok(OwnershipAnalysis { cfg, parties, registry, summary: s })
    }

    /// Per-account ownership context for the graph model.
    pub fn graph_map(&self) -> OwnershipMap {
        let mut m = OwnershipMap::new();
        for (account, pid) in self.registry.iter() {
            let Some(bo) = self.parties.get(&pid) else { continue };
            m.insert(
                account,
                OwnerInfo {
                    party: pid,
                    group: bo.group,
                    owners: bo.related_owners.clone(),
                    layers: bo.layers.min(255) as u8,
                    unresolved: bo.unresolved as f32,
                    opacity: bo.opacity as f32,
                    sanctioned_pct: bo.blocked_ownership as f32,
                    exposure: bo.exposure as f32,
                    pep: bo.pep,
                    high_risk_jurisdiction: bo.high_risk_jurisdiction,
                },
            );
        }
        m
    }

    pub fn top_opaque(&self, n: usize) -> Vec<&PartyBo> {
        let mut v: Vec<&PartyBo> = self.parties.values().filter(|b| b.kind == PartyKind::Entity).collect();
        v.sort_by(|a, b| b.opacity.partial_cmp(&a.opacity).unwrap_or(std::cmp::Ordering::Equal).then(a.party.cmp(&b.party)));
        v.truncate(n);
        v
    }
}

impl PartyBo {
    /// Analyst-readable ownership profile.
    pub fn render(&self) -> String {
        use std::fmt::Write;
        let mut s = String::new();
        let _ = writeln!(s, "party {} {:?} \"{}\"   group {}   opacity {:.2}", self.party, self.kind, self.name, self.group, self.opacity);
        if self.kind == PartyKind::Entity {
            let _ = writeln!(s, "  layers {}  unresolved {:.0}%  nominee {}  high-risk jurisdiction {}  pep {}", self.layers, 100.0 * self.unresolved, self.nominee_in_chain, self.high_risk_jurisdiction, self.pep);
            for u in &self.ubos {
                let _ = writeln!(s, "  UBO {:>6} {:<28} {:>5.1}%  paths {}{}{}{}", u.party, u.name, 100.0 * u.effective, u.paths, if u.meets_threshold { "" } else { "  (below threshold)" }, if u.controls { "  [control]" } else { "" }, if u.pep { "  [PEP]" } else { "" });
            }
            if self.no_owner_identified {
                let _ = writeln!(s, "  NO BENEFICIAL OWNER IDENTIFIED");
            }
            for (u, f) in &self.unresolved_parties {
                let _ = writeln!(s, "  unresolved corporate owner {u} ({:.1}%)", 100.0 * f);
            }
            for c in &self.circular {
                let _ = writeln!(s, "  circular holding: {}", c.iter().map(|p| p.to_string()).collect::<Vec<_>>().join(" -> "));
            }
        }
        if self.blocked_ownership > 0.0 {
            let _ = writeln!(s, "  owned {:.0}% by blocked parties (50% rule {})", 100.0 * self.blocked_ownership, if self.blocked_ownership >= 0.5 { "APPLIES" } else { "does not apply" });
        }
        if self.exposure > 0.0 {
            let _ = writeln!(s, "  indirect sanctions exposure {:.4} via party {:?}", self.exposure, self.exposure_source);
        }
        if !self.opacity_reasons.is_empty() {
            let _ = writeln!(s, "  opacity because: {}", self.opacity_reasons.join("; "));
        }
        s
    }
}
