//! Sanctions screening: an inverted index over canonical list entries with
//! blocking (fast candidate generation) followed by multi-signal scoring.

use crate::country;
use crate::ml::{DobCode, IdCode, MatchFeatures, NatCode};
use crate::model::*;
use crate::similarity::{name_score, NameScore, PreparedName};
use crate::text::{skeleton, tokens};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Subject {
    pub kind: SubjectKind,
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
}

impl Subject {
    pub fn new(kind: SubjectKind, name: &str) -> Self {
        Self {
            kind,
            name: name.to_string(),
            aliases: vec![],
            dob: vec![],
            nationalities: vec![],
            countries: vec![],
            identifiers: vec![],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ScreeningConfig {
    /// Minimum score to raise an alert.
    pub review_threshold: f64,
    /// Minimum score (with corroboration) to rate a hit as high confidence.
    pub high_threshold: f64,
    pub max_hits: usize,
    /// Multiplier applied to weak/former-name aliases.
    pub weak_alias_factor: f64,
    /// If set, CDD approval is blocked while any loaded list is older than this many days.
    pub max_list_age_days: Option<u32>,
}

impl Default for ScreeningConfig {
    fn default() -> Self {
        Self { review_threshold: 0.85, high_threshold: 0.95, max_hits: 50, weak_alias_factor: 0.92, max_list_age_days: None }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HitLevel {
    Review,
    High,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Hit {
    pub list: String,
    pub uid: String,
    pub entry_hash: String,
    pub entry_kind: SubjectKind,
    pub primary_name: String,
    pub matched_name: String,
    pub name_score: f64,
    pub score: f64,
    pub level: HitLevel,
    pub reasons: Vec<String>,
    pub programs: Vec<String>,
    /// Structured evidence behind the score (input to the learned alert prioritiser).
    #[serde(default)]
    pub features: MatchFeatures,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ScreeningResult {
    pub hits: Vec<Hit>,
    pub candidates_evaluated: usize,
}

#[derive(Default)]
pub struct SanctionsIndex {
    entries: Vec<ListEntry>,
    prepared: Vec<Vec<PreparedName>>,
    by_key: HashMap<(String, String), usize>,
    postings: HashMap<String, Vec<u32>>,
    ids: HashMap<String, Vec<u32>>,
}

fn keys_for_token(tok: &str) -> Vec<String> {
    let mut k = vec![format!("t:{tok}")];
    let sk = skeleton(tok);
    if sk.chars().count() >= 2 {
        k.push(format!("p:{sk}"));
    }
    let cs: Vec<char> = tok.chars().collect();
    if cs.len() >= 5 {
        k.push(format!("d:{tok}"));
        for i in 0..cs.len() {
            let s: String = cs.iter().enumerate().filter(|(j, _)| *j != i).map(|(_, c)| *c).collect();
            k.push(format!("d:{s}"));
        }
    }
    k
}

impl SanctionsIndex {
    pub fn build(entries: Vec<ListEntry>) -> Self {
        let mut idx = SanctionsIndex::default();
        for (i, e) in entries.iter().enumerate() {
            let prep: Vec<PreparedName> =
                e.names.iter().map(|n| PreparedName::new(tokens(&n.name, e.kind))).collect();
            let mut seen: HashSet<String> = HashSet::new();
            for p in &prep {
                for t in &p.tokens {
                    if t.chars().count() < 2 {
                        continue;
                    }
                    for k in keys_for_token(t) {
                        if seen.contains(&k) {
                            continue;
                        }
                        idx.postings.entry(k.clone()).or_default().push(i as u32);
                        seen.insert(k);
                    }
                }
            }
            for id in &e.identifiers {
                let k = id.key();
                if k.len() >= 5 {
                    let v = idx.ids.entry(k).or_default();
                    if v.last() != Some(&(i as u32)) {
                        v.push(i as u32);
                    }
                }
            }
            idx.by_key.insert((e.list.clone(), e.uid.clone()), i);
            idx.prepared.push(prep);
        }
        idx.entries = entries;
        idx
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn entry(&self, list: &str, uid: &str) -> Option<&ListEntry> {
        self.by_key.get(&(list.to_string(), uid.to_string())).map(|&i| &self.entries[i])
    }

    /// ln(N/df) of the rarest and the mean name token that the screened and the listed name share exactly.
    /// A shared rare token is far more specific than a shared common surname. `(0, 0)` if nothing is shared.
    fn shared_token_idf(&self, q: &PreparedName, c: &PreparedName) -> (f64, f64) {
        let n = self.entries.len().max(1) as f64;
        let idfs: Vec<f64> = q
            .tokens
            .iter()
            .filter(|t| c.tokens.contains(t))
            .map(|t| {
                let df = self.postings.get(&format!("t:{t}")).map_or(1, |p| p.len().max(1)) as f64;
                (n / df).ln().max(0.0)
            })
            .collect();
        if idfs.is_empty() {
            return (0.0, 0.0);
        }
        (idfs.iter().cloned().fold(0.0, f64::max), idfs.iter().sum::<f64>() / idfs.len() as f64)
    }

    pub fn screen(&self, s: &Subject, cfg: &ScreeningConfig) -> ScreeningResult {
        if self.entries.is_empty() {
            return ScreeningResult::default();
        }
        let mut variants: Vec<(&str, PreparedName)> = Vec::new();
        for name in std::iter::once(&s.name).chain(s.aliases.iter()) {
            let t = tokens(name, s.kind);
            if !t.is_empty() {
                variants.push((name.as_str(), PreparedName::new(t)));
            }
        }

        // --- blocking: entries sharing >= min(2, n) distinct query tokens (exact, phonetic or 1-edit)
        let mut cands: HashSet<u32> = HashSet::new();
        for (_, v) in &variants {
            let long: Vec<&String> = v.tokens.iter().filter(|t| t.chars().count() >= 2).collect();
            if long.is_empty() {
                continue;
            }
            let need = long.len().min(2);
            let mut counts: HashMap<u32, usize> = HashMap::new();
            for t in &long {
                let mut set: HashSet<u32> = HashSet::new();
                for k in keys_for_token(t) {
                    if let Some(p) = self.postings.get(&k) {
                        set.extend(p.iter().copied());
                    }
                }
                for id in set {
                    *counts.entry(id).or_insert(0) += 1;
                }
            }
            cands.extend(counts.into_iter().filter(|(_, c)| *c >= need).map(|(id, _)| id));
        }

        // --- identifier matches
        let mut id_hits: HashMap<u32, (bool, String)> = HashMap::new();
        for ident in &s.identifiers {
            let k = ident.key();
            if k.len() < 5 {
                continue;
            }
            let Some(list) = self.ids.get(&k) else { continue };
            for &ei in list {
                let e = &self.entries[ei as usize];
                let Some(eid) = e.identifiers.iter().find(|x| x.key() == k) else { continue };
                let kinds_ok = ident.kind == eid.kind || ident.kind == IdKind::Other || eid.kind == IdKind::Other;
                let country_ok = ident.kind.globally_unique()
                    || match (&ident.country, &eid.country) {
                        (Some(a), Some(b)) => a == b,
                        _ => true,
                    };
                let strong = kinds_ok && country_ok;
                let reason = format!("identifier match ({:?} {})", eid.kind, eid.value);
                id_hits
                    .entry(ei)
                    .and_modify(|x| {
                        if strong && !x.0 {
                            *x = (true, reason.clone());
                        }
                    })
                    .or_insert((strong, reason));
            }
        }
        cands.extend(id_hits.keys().copied());

        let subj_nat: HashSet<String> = s.nationalities.iter().map(|c| country::normalize(c)).collect();
        let subj_geo: HashSet<String> =
            subj_nat.iter().cloned().chain(s.countries.iter().map(|c| country::normalize(c))).collect();

        let evaluated = cands.len();
        let mut hits: Vec<Hit> = Vec::new();
        for ei in cands {
            let e = &self.entries[ei as usize];
            let idh = id_hits.get(&ei);
            if !s.kind.compatible(e.kind) && idh.is_none() {
                continue;
            }
            // (weighted, raw, variant index, list-name index, token alignment)
            let mut best: Option<(f64, f64, usize, usize, NameScore)> = None;
            for (vi, (_, sp)) in variants.iter().enumerate() {
                for (ci, cp) in self.prepared[ei as usize].iter().enumerate() {
                    let ns = name_score(sp, cp);
                    let raw = ns.score;
                    let f = match e.names[ci].strength {
                        AliasStrength::Weak => cfg.weak_alias_factor,
                        _ => 1.0,
                    };
                    if best.as_ref().map_or(true, |b| raw * f > b.0) {
                        best = Some((raw * f, raw, vi, ci, ns));
                    }
                }
            }
            let Some((name_sc, _raw, vi, ci, ns)) = best else { continue };
            let sname = variants[vi].0;
            let cname = e.names[ci].name.as_str();
            let mut score = name_sc;
            let mut reasons = vec![format!("name similarity {name_sc:.2}: '{sname}' ~ '{cname}'")];
            let mut corroborated = e.kind != SubjectKind::Individual && name_sc >= 0.97;
            let (mut nat_code, mut country_assoc) = (NatCode::Unknown, false);

            let dob_rel = compare_dates(&s.dob, &e.dobs);
            match dob_rel {
                DobRel::Exact => {
                    score += 0.10;
                    corroborated = true;
                    reasons.push("date of birth matches exactly".into());
                }
                DobRel::YearOnly => {
                    score += 0.03;
                    corroborated = true;
                    reasons.push("birth year consistent".into());
                }
                DobRel::NearMiss => {
                    score -= 0.03;
                    reasons.push("birth year matches but month/day differ".into());
                }
                DobRel::Conflict => {
                    score -= 0.25;
                    reasons.push("date of birth conflicts with listing".into());
                }
                DobRel::Unknown => {}
            }

            let ent_nat: HashSet<String> = e.nationalities.iter().cloned().collect();
            if !subj_nat.is_empty() && !ent_nat.is_empty() {
                if subj_nat.intersection(&ent_nat).next().is_some() {
                    score += 0.03;
                    corroborated = true;
                    nat_code = NatCode::Match;
                    reasons.push("nationality matches".into());
                } else {
                    nat_code = NatCode::Differs;
                    score -= 0.05;
                    reasons.push("nationality differs".into());
                }
            } else {
                let ent_geo: HashSet<String> = ent_nat.iter().cloned().chain(e.countries.iter().cloned()).collect();
                if subj_geo.intersection(&ent_geo).next().is_some() {
                    score += 0.02;
                    country_assoc = true;
                    reasons.push("country association matches".into());
                }
            }

            let mut id_strong = false;
            let id_code = match idh {
                Some((true, _)) => IdCode::Strong,
                Some((false, _)) => IdCode::Weak,
                None => IdCode::None,
            };
            if let Some((strong, reason)) = idh {
                if *strong && name_sc >= 0.5 {
                    score = score.max(0.97);
                    id_strong = true;
                    reasons.push(reason.clone());
                } else if *strong {
                    score = score.max(0.90);
                    reasons.push(format!("{reason}; names differ — verify"));
                } else {
                    score = score.max(0.88);
                    reasons.push(format!("{reason}; id type or country differs"));
                }
            }

            let score = score.clamp(0.0, 1.0);
            if score < cfg.review_threshold {
                continue;
            }
            let level = if score >= cfg.high_threshold && (id_strong || (name_sc >= cfg.review_threshold && corroborated)) {
                HitLevel::High
            } else {
                HitLevel::Review
            };
            let (rarest_idf, mean_idf) = self.shared_token_idf(&variants[vi].1, &self.prepared[ei as usize][ci]);
            let features = MatchFeatures {
                rule_score: score,
                name_score: name_sc,
                strong_tokens: ns.strong_matches as u32,
                subject_tokens: variants[vi].1.tokens.len() as u32,
                entry_tokens: self.prepared[ei as usize][ci].tokens.len() as u32,
                weak_alias: e.names[ci].strength == AliasStrength::Weak,
                subject_alias_used: sname != s.name.as_str(),
                exact_name: variants[vi].1.sorted_joined == self.prepared[ei as usize][ci].sorted_joined,
                dob: DobCode::from(dob_rel),
                nationality: nat_code,
                country_assoc,
                id: id_code,
                entry_non_individual: e.kind != SubjectKind::Individual,
                rarest_shared_idf: rarest_idf,
                mean_shared_idf: mean_idf,
                competing_hits: 0,
            };
            hits.push(Hit {
                list: e.list.clone(),
                uid: e.uid.clone(),
                entry_hash: e.content_hash.clone(),
                entry_kind: e.kind,
                primary_name: e.primary_name().to_string(),
                matched_name: cname.to_string(),
                name_score: name_sc,
                score,
                level,
                reasons,
                programs: e.programs.clone(),
                features,
            });
        }
        let competing = hits.len().saturating_sub(1) as u32;
        for h in hits.iter_mut() {
            h.features.competing_hits = competing;
        }
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap().then_with(|| a.uid.cmp(&b.uid)));
        hits.truncate(cfg.max_hits);
        ScreeningResult { hits, candidates_evaluated: evaluated }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn entry(uid: &str, kind: SubjectKind, name: &str) -> ListEntry {
        ListEntry {
            list: "TEST".into(),
            uid: uid.into(),
            kind,
            names: vec![NameVariant { name: name.into(), strength: AliasStrength::Primary }],
            dobs: vec![],
            nationalities: vec![],
            countries: vec![],
            identifiers: vec![],
            programs: vec!["TESTPROG".into()],
            remarks: None,
            listed_on: None,
            content_hash: String::new(),
        }
    }

    fn fixture() -> SanctionsIndex {
        let mut petrov = entry("1", SubjectKind::Individual, "PETROV, Ivan Sergeyevich");
        petrov.dobs = vec![PartialDate::ymd(1965, 3, 12)];
        petrov.nationalities = vec!["RU".into()];
        petrov.identifiers = vec![Identifier::new(IdKind::Passport, "75 1234567", Some("RU"))];
        petrov.names.push(NameVariant { name: "Vanya Petrof".into(), strength: AliasStrength::Strong });
        petrov.names.push(NameVariant { name: "Johnny Pet".into(), strength: AliasStrength::Weak });
        let star = entry("2", SubjectKind::Entity, "NORTHERN STAR TRADING LLC");
        let mut ship = entry("3", SubjectKind::Vessel, "MV TEST HORIZON");
        ship.identifiers = vec![Identifier::new(IdKind::Imo, "9876543", None)];
        let other = entry("4", SubjectKind::Individual, "GONZALEZ, Maria Elena");
        SanctionsIndex::build(vec![petrov, star, ship, other].into_iter().map(|e| e.finalize()).collect())
    }

    fn subj(name: &str) -> Subject {
        Subject::new(SubjectKind::Individual, name)
    }

    #[test]
    fn exact_name_and_dob_is_high() {
        let mut s = subj("Ivan Petrov");
        s.dob = vec![PartialDate::ymd(1965, 3, 12)];
        let r = fixture().screen(&s, &ScreeningConfig::default());
        assert_eq!(r.hits.len(), 1);
        let h = &r.hits[0];
        assert_eq!(h.uid, "1");
        assert_eq!(h.level, HitLevel::High);
        assert!(h.reasons.iter().any(|x| x.contains("date of birth matches")));
    }

    #[test]
    fn name_only_individual_is_review_not_high() {
        let r = fixture().screen(&subj("Ivan Sergeyevich Petrov"), &ScreeningConfig::default());
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.hits[0].level, HitLevel::Review);
    }

    #[test]
    fn conflicting_dob_suppresses_hit() {
        let mut s = subj("Ivan Sergeyevich Petrov");
        s.dob = vec![PartialDate::ymd(1982, 7, 1)];
        assert!(fixture().screen(&s, &ScreeningConfig::default()).hits.is_empty());
    }

    #[test]
    fn typo_and_alias_matching() {
        let mut s = subj("Ivan Petrof");
        s.dob = vec![PartialDate::year(1965)];
        assert_eq!(fixture().screen(&s, &ScreeningConfig::default()).hits[0].uid, "1");
        let mut a = subj("Vanya Petrov");
        a.dob = vec![PartialDate::ymd(1965, 3, 12)];
        assert_eq!(fixture().screen(&a, &ScreeningConfig::default()).hits[0].uid, "1");
    }

    #[test]
    fn subject_aliases_are_screened() {
        let mut s = subj("Completely Different");
        s.aliases = vec!["Ivan Sergeyevich Petrov".into()];
        assert_eq!(fixture().screen(&s, &ScreeningConfig::default()).hits.len(), 1);
    }

    #[test]
    fn identifier_match_overrides_name() {
        let mut s = subj("Totally Unrelated Person");
        s.identifiers = vec![Identifier::new(IdKind::Passport, "751234567", Some("RU"))];
        let r = fixture().screen(&s, &ScreeningConfig::default());
        assert_eq!(r.hits.len(), 1);
        assert_eq!(r.hits[0].level, HitLevel::Review); // names differ => verify, not auto-high
        assert!(r.hits[0].score >= 0.90);

        let mut s2 = subj("Ivan Petrov");
        s2.identifiers = vec![Identifier::new(IdKind::Passport, "75 1234567", Some("RU"))];
        let r2 = fixture().screen(&s2, &ScreeningConfig::default());
        assert_eq!(r2.hits[0].level, HitLevel::High);

        // same number, different issuing country: only a weak signal
        let mut s3 = subj("Totally Unrelated Person");
        s3.identifiers = vec![Identifier::new(IdKind::Passport, "75 1234567", Some("FR"))];
        let r3 = fixture().screen(&s3, &ScreeningConfig::default());
        assert!(r3.hits.is_empty() || r3.hits[0].level == HitLevel::Review);
    }

    #[test]
    fn entity_legal_suffix_variants() {
        let s = Subject::new(SubjectKind::Entity, "Northern Star Trading Limited");
        let r = fixture().screen(&s, &ScreeningConfig::default());
        assert_eq!(r.hits[0].uid, "2");
        assert_eq!(r.hits[0].level, HitLevel::High);
    }

    #[test]
    fn vessel_by_imo_even_with_renamed_ship() {
        let mut s = Subject::new(SubjectKind::Vessel, "MV RENAMED SEA");
        s.identifiers = vec![Identifier::new(IdKind::Imo, "9876543", None)];
        let r = fixture().screen(&s, &ScreeningConfig::default());
        assert_eq!(r.hits[0].uid, "3");
    }

    #[test]
    fn unrelated_subject_has_no_hits() {
        let r = fixture().screen(&subj("Chen Wei"), &ScreeningConfig::default());
        assert!(r.hits.is_empty());
    }

    #[test]
    fn weak_alias_is_discounted() {
        let r = fixture().screen(&subj("Johnny Pet"), &ScreeningConfig { review_threshold: 0.5, ..Default::default() });
        let h = r.hits.iter().find(|h| h.uid == "1").unwrap();
        assert!(h.name_score <= 0.92 + 1e-9);
    }

    #[test]
    fn blocking_keeps_candidate_set_small() {
        // 20k synthetic entries; a query must not evaluate more than a small fraction.
        let syl = ["ka", "lo", "mi", "ra", "su", "te", "vo", "ni", "da", "be", "zu", "fa", "go", "hi", "je", "pu"];
        let mut v = Vec::new();
        for i in 0..20_000usize {
            let n = |k: usize| format!("{}{}{}", syl[(i >> k) % 16], syl[(i / 7 >> k) % 16], syl[(i / 13 + k) % 16]);
            v.push(entry(&i.to_string(), SubjectKind::Individual, &format!("{} {}", n(0), n(3))).finalize());
        }
        let idx = SanctionsIndex::build(v);
        let r = idx.screen(&subj("Ivan Petrov"), &ScreeningConfig::default());
        assert!(r.candidates_evaluated < 2_000, "evaluated {}", r.candidates_evaluated);
        assert_eq!(idx.len(), 20_000);
    }
}
