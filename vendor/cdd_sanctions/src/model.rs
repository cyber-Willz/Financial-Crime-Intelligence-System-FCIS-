//! Core data model shared by list parsers, the screening index and the CDD layer.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubjectKind {
    Individual,
    Entity,
    Vessel,
    Aircraft,
    Unknown,
}

impl SubjectKind {
    pub fn compatible(self, other: Self) -> bool {
        self == Self::Unknown || other == Self::Unknown || self == other
    }
    pub fn parse(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "individual" | "person" | "p" => Self::Individual,
            "entity" | "organization" | "organisation" | "business" | "enterprise" | "e" => Self::Entity,
            "vessel" => Self::Vessel,
            "aircraft" => Self::Aircraft,
            _ => Self::Unknown,
        }
    }
}

/// A date that may be known only to year or month precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PartialDate {
    pub year: Option<i32>,
    pub month: Option<u8>,
    pub day: Option<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DobRel {
    Exact,
    YearOnly,
    NearMiss,
    Conflict,
    Unknown,
}

impl DobRel {
    fn rank(self) -> u8 {
        match self {
            Self::Exact => 4,
            Self::YearOnly => 3,
            Self::NearMiss => 2,
            Self::Conflict => 1,
            Self::Unknown => 0,
        }
    }
}

fn month_num(s: &str) -> Option<u8> {
    let m = s.get(..3)?.to_ascii_lowercase();
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        .iter()
        .position(|x| *x == m)
        .map(|i| i as u8 + 1)
}

impl PartialDate {
    pub fn year(y: i32) -> Self {
        Self { year: Some(y), month: None, day: None }
    }
    pub fn ymd(y: i32, m: u8, d: u8) -> Self {
        Self { year: Some(y), month: Some(m), day: Some(d) }
    }

    /// Parses `YYYY`, `YYYY-MM`, `YYYY-MM-DD`, `DD Mon YYYY`, `Mon YYYY`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().trim_matches(|c: char| c == '.' || c == ',');
        if s.is_empty() {
            return None;
        }
        let valid = |m: u8, d: u8| (1..=12).contains(&m) && (1..=31).contains(&d);
        if s.contains('-') && s.chars().next()?.is_ascii_digit() {
            let p: Vec<&str> = s.split('-').collect();
            let y: i32 = p.first()?.parse().ok()?;
            if p[0].len() != 4 {
                return None;
            }
            return match p.len() {
                1 => Some(Self::year(y)),
                2 => {
                    let m: u8 = p[1].parse().ok()?;
                    valid(m, 1).then_some(Self { year: Some(y), month: Some(m), day: None })
                }
                3 => {
                    let m: u8 = p[1].parse().ok()?;
                    let d: u8 = p[2].get(..2).unwrap_or(p[2]).parse().ok()?;
                    valid(m, d).then_some(Self::ymd(y, m, d))
                }
                _ => None,
            };
        }
        let p: Vec<&str> = s.split_whitespace().collect();
        match p.as_slice() {
            [y] if y.len() == 4 => Some(Self::year(y.parse().ok()?)),
            [m, y] if y.len() == 4 => Some(Self { year: Some(y.parse().ok()?), month: Some(month_num(m)?), day: None }),
            [d, m, y] if y.len() == 4 => {
                let (d, m) = (d.parse::<u8>().ok()?, month_num(m)?);
                valid(m, d).then_some(Self::ymd(y.parse().ok()?, m, d))
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for PartialDate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.year, self.month, self.day) {
            (Some(y), Some(m), Some(d)) => write!(f, "{y:04}-{m:02}-{d:02}"),
            (Some(y), Some(m), None) => write!(f, "{y:04}-{m:02}"),
            (Some(y), _, _) => write!(f, "{y:04}"),
            _ => write!(f, "?"),
        }
    }
}

fn pair_rel(x: &PartialDate, y: &PartialDate) -> DobRel {
    if x.year != y.year {
        return DobRel::Conflict;
    }
    match (x.month, y.month) {
        (Some(m1), Some(m2)) if m1 != m2 => DobRel::NearMiss,
        (Some(_), Some(_)) => match (x.day, y.day) {
            (Some(d1), Some(d2)) if d1 == d2 => DobRel::Exact,
            (Some(_), Some(_)) => DobRel::NearMiss,
            _ => DobRel::YearOnly,
        },
        _ => DobRel::YearOnly,
    }
}

/// Best relationship between two sets of (possibly partial) dates.
pub fn compare_dates(a: &[PartialDate], b: &[PartialDate]) -> DobRel {
    let a: Vec<&PartialDate> = a.iter().filter(|d| d.year.is_some()).collect();
    let b: Vec<&PartialDate> = b.iter().filter(|d| d.year.is_some()).collect();
    if a.is_empty() || b.is_empty() {
        return DobRel::Unknown;
    }
    let mut best = DobRel::Conflict;
    for x in &a {
        for y in &b {
            let r = pair_rel(x, y);
            if r.rank() > best.rank() {
                best = r;
            }
        }
    }
    best
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IdKind {
    Passport,
    NationalId,
    TaxId,
    Registration,
    Imo,
    Swift,
    Crypto,
    Other,
}

impl IdKind {
    /// Kinds whose values are globally unique regardless of issuing country.
    pub fn globally_unique(self) -> bool {
        matches!(self, Self::Imo | Self::Swift | Self::Crypto)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Identifier {
    pub kind: IdKind,
    pub value: String,
    #[serde(default)]
    pub country: Option<String>,
}

impl Identifier {
    pub fn new(kind: IdKind, value: &str, country: Option<&str>) -> Self {
        Self {
            kind,
            value: value.trim().to_string(),
            country: country.filter(|c| !c.trim().is_empty()).map(crate::country::normalize),
        }
    }
    /// Comparison key: upper-case alphanumerics only.
    pub fn key(&self) -> String {
        self.value.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_uppercase()).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AliasStrength {
    Primary,
    Strong,
    Weak,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NameVariant {
    pub name: String,
    pub strength: AliasStrength,
}

/// One designated person/entity/vessel from a sanctions list, in canonical form.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ListEntry {
    pub list: String,
    pub uid: String,
    pub kind: SubjectKind,
    pub names: Vec<NameVariant>,
    #[serde(default)]
    pub dobs: Vec<PartialDate>,
    #[serde(default)]
    pub nationalities: Vec<String>,
    #[serde(default)]
    pub countries: Vec<String>,
    #[serde(default)]
    pub identifiers: Vec<Identifier>,
    #[serde(default)]
    pub programs: Vec<String>,
    #[serde(default)]
    pub remarks: Option<String>,
    #[serde(default)]
    pub listed_on: Option<String>,
    #[serde(default)]
    pub content_hash: String,
}

impl ListEntry {
    /// Normalises countries and computes the content hash used for change detection.
    pub fn finalize(mut self) -> Self {
        for c in self.nationalities.iter_mut().chain(self.countries.iter_mut()) {
            *c = crate::country::normalize(c);
        }
        self.nationalities.sort();
        self.nationalities.dedup();
        self.countries.sort();
        self.countries.dedup();
        self.content_hash.clear();
        let bytes = serde_json::to_vec(&self).expect("serialisable");
        self.content_hash = sha256_hex(&bytes);
        self
    }
    pub fn primary_name(&self) -> &str {
        self.names.first().map(|n| n.name.as_str()).unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn date_parsing() {
        assert_eq!(PartialDate::parse("1965-03-12"), Some(PartialDate::ymd(1965, 3, 12)));
        assert_eq!(PartialDate::parse("12 Mar 1965"), Some(PartialDate::ymd(1965, 3, 12)));
        assert_eq!(PartialDate::parse("Mar 1965"), Some(PartialDate { year: Some(1965), month: Some(3), day: None }));
        assert_eq!(PartialDate::parse("1965"), Some(PartialDate::year(1965)));
        assert_eq!(PartialDate::parse("1965-13-40"), None);
        assert_eq!(PartialDate::parse("garbage"), None);
    }

    #[test]
    fn date_comparison() {
        let full = [PartialDate::ymd(1965, 3, 12)];
        assert_eq!(compare_dates(&full, &[PartialDate::ymd(1965, 3, 12)]), DobRel::Exact);
        assert_eq!(compare_dates(&full, &[PartialDate::year(1965)]), DobRel::YearOnly);
        assert_eq!(compare_dates(&full, &[PartialDate::ymd(1965, 4, 12)]), DobRel::NearMiss);
        assert_eq!(compare_dates(&full, &[PartialDate::year(1970)]), DobRel::Conflict);
        assert_eq!(compare_dates(&full, &[]), DobRel::Unknown);
        // best of several candidate dates wins
        assert_eq!(compare_dates(&full, &[PartialDate::year(1970), PartialDate::ymd(1965, 3, 12)]), DobRel::Exact);
    }

    #[test]
    fn hash_changes_with_content() {
        let mk = |n: &str| ListEntry {
            list: "T".into(), uid: "1".into(), kind: SubjectKind::Individual,
            names: vec![NameVariant { name: n.into(), strength: AliasStrength::Primary }],
            dobs: vec![], nationalities: vec!["Syria".into()], countries: vec![], identifiers: vec![],
            programs: vec![], remarks: None, listed_on: None, content_hash: String::new(),
        }.finalize();
        assert_eq!(mk("A B").content_hash, mk("A B").content_hash);
        assert_ne!(mk("A B").content_hash, mk("A C").content_hash);
        assert_eq!(mk("A B").nationalities, vec!["SY"]);
    }
}
