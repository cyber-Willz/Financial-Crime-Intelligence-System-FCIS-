//! OFAC SDN / Consolidated (non-SDN) legacy CSV format: `sdn.csv`, `alt.csv`, `add.csv`.
//! The files have no header row; empty values are written as `-0- `.

use crate::error::Result;
use crate::model::*;
use std::collections::HashMap;

fn clean(s: &str) -> String {
    let t = s.trim();
    if t == "-0-" { String::new() } else { t.to_string() }
}

fn reader(text: &str) -> csv::Reader<&[u8]> {
    csv::ReaderBuilder::new().has_headers(false).flexible(true).from_reader(text.as_bytes())
}

#[derive(Default, Debug)]
pub struct Remarks {
    pub dobs: Vec<PartialDate>,
    pub nationalities: Vec<String>,
    pub identifiers: Vec<Identifier>,
}

fn strip_prefix_ci<'a>(s: &'a str, prefix: &str) -> Option<&'a str> {
    if s.len() >= prefix.len() && s.is_char_boundary(prefix.len()) && s[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(s[prefix.len()..].trim())
    } else {
        None
    }
}

fn years_in(s: &str) -> Vec<i32> {
    s.split(|c: char| !c.is_ascii_digit())
        .filter(|t| t.len() == 4)
        .filter_map(|t| t.parse().ok())
        .collect()
}

const ID_LABELS: &[(&str, IdKind)] = &[
    ("vessel registration identification imo ", IdKind::Imo),
    ("digital currency address - ", IdKind::Crypto),
    ("national id no.", IdKind::NationalId),
    ("business registration number", IdKind::Registration),
    ("registration number", IdKind::Registration),
    ("tax id no.", IdKind::TaxId),
    ("passport", IdKind::Passport),
    ("swift/bic", IdKind::Swift),
];

/// Parses the semicolon-separated `Remarks` column.
pub fn parse_remarks(remarks: &str) -> Remarks {
    let mut r = Remarks::default();
    for seg in remarks.split(';') {
        let seg = seg.trim();
        let seg = strip_prefix_ci(seg, "alt.").unwrap_or(seg);
        if let Some(rest) = strip_prefix_ci(seg, "dob") {
            let rest = strip_prefix_ci(rest, "circa").unwrap_or(rest);
            if let Some(d) = PartialDate::parse(rest) {
                r.dobs.push(d);
            } else {
                // ranges such as "1960 to 1962" or "1960-1962"
                let ys = years_in(rest);
                if let (Some(a), Some(b)) = (ys.first(), ys.last()) {
                    for y in *a..=(*b).min(*a + 10) {
                        r.dobs.push(PartialDate::year(y));
                    }
                }
            }
        } else if let Some(rest) = strip_prefix_ci(seg, "nationality ").or_else(|| strip_prefix_ci(seg, "citizen ")) {
            r.nationalities.push(rest.to_string());
        } else {
            for (label, kind) in ID_LABELS {
                if let Some(rest) = strip_prefix_ci(seg, label) {
                    let country = rest.rfind('(').and_then(|i| rest[i + 1..].split(')').next()).map(str::trim);
                    let body = rest.split('(').next().unwrap_or("").trim();
                    let body = body.split(" expires").next().unwrap_or(body).trim();
                    let value = if *kind == IdKind::Crypto {
                        body.split_whitespace().last().unwrap_or("")
                    } else if *kind == IdKind::Imo || *kind == IdKind::Swift {
                        body.split_whitespace().next().unwrap_or("")
                    } else {
                        body
                    };
                    if !value.is_empty() {
                        r.identifiers.push(Identifier::new(*kind, value, country));
                    }
                    break;
                }
            }
        }
    }
    r
}

fn programs(s: &str) -> Vec<String> {
    s.split("] [")
        .flat_map(|p| p.split(';'))
        .map(|p| p.trim().trim_matches(|c| c == '[' || c == ']').trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// `alt` and `add` are optional companion files.
pub fn parse(sdn: &str, alt: Option<&str>, add: Option<&str>, list_name: &str) -> Result<Vec<ListEntry>> {
    let mut aliases: HashMap<String, Vec<NameVariant>> = HashMap::new();
    if let Some(alt) = alt {
        for rec in reader(alt).records() {
            let rec = rec?;
            if rec.len() < 4 {
                continue;
            }
            let (ent, ty, name) = (clean(&rec[0]), clean(&rec[2]).to_ascii_lowercase(), clean(&rec[3]));
            if name.is_empty() || ent.parse::<u64>().is_err() {
                continue;
            }
            let strength = if ty == "aka" { AliasStrength::Strong } else { AliasStrength::Weak };
            aliases.entry(ent).or_default().push(NameVariant { name, strength });
        }
    }
    let mut addresses: HashMap<String, Vec<String>> = HashMap::new();
    if let Some(add) = add {
        for rec in reader(add).records() {
            let rec = rec?;
            if rec.len() < 5 {
                continue;
            }
            let (ent, country) = (clean(&rec[0]), clean(&rec[4]));
            if !country.is_empty() && ent.parse::<u64>().is_ok() {
                addresses.entry(ent).or_default().push(country);
            }
        }
    }

    let mut out = Vec::new();
    for rec in reader(sdn).records() {
        let rec = rec?;
        if rec.len() < 4 {
            continue; // trailing Ctrl-Z line etc.
        }
        let uid = clean(&rec[0]);
        if uid.parse::<u64>().is_err() {
            continue;
        }
        let name = clean(&rec[1]);
        if name.is_empty() {
            continue;
        }
        let kind = match clean(&rec[2]).to_ascii_lowercase().as_str() {
            "individual" => SubjectKind::Individual,
            "vessel" => SubjectKind::Vessel,
            "aircraft" => SubjectKind::Aircraft,
            _ => SubjectKind::Entity,
        };
        let remarks = rec.get(11).map(clean).unwrap_or_default();
        let parsed = parse_remarks(&remarks);
        let mut names = vec![NameVariant { name, strength: AliasStrength::Primary }];
        names.extend(aliases.remove(&uid).unwrap_or_default());
        out.push(
            ListEntry {
                list: list_name.to_string(),
                uid: uid.clone(),
                kind,
                names,
                dobs: parsed.dobs,
                nationalities: parsed.nationalities,
                countries: addresses.remove(&uid).unwrap_or_default(),
                identifiers: parsed.identifiers,
                programs: programs(&clean(&rec[3])),
                remarks: (!remarks.is_empty()).then_some(remarks),
                listed_on: None,
                content_hash: String::new(),
            }
            .finalize(),
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remarks_parsing() {
        let r = parse_remarks("DOB 12 Mar 1965; alt. DOB 1966; POB Aleppo, Syria; nationality Syria; Gender Male; \
            Passport N0123456 (Syria) expires 01 Jan 2030; National ID No. 99887766 (Syria); \
            Digital Currency Address - XBT 1AbCdEfGhIjKlMnOpQrStUvWxYz123456; Linked To: SOMEBODY");
        assert_eq!(r.dobs, vec![PartialDate::ymd(1965, 3, 12), PartialDate::year(1966)]);
        assert_eq!(r.nationalities, vec!["Syria"]);
        assert_eq!(r.identifiers.len(), 3);
        assert_eq!(r.identifiers[0], Identifier::new(IdKind::Passport, "N0123456", Some("SY")));
        assert_eq!(r.identifiers[1].kind, IdKind::NationalId);
        assert_eq!(r.identifiers[2].kind, IdKind::Crypto);
        assert_eq!(r.identifiers[2].value, "1AbCdEfGhIjKlMnOpQrStUvWxYz123456");
    }

    #[test]
    fn full_parse_with_alt_and_add() {
        let sdn = "\
10001,\"PETROV, Ivan Sergeyevich\",\"individual\",\"TESTPROG\",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,\"DOB 12 Mar 1965; nationality Russia; Passport 75 1234567 (Russia)\"
10002,\"NORTHERN STAR TRADING LLC\",-0- ,\"TESTPROG] [OTHERPROG\",-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,-0- ,\"Registration Number 1177746000001 (Russia)\"
10003,\"MV TEST HORIZON\",\"vessel\",\"TESTPROG\",-0- ,-0- ,\"Cargo\",-0- ,-0- ,\"Panama\",-0- ,\"Vessel Registration Identification IMO 9876543\"
\x1a";
        let alt = "10001,5001,\"aka\",\"PETROFF, Ivan\",-0- \n10001,5002,\"fka\",\"PETROV, Vanya\",-0- \n";
        let add = "10001,3001,\"1 Test St\",\"Moscow\",\"Russia\",-0- \n10002,3002,\"Dock 4\",\"Limassol\",\"Cyprus\",-0- \n";
        let v = parse(sdn, Some(alt), Some(add), "OFAC-SDN").unwrap();
        assert_eq!(v.len(), 3);
        let p = &v[0];
        assert_eq!(p.kind, SubjectKind::Individual);
        assert_eq!(p.names.len(), 3);
        assert_eq!(p.names[1].strength, AliasStrength::Strong);
        assert_eq!(p.names[2].strength, AliasStrength::Weak);
        assert_eq!(p.nationalities, vec!["RU"]);
        assert_eq!(p.countries, vec!["RU"]);
        assert_eq!(v[1].kind, SubjectKind::Entity);
        assert_eq!(v[1].programs, vec!["TESTPROG", "OTHERPROG"]);
        assert_eq!(v[1].countries, vec!["CY"]);
        assert_eq!(v[2].kind, SubjectKind::Vessel);
        assert_eq!(v[2].identifiers[0].kind, IdKind::Imo);
        assert_eq!(v[2].identifiers[0].value, "9876543");
        assert!(!p.content_hash.is_empty());
    }
}
