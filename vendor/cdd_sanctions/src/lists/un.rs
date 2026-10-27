//! UN Security Council Consolidated List (XML).

use super::xmltree::{parse as parse_xml, Node};
use crate::error::Result;
use crate::model::*;

fn name_of(n: &Node, fields: &[&str]) -> String {
    fields.iter().filter_map(|f| n.text_of(f)).collect::<Vec<_>>().join(" ")
}

fn dob_from(n: &Node, out: &mut Vec<PartialDate>) {
    for d in n.all("INDIVIDUAL_DATE_OF_BIRTH") {
        let approx = d.text_of("TYPE_OF_DATE").map(|t| t.to_ascii_lowercase().contains("approx")).unwrap_or(false);
        let before = out.len();
        let mut years: Vec<i32> = Vec::new();
        if let Some(date) = d.text_of("DATE").and_then(|t| PartialDate::parse(t.get(..10).unwrap_or(&t))) {
            out.push(date);
            years.extend(date.year);
        }
        if let Some(y) = d.text_of("YEAR").and_then(|t| t.parse::<i32>().ok()) {
            out.push(PartialDate::year(y));
            years.push(y);
        }
        let from = d.text_of("FROM_YEAR").and_then(|t| t.parse::<i32>().ok());
        let to = d.text_of("TO_YEAR").and_then(|t| t.parse::<i32>().ok());
        if let (Some(a), Some(b)) = (from, to) {
            for y in a..=b.min(a + 10) {
                out.push(PartialDate::year(y));
            }
        }
        if out.len() == before {
            // some lists carry only free text, e.g. "Nov. 1973"
            if let Some(date) = d.text_of("NOTE").and_then(|t| PartialDate::parse(&t)) {
                out.push(date);
            }
        }
        if approx {
            // "approximately 1977": accept +/- 2 years so a small error does not read as a conflict
            for y in years {
                for delta in [-2, -1, 1, 2] {
                    out.push(PartialDate::year(y + delta));
                }
            }
        }
    }
}

fn aliases(n: &Node, container: &str, out: &mut Vec<NameVariant>) {
    for a in n.all(container) {
        if let Some(name) = a.text_of("ALIAS_NAME") {
            let low = a.text_of("QUALITY").map(|q| q.to_ascii_lowercase().contains("low")).unwrap_or(false);
            out.push(NameVariant { name, strength: if low { AliasStrength::Weak } else { AliasStrength::Strong } });
        }
    }
}

fn countries(n: &Node, container: &str) -> Vec<String> {
    n.values_under(container, "COUNTRY")
}

fn programs(n: &Node) -> Vec<String> {
    let mut p = n.values_under("LIST_TYPE", "VALUE");
    if let Some(t) = n.text_of("UN_LIST_TYPE") {
        p.push(t);
    }
    p.sort();
    p.dedup();
    p
}

pub fn parse(xml: &str, list_name: &str) -> Result<Vec<ListEntry>> {
    let root = parse_xml(xml)?;
    let mut out = Vec::new();

    if let Some(inds) = root.child("INDIVIDUALS") {
        for n in inds.all("INDIVIDUAL") {
            let name = name_of(n, &["FIRST_NAME", "SECOND_NAME", "THIRD_NAME", "FOURTH_NAME"]);
            let uid = n.text_of("REFERENCE_NUMBER").or_else(|| n.text_of("DATAID")).or_else(|| n.text_of("DABZTAiD"));
            let (Some(uid), false) = (uid, name.is_empty()) else { continue };
            let mut names = vec![NameVariant { name, strength: AliasStrength::Primary }];
            aliases(n, "INDIVIDUAL_ALIAS", &mut names);
            let mut dobs = Vec::new();
            dob_from(n, &mut dobs);
            let identifiers = n
                .all("INDIVIDUAL_DOCUMENT")
                .filter_map(|d| {
                    let number = d.text_of("NUMBER")?;
                    let ty = d.text_of("TYPE_OF_DOCUMENT").unwrap_or_default().to_ascii_lowercase();
                    let kind = if ty.contains("passport") {
                        IdKind::Passport
                    } else if ty.contains("national") || ty.contains("identification") {
                        IdKind::NationalId
                    } else {
                        IdKind::Other
                    };
                    Some(Identifier::new(kind, &number, d.text_of("ISSUING_COUNTRY").as_deref()))
                })
                .collect();
            out.push(
                ListEntry {
                    list: list_name.into(),
                    uid,
                    kind: SubjectKind::Individual,
                    names,
                    dobs,
                    nationalities: n.values_under("NATIONALITY", "VALUE"),
                    countries: countries(n, "INDIVIDUAL_ADDRESS"),
                    identifiers,
                    programs: programs(n),
                    remarks: n.text_of("COMMENTS1"),
                    listed_on: n.text_of("LISTED_ON"),
                    content_hash: String::new(),
                }
                .finalize(),
            );
        }
    }

    if let Some(ents) = root.child("ENTITIES") {
        for n in ents.all("ENTITY") {
            let name = name_of(n, &["FIRST_NAME"]);
            let uid = n.text_of("REFERENCE_NUMBER").or_else(|| n.text_of("DATAID")).or_else(|| n.text_of("DABZTAiD"));
            let (Some(uid), false) = (uid, name.is_empty()) else { continue };
            let mut names = vec![NameVariant { name, strength: AliasStrength::Primary }];
            aliases(n, "ENTITY_ALIAS", &mut names);
            out.push(
                ListEntry {
                    list: list_name.into(),
                    uid,
                    kind: SubjectKind::Entity,
                    names,
                    dobs: vec![],
                    nationalities: vec![],
                    countries: countries(n, "ENTITY_ADDRESS"),
                    identifiers: vec![],
                    programs: programs(n),
                    remarks: n.text_of("COMMENTS1"),
                    listed_on: n.text_of("LISTED_ON"),
                    content_hash: String::new(),
                }
                .finalize(),
            );
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const XML: &str = r#"<?xml version="1.0"?>
<CONSOLIDATED_LIST>
 <INDIVIDUALS>
  <INDIVIDUAL>
   <DATAID>900001</DATAID>
   <FIRST_NAME>Ivan</FIRST_NAME><SECOND_NAME>Sergeyevich</SECOND_NAME><THIRD_NAME>Petrov</THIRD_NAME>
   <UN_LIST_TYPE>TestRegime</UN_LIST_TYPE><REFERENCE_NUMBER>TST.001</REFERENCE_NUMBER>
   <LISTED_ON>2020-01-15</LISTED_ON><COMMENTS1>Fictional test record</COMMENTS1>
   <LIST_TYPE><VALUE>UN List</VALUE></LIST_TYPE>
   <NATIONALITY><VALUE>Russian Federation</VALUE></NATIONALITY>
   <INDIVIDUAL_ALIAS><QUALITY>Good</QUALITY><ALIAS_NAME>Vanya Petrof</ALIAS_NAME></INDIVIDUAL_ALIAS>
   <INDIVIDUAL_ALIAS><QUALITY>Low</QUALITY><ALIAS_NAME>I S P</ALIAS_NAME></INDIVIDUAL_ALIAS>
   <INDIVIDUAL_ALIAS><QUALITY></QUALITY><ALIAS_NAME></ALIAS_NAME></INDIVIDUAL_ALIAS>
   <INDIVIDUAL_ADDRESS><COUNTRY>Cyprus</COUNTRY></INDIVIDUAL_ADDRESS>
   <INDIVIDUAL_DATE_OF_BIRTH><TYPE_OF_DATE>EXACT</TYPE_OF_DATE><DATE>1965-03-12</DATE></INDIVIDUAL_DATE_OF_BIRTH>
   <INDIVIDUAL_DATE_OF_BIRTH><TYPE_OF_DATE>BETWEEN</TYPE_OF_DATE><FROM_YEAR>1964</FROM_YEAR><TO_YEAR>1966</TO_YEAR></INDIVIDUAL_DATE_OF_BIRTH>
   <INDIVIDUAL_DOCUMENT><TYPE_OF_DOCUMENT>Passport</TYPE_OF_DOCUMENT><NUMBER>75 1234567</NUMBER><ISSUING_COUNTRY>Russian Federation</ISSUING_COUNTRY></INDIVIDUAL_DOCUMENT>
  </INDIVIDUAL>
 </INDIVIDUALS>
 <ENTITIES>
  <ENTITY>
   <DATAID>900002</DATAID><FIRST_NAME>Northern Star Trading LLC</FIRST_NAME>
   <REFERENCE_NUMBER>TST.002</REFERENCE_NUMBER><UN_LIST_TYPE>TestRegime</UN_LIST_TYPE>
   <ENTITY_ALIAS><QUALITY>a.k.a.</QUALITY><ALIAS_NAME>NS Trading</ALIAS_NAME></ENTITY_ALIAS>
   <ENTITY_ADDRESS><COUNTRY>Cyprus</COUNTRY></ENTITY_ADDRESS>
  </ENTITY>
 </ENTITIES>
</CONSOLIDATED_LIST>"#;

    #[test]
    fn parses_individuals_and_entities() {
        let v = parse(XML, "UN-SC").unwrap();
        assert_eq!(v.len(), 2);
        let p = &v[0];
        assert_eq!(p.uid, "TST.001");
        assert_eq!(p.primary_name(), "Ivan Sergeyevich Petrov");
        assert_eq!(p.names.len(), 3); // empty alias skipped
        assert_eq!(p.names[2].strength, AliasStrength::Weak);
        assert_eq!(p.nationalities, vec!["RU"]);
        assert_eq!(p.countries, vec!["CY"]);
        assert!(p.dobs.contains(&PartialDate::ymd(1965, 3, 12)));
        assert!(p.dobs.contains(&PartialDate::year(1964)) && p.dobs.contains(&PartialDate::year(1966)));
        assert_eq!(p.identifiers[0], Identifier::new(IdKind::Passport, "75 1234567", Some("RU")));
        assert_eq!(p.programs, vec!["TestRegime", "UN List"]);
        assert_eq!(v[1].kind, SubjectKind::Entity);
        assert_eq!(v[1].names[1].name, "NS Trading");
    }
}
