//! Consolidated Belize Sanctions List, published by the Financial Intelligence Unit of Belize
//! (https://fiubelize.org/consolidated-belize-sanctions-list/).
//!
//! The XML uses the UN consolidated-list schema (INDIVIDUALS/INDIVIDUAL, ENTITIES/ENTITY, …) with
//! Belize reference numbers (`BZCDi.001` is Belize's number for the UN DRC entry `CDi.001`), so
//! [`super::un::parse`] does the structural work. This module adds handling for a defect seen in the
//! July 2026 file: a stray `BZ` is inserted before some letter groups (`TAI`, `SSI`, `IRE`, `IRI`),
//! corrupting words — including names and aliases (e.g. `NSANZUBUKBZIRe` for `NSANZUBUKIRE`).
//! Names are kept exactly as published (provenance) and a repaired spelling is added as an extra
//! variant, so a customer is matched whichever spelling is used. The FIU should be told about the defect.

use super::un;
use crate::error::Result;
use crate::model::*;

#[derive(Clone, Debug, Default)]
pub struct ParseReport {
    pub entries: usize,
    /// Entries that had at least one name repaired.
    pub repaired_entries: usize,
}

/// Removes the stray `BZ` marker (only when it precedes `TAI`/`SSI`/`IRE`/`IRI`, any case).
pub fn repair(s: &str) -> String {
    let b: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == 'B' && b.get(i + 1) == Some(&'Z') {
            let tail: String = b.iter().skip(i + 2).take(3).collect::<String>().to_ascii_uppercase();
            if matches!(tail.as_str(), "TAI" | "SSI" | "IRE" | "IRI") {
                i += 2;
                continue;
            }
        }
        out.push(b[i]);
        i += 1;
    }
    out
}

pub fn parse(xml: &str, list_name: &str) -> Result<(Vec<ListEntry>, ParseReport)> {
    let mut entries = un::parse(xml, list_name)?;
    let mut report = ParseReport { entries: entries.len(), repaired_entries: 0 };
    for e in &mut entries {
        let extra: Vec<NameVariant> = e
            .names
            .iter()
            .filter_map(|n| {
                let r = repair(&n.name);
                (r != n.name && !e.names.iter().any(|x| x.name == r)).then(|| NameVariant {
                    name: r,
                    strength: if n.strength == AliasStrength::Primary { AliasStrength::Strong } else { n.strength },
                })
            })
            .collect();
        if !extra.is_empty() {
            report.repaired_entries += 1;
            e.names.extend(extra);
        }
    }
    Ok((entries.into_iter().map(|e| e.finalize()).collect(), report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screening::{HitLevel, SanctionsIndex, ScreeningConfig, Subject};

    // Structure and quirks follow the July 2026 Belize file (trimmed). The <ENTITIES> block is
    // assumed to use the UN layout; it was not visible in the portion of the file inspected.
    const XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<CONSOLIDATED_LIST>
  <INDIVIDUALS>
    <INDIVIDUAL>
      <DABZTAiD>6908014</DABZTAiD>
      <VERSIONNUM>1</VERSIONNUM>
      <FIRST_NAME>FELICIEN</FIRST_NAME>
      <SECOND_NAME>NSANZUBUKBZIRe</SECOND_NAME>
      <UN_LIST_TYPE>DRC</UN_LIST_TYPE>
      <REFERENCE_NUMBER>BZCDi.023</REFERENCE_NUMBER>
      <LISTED_ON>2010-12-01</LISTED_ON>
      <COMMENTS1>Remarks with the same defect: deBZTAined and mainBZTAined.</COMMENTS1>
      <DESIGNATION><VALUE>FDLR-FOCA Subsector Commander </VALUE><VALUE>FDLR-FOCA Colonel </VALUE></DESIGNATION>
      <NATIONALITY><VALUE>Rwanda</VALUE></NATIONALITY>
      <LIST_TYPE><VALUE>UN List</VALUE></LIST_TYPE>
      <INDIVIDUAL_ALIAS><QUALITY>Good</QUALITY><ALIAS_NAME>Fred Irakeza</ALIAS_NAME></INDIVIDUAL_ALIAS>
      <INDIVIDUAL_ALIAS><QUALITY/><ALIAS_NAME/></INDIVIDUAL_ALIAS>
      <INDIVIDUAL_ADDRESS><COUNTRY>Democratic Republic of the Congo</COUNTRY></INDIVIDUAL_ADDRESS>
      <INDIVIDUAL_DATE_OF_BIRTH><TYPE_OF_DATE>EXACT</TYPE_OF_DATE><YEAR>1967</YEAR></INDIVIDUAL_DATE_OF_BIRTH>
      <INDIVIDUAL_DOCUMENT/>
    </INDIVIDUAL>
    <INDIVIDUAL>
      <DABZTAiD>6907996</DABZTAiD>
      <FIRST_NAME>INNOCENT</FIRST_NAME><SECOND_NAME>KAINA</SECOND_NAME>
      <UN_LIST_TYPE>DRC</UN_LIST_TYPE><REFERENCE_NUMBER>BZCDi.004</REFERENCE_NUMBER>
      <INDIVIDUAL_ALIAS><QUALITY>Good</QUALITY><ALIAS_NAME>Colonel Innocent KAINA</ALIAS_NAME></INDIVIDUAL_ALIAS>
      <INDIVIDUAL_DATE_OF_BIRTH><TYPE_OF_DATE>EXACT</TYPE_OF_DATE><NOTE>Nov. 1973 </NOTE></INDIVIDUAL_DATE_OF_BIRTH>
    </INDIVIDUAL>
    <INDIVIDUAL>
      <DABZTAiD>6908015</DABZTAiD>
      <FIRST_NAME>PACIFIQUE</FIRST_NAME><SECOND_NAME>NTAWUNGUKA</SECOND_NAME>
      <UN_LIST_TYPE>DRC</UN_LIST_TYPE><REFERENCE_NUMBER>BZCDi.024</REFERENCE_NUMBER>
      <INDIVIDUAL_DATE_OF_BIRTH><TYPE_OF_DATE>EXACT</TYPE_OF_DATE><DATE>1964-01-01</DATE></INDIVIDUAL_DATE_OF_BIRTH>
      <INDIVIDUAL_DATE_OF_BIRTH><TYPE_OF_DATE>APPROXIMATELY</TYPE_OF_DATE><YEAR>1964</YEAR></INDIVIDUAL_DATE_OF_BIRTH>
      <INDIVIDUAL_DOCUMENT><TYPE_OF_DOCUMENT>Passport</TYPE_OF_DOCUMENT><NUMBER>OP 1097934 </NUMBER><ISSUING_COUNTRY>Democratic Republic of the Congo</ISSUING_COUNTRY><DATE_OF_EXPIRY>2026-12-11</DATE_OF_EXPIRY></INDIVIDUAL_DOCUMENT>
    </INDIVIDUAL>
  </INDIVIDUALS>
  <ENTITIES>
    <ENTITY>
      <DABZTAiD>6909001</DABZTAiD><FIRST_NAME>Allied Democratic Forces</FIRST_NAME>
      <UN_LIST_TYPE>DRC</UN_LIST_TYPE><REFERENCE_NUMBER>BZCDe.001</REFERENCE_NUMBER>
      <ENTITY_ALIAS><QUALITY>Good</QUALITY><ALIAS_NAME>ADF</ALIAS_NAME></ENTITY_ALIAS>
    </ENTITY>
  </ENTITIES>
</CONSOLIDATED_LIST>"#;

    #[test]
    fn repair_rules() {
        assert_eq!(repair("NSANZUBUKBZIRe"), "NSANZUBUKIRe");
        assert_eq!(repair("ByBZIRingiro Victor"), "ByIRingiro Victor");
        assert_eq!(repair("deBZTAined"), "deTAined");
        assert_eq!(repair("commiBZSSion"), "commiSSion");
        assert_eq!(repair("BZCDi.001"), "BZCDi.001"); // genuine Belize reference prefix untouched
        assert_eq!(repair("Bzira Abzug"), "Bzira Abzug"); // only the uppercase marker, only before the 4 triples
        assert_eq!(repair("Plain Name"), "Plain Name");
    }

    #[test]
    fn parses_belize_quirks() {
        let (v, rep) = parse(XML, "BZ-FIU").unwrap();
        assert_eq!((v.len(), rep.repaired_entries), (4, 1));
        let f = v.iter().find(|e| e.uid == "BZCDi.023").unwrap();
        assert_eq!(f.list, "BZ-FIU");
        assert_eq!(f.primary_name(), "FELICIEN NSANZUBUKBZIRe"); // as published
        assert!(f.names.iter().any(|n| n.name == "FELICIEN NSANZUBUKIRe" && n.strength == AliasStrength::Strong));
        assert_eq!(f.names.len(), 3); // primary, Fred Irakeza, repaired; the empty alias is skipped
        assert_eq!(f.nationalities, vec!["RW"]);
        assert_eq!(f.countries, vec!["CD"]);
        assert_eq!(f.dobs, vec![PartialDate::year(1967)]);

        let k = v.iter().find(|e| e.uid == "BZCDi.004").unwrap();
        assert_eq!(k.dobs, vec![PartialDate { year: Some(1973), month: Some(11), day: None }]); // from free-text note

        let n = v.iter().find(|e| e.uid == "BZCDi.024").unwrap();
        for y in 1962..=1966 {
            assert!(n.dobs.contains(&PartialDate::year(y)), "{y}"); // approximate year widened
        }
        assert_eq!(n.identifiers[0].key(), "OP1097934");
        assert_eq!(v.iter().find(|e| e.uid == "BZCDe.001").unwrap().kind, SubjectKind::Entity);
    }

    #[test]
    fn customer_is_found_with_correct_or_published_spelling() {
        let (v, _) = parse(XML, "BZ-FIU").unwrap();
        let idx = SanctionsIndex::build(v);
        let cfg = ScreeningConfig::default();
        let mut correct = Subject::new(SubjectKind::Individual, "Felicien Nsanzubukire");
        correct.dob = vec![PartialDate::ymd(1967, 6, 1)];
        correct.nationalities = vec!["Rwanda".into()];
        let r = idx.screen(&correct, &cfg);
        assert_eq!((r.hits[0].uid.as_str(), r.hits[0].level), ("BZCDi.023", HitLevel::High));

        let as_published = Subject::new(SubjectKind::Individual, "Felicien Nsanzubukbzire");
        assert_eq!(idx.screen(&as_published, &cfg).hits[0].uid, "BZCDi.023");

        // approximate DOB (1964 +/- 2) must not read as a conflict for a 1965 birth year
        let mut s = Subject::new(SubjectKind::Individual, "Pacifique Ntawunguka");
        s.dob = vec![PartialDate::year(1965)];
        assert!(!idx.screen(&s, &cfg).hits.is_empty());
    }
}
