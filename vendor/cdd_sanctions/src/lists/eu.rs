//! EU consolidated financial sanctions list, XML "FSF" layout (attribute-based).
//!
//! NOTE: written against the published FSF schema from documentation; it has been
//! exercised only with a synthetic fixture, not a live download. Validate against
//! a real file before relying on it, or convert the EU file to canonical JSONL.

use super::xmltree::parse as parse_xml;
use crate::error::Result;
use crate::model::*;

pub fn parse(xml: &str, list_name: &str) -> Result<Vec<ListEntry>> {
    let root = parse_xml(xml)?;
    let mut out = Vec::new();
    for n in root.all("sanctionEntity") {
        let uid = n
            .attr("euReferenceNumber")
            .filter(|s| !s.is_empty())
            .or_else(|| n.attr("logicalId"))
            .map(str::to_string);
        let Some(uid) = uid else { continue };
        let kind = match n.child("subjectType").and_then(|s| s.attr("classificationCode")) {
            Some("P") => SubjectKind::Individual,
            Some("E") => SubjectKind::Entity,
            _ => SubjectKind::Unknown,
        };
        let mut names: Vec<NameVariant> = Vec::new();
        for a in n.all("nameAlias") {
            let whole = a.attr("wholeName").filter(|s| !s.trim().is_empty()).map(str::to_string).or_else(|| {
                let parts: Vec<&str> = ["firstName", "middleName", "lastName"]
                    .iter()
                    .filter_map(|k| a.attr(k))
                    .filter(|s| !s.trim().is_empty())
                    .collect();
                (!parts.is_empty()).then(|| parts.join(" "))
            });
            if let Some(name) = whole {
                let strong = a.attr("strong").map(|s| s.eq_ignore_ascii_case("true")).unwrap_or(true);
                let strength = if names.is_empty() {
                    AliasStrength::Primary
                } else if strong {
                    AliasStrength::Strong
                } else {
                    AliasStrength::Weak
                };
                names.push(NameVariant { name, strength });
            }
        }
        if names.is_empty() {
            continue;
        }
        let dobs = n
            .all("birthdate")
            .filter_map(|b| {
                b.attr("birthdate")
                    .and_then(PartialDate::parse)
                    .or_else(|| b.attr("year").and_then(|y| y.parse().ok()).map(PartialDate::year))
            })
            .collect();
        let nationalities = n
            .all("citizenship")
            .filter_map(|c| c.attr("countryIso2Code").or_else(|| c.attr("countryDescription")))
            .map(str::to_string)
            .collect();
        let countries = n
            .all("address")
            .filter_map(|a| a.attr("countryIso2Code").or_else(|| a.attr("countryDescription")))
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        let identifiers = n
            .all("identification")
            .filter_map(|i| {
                let number = i.attr("number")?;
                let t = i.attr("identificationTypeCode").unwrap_or("").to_ascii_lowercase();
                let kind = match t.as_str() {
                    "passport" => IdKind::Passport,
                    "id" | "nationalid" => IdKind::NationalId,
                    "tax" | "taxid" => IdKind::TaxId,
                    "reg" | "registration" => IdKind::Registration,
                    _ => IdKind::Other,
                };
                Some(Identifier::new(kind, number, i.attr("countryIso2Code")))
            })
            .collect();
        let programs: Vec<String> =
            n.all("regulation").filter_map(|r| r.attr("programme")).map(str::to_string).collect();
        out.push(
            ListEntry {
                list: list_name.into(),
                uid,
                kind,
                names,
                dobs,
                nationalities,
                countries,
                identifiers,
                programs,
                remarks: n.child("remark").map(|r| r.text.trim().to_string()).filter(|s| !s.is_empty()),
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
    fn synthetic_fixture() {
        let xml = r#"<export xmlns="http://eu.europa.ec/fsd/export">
 <sanctionEntity logicalId="1" euReferenceNumber="EU.9.1">
  <regulation programme="TESTREG"/>
  <subjectType code="person" classificationCode="P"/>
  <nameAlias wholeName="Ivan Sergeyevich Petrov" strong="true"/>
  <nameAlias firstName="Vanya" lastName="Petrof" strong="false"/>
  <birthdate birthdate="1965-03-12" year="1965"/>
  <citizenship countryIso2Code="RU"/>
  <identification number="75 1234567" identificationTypeCode="passport" countryIso2Code="RU"/>
 </sanctionEntity>
 <sanctionEntity logicalId="2"><subjectType classificationCode="E"/><nameAlias wholeName="Northern Star Trading LLC"/></sanctionEntity>
</export>"#;
        let v = parse(xml, "EU-FSF").unwrap();
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].uid, "EU.9.1");
        assert_eq!(v[0].kind, SubjectKind::Individual);
        assert_eq!(v[0].names[1].name, "Vanya Petrof");
        assert_eq!(v[0].names[1].strength, AliasStrength::Weak);
        assert_eq!(v[0].dobs, vec![PartialDate::ymd(1965, 3, 12)]);
        assert_eq!(v[0].identifiers[0].kind, IdKind::Passport);
        assert_eq!(v[1].uid, "2");
        assert_eq!(v[1].kind, SubjectKind::Entity);
    }
}
