//! Country normalisation to ISO 3166-1 alpha-2 (best effort, name based).

use crate::text::fold;
use std::collections::HashMap;
use std::sync::OnceLock;

const TABLE: &[(&str, &str)] = &[
    ("AF", "afghanistan"), ("AL", "albania"), ("DZ", "algeria"), ("AD", "andorra"), ("AO", "angola"),
    ("AG", "antigua and barbuda"), ("AR", "argentina"), ("AM", "armenia"), ("AU", "australia"),
    ("AT", "austria"), ("AZ", "azerbaijan"), ("BS", "bahamas|the bahamas"), ("BH", "bahrain"),
    ("BD", "bangladesh"), ("BB", "barbados"), ("BY", "belarus"), ("BE", "belgium"), ("BZ", "belize"),
    ("BJ", "benin"), ("BT", "bhutan"), ("BO", "bolivia|plurinational state of bolivia"),
    ("BA", "bosnia and herzegovina|bosnia"), ("BW", "botswana"), ("BR", "brazil"),
    ("BN", "brunei|brunei darussalam"), ("BG", "bulgaria"), ("BF", "burkina faso"), ("BI", "burundi"),
    ("CV", "cabo verde|cape verde"), ("KH", "cambodia"), ("CM", "cameroon"), ("CA", "canada"),
    ("CF", "central african republic"), ("TD", "chad"), ("CL", "chile"), ("CN", "china|peoples republic of china|prc"),
    ("CO", "colombia"), ("KM", "comoros"), ("CG", "congo|republic of the congo|congo brazzaville"),
    ("CD", "democratic republic of the congo|congo democratic republic|congo kinshasa|drc|zaire"),
    ("CR", "costa rica"), ("CI", "cote divoire|ivory coast"), ("HR", "croatia"), ("CU", "cuba"),
    ("CY", "cyprus"), ("CZ", "czechia|czech republic"), ("DK", "denmark"), ("DJ", "djibouti"),
    ("DM", "dominica"), ("DO", "dominican republic"), ("EC", "ecuador"), ("EG", "egypt"),
    ("SV", "el salvador"), ("GQ", "equatorial guinea"), ("ER", "eritrea"), ("EE", "estonia"),
    ("SZ", "eswatini|swaziland"), ("ET", "ethiopia"), ("FJ", "fiji"), ("FI", "finland"), ("FR", "france"),
    ("GA", "gabon"), ("GM", "gambia|the gambia"), ("GE", "georgia"), ("DE", "germany"), ("GH", "ghana"),
    ("GR", "greece"), ("GD", "grenada"), ("GT", "guatemala"), ("GN", "guinea"), ("GW", "guinea bissau"),
    ("GY", "guyana"), ("HT", "haiti"), ("HN", "honduras"), ("HK", "hong kong|hong kong sar"), ("HU", "hungary"),
    ("IS", "iceland"), ("IN", "india"), ("ID", "indonesia"),
    ("IR", "iran|islamic republic of iran|iran islamic republic of"), ("IQ", "iraq"), ("IE", "ireland"),
    ("IL", "israel"), ("IT", "italy"), ("JM", "jamaica"), ("JP", "japan"), ("JO", "jordan"),
    ("KZ", "kazakhstan"), ("KE", "kenya"), ("KI", "kiribati"),
    ("KP", "north korea|korea north|democratic peoples republic of korea|dprk"),
    ("KR", "south korea|korea south|republic of korea|korea republic of"), ("XK", "kosovo"),
    ("KW", "kuwait"), ("KG", "kyrgyzstan|kyrgyz republic"),
    ("LA", "laos|lao peoples democratic republic"), ("LV", "latvia"), ("LB", "lebanon"), ("LS", "lesotho"),
    ("LR", "liberia"), ("LY", "libya"), ("LI", "liechtenstein"), ("LT", "lithuania"), ("LU", "luxembourg"),
    ("MO", "macau|macao"), ("MG", "madagascar"), ("MW", "malawi"), ("MY", "malaysia"), ("MV", "maldives"),
    ("ML", "mali"), ("MT", "malta"), ("MH", "marshall islands"), ("MR", "mauritania"), ("MU", "mauritius"),
    ("MX", "mexico"), ("FM", "micronesia"), ("MD", "moldova|republic of moldova"), ("MC", "monaco"),
    ("MN", "mongolia"), ("ME", "montenegro"), ("MA", "morocco"), ("MZ", "mozambique"),
    ("MM", "myanmar|burma"), ("NA", "namibia"), ("NR", "nauru"), ("NP", "nepal"), ("NL", "netherlands|the netherlands|holland"),
    ("NZ", "new zealand"), ("NI", "nicaragua"), ("NE", "niger"), ("NG", "nigeria"),
    ("MK", "north macedonia|macedonia"), ("NO", "norway"), ("OM", "oman"), ("PK", "pakistan"),
    ("PW", "palau"), ("PS", "palestine|state of palestine|occupied palestinian territory|west bank|gaza"),
    ("PA", "panama"), ("PG", "papua new guinea"), ("PY", "paraguay"), ("PE", "peru"), ("PH", "philippines"),
    ("PL", "poland"), ("PT", "portugal"), ("QA", "qatar"), ("RO", "romania"),
    ("RU", "russia|russian federation"), ("RW", "rwanda"), ("KN", "saint kitts and nevis|st kitts and nevis"),
    ("LC", "saint lucia|st lucia"), ("VC", "saint vincent and the grenadines|st vincent and the grenadines"),
    ("WS", "samoa"), ("SM", "san marino"), ("ST", "sao tome and principe"), ("SA", "saudi arabia"),
    ("SN", "senegal"), ("RS", "serbia"), ("SC", "seychelles"), ("SL", "sierra leone"), ("SG", "singapore"),
    ("SK", "slovakia"), ("SI", "slovenia"), ("SB", "solomon islands"), ("SO", "somalia"),
    ("ZA", "south africa"), ("SS", "south sudan"), ("ES", "spain"), ("LK", "sri lanka"), ("SD", "sudan"),
    ("SR", "suriname"), ("SE", "sweden"), ("CH", "switzerland"), ("SY", "syria|syrian arab republic"),
    ("TW", "taiwan|taiwan province of china"), ("TJ", "tajikistan"), ("TZ", "tanzania|united republic of tanzania"),
    ("TH", "thailand"), ("TL", "timor leste|east timor"), ("TG", "togo"), ("TO", "tonga"),
    ("TT", "trinidad and tobago"), ("TN", "tunisia"), ("TR", "turkey|turkiye"), ("TM", "turkmenistan"),
    ("TV", "tuvalu"), ("UG", "uganda"), ("UA", "ukraine"), ("AE", "united arab emirates|uae"),
    ("GB", "united kingdom|uk|great britain|england|scotland|wales|northern ireland"),
    ("US", "united states|united states of america|usa"), ("UY", "uruguay"), ("UZ", "uzbekistan"),
    ("VU", "vanuatu"), ("VA", "holy see|vatican|vatican city"), ("VE", "venezuela|bolivarian republic of venezuela|venezuela bolivarian republic of"),
    ("VN", "vietnam|viet nam"), ("YE", "yemen"), ("ZM", "zambia"), ("ZW", "zimbabwe"),
    ("KY", "cayman islands"), ("VG", "british virgin islands"), ("BM", "bermuda"), ("PR", "puerto rico"),
];

fn map() -> &'static HashMap<String, &'static str> {
    static M: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        for (code, names) in TABLE {
            for n in names.split('|') {
                m.insert(fold(n), *code);
            }
        }
        m
    })
}

/// Normalise a country name or code to ISO alpha-2. Unknown strings come back
/// folded and upper-cased so they can still be compared for equality.
pub fn normalize(s: &str) -> String {
    let t = s.trim();
    if t.len() == 2 && t.chars().all(|c| c.is_ascii_alphabetic()) {
        return t.to_ascii_uppercase();
    }
    let f = fold(t);
    match map().get(&f) {
        Some(c) => (*c).to_string(),
        None => f.to_uppercase(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn names_and_codes() {
        assert_eq!(normalize("Syrian Arab Republic"), "SY");
        assert_eq!(normalize("sy"), "SY");
        assert_eq!(normalize("Côte d'Ivoire"), "CI");
        assert_eq!(normalize("Korea, North"), "KP");
        assert_eq!(normalize("Atlantis"), "ATLANTIS");
    }
}
