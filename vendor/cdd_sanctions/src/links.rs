//! Association links between parties: shared addresses, phones, emails, bank accounts and
//! registered agents.
//!
//! These are deliberately **not** [`crate::model::Identifier`]s. Identifiers are matched against
//! sanctions-list entries and are part of the screening profile hash (changing one re-opens
//! false-positive dispositions). Links are only ever used to build the customer network in
//! [`crate::hypernet`]: a value that two customers quote makes them *associated*, which
//! feeds the advisory exposure score. A link never takes part in screening, and editing a
//! link never invalidates a screening result.
//!
//! Matching is only as good as the normalisation, so each kind is canonicalised before two
//! values are compared ("12 Main St." and "12 main street" are the same address).

use crate::text::{fold, tokens};
use crate::model::SubjectKind;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LinkKind {
    Address,
    Phone,
    Email,
    BankAccount,
    /// Name (or registration number) of the registered agent / corporate service provider.
    RegisteredAgent,
}

impl LinkKind {
    pub const ALL: [LinkKind; 5] = [Self::Address, Self::Phone, Self::Email, Self::BankAccount, Self::RegisteredAgent];

    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "address" => Self::Address,
            "phone" | "telephone" | "mobile" => Self::Phone,
            "email" | "e_mail" => Self::Email,
            "bank_account" | "bank" | "account" | "iban" => Self::BankAccount,
            "registered_agent" | "agent" => Self::RegisteredAgent,
            _ => return None,
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Address => "address",
            Self::Phone => "phone",
            Self::Email => "email",
            Self::BankAccount => "bank_account",
            Self::RegisteredAgent => "registered_agent",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Link {
    pub kind: LinkKind,
    pub value: String,
    /// Country qualifier. Used to separate same-looking values from different places
    /// (an address, a local account number, an agent name); ignored where the value is
    /// globally unique (phone, email, IBAN).
    #[serde(default)]
    pub country: Option<String>,
}

/// A normalised link value, ready to be grouped on.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct LinkKey {
    pub kind: LinkKind,
    pub key: String,
    pub scope: Option<String>,
}

impl Link {
    pub fn new(kind: LinkKind, value: &str, country: Option<&str>) -> Self {
        Self {
            kind,
            value: value.trim().to_string(),
            country: country.filter(|c| !c.trim().is_empty()).map(crate::country::normalize),
        }
    }

    /// Canonical comparison key, or `None` when the value is unusable (empty, a placeholder,
    /// too short to be discriminating). `home_dial_code` (digits, e.g. `"501"`) lets a local
    /// phone number match the same number written in international form.
    pub fn key(&self, home_dial_code: Option<&str>) -> Option<LinkKey> {
        let (key, scoped) = match self.kind {
            LinkKind::Address => (address_key(&self.value)?, true),
            LinkKind::Phone => (phone_key(&self.value, home_dial_code)?, false),
            LinkKind::Email => (email_key(&self.value)?, false),
            LinkKind::BankAccount => {
                let k = account_key(&self.value)?;
                let scoped = !looks_like_iban(&k);
                (k, scoped)
            }
            LinkKind::RegisteredAgent => (agent_key(&self.value)?, true),
        };
        Some(LinkKey { kind: self.kind, key, scope: if scoped { self.country.clone() } else { None } })
    }
}

impl LinkKey {
    /// What to show an analyst to recognise the link without reproducing the whole value:
    /// numbers keep their last 3 characters, emails keep the domain, and addresses and agent
    /// names (business information the analyst has to be able to read) are shown in full,
    /// capped at 40 characters.
    pub fn hint(&self) -> String {
        let tail = |n: usize| -> String { self.key.chars().rev().take(n).collect::<Vec<_>>().into_iter().rev().collect() };
        match self.kind {
            LinkKind::Phone | LinkKind::BankAccount => format!("…{}", tail(3)),
            LinkKind::Email => format!("…@{}", self.key.rsplit('@').next().unwrap_or("")),
            LinkKind::Address | LinkKind::RegisteredAgent => self.key.chars().take(40).collect(),
        }
    }
}

const JUNK: [&str; 12] = [
    "NA", "NONE", "NULL", "UNKNOWN", "NOTAVAILABLE", "NOTAPPLICABLE", "TBD", "PENDING", "SAMEASABOVE", "NOADDRESS", "TEST", "NIL",
];

fn is_junk(upper_alnum: &str) -> bool {
    JUNK.contains(&upper_alnum)
}

fn upper_alnum(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_uppercase()).collect()
}

fn all_same(s: &str) -> bool {
    let first = s.chars().next();
    s.chars().all(|c| Some(c) == first)
}

// ---------------------------------------------------------------------------------------
// Address
// ---------------------------------------------------------------------------------------

fn canonical_address_token(t: &str) -> Option<&str> {
    // Some(..) = replacement, None = drop
    Some(match t {
        "st" | "str" => "street",
        "rd" => "road",
        "ave" | "av" => "avenue",
        "blvd" => "boulevard",
        "hwy" => "highway",
        "dr" => "drive",
        "ln" => "lane",
        "ct" => "court",
        "pl" => "place",
        "sq" => "square",
        "bldg" => "building",
        "fl" => "floor",
        "ste" => "suite",
        "apt" => "apartment",
        "no" | "number" => return None,
        other => other,
    })
}

fn address_key(raw: &str) -> Option<String> {
    let folded = fold(raw);
    let toks: Vec<&str> = folded.split_whitespace().collect();
    let mut out: Vec<String> = Vec::with_capacity(toks.len());
    let mut i = 0;
    while i < toks.len() {
        // "p o box" / "po box" / "post office box" -> "pobox"
        if toks[i] == "p" && toks.get(i + 1) == Some(&"o") && toks.get(i + 2) == Some(&"box") {
            out.push("pobox".into());
            i += 3;
            continue;
        }
        if toks[i] == "po" && toks.get(i + 1) == Some(&"box") {
            out.push("pobox".into());
            i += 2;
            continue;
        }
        if toks[i] == "post" && toks.get(i + 1) == Some(&"office") && toks.get(i + 2) == Some(&"box") {
            out.push("pobox".into());
            i += 3;
            continue;
        }
        if let Some(t) = canonical_address_token(toks[i]) {
            out.push(t.to_string());
        }
        i += 1;
    }
    let joined = out.join(" ").to_uppercase();
    let alnum = upper_alnum(&joined);
    // an address with a single word ("Belize") cannot tell two customers apart
    if out.len() < 2 || alnum.chars().count() < 8 || is_junk(&alnum) {
        return None;
    }
    Some(joined)
}

// ---------------------------------------------------------------------------------------
// Phone
// ---------------------------------------------------------------------------------------

fn phone_key(raw: &str, home_dial_code: Option<&str>) -> Option<String> {
    let trimmed = raw.trim();
    let mut d: String = trimmed.chars().filter(|c| c.is_ascii_digit()).collect();
    // "00" is the international call prefix, equivalent to "+"
    if d.starts_with("00") {
        d = d[2..].to_string();
    }
    if let Some(code) = home_dial_code.map(|c| c.trim().trim_start_matches('+')).filter(|c| !c.is_empty() && c.chars().all(|x| x.is_ascii_digit())) {
        if d.starts_with(code) && d.len() >= code.len() + 6 {
            d = d[code.len()..].to_string();
        }
    }
    if d.len() < 7 || d.len() > 15 || all_same(&d) || d == "1234567" || d == "12345678" {
        return None;
    }
    Some(d)
}

// ---------------------------------------------------------------------------------------
// Email
// ---------------------------------------------------------------------------------------

fn email_key(raw: &str) -> Option<String> {
    let s = raw.trim().to_lowercase();
    let mut parts = s.split('@');
    let (local, domain) = (parts.next()?, parts.next()?);
    if parts.next().is_some() || local.is_empty() || !domain.contains('.') || domain.starts_with('.') || domain.ends_with('.') {
        return None;
    }
    let local = local.split('+').next().unwrap_or(local);
    let (local, domain) = if domain == "gmail.com" || domain == "googlemail.com" {
        (local.replace('.', ""), "gmail.com".to_string())
    } else {
        (local.to_string(), domain.to_string())
    };
    if local.is_empty() || local.chars().any(char::is_whitespace) || domain.chars().any(char::is_whitespace) {
        return None;
    }
    Some(format!("{local}@{domain}"))
}

// ---------------------------------------------------------------------------------------
// Bank account
// ---------------------------------------------------------------------------------------

fn account_key(raw: &str) -> Option<String> {
    let k = upper_alnum(raw);
    let n = k.chars().count();
    if !(6..=34).contains(&n) || all_same(&k) || is_junk(&k) {
        return None;
    }
    Some(k)
}

/// Two letters, two digits, then 11-30 alphanumerics: the IBAN shape (no checksum test).
fn looks_like_iban(k: &str) -> bool {
    let c: Vec<char> = k.chars().collect();
    c.len() >= 15 && c.len() <= 34 && c[0].is_ascii_alphabetic() && c[1].is_ascii_alphabetic() && c[2].is_ascii_digit() && c[3].is_ascii_digit()
}

// ---------------------------------------------------------------------------------------
// Registered agent
// ---------------------------------------------------------------------------------------

fn agent_key(raw: &str) -> Option<String> {
    let t = tokens(raw, SubjectKind::Entity);
    let joined = t.join(" ").to_uppercase();
    let alnum = upper_alnum(&joined);
    if alnum.chars().count() < 4 || is_junk(&alnum) {
        return None;
    }
    Some(joined)
}

// ---------------------------------------------------------------------------------------
// Policy
// ---------------------------------------------------------------------------------------

/// How strongly one kind of link ties parties together, and how many parties may share a value
/// before it is treated as a service provider or bad data rather than an association.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkPolicy {
    /// Hyperedge weight. `0` switches the kind off.
    pub weight: f64,
    /// A value quoted by more parties than this is skipped entirely.
    pub max_group: usize,
}

/// Per-kind policy. **Starting points set by judgement, not calibrated on real cases**:
/// run `cddctl calibrate` on labelled data (it reports what each kind contributes) before relying on them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LinkPolicies {
    pub address: LinkPolicy,
    pub phone: LinkPolicy,
    pub email: LinkPolicy,
    pub bank_account: LinkPolicy,
    pub registered_agent: LinkPolicy,
}

impl Default for LinkPolicies {
    fn default() -> Self {
        Self {
            // a shared settlement account is the strongest tie short of a shared identity document
            bank_account: LinkPolicy { weight: 1.0, max_group: 6 },
            phone: LinkPolicy { weight: 0.6, max_group: 6 },
            email: LinkPolicy { weight: 0.6, max_group: 6 },
            // households, offices, PO boxes: legitimately shared, so weaker and capped lower
            address: LinkPolicy { weight: 0.5, max_group: 5 },
            // professional agents serve many unrelated clients: weakest tie, larger cap
            registered_agent: LinkPolicy { weight: 0.3, max_group: 15 },
        }
    }
}

impl LinkPolicies {
    pub fn get(&self, kind: LinkKind) -> LinkPolicy {
        match kind {
            LinkKind::Address => self.address,
            LinkKind::Phone => self.phone,
            LinkKind::Email => self.email,
            LinkKind::BankAccount => self.bank_account,
            LinkKind::RegisteredAgent => self.registered_agent,
        }
    }
    pub fn set_weight(&mut self, kind: LinkKind, weight: f64) {
        match kind {
            LinkKind::Address => self.address.weight = weight,
            LinkKind::Phone => self.phone.weight = weight,
            LinkKind::Email => self.email.weight = weight,
            LinkKind::BankAccount => self.bank_account.weight = weight,
            LinkKind::RegisteredAgent => self.registered_agent.weight = weight,
        }
    }
}

impl Default for LinkPolicy {
    fn default() -> Self {
        Self { weight: 0.5, max_group: 6 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn k(kind: LinkKind, v: &str, c: Option<&str>) -> Option<LinkKey> {
        Link::new(kind, v, c).key(None)
    }

    #[test]
    fn addresses_match_across_abbreviation_case_and_punctuation() {
        let a = k(LinkKind::Address, "12 Main St., Belize City", Some("BZ")).unwrap();
        let b = k(LinkKind::Address, "No. 12  main STREET,  belize city", Some("Belize")).unwrap();
        assert_eq!(a, b);
        let c = k(LinkKind::Address, "14 Main St., Belize City", Some("BZ")).unwrap();
        assert_ne!(a, c);
        // same text in another country is another place
        let d = k(LinkKind::Address, "12 Main St., Belize City", Some("US")).unwrap();
        assert_ne!(a, d);
    }

    #[test]
    fn po_box_spellings_agree_and_junk_addresses_are_dropped() {
        let a = k(LinkKind::Address, "P.O. Box 123, Belize City", None).unwrap();
        let b = k(LinkKind::Address, "PO Box 123 Belize City", None).unwrap();
        let c = k(LinkKind::Address, "Post Office Box 123, Belize City", None).unwrap();
        assert_eq!(a, b);
        assert_eq!(b, c);
        for junk in ["", "N/A", "Belize", "same as above", "none", "  "] {
            assert!(k(LinkKind::Address, junk, None).is_none(), "{junk:?} must not be a link");
        }
    }

    #[test]
    fn phones_ignore_formatting_and_optionally_the_home_dial_code() {
        let p = |v: &str, code: Option<&str>| Link::new(LinkKind::Phone, v, None).key(code).map(|x| x.key);
        assert_eq!(p("+501 622-1234", None), Some("5016221234".into()));
        assert_eq!(p("00501 622 1234", None), Some("5016221234".into()));
        assert_eq!(p("(501) 622-1234", None), Some("5016221234".into()));
        // with the home code configured, local and international forms agree
        assert_eq!(p("622-1234", Some("501")), Some("6221234".into()));
        assert_eq!(p("+501 622 1234", Some("501")), Some("6221234".into()));
        assert_eq!(p("+501 622 1234", Some("+501")), Some("6221234".into()));
        // too short, repeated digit, placeholder
        assert_eq!(p("123", None), None);
        assert_eq!(p("0000000", None), None);
        assert_eq!(p("n/a", None), None);
    }

    #[test]
    fn emails_are_case_plus_tag_and_gmail_dot_insensitive() {
        let e = |v: &str| k(LinkKind::Email, v, None).map(|x| x.key);
        assert_eq!(e(" John.Smith+kyc@Gmail.com "), Some("johnsmith@gmail.com".into()));
        assert_eq!(e("johnsmith@googlemail.com"), Some("johnsmith@gmail.com".into()));
        // dots matter outside gmail
        assert_ne!(e("john.smith@corp.bz"), e("johnsmith@corp.bz"));
        for bad in ["", "nobody", "a@b", "@corp.bz", "a@@corp.bz", "a b@corp.bz"] {
            assert_eq!(e(bad), None, "{bad:?}");
        }
    }

    #[test]
    fn bank_accounts_ignore_spacing_and_only_iban_shaped_ones_are_global() {
        let iban1 = k(LinkKind::BankAccount, "GB82 WEST 1234 5698 7654 32", Some("GB")).unwrap();
        let iban2 = k(LinkKind::BankAccount, "gb82west12345698765432", Some("US")).unwrap();
        assert_eq!(iban1, iban2, "an IBAN is the same account whatever country is recorded");
        let loc1 = k(LinkKind::BankAccount, "0012345678", Some("BZ")).unwrap();
        let loc2 = k(LinkKind::BankAccount, "0012345678", Some("US")).unwrap();
        assert_ne!(loc1, loc2, "a local account number is only unique within its country");
        assert!(k(LinkKind::BankAccount, "12345", None).is_none());
        assert!(k(LinkKind::BankAccount, "00000000", None).is_none());
    }

    #[test]
    fn registered_agent_names_drop_legal_forms() {
        let a = k(LinkKind::RegisteredAgent, "Caribbean Corporate Services Ltd.", Some("BZ")).unwrap();
        let b = k(LinkKind::RegisteredAgent, "CARIBBEAN CORPORATE SERVICES LIMITED", Some("bz")).unwrap();
        assert_eq!(a, b);
        assert!(k(LinkKind::RegisteredAgent, "Ltd", None).is_none());
    }

    #[test]
    fn hints_do_not_reveal_full_numbers_or_mailboxes() {
        let h = |kind, v: &str| Link::new(kind, v, None).key(None).unwrap().hint();
        assert_eq!(h(LinkKind::Phone, "+501 622-1234"), "…234");
        assert_eq!(h(LinkKind::BankAccount, "0012345678"), "…678");
        assert_eq!(h(LinkKind::Email, "john@corp.bz"), "…@corp.bz");
        assert_eq!(h(LinkKind::Address, "12 Main St, Belize City"), "12 MAIN STREET BELIZE CITY");
    }

    #[test]
    fn kind_names_parse() {
        assert_eq!(LinkKind::parse("Bank-Account"), Some(LinkKind::BankAccount));
        assert_eq!(LinkKind::parse("agent"), Some(LinkKind::RegisteredAgent));
        assert_eq!(LinkKind::parse("fax"), None);
    }
}
