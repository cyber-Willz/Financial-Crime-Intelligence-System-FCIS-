//! The evidence packet and the numbered fact list built from it.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuleFact {
    pub code: String,
    pub weight: f64,
    #[serde(default)]
    pub detail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AlertFacts {
    pub id: i64,
    pub score: u8,
    pub severity: String,
    pub status: String,
    pub rules: Vec<RuleFact>,
    pub txn_count: i32,
    #[serde(default)]
    pub created_at: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AccountFacts {
    pub account: i64,
    pub country: Option<String>,
    pub risk_level: String,
    pub risk_score: i16,
    pub alert_count: i32,
    pub watchlisted: bool,
    pub first_seen: String,
    pub last_seen: String,
    pub txn_sent: i64,
    pub txn_received: i64,
    pub total_sent_gbp: f64,
    pub total_received_gbp: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CustomerFacts {
    pub reference: String,
    pub customer_type: String,
    pub country: Option<String>,
    pub sector: Option<String>,
    pub risk_rating: String,
    pub pep: bool,
    pub annual_income_gbp: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TxnFact {
    pub id: i64,
    pub at: String,
    pub sender: i64,
    pub receiver: i64,
    pub amount_gbp: f64,
    pub payment_type: String,
    pub sender_country: String,
    pub receiver_country: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphFacts {
    /// graph-model risk in `[0, 1]` for the subject account, when a graph model has scored it
    pub risk: Option<f32>,
    /// date of the scored window
    pub window: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UboFact {
    pub name: String,
    pub effective_pct: f64,
    pub controls: bool,
    pub pep: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OwnershipFacts {
    pub holder: String,
    pub ubos: Vec<UboFact>,
    pub opacity: f64,
    pub opacity_reasons: Vec<String>,
    pub layers: usize,
    pub blocked_ownership_pct: f64,
    pub exposure: f64,
    pub group: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SanctionsFacts {
    pub watchlist_hit: bool,
    pub watchlist_name: Option<String>,
    pub sanctioned_jurisdiction_countries: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NoteFact {
    pub author: String,
    pub at: String,
    pub body: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HistoryFacts {
    pub prior_alerts: u32,
    pub prior_sar: u32,
    pub prior_confirmed: u32,
    pub prior_false_positive: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FeedbackFacts {
    /// share of comparable reviewed alerts that were confirmed (isotonic calibration of the engine score)
    pub calibrated_probability: Option<f64>,
    pub priority_factor: f64,
    pub factor_reason: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Evidence {
    pub subject: i64,
    pub alert: Option<AlertFacts>,
    pub account: Option<AccountFacts>,
    pub customer: Option<CustomerFacts>,
    pub transactions: Vec<TxnFact>,
    pub graph: GraphFacts,
    pub ownership: Option<OwnershipFacts>,
    pub sanctions: SanctionsFacts,
    pub notes: Vec<NoteFact>,
    pub history: HistoryFacts,
    pub feedback: Option<FeedbackFacts>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub id: String,
    pub text: String,
}

/// Numbered facts. Ids are `F1`, `F2`, … and stay stable while facts are appended.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FactBook {
    pub facts: Vec<Fact>,
}

impl FactBook {
    pub fn add(&mut self, text: impl Into<String>) -> String {
        let id = format!("F{}", self.facts.len() + 1);
        self.facts.push(Fact { id: id.clone(), text: text.into() });
        id
    }
    pub fn get(&self, id: &str) -> Option<&Fact> {
        self.facts.iter().find(|f| f.id == id)
    }
    pub fn corpus(&self) -> String {
        self.facts.iter().map(|f| f.text.as_str()).collect::<Vec<_>>().join("\n")
    }
    pub fn len(&self) -> usize {
        self.facts.len()
    }
    pub fn is_empty(&self) -> bool {
        self.facts.is_empty()
    }
}

/// Which fact ids each part of an evidence packet produced (for section building).
#[derive(Clone, Debug, Default)]
pub struct FactIndex {
    pub alert: Vec<String>,
    pub rules: Vec<(String, String)>,
    pub account: Vec<String>,
    pub customer: Vec<String>,
    pub transactions: Vec<String>,
    pub graph: Vec<String>,
    pub ownership: Vec<String>,
    pub sanctions: Vec<String>,
    pub history: Vec<String>,
    pub notes: Vec<String>,
    pub feedback: Vec<String>,
}

/// Amount with thousands separators and two decimals: `12,345.60`.
pub fn money(x: f64) -> String {
    let neg = x < 0.0;
    let cents = (x.abs() * 100.0).round() as u128;
    let (whole, frac) = (cents / 100, cents % 100);
    let s = whole.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    format!("{}{}.{:02}", if neg { "-" } else { "" }, out, frac)
}

fn trunc(s: &str, n: usize) -> String {
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if s.chars().count() <= n {
        s
    } else {
        format!("{}…", s.chars().take(n).collect::<String>())
    }
}

impl Evidence {
    /// Render every part of the packet as numbered facts appended to `book`.
    pub fn facts_into(&self, book: &mut FactBook) -> FactIndex {
        let mut ix = FactIndex::default();
        if let Some(a) = &self.alert {
            ix.alert.push(book.add(format!(
                "Alert {} on account {} scored {}/100 ({} severity, status {}) and covers {} flagged transaction(s).",
                a.id, self.subject, a.score, a.severity, a.status, a.txn_count
            )));
            for r in &a.rules {
                let id = book.add(format!("Rule {} fired with weight {:.2}{}", r.code, r.weight, if r.detail.is_empty() { ".".into() } else { format!(": {}", trunc(&r.detail, 240)) }));
                ix.rules.push((r.code.clone(), id));
            }
        }
        if let Some(a) = &self.account {
            ix.account.push(book.add(format!(
                "Account {}{} was first seen {} and last seen {}; it sent {} payment(s) totalling {} GBP and received {} totalling {} GBP; risk level {}, {} earlier alert(s).",
                a.account,
                a.country.as_ref().map(|c| format!(" ({c})")).unwrap_or_default(),
                a.first_seen, a.last_seen, a.txn_sent, money(a.total_sent_gbp), a.txn_received, money(a.total_received_gbp), a.risk_level, a.alert_count
            )));
        }
        if let Some(c) = &self.customer {
            ix.customer.push(book.add(format!(
                "Customer {} is a {}{}, risk rating {}, politically exposed person: {}{}.",
                c.reference,
                c.customer_type,
                c.sector.as_ref().map(|s| format!(" in sector {s}")).unwrap_or_default(),
                c.risk_rating,
                if c.pep { "yes" } else { "no" },
                c.annual_income_gbp.map(|i| format!(", declared annual income {} GBP", money(i))).unwrap_or_default()
            )));
        }
        for t in &self.transactions {
            ix.transactions.push(book.add(format!(
                "Transaction {} on {}: {} GBP by {} from account {} ({}) to account {} ({}).",
                t.id, t.at, money(t.amount_gbp), t.payment_type, t.sender, t.sender_country, t.receiver, t.receiver_country
            )));
        }
        if let Some(r) = self.graph.risk {
            ix.graph.push(book.add(format!(
                "The graph model rates account {} at risk {:.2} on a 0 to 1 scale{}.",
                self.subject, r, self.graph.window.as_ref().map(|w| format!(" (window {w})")).unwrap_or_default()
            )));
        }
        if let Some(o) = &self.ownership {
            let owners = if o.ubos.is_empty() {
                "no beneficial owner could be identified".to_string()
            } else {
                o.ubos
                    .iter()
                    .take(5)
                    .map(|u| format!("{} ({:.1}% effective{}{})", u.name, u.effective_pct, if u.controls { ", controls" } else { "" }, if u.pep { ", PEP" } else { "" }))
                    .collect::<Vec<_>>()
                    .join("; ")
            };
            ix.ownership.push(book.add(format!("Account holder {}: {} layer(s) of ownership; {}.", o.holder, o.layers, owners)));
            if o.opacity > 0.0 {
                ix.ownership.push(book.add(format!(
                    "Ownership opacity of {} is {:.2} on a 0 to 1 scale{}.",
                    o.holder, o.opacity, if o.opacity_reasons.is_empty() { String::new() } else { format!(" ({})", o.opacity_reasons.join("; ")) }
                )));
            }
            if o.blocked_ownership_pct > 0.0 {
                ix.ownership.push(book.add(format!("{:.1}% of {} is owned, directly or through chains, by parties blocked in sanctions screening.", o.blocked_ownership_pct, o.holder)));
            }
            if o.exposure > 0.0 {
                ix.ownership.push(book.add(format!("Sanctions network exposure of {} is {:.2}.", o.holder, o.exposure)));
            }
        }
        if self.sanctions.watchlist_hit {
            ix.sanctions.push(book.add(format!("Account {} is on watchlist '{}'.", self.subject, self.sanctions.watchlist_name.clone().unwrap_or_else(|| "internal".into()))));
        }
        if !self.sanctions.sanctioned_jurisdiction_countries.is_empty() {
            ix.sanctions.push(book.add(format!("Counterparties in sanctioned jurisdiction(s): {}.", self.sanctions.sanctioned_jurisdiction_countries.join(", "))));
        }
        let h = &self.history;
        if h.prior_alerts > 0 {
            ix.history.push(book.add(format!(
                "Account {} has {} earlier reviewed alert(s): {} led to a SAR, {} were confirmed, {} were closed as false positives.",
                self.subject, h.prior_alerts, h.prior_sar, h.prior_confirmed, h.prior_false_positive
            )));
        }
        for n in &self.notes {
            ix.notes.push(book.add(format!("Analyst note by {} on {}: {}", n.author, n.at, trunc(&n.body, 300))));
        }
        if let Some(f) = &self.feedback {
            if let Some(p) = f.calibrated_probability {
                ix.feedback.push(book.add(format!("Among comparable reviewed alerts, {:.0}% were confirmed (calibrated from analyst outcomes).", p * 100.0)));
            }
            if (f.priority_factor - 1.0).abs() > 1e-9 {
                ix.feedback.push(book.add(format!("Review priority factor {:.2}: {}.", f.priority_factor, f.factor_reason)));
            }
        }
        ix
    }

    pub fn fact_book(&self) -> (FactBook, FactIndex) {
        let mut b = FactBook::default();
        let ix = self.facts_into(&mut b);
        (b, ix)
    }

    /// FNV-1a over the canonical JSON: lets an audit record show which evidence an output was based on.
    pub fn digest(&self) -> String {
        let s = serde_json::to_string(self).unwrap_or_default();
        let mut h: u64 = 0xcbf29ce484222325;
        for b in s.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    }

    /// Rule codes of the alert (empty without an alert).
    pub fn rule_codes(&self) -> Vec<&str> {
        self.alert.as_ref().map(|a| a.rules.iter().map(|r| r.code.as_str()).collect()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_format() {
        assert_eq!(money(0.0), "0.00");
        assert_eq!(money(1234567.891), "1,234,567.89");
        assert_eq!(money(999.999), "1,000.00");
        assert_eq!(money(12.5), "12.50");
    }

    #[test]
    fn digest_is_stable_and_sensitive() {
        let mut e = Evidence { subject: 5, ..Default::default() };
        let a = e.digest();
        assert_eq!(a, e.digest());
        e.subject = 6;
        assert_ne!(a, e.digest());
    }
}
