//! Typology identification from rule activations and the other evidence.
//!
//! Each typology is supported by a set of rules. Confidence is the noisy-OR of the weights the engine gave
//! the supporting rules in this alert — the same aggregation the engine itself uses — plus pseudo-rules for
//! graph risk and ownership opacity. It says how strongly the *evidence* points at a typology, not how likely
//! laundering is; probabilities come from `aml_feedback::Calibrator`.

use crate::evidence::Evidence;
use serde::{Deserialize, Serialize};

pub struct TypologyDef {
    pub code: &'static str,
    pub name: &'static str,
    pub summary: &'static str,
    pub rules: &'static [&'static str],
    pub next_steps: &'static [&'static str],
}

pub const TYPOLOGIES: &[TypologyDef] = &[
    TypologyDef {
        code: "structuring",
        name: "structuring / smurfing",
        summary: "many payments kept below reporting thresholds or fed in small pieces from many senders",
        rules: &["STRUCTURING_HUB", "STRUCTURING_BAND", "REPEATED_PAIR", "CASH_VELOCITY", "owner_structuring"],
        next_steps: &["List every deposit in the month before the first alert and look for amounts clustered just under the reporting threshold.", "Check whether the senders are related to each other or to the account holder."],
    },
    TypologyDef {
        code: "funnel",
        name: "funnel / mule account",
        summary: "funds gathered from many sources into one account and moved on quickly",
        rules: &["FAN_IN", "PASS_THROUGH"],
        next_steps: &["Establish whether the inbound senders have a plausible commercial or family reason to pay this account.", "Trace where the collected funds went next and how soon."],
    },
    TypologyDef {
        code: "layering",
        name: "layering",
        summary: "funds forwarded onward in chains or spread to many recipients to obscure their origin",
        rules: &["LAYERING_CHAIN", "PASS_THROUGH", "FAN_OUT", "ownership_pass_through"],
        next_steps: &["Follow the chain forward to its end and identify the final recipients.", "Compare timing and amounts hop by hop for near-identical pass-through."],
    },
    TypologyDef {
        code: "round_tripping",
        name: "round-tripping",
        summary: "funds that leave an account and come back to it, or circulate inside a group, with no apparent purpose",
        rules: &["CIRCULAR_FLOW", "group_circulation"],
        next_steps: &["Confirm each hop of the loop and who controls the accounts on it.", "Ask for the commercial justification for the circular flow."],
    },
    TypologyDef {
        code: "dormant_misuse",
        name: "dormant or changed-behaviour account",
        summary: "an account that is quiet for a long time or abruptly departs from its own pattern",
        rules: &["DORMANT_REACTIVATION", "BEHAVIOUR_CHANGE"],
        next_steps: &["Check for a change of control, new signatories or a recent contact-detail change.", "Compare the new activity with the account's stated purpose."],
    },
    TypologyDef {
        code: "trade_based",
        name: "trade-based laundering",
        summary: "large payments from an import/export business that may not match real trade",
        rules: &["TRADE_MISPRICING", "LARGE_TXN"],
        next_steps: &["Request invoices, bills of lading and customs documents for the payments.", "Compare invoiced prices with market prices for the goods."],
    },
    TypologyDef {
        code: "sanctions",
        name: "sanctions exposure",
        summary: "a watchlisted party, a sanctioned jurisdiction or a blocked owner is involved",
        rules: &["WATCHLIST_HIT", "SANCTIONED_JURISDICTION", "material_sanctioned_stake", "network_exposure"],
        next_steps: &["Escalate to the sanctions officer before any further processing.", "Confirm the screening match and the ownership chain that creates the exposure."],
    },
    TypologyDef {
        code: "high_risk_geography",
        name: "high-risk geography",
        summary: "payments to or from jurisdictions rated as high risk",
        rules: &["HIGH_RISK_JURISDICTION", "CUSTOMER_JURISDICTION", "CURRENCY_CONVERSION"],
        next_steps: &["Document the business reason for each high-risk corridor."],
    },
    TypologyDef {
        code: "source_of_funds",
        name: "source-of-funds mismatch",
        summary: "activity out of proportion to the customer's declared means or risk rating",
        rules: &["INCOME_MISMATCH", "HIGH_RISK_CUSTOMER"],
        next_steps: &["Refresh source-of-funds and source-of-wealth information and compare it with the activity."],
    },
    TypologyDef {
        code: "opaque_ownership",
        name: "opaque or fragmented ownership",
        summary: "the beneficial owner is hard to identify through layers, nominees or stakes split below the threshold",
        rules: &["opaque_owner", "fragmented_ownership"],
        next_steps: &["Obtain the full ownership chain and the identity of each natural person behind it.", "Ask why the structure needs this many layers."],
    },
    TypologyDef {
        code: "network_anomaly",
        name: "network anomaly",
        summary: "the transaction-graph model sees a multi-day pattern around the account that individual rules do not",
        rules: &["graph_model_risk"],
        next_steps: &["Inspect the account's neighbourhood: who pays it, who it pays and how those accounts relate."],
    },
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TypologyHit {
    pub code: String,
    pub name: String,
    pub summary: String,
    pub confidence: f64,
    pub supporting_rules: Vec<String>,
}

fn noisy_or(ws: impl Iterator<Item = f64>) -> f64 {
    1.0 - ws.fold(1.0, |a, w| a * (1.0 - w.clamp(0.0, 0.999)))
}

pub fn def(code: &str) -> Option<&'static TypologyDef> {
    TYPOLOGIES.iter().find(|t| t.code == code)
}

/// Typologies the evidence supports, strongest first.
pub fn identify(ev: &Evidence) -> Vec<TypologyHit> {
    let mut weights: Vec<(String, f64)> = ev.alert.as_ref().map(|a| a.rules.iter().map(|r| (r.code.clone(), r.weight)).collect()).unwrap_or_default();
    if let Some(r) = ev.graph.risk {
        if r >= 0.4 {
            weights.push(("graph_model_risk".into(), r as f64));
        }
    }
    if let Some(o) = &ev.ownership {
        if o.opacity >= 0.6 {
            weights.push(("opaque_owner".into(), (o.opacity - 0.2).min(0.8)));
        }
        if o.blocked_ownership_pct >= 25.0 {
            weights.push(("material_sanctioned_stake".into(), 0.55));
        }
    }
    if ev.sanctions.watchlist_hit {
        weights.push(("WATCHLIST_HIT".into(), 0.9));
    }
    let mut hits: Vec<TypologyHit> = TYPOLOGIES
        .iter()
        .filter_map(|t| {
            let mut matched: Vec<&(String, f64)> = weights.iter().filter(|(c, _)| t.rules.contains(&c.as_str())).collect();
            matched.sort_by(|a, b| a.0.cmp(&b.0));
            matched.dedup_by(|a, b| a.0 == b.0);
            if matched.is_empty() {
                return None;
            }
            Some(TypologyHit {
                code: t.code.into(),
                name: t.name.into(),
                summary: t.summary.into(),
                confidence: noisy_or(matched.iter().map(|m| m.1)),
                supporting_rules: matched.iter().map(|m| m.0.clone()).collect(),
            })
        })
        .collect();
    hits.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal).then(a.code.cmp(&b.code)));
    hits
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{AlertFacts, GraphFacts, OwnershipFacts, RuleFact};

    fn ev(rules: &[(&str, f64)]) -> Evidence {
        Evidence {
            subject: 1,
            alert: Some(AlertFacts { id: 1, score: 80, severity: "high".into(), status: "open".into(), txn_count: 1, rules: rules.iter().map(|(c, w)| RuleFact { code: c.to_string(), weight: *w, detail: String::new() }).collect(), created_at: String::new() }),
            ..Default::default()
        }
    }

    #[test]
    fn fan_in_and_hub_read_as_structuring_and_funnel() {
        let h = identify(&ev(&[("STRUCTURING_HUB", 0.7), ("FAN_IN", 0.4), ("HIGH_RISK_JURISDICTION", 0.2)]));
        let codes: Vec<&str> = h.iter().map(|t| t.code.as_str()).collect();
        assert_eq!(codes[0], "structuring");
        assert!(codes.contains(&"funnel") && codes.contains(&"high_risk_geography"));
        assert!(h.windows(2).all(|w| w[0].confidence >= w[1].confidence));
        assert!((h[0].confidence - 0.7).abs() < 1e-9);
    }

    #[test]
    fn noisy_or_combines_supporting_rules() {
        let h = identify(&ev(&[("PASS_THROUGH", 0.5), ("LAYERING_CHAIN", 0.5)]));
        let l = h.iter().find(|t| t.code == "layering").unwrap();
        assert!((l.confidence - 0.75).abs() < 1e-9);
        assert_eq!(l.supporting_rules.len(), 2);
    }

    #[test]
    fn graph_and_ownership_add_typologies_without_rules() {
        let mut e = ev(&[]);
        e.graph = GraphFacts { risk: Some(0.9), window: None };
        e.ownership = Some(OwnershipFacts { holder: "X Ltd".into(), opacity: 0.8, blocked_ownership_pct: 40.0, ..Default::default() });
        let codes: Vec<String> = identify(&e).into_iter().map(|t| t.code).collect();
        assert!(codes.contains(&"network_anomaly".to_string()));
        assert!(codes.contains(&"opaque_ownership".to_string()));
        assert!(codes.contains(&"sanctions".to_string()));
        assert!(identify(&ev(&[])).is_empty());
    }

    #[test]
    fn every_listed_rule_belongs_to_a_known_code_and_defs_are_unique() {
        let mut codes: Vec<&str> = TYPOLOGIES.iter().map(|t| t.code).collect();
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), TYPOLOGIES.len());
        assert!(TYPOLOGIES.iter().all(|t| !t.rules.is_empty() && !t.next_steps.is_empty()));
    }
}
