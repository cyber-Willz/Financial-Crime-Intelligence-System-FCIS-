//! Outcome memory: what analysts concluded the last times this account (or this account + rule) alerted.
//!
//! This is a *ranking aid for the review queue*, not a detector and not a suppressor:
//! * it changes the order in which alerts are shown, never whether one is raised or closed;
//! * decayed counts (half-life `half_life_days`) mean old decisions fade;
//! * hard controls (watchlist, sanctions, blocked ownership, PEP, SAR history) are never down-weighted;
//! * the multiplier is bounded to `[min_factor, max_factor]`.
//!
//! The recurrent network's hidden states are deliberately *not* edited by feedback — they are a
//! deterministic function of the transaction stream and the trained weights, and rewriting them by hand
//! would make them inconsistent with the model. Feedback reaches the network only through [`crate::tune`].

use crate::signal::{Decision, FeedbackEvent};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub half_life_days: f64,
    /// decayed false positives (with no confirmation) needed before an account is down-weighted
    pub fp_floor: f64,
    pub min_factor: f64,
    pub max_factor: f64,
    /// rules that always keep full priority
    pub never_suppress: Vec<String>,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        MemoryConfig {
            half_life_days: 90.0,
            fp_floor: 2.5,
            min_factor: 0.6,
            max_factor: 1.5,
            never_suppress: vec!["WATCHLIST_HIT".into(), "SANCTIONED_JURISDICTION".into(), "material_sanctioned_stake".into()],
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Tally {
    tp: f64,
    fp: f64,
    sar: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutcomeMemory {
    now: i64,
    cfg_half_life_days: f64,
    accounts: BTreeMap<i64, Tally>,
    /// (account, rule) -> tally
    pairs: BTreeMap<(i64, String), Tally>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Adjusted {
    pub factor: f64,
    pub reason: String,
    pub decayed_confirmed: f64,
    pub decayed_false_positive: f64,
    pub sar_history: bool,
}

impl OutcomeMemory {
    /// Build from reviewed alerts. `now` is unix seconds.
    pub fn from_events(events: &[FeedbackEvent], now: i64, cfg: &MemoryConfig) -> Self {
        let mut m = OutcomeMemory { now, cfg_half_life_days: cfg.half_life_days, ..Default::default() };
        for e in events {
            let age_days = ((now - e.decided_at).max(0) as f64) / 86_400.0;
            let d = 0.5f64.powf(age_days / cfg.half_life_days.max(1.0));
            let add = |t: &mut Tally| match e.decision {
                Decision::FalsePositive => t.fp += d,
                Decision::Confirmed => t.tp += d,
                Decision::SarFiled => {
                    t.tp += d;
                    t.sar += 1.0; // SAR history never decays: it is a regulatory fact
                }
            };
            add(m.accounts.entry(e.account).or_default());
            for r in &e.rules {
                add(m.pairs.entry((e.account, r.rule.clone())).or_default());
            }
        }
        m
    }

    pub fn accounts_tracked(&self) -> usize {
        self.accounts.len()
    }

    /// Priority multiplier for an alert on `account` raised by `rules`.
    pub fn adjust(&self, account: i64, rules: &[&str], cfg: &MemoryConfig) -> Adjusted {
        let t = self.accounts.get(&account).cloned().unwrap_or_default();
        let neutral = |reason: &str| Adjusted { factor: 1.0, reason: reason.into(), decayed_confirmed: t.tp, decayed_false_positive: t.fp, sar_history: t.sar > 0.0 };
        if t.sar > 0.0 {
            let f = (1.0 + 0.25 * t.sar.min(2.0)).min(cfg.max_factor);
            return Adjusted { factor: f, reason: format!("{} earlier SAR(s) on this account", t.sar as u32), ..neutral("") };
        }
        if t.tp > 0.0 {
            let f = (1.0 + 0.3 * t.tp.min(1.7)).min(cfg.max_factor);
            return Adjusted { factor: f, reason: "an earlier alert on this account was confirmed".into(), ..neutral("") };
        }
        if rules.iter().any(|r| cfg.never_suppress.iter().any(|n| n == r)) {
            return neutral("hard control rule: never down-weighted");
        }
        // repeated false positives for the *same* rule pattern on this account
        let pair_fp = rules.iter().map(|r| self.pairs.get(&(account, r.to_string())).map(|x| x.fp).unwrap_or(0.0)).fold(0.0, f64::max);
        if pair_fp >= cfg.fp_floor {
            let f = (1.0 - 0.12 * (pair_fp - cfg.fp_floor + 1.0)).clamp(cfg.min_factor, 1.0);
            return Adjusted { factor: f, reason: format!("this rule pattern was closed as a false positive {:.1} times (decayed) on this account and never confirmed", pair_fp), ..neutral("") };
        }
        neutral("no relevant history")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::RuleHit;

    fn ev(id: i64, acct: i64, day: i64, rule: &str, d: Decision) -> FeedbackEvent {
        FeedbackEvent { alert_id: id, account: acct, decided_at: day * 86_400, score: 70, rules: vec![RuleHit { rule: rule.into(), weight: 0.5 }], graph_risk: None, decision: d, decided_by: "t".into(), case_id: None }
    }

    #[test]
    fn repeated_false_positives_lower_priority_within_bounds() {
        let cfg = MemoryConfig::default();
        let evs: Vec<_> = (0..6).map(|i| ev(i, 7, 100 + i, "FAN_IN", Decision::FalsePositive)).collect();
        let m = OutcomeMemory::from_events(&evs, 110 * 86_400, &cfg);
        let a = m.adjust(7, &["FAN_IN"], &cfg);
        assert!(a.factor < 1.0 && a.factor >= cfg.min_factor, "{a:?}");
        // a different rule on the same account is not penalised
        assert_eq!(m.adjust(7, &["STRUCTURING_HUB"], &cfg).factor, 1.0);
        // an unknown account is neutral
        assert_eq!(m.adjust(8, &["FAN_IN"], &cfg).factor, 1.0);
    }

    #[test]
    fn hard_controls_and_confirmations_are_never_suppressed() {
        let cfg = MemoryConfig::default();
        let mut evs: Vec<_> = (0..10).map(|i| ev(i, 7, 100, "WATCHLIST_HIT", Decision::FalsePositive)).collect();
        let m = OutcomeMemory::from_events(&evs, 101 * 86_400, &cfg);
        assert_eq!(m.adjust(7, &["WATCHLIST_HIT"], &cfg).factor, 1.0);
        evs.push(ev(99, 7, 100, "FAN_IN", Decision::Confirmed));
        let m = OutcomeMemory::from_events(&evs, 101 * 86_400, &cfg);
        assert!(m.adjust(7, &["FAN_IN"], &cfg).factor > 1.0);
    }

    #[test]
    fn old_decisions_fade_but_sar_history_does_not() {
        let cfg = MemoryConfig::default();
        let evs: Vec<_> = (0..6).map(|i| ev(i, 7, 0, "FAN_IN", Decision::FalsePositive)).collect();
        let old = OutcomeMemory::from_events(&evs, 900 * 86_400, &cfg);
        assert_eq!(old.adjust(7, &["FAN_IN"], &cfg).factor, 1.0);
        let sar = OutcomeMemory::from_events(&[ev(1, 9, 0, "FAN_IN", Decision::SarFiled)], 900 * 86_400, &cfg);
        let a = sar.adjust(9, &["FAN_IN"], &cfg);
        assert!(a.sar_history && a.factor > 1.0 && a.factor <= cfg.max_factor);
    }
}
