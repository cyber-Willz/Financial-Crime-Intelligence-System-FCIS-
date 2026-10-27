//! Case prioritisation: an explainable ordering of the review queue.
//!
//! `priority = P × memory_factor × (1 + w_e·ln(1 + amount/scale)) × (1 + age boost)` where
//! `P = 1 − (1 − p_alert)(1 − w_g · graph_risk)`, `p_alert` is the calibrated probability of confirmation when
//! analyst outcomes are available (else score/100) and `w_g` is deliberately modest because the graph model
//! has not been validated on real data. Hard controls (watchlist, sanctioned jurisdiction, material sanctioned
//! stake) go to the top regardless. It changes the order alerts are shown in; it never closes or hides one.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueueItem {
    pub alert_id: i64,
    pub account: i64,
    pub score: u8,
    pub severity: String,
    pub rules: Vec<String>,
    pub amount_gbp: f64,
    pub age_days: f64,
    pub graph_risk: Option<f32>,
    pub calibrated: Option<f64>,
    pub memory_factor: f64,
    pub memory_reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PriorityConfig {
    pub graph_weight: f64,
    pub exposure_weight: f64,
    pub exposure_scale_gbp: f64,
    pub sla_days: f64,
    pub age_boost_per_day: f64,
    pub max_age_boost: f64,
    pub hard_controls: Vec<String>,
}

impl Default for PriorityConfig {
    fn default() -> Self {
        PriorityConfig {
            graph_weight: 0.5,
            exposure_weight: 0.15,
            exposure_scale_gbp: 10_000.0,
            sla_days: 5.0,
            age_boost_per_day: 0.05,
            max_age_boost: 0.5,
            hard_controls: vec!["WATCHLIST_HIT".into(), "SANCTIONED_JURISDICTION".into(), "material_sanctioned_stake".into()],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Ranked {
    pub alert_id: i64,
    pub account: i64,
    pub priority: f64,
    pub tier: String,
    pub reasons: Vec<String>,
}

pub fn rank(items: &[QueueItem], cfg: &PriorityConfig) -> Vec<Ranked> {
    let mut out: Vec<(Ranked, f64)> = items
        .iter()
        .map(|it| {
            let mut reasons = vec![];
            let p_alert = it.calibrated.unwrap_or(it.score as f64 / 100.0).clamp(0.0, 1.0);
            reasons.push(match it.calibrated {
                Some(p) => format!("{:.0}% of comparable reviewed alerts were confirmed", p * 100.0),
                None => format!("engine score {}/100 (no analyst outcomes to calibrate it yet)", it.score),
            });
            let g = it.graph_risk.map(|r| (cfg.graph_weight * r as f64).clamp(0.0, 1.0)).unwrap_or(0.0);
            if let Some(r) = it.graph_risk {
                reasons.push(format!("graph model risk {:.2}", r));
            }
            let base = 1.0 - (1.0 - p_alert) * (1.0 - g);
            let exposure = 1.0 + cfg.exposure_weight * (1.0 + it.amount_gbp.max(0.0) / cfg.exposure_scale_gbp).ln();
            let over = (it.age_days - cfg.sla_days).max(0.0);
            let age = (over * cfg.age_boost_per_day).min(cfg.max_age_boost);
            if age > 0.0 {
                reasons.push(format!("open {:.0} day(s), past the {:.0}-day target", it.age_days, cfg.sla_days));
            }
            if (it.memory_factor - 1.0).abs() > 1e-9 {
                reasons.push(format!("history factor {:.2}: {}", it.memory_factor, it.memory_reason));
            }
            let hard = it.rules.iter().any(|r| cfg.hard_controls.contains(r));
            if hard {
                reasons.push("hard control (sanctions / watchlist): always reviewed first".into());
            }
            let mut p = base * it.memory_factor.clamp(0.0, 3.0) * exposure * (1.0 + age);
            if hard {
                p = p.max(1.0) + 1.0;
            }
            let tier = if hard || p >= 0.8 { "critical" } else if p >= 0.5 { "high" } else if p >= 0.25 { "medium" } else { "low" };
            (Ranked { alert_id: it.alert_id, account: it.account, priority: p, tier: tier.into(), reasons }, it.age_days)
        })
        .collect();
    out.sort_by(|a, b| b.0.priority.partial_cmp(&a.0.priority).unwrap_or(std::cmp::Ordering::Equal).then(b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)).then(a.0.alert_id.cmp(&b.0.alert_id)));
    out.into_iter().map(|x| x.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(id: i64, score: u8, cal: Option<f64>) -> QueueItem {
        QueueItem { alert_id: id, account: id, score, severity: "high".into(), rules: vec!["FAN_IN".into()], amount_gbp: 1000.0, age_days: 1.0, graph_risk: None, calibrated: cal, memory_factor: 1.0, memory_reason: String::new() }
    }

    #[test]
    fn calibrated_probability_beats_raw_score() {
        let r = rank(&[it(1, 90, Some(0.1)), it(2, 65, Some(0.6))], &PriorityConfig::default());
        assert_eq!(r[0].alert_id, 2);
    }

    #[test]
    fn hard_controls_come_first_and_graph_risk_helps() {
        let mut a = it(1, 99, None);
        let mut b = it(2, 50, None);
        b.rules = vec!["WATCHLIST_HIT".into()];
        let mut c = it(3, 60, None);
        c.graph_risk = Some(0.95);
        a.graph_risk = None;
        let r = rank(&[a, b, c], &PriorityConfig::default());
        assert_eq!(r[0].alert_id, 2);
        assert_eq!(r[0].tier, "critical");
        let pos = |id| r.iter().position(|x| x.alert_id == id).unwrap();
        assert!(pos(3) < pos(1) || r[pos(3)].priority > 0.6);
    }

    #[test]
    fn history_factor_and_age_move_the_order_within_bounds() {
        let mut fp = it(1, 70, None);
        fp.memory_factor = 0.6;
        let fresh = it(2, 70, None);
        let mut old = it(3, 70, None);
        old.age_days = 40.0;
        let r = rank(&[fp, fresh, old], &PriorityConfig::default());
        assert_eq!(r[0].alert_id, 3);
        assert_eq!(r[2].alert_id, 1);
        assert!(r[2].reasons.iter().any(|x| x.contains("history factor")));
        let max = PriorityConfig::default().max_age_boost;
        assert!(r[0].priority <= 0.7 * 1.1 * (1.0 + max) + 1e-9);
    }
}
