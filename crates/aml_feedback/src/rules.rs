//! "Recommend Rule Tuning": which rules earn their weight according to analyst outcomes.
//!
//! Method (all numbers reproducible from the events):
//! 1. *Marginal precision* of each rule = share of alerts containing the rule that analysts confirmed,
//!    shrunk toward the overall precision with `prior_strength` pseudo-alerts, with a Wilson interval.
//! 2. A ridge-logistic regression of the outcome on rule indicators gives each rule's *conditional* effect
//!    with the other rules held fixed. Rules co-fire constantly (a fan-in alert also carries the jurisdiction
//!    rule), so a marginal lift alone would credit or blame the wrong rule.
//! 3. A rule is recommended **up** only if the whole interval is above the baseline by `margin` *and* the
//!    regression coefficient is positive; **down** only if the interval is entirely below *and* the
//!    coefficient is negative. Anything else is `Keep`. The step is a bounded move in logit space.
//!
//! Limits (also returned in `RuleReport::caveats`): outcomes exist only for alerts that fired, so this
//! cannot find rules that are *missing*, nor say anything about transactions below the alert threshold;
//! reviewers' decisions are noisy; a rule that is always present has no contrast and stays `Keep`.

use crate::signal::FeedbackEvent;
use crate::stats::{logit, sigmoid, wilson};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleLearnerConfig {
    /// alerts containing the rule required before any recommendation
    pub min_events: usize,
    /// pseudo-alerts pulling a rule's precision toward the overall precision
    pub prior_strength: f64,
    /// z of the Wilson interval (1.645 = 90 %)
    pub z: f64,
    /// relative distance from the baseline the interval must clear
    pub margin: f64,
    /// largest change of logit(weight) per recommendation cycle
    pub max_logit_step: f64,
    pub min_weight: f64,
    pub max_weight: f64,
    /// rules that are never recommended for a decrease (hard sanctions / watchlist controls)
    pub protected: Vec<String>,
    /// rules that describe context, not a typology; confirmed alerts carrying only these point at a coverage gap
    pub context_rules: Vec<String>,
    pub ridge: f64,
}

impl Default for RuleLearnerConfig {
    fn default() -> Self {
        RuleLearnerConfig {
            min_events: 30,
            prior_strength: 10.0,
            z: 1.645,
            margin: 0.25,
            max_logit_step: 0.6,
            min_weight: 0.02,
            max_weight: 0.95,
            protected: vec!["WATCHLIST_HIT".into(), "SANCTIONED_JURISDICTION".into()],
            context_rules: vec![
                "HIGH_RISK_JURISDICTION".into(),
                "CURRENCY_CONVERSION".into(),
                "CUSTOMER_JURISDICTION".into(),
                "HIGH_RISK_CUSTOMER".into(),
                "LARGE_TXN".into(),
            ],
            ridge: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Increase,
    Decrease,
    Keep,
    InsufficientData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleStat {
    pub rule: String,
    pub alerts: usize,
    pub confirmed: usize,
    pub false_positives: usize,
    /// shrunk marginal precision
    pub precision: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    /// precision / baseline precision
    pub lift: f64,
    /// ridge-logistic coefficient (log-odds of "confirmed" when the rule is present, others fixed)
    pub coef: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuleRecommendation {
    pub rule: String,
    pub action: Action,
    pub current_weight: Option<f64>,
    pub suggested_weight: Option<f64>,
    pub stat: RuleStat,
    pub rationale: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuleReport {
    pub events: usize,
    pub confirmed: usize,
    pub baseline_precision: f64,
    pub stats: Vec<RuleStat>,
    pub recommendations: Vec<RuleRecommendation>,
    /// confirmed alerts whose rules were all context rules
    pub uncovered_confirmed: usize,
    pub caveats: Vec<String>,
}

/// Solve `A x = b` (dense, partial pivoting). Returns `None` for a singular system.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    for c in 0..n {
        let p = (c..n).max_by(|&i, &j| a[i][c].abs().partial_cmp(&a[j][c].abs()).unwrap_or(std::cmp::Ordering::Equal))?;
        if a[p][c].abs() < 1e-12 {
            return None;
        }
        a.swap(c, p);
        b.swap(c, p);
        for r in c + 1..n {
            let f = a[r][c] / a[c][c];
            if f != 0.0 {
                for k in c..n {
                    a[r][k] -= f * a[c][k];
                }
                b[r] -= f * b[c];
            }
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|k| a[r][k] * x[k]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

/// Ridge-penalised weighted logistic regression by IRLS; the intercept (index 0) is not penalised.
/// `rows[i]` lists the feature indices (1-based into beta) that are 1 for sample `i`.
fn ridge_logistic(rows: &[Vec<usize>], y: &[bool], w: &[f64], d: usize, ridge: f64) -> Vec<f64> {
    let k = d + 1;
    let mut beta = vec![0.0; k];
    for _ in 0..30 {
        let mut h = vec![vec![0.0; k]; k];
        let mut g = vec![0.0; k];
        for (i, feats) in rows.iter().enumerate() {
            let eta: f64 = beta[0] + feats.iter().map(|&f| beta[f]).sum::<f64>();
            let p = sigmoid(eta);
            let s = (p * (1.0 - p)).max(1e-6) * w[i];
            let r = w[i] * (y[i] as u8 as f64 - p);
            g[0] += r;
            h[0][0] += s;
            for &f in feats {
                g[f] += r;
                h[0][f] += s;
                h[f][0] += s;
                for &f2 in feats {
                    h[f][f2] += s;
                }
            }
        }
        for j in 1..k {
            g[j] -= ridge * beta[j];
            h[j][j] += ridge;
        }
        h[0][0] += 1e-9;
        let Some(step) = solve(h, g) else { break };
        let mut m = 0.0f64;
        for j in 0..k {
            beta[j] += step[j];
            m = m.max(step[j].abs());
        }
        if m < 1e-7 {
            break;
        }
    }
    beta
}

/// Analyse reviewed alerts. `weights` are the current rule weights (code -> weight), when known.
pub fn analyse(events: &[FeedbackEvent], weights: &BTreeMap<String, f64>, cfg: &RuleLearnerConfig) -> RuleReport {
    let mut rep = RuleReport { events: events.len(), ..Default::default() };
    rep.caveats.push(
        "Outcomes exist only for alerts that were raised and reviewed: this cannot reveal missing rules or anything below the alert threshold, and it inherits reviewer error.".into(),
    );
    if events.is_empty() {
        return rep;
    }
    let tot_w: f64 = events.iter().map(|e| e.weight()).sum();
    let pos_w: f64 = events.iter().filter(|e| e.label()).map(|e| e.weight()).sum();
    rep.confirmed = events.iter().filter(|e| e.label()).count();
    let base = if tot_w > 0.0 { pos_w / tot_w } else { 0.0 };
    rep.baseline_precision = base;

    let ctx: BTreeSet<&str> = cfg.context_rules.iter().map(String::as_str).collect();
    rep.uncovered_confirmed = events.iter().filter(|e| e.label() && !e.rules.is_empty() && e.rules.iter().all(|r| ctx.contains(r.rule.as_str()))).count()
        + events.iter().filter(|e| e.label() && e.rules.is_empty()).count();

    // rule index
    let codes: Vec<String> = events.iter().flat_map(|e| e.rules.iter().map(|r| r.rule.clone())).collect::<BTreeSet<_>>().into_iter().collect();
    let idx: BTreeMap<&str, usize> = codes.iter().enumerate().map(|(i, c)| (c.as_str(), i + 1)).collect();
    let rows: Vec<Vec<usize>> = events
        .iter()
        .map(|e| e.rules.iter().filter_map(|r| idx.get(r.rule.as_str()).copied()).collect::<BTreeSet<_>>().into_iter().collect())
        .collect();
    let y: Vec<bool> = events.iter().map(|e| e.label()).collect();
    let w: Vec<f64> = events.iter().map(|e| e.weight()).collect();
    let degenerate = base <= 0.0 || base >= 1.0;
    let beta = if degenerate { vec![0.0; codes.len() + 1] } else { ridge_logistic(&rows, &y, &w, codes.len(), cfg.ridge) };
    if degenerate {
        rep.caveats.push("Every reviewed alert has the same outcome, so no rule can be distinguished; no recommendations were made.".into());
    }

    for code in &codes {
        let (mut n, mut pos, mut cnt, mut conf) = (0.0, 0.0, 0usize, 0usize);
        for e in events.iter().filter(|e| e.has_rule(code)) {
            n += e.weight();
            cnt += 1;
            if e.label() {
                pos += e.weight();
                conf += 1;
            }
        }
        let shrunk = (pos + cfg.prior_strength * base) / (n + cfg.prior_strength);
        let (_, lo, hi) = wilson(pos, n, cfg.z);
        rep.stats.push(RuleStat {
            rule: code.clone(),
            alerts: cnt,
            confirmed: conf,
            false_positives: cnt - conf,
            precision: shrunk,
            ci_low: lo,
            ci_high: hi,
            lift: if base > 0.0 { shrunk / base } else { 0.0 },
            coef: beta[idx[code.as_str()]],
        });
    }
    rep.stats.sort_by(|a, b| b.alerts.cmp(&a.alerts).then(a.rule.cmp(&b.rule)));

    for s in &rep.stats {
        let cur = weights.get(&s.rule).copied();
        let (action, why) = if s.alerts < cfg.min_events {
            (Action::InsufficientData, format!("only {} reviewed alerts contain it (need {})", s.alerts, cfg.min_events))
        } else if degenerate {
            (Action::Keep, "no outcome contrast".into())
        } else if s.ci_low > base * (1.0 + cfg.margin) && s.coef > 0.0 {
            (
                Action::Increase,
                format!(
                    "{}/{} reviewed alerts confirmed ({:.0}% shrunk, 90% interval {:.0}-{:.0}%) against {:.0}% overall, and still positive with the other rules held fixed (coef {:+.2})",
                    s.confirmed, s.alerts, s.precision * 100.0, s.ci_low * 100.0, s.ci_high * 100.0, base * 100.0, s.coef
                ),
            )
        } else if s.ci_high < base * (1.0 - cfg.margin) && s.coef < 0.0 {
            if cfg.protected.iter().any(|p| p == &s.rule) {
                (Action::Keep, format!("below baseline ({}/{} confirmed) but '{}' is a protected control and is never lowered automatically", s.confirmed, s.alerts, s.rule))
            } else {
                (
                    Action::Decrease,
                    format!(
                        "only {}/{} reviewed alerts confirmed ({:.0}% shrunk, 90% interval {:.0}-{:.0}%) against {:.0}% overall, and still negative with the other rules held fixed (coef {:+.2})",
                        s.confirmed, s.alerts, s.precision * 100.0, s.ci_low * 100.0, s.ci_high * 100.0, base * 100.0, s.coef
                    ),
                )
            }
        } else {
            (Action::Keep, format!("{}/{} confirmed ({:.0}%) is not clearly different from the {:.0}% baseline", s.confirmed, s.alerts, s.precision * 100.0, base * 100.0))
        };
        let suggested = match (action, cur) {
            (Action::Increase | Action::Decrease, Some(w0)) => {
                let step = (0.5 * s.lift.max(1e-6).ln()).clamp(-cfg.max_logit_step, cfg.max_logit_step);
                Some(sigmoid(logit(w0) + step).clamp(cfg.min_weight, cfg.max_weight))
            }
            _ => None,
        };
        // a recommendation whose rounded weight does not move is a Keep in practice
        let action = match (action, cur, suggested) {
            (Action::Increase | Action::Decrease, Some(w0), Some(w1)) if (w1 - w0).abs() < 0.005 => Action::Keep,
            (a, _, _) => a,
        };
        rep.recommendations.push(RuleRecommendation { rule: s.rule.clone(), action, current_weight: cur, suggested_weight: suggested, stat: s.clone(), rationale: why });
    }
    if rep.uncovered_confirmed > 0 {
        rep.caveats.push(format!(
            "{} confirmed alert(s) were raised by context rules only (jurisdiction / currency / customer risk / size): the typology rules did not recognise them. Review those cases for a missing typology.",
            rep.uncovered_confirmed
        ));
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::{Decision, RuleHit};

    struct Lcg(u64);
    impl Lcg {
        fn u(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    fn ev(id: i64, rules: &[&str], pos: bool) -> FeedbackEvent {
        FeedbackEvent {
            alert_id: id,
            account: id,
            decided_at: 1_700_000_000 + id,
            score: 70,
            rules: rules.iter().map(|r| RuleHit { rule: r.to_string(), weight: 0.5 }).collect(),
            graph_risk: None,
            decision: if pos { Decision::SarFiled } else { Decision::FalsePositive },
            decided_by: "t".into(),
            case_id: None,
        }
    }

    /// GOOD fires with precision .85, BAD with .05, NEUTRAL with .30; JUR co-fires with everything.
    fn world(n: usize, seed: u64) -> Vec<FeedbackEvent> {
        let mut r = Lcg(seed);
        (0..n)
            .map(|i| {
                let u = r.u();
                let (main, p) = if u < 0.3 { ("GOOD", 0.85) } else if u < 0.6 { ("BAD", 0.05) } else { ("NEUTRAL", 0.30) };
                let pos = r.u() < p;
                if r.u() < 0.5 {
                    ev(i as i64, &[main, "JUR"], pos)
                } else {
                    ev(i as i64, &[main], pos)
                }
            })
            .collect()
    }

    fn weights() -> BTreeMap<String, f64> {
        [("GOOD", 0.4), ("BAD", 0.4), ("NEUTRAL", 0.4), ("JUR", 0.3)].iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn act(rep: &RuleReport, code: &str) -> Action {
        rep.recommendations.iter().find(|r| r.rule == code).unwrap().action
    }

    #[test]
    fn finds_good_and_bad_rules() {
        let rep = analyse(&world(1500, 5), &weights(), &RuleLearnerConfig::default());
        assert_eq!(act(&rep, "GOOD"), Action::Increase);
        assert_eq!(act(&rep, "BAD"), Action::Decrease);
        assert_eq!(act(&rep, "NEUTRAL"), Action::Keep);
        // JUR co-fires at random: it must not be blamed or credited for its partners' outcomes
        assert_eq!(act(&rep, "JUR"), Action::Keep, "{:?}", rep.stats.iter().find(|s| s.rule == "JUR"));
        let g = rep.recommendations.iter().find(|r| r.rule == "GOOD").unwrap();
        let b = rep.recommendations.iter().find(|r| r.rule == "BAD").unwrap();
        assert!(g.suggested_weight.unwrap() > 0.4 && b.suggested_weight.unwrap() < 0.4);
        // bounded step
        assert!((logit(g.suggested_weight.unwrap()) - logit(0.4)).abs() <= 0.6 + 1e-9);
    }

    #[test]
    fn small_samples_are_not_acted_on() {
        let rep = analyse(&world(40, 9), &weights(), &RuleLearnerConfig::default());
        assert!(rep.recommendations.iter().all(|r| matches!(r.action, Action::InsufficientData | Action::Keep)));
    }

    #[test]
    fn protected_rules_are_never_lowered() {
        let mut evs = vec![];
        for i in 0..200 {
            evs.push(ev(i, &["WATCHLIST_HIT"], i % 40 == 0));
            evs.push(ev(1000 + i, &["OTHER"], i % 2 == 0));
        }
        let w: BTreeMap<String, f64> = [("WATCHLIST_HIT".to_string(), 0.9), ("OTHER".to_string(), 0.4)].into_iter().collect();
        let rep = analyse(&evs, &w, &RuleLearnerConfig::default());
        let r = rep.recommendations.iter().find(|r| r.rule == "WATCHLIST_HIT").unwrap();
        assert_eq!(r.action, Action::Keep);
        assert!(r.rationale.contains("protected"));
    }

    #[test]
    fn degenerate_outcomes_give_no_advice() {
        let evs: Vec<_> = (0..100).map(|i| ev(i, &["A", "B"], false)).collect();
        let rep = analyse(&evs, &BTreeMap::new(), &RuleLearnerConfig::default());
        assert!(rep.recommendations.iter().all(|r| r.action == Action::Keep));
        assert!(rep.caveats.iter().any(|c| c.contains("same outcome")));
    }

    #[test]
    fn context_only_confirmations_are_reported() {
        let mut evs = world(200, 3);
        for i in 0..5 {
            evs.push(ev(9000 + i, &["HIGH_RISK_JURISDICTION"], true));
        }
        let rep = analyse(&evs, &weights(), &RuleLearnerConfig::default());
        assert_eq!(rep.uncovered_confirmed, 5);
        assert!(rep.caveats.iter().any(|c| c.contains("context rules only")));
    }

    #[test]
    fn solver_matches_known_system() {
        let x = solve(vec![vec![2.0, 1.0], vec![1.0, 3.0]], vec![5.0, 10.0]).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);
        assert!(solve(vec![vec![1.0, 2.0], vec![2.0, 4.0]], vec![1.0, 2.0]).is_none());
    }
}
