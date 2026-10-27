//! "Calibrate Risk Thresholds": turn engine scores into observed probabilities and advise the alert threshold.
//!
//! * [`Calibrator`] — isotonic regression (pool-adjacent-violators) of *confirmed* on score. Monotone by
//!   construction, no parametric shape assumed.
//! * [`advise`] — for each candidate threshold: reviewed alerts at or above it, how many were confirmed, a
//!   Wilson lower bound on precision, and the share of all confirmed alerts that would be kept. It recommends
//!   the **lowest** threshold whose precision lower bound reaches the target while keeping at least
//!   `min_retained` of the confirmed alerts, and never moves more than `max_step` points per cycle.
//!
//! What this cannot do: reviewed alerts all scored at least the threshold in force when they were raised, so
//! nothing here can say what lies *below* the floor of the data. A recommendation to lower the threshold is
//! therefore never produced; run a shadow period at a lower threshold to learn that.

use crate::signal::FeedbackEvent;
use crate::stats::wilson;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Calibrator {
    /// (score, probability) knots, score ascending, probability non-decreasing
    pub knots: Vec<(f64, f64)>,
}

impl Calibrator {
    /// `points`: (score, positive, weight).
    pub fn fit(points: &[(f64, bool, f64)]) -> Option<Calibrator> {
        let mut p: Vec<(f64, f64, f64)> = points.iter().filter(|x| x.2 > 0.0 && x.0.is_finite()).map(|&(s, y, w)| (s, y as u8 as f64, w)).collect();
        if p.len() < 2 {
            return None;
        }
        p.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // blocks: (sum_w, sum_wy, sum_wx)
        let mut blocks: Vec<(f64, f64, f64)> = Vec::new();
        for (x, y, w) in p {
            blocks.push((w, w * y, w * x));
            while blocks.len() >= 2 {
                let n = blocks.len();
                let (a, b) = (blocks[n - 2], blocks[n - 1]);
                if a.1 / a.0 > b.1 / b.0 {
                    blocks.truncate(n - 2);
                    blocks.push((a.0 + b.0, a.1 + b.1, a.2 + b.2));
                } else {
                    break;
                }
            }
        }
        // Jeffreys smoothing (half a pseudo-confirmation, half a pseudo-false-positive): a block of three
        // confirmed alerts is evidence, not certainty. A final running maximum keeps the curve monotone.
        let mut knots: Vec<(f64, f64)> = blocks.iter().map(|b| (b.2 / b.0, (b.1 + 0.5) / (b.0 + 1.0))).collect();
        let mut top = 0.0f64;
        for k in knots.iter_mut() {
            top = top.max(k.1);
            k.1 = top;
        }
        knots.dedup_by(|b, a| (b.0 - a.0).abs() < 1e-12);
        Some(Calibrator { knots })
    }

    pub fn predict(&self, score: f64) -> f64 {
        let k = &self.knots;
        if k.is_empty() {
            return 0.0;
        }
        if score <= k[0].0 {
            return k[0].1;
        }
        if score >= k[k.len() - 1].0 {
            return k[k.len() - 1].1;
        }
        let i = k.partition_point(|x| x.0 <= score);
        let (x0, y0) = k[i - 1];
        let (x1, y1) = k[i];
        y0 + (y1 - y0) * (score - x0) / (x1 - x0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// precision (lower confidence bound) the alert queue should reach
    pub target_precision: f64,
    pub z: f64,
    /// smallest number of reviewed alerts a candidate threshold must rest on
    pub min_alerts: usize,
    /// share of confirmed alerts a higher threshold must keep
    pub min_retained: f64,
    /// largest move per cycle, in score points
    pub max_step: u8,
    pub grid_step: u8,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        ThresholdConfig { target_precision: 0.25, z: 1.645, min_alerts: 50, min_retained: 0.7, max_step: 10, grid_step: 5 }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdRow {
    pub threshold: u8,
    pub alerts: usize,
    pub confirmed: usize,
    pub precision: f64,
    pub precision_low: f64,
    /// share of all reviewed-confirmed alerts at or above the threshold
    pub retained: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdAdvice {
    pub current: u8,
    pub recommended: u8,
    pub changed: bool,
    /// lowest score present in the reviewed data; nothing is known below it
    pub observed_floor: Option<u8>,
    pub table: Vec<ThresholdRow>,
    pub calibrator: Option<Calibrator>,
    pub rationale: String,
    pub caveats: Vec<String>,
}

pub fn advise(events: &[FeedbackEvent], current: u8, cfg: &ThresholdConfig) -> ThresholdAdvice {
    let mut adv = ThresholdAdvice {
        current,
        recommended: current,
        changed: false,
        observed_floor: events.iter().map(|e| e.score).min(),
        table: vec![],
        calibrator: None,
        rationale: String::new(),
        caveats: vec![
            "Only alerts that were raised and reviewed have outcomes: the data cannot say whether a lower threshold would find more laundering, so lowering is never recommended from it.".into(),
        ],
    };
    if events.len() < cfg.min_alerts {
        adv.rationale = format!("{} reviewed alerts; at least {} are needed before the threshold is assessed", events.len(), cfg.min_alerts);
        return adv;
    }
    let pts: Vec<(f64, bool, f64)> = events.iter().map(|e| (e.score as f64, e.label(), e.weight())).collect();
    adv.calibrator = Calibrator::fit(&pts);
    let tot_conf: f64 = events.iter().filter(|e| e.label()).map(|e| e.weight()).sum();
    let floor = adv.observed_floor.unwrap_or(current).min(current);
    let mut t = floor;
    while t <= 100 {
        let sel: Vec<&FeedbackEvent> = events.iter().filter(|e| e.score >= t).collect();
        let n: f64 = sel.iter().map(|e| e.weight()).sum();
        let pos: f64 = sel.iter().filter(|e| e.label()).map(|e| e.weight()).sum();
        let (p, lo, _) = wilson(pos, n, cfg.z);
        adv.table.push(ThresholdRow { threshold: t, alerts: sel.len(), confirmed: sel.iter().filter(|e| e.label()).count(), precision: p, precision_low: lo, retained: if tot_conf > 0.0 { pos / tot_conf } else { 0.0 } });
        if t == 100 {
            break;
        }
        t = (t + cfg.grid_step).min(100);
    }
    let cur_row = adv.table.iter().find(|r| r.threshold >= current).cloned();
    let meets = |r: &ThresholdRow| r.alerts >= cfg.min_alerts && r.precision_low >= cfg.target_precision && r.retained >= cfg.min_retained;
    match cur_row {
        Some(r) if meets(&r) || r.threshold < current => {
            adv.rationale = format!(
                "At {} the reviewed alerts reach {:.0}% precision (lower bound {:.0}%) against a target of {:.0}%: keep the threshold.",
                current, r.precision * 100.0, r.precision_low * 100.0, cfg.target_precision * 100.0
            );
        }
        _ => {
            let cand = adv.table.iter().filter(|r| r.threshold > current && meets(r)).min_by_key(|r| r.threshold).cloned();
            match cand {
                Some(c) => {
                    let to = c.threshold.min(current.saturating_add(cfg.max_step));
                    adv.recommended = to;
                    adv.changed = to != current;
                    adv.rationale = format!(
                        "Precision at {} is below the {:.0}% target. At {} the reviewed alerts reach {:.0}% (lower bound {:.0}%) while keeping {:.0}% of the confirmed ones; moving to {}{}.",
                        current, cfg.target_precision * 100.0, c.threshold, c.precision * 100.0, c.precision_low * 100.0, c.retained * 100.0, to,
                        if to != c.threshold { format!(" (capped at +{} points per cycle)", cfg.max_step) } else { String::new() }
                    );
                }
                None => {
                    adv.rationale = format!(
                        "No threshold in the reviewed data reaches a {:.0}% precision lower bound while keeping {:.0}% of confirmed alerts; keep {} and look at rule weights instead.",
                        cfg.target_precision * 100.0, cfg.min_retained * 100.0, current
                    );
                }
            }
        }
    }
    adv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::Decision;

    fn ev(id: i64, score: u8, pos: bool) -> FeedbackEvent {
        FeedbackEvent {
            alert_id: id,
            account: id,
            decided_at: 0,
            score,
            rules: vec![],
            graph_risk: None,
            decision: if pos { Decision::SarFiled } else { Decision::FalsePositive },
            decided_by: "t".into(),
            case_id: None,
        }
    }

    #[test]
    fn pav_is_monotone_and_pools() {
        let pts = [(1.0, false, 1.0), (2.0, true, 1.0), (3.0, false, 1.0), (4.0, true, 1.0), (5.0, true, 1.0)];
        let c = Calibrator::fit(&pts).unwrap();
        assert!(c.knots.windows(2).all(|w| w[0].1 <= w[1].1 + 1e-12));
        // 2 and 3 violate monotonicity (1, 0) and pool; smoothing keeps every value strictly inside (0, 1)
        assert!(c.predict(2.5) > 0.3 && c.predict(2.5) < 0.7, "{:?}", c.knots);
        assert!(c.predict(0.0) > 0.0 && c.predict(0.0) < 0.5);
        assert!(c.predict(9.0) < 1.0 && c.predict(9.0) > 0.5);
    }

    #[test]
    fn pav_recovers_a_noisy_ramp() {
        let mut s = 7u64;
        let mut pts = vec![];
        for i in 0..4000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
            let x = 60.0 + (i % 41) as f64;
            let p = (x - 60.0) / 40.0 * 0.8 + 0.1;
            pts.push((x, u < p, 1.0));
        }
        let c = Calibrator::fit(&pts).unwrap();
        assert!((c.predict(65.0) - 0.20).abs() < 0.09, "{}", c.predict(65.0));
        assert!((c.predict(95.0) - 0.70).abs() < 0.08, "{}", c.predict(95.0));
    }

    fn graded(n: usize) -> Vec<FeedbackEvent> {
        // precision rises with score: 60-69 -> 5%, 70-79 -> 15%, 80-89 -> 40%, 90+ -> 70%
        let mut s = 3u64;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 11) as f64) / ((1u64 << 53) as f64);
                let score = 60 + (i % 40) as u8;
                let p = match score {
                    60..=69 => 0.05,
                    70..=79 => 0.15,
                    80..=89 => 0.40,
                    _ => 0.70,
                };
                ev(i as i64, score, u < p)
            })
            .collect()
    }

    #[test]
    fn raises_threshold_to_reach_target_but_caps_the_step() {
        let evs = graded(4000);
        let cfg = ThresholdConfig { target_precision: 0.35, min_retained: 0.5, ..Default::default() };
        let a = advise(&evs, 60, &cfg);
        assert!(a.changed && a.recommended > 60 && a.recommended <= 70, "{a:?}");
        // an uncapped move lands where the lower bound first reaches the target
        let uncapped = advise(&evs, 60, &ThresholdConfig { max_step: 100, ..cfg.clone() });
        assert!(uncapped.recommended >= 65 && uncapped.recommended <= 85, "{}", uncapped.recommended);
        // a smaller cap really caps
        let capped = advise(&evs, 60, &ThresholdConfig { max_step: 5, ..cfg.clone() });
        assert_eq!(capped.recommended, 65);
        assert!(capped.rationale.contains("capped"));
    }

    #[test]
    fn keeps_a_threshold_that_already_meets_the_target() {
        let evs = graded(4000);
        let a = advise(&evs, 60, &ThresholdConfig { target_precision: 0.05, min_retained: 0.5, ..Default::default() });
        assert!(!a.changed && a.recommended == 60);
    }

    #[test]
    fn never_recommends_lowering_and_needs_data() {
        let evs: Vec<_> = (0..300).map(|i| ev(i, 80 + (i % 20) as u8, i % 2 == 0)).collect();
        let a = advise(&evs, 70, &ThresholdConfig::default());
        assert!(a.recommended >= 70);
        let few = advise(&evs[..10], 70, &ThresholdConfig::default());
        assert!(!few.changed && few.rationale.contains("needed"));
    }
}
