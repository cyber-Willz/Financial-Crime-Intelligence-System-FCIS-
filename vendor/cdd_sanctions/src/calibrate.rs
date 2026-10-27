//! Measuring the network exposure score against labelled cases.
//!
//! The exposure thresholds (`moderate_exposure`, `strong_exposure`), the damping and the link
//! weights ship as starting points tuned on a synthetic network. This module is how you replace
//! them with evidence: give it parties your compliance team has judged ("really associated with a
//! sanctioned structure" / "reviewed and not associated") and it reports how well the score
//! separates the two groups, which edge families carry signal, and thresholds that meet a recall
//! and precision target.
//!
//! Everything here is **advisory output for a human**. Nothing is applied automatically.
//!
//! What it can and cannot tell you:
//! * AUC is the probability that a random linked party scores above a random unlinked one
//!   (0.5 = no better than chance, 1.0 = perfect). It does not depend on any threshold.
//! * Suggested thresholds are only as good as the labels. Few labels give unstable thresholds
//!   (a warning is raised below 30 of either class and no thresholds are suggested below 10).
//! * Labels should be parties **not** in the seed set. Seeds are skipped (they are trivially
//!   "linked"), and if your positives became seeds only *after* a human found the link, you are
//!   calibrating on an outcome the network could not have seen. Use labels from a time before
//!   the confirmation where you can.

use crate::cdd::PartyId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Label {
    pub party: PartyId,
    /// `true`: a person judged this party associated with a sanctioned structure.
    pub linked: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct CalibrationOptions {
    /// The moderate threshold catches at least this share of linked parties. It is placed in the gap
    /// just below the weakest linked party needed to reach the target (geometric midpoint to the next
    /// lower observed score), not on that party's own score, so a party slightly less exposed is not
    /// missed. Every labelled party is classified the same as at the score itself.
    pub recall_target: f64,
    /// The strong threshold is the lowest observed score at which at least this share of flagged
    /// parties is linked (and at least 5 of them are). It is deliberately not extended downward into
    /// ground where there is no evidence, because it triggers enhanced due diligence.
    pub precision_target: f64,
}

impl Default for CalibrationOptions {
    fn default() -> Self {
        Self { recall_target: 0.9, precision_target: 0.8 }
    }
}

/// Classification quality when every party with exposure `>= threshold` is flagged.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub threshold: f64,
    pub tp: usize,
    pub fp: usize,
    pub fn_: usize,
    pub tn: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Evaluation {
    pub positives: usize,
    pub negatives: usize,
    /// Linked parties with exposure above zero, i.e. reachable from a seed at all.
    pub detected_positives: usize,
    pub auc: Option<f64>,
    /// One point per distinct positive exposure, highest threshold first.
    pub curve: Vec<Point>,
    pub best_f1: Option<Point>,
    pub suggested_moderate: Option<f64>,
    pub suggested_strong: Option<f64>,
    pub warnings: Vec<String>,
}

/// What switching one edge family off does to the AUC. `delta_auc < 0` means the family helps
/// separate linked from unlinked parties on this data; `> 0` means it hurts; `0` that it does nothing.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FamilyEffect {
    pub family: String,
    /// Hyperedges of this family in the current data (a family with none cannot matter).
    pub edges: usize,
    pub auc_without: Option<f64>,
    pub delta_auc: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub baseline_damping: f64,
    pub baseline: Evaluation,
    /// Labels dropped because the party is itself a seed (trivially "linked").
    pub skipped_seed_labels: usize,
    /// Labels dropped because no such party exists.
    pub skipped_unknown_labels: usize,
    /// AUC at each tried damping (other settings unchanged).
    pub damping_sweep: Vec<(f64, Option<f64>)>,
    pub families: Vec<FamilyEffect>,
}

pub const MIN_PER_CLASS_FOR_SUGGESTION: usize = 10;
pub const COMFORTABLE_PER_CLASS: usize = 30;
pub const MIN_TRUE_POSITIVES_FOR_STRONG: usize = 5;

fn same(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs()).max(1e-300)
}

/// Area under the ROC curve by the rank-sum (Mann-Whitney) method; ties count one half.
/// `None` when either class is empty.
pub fn auc(scores: &[(bool, f64)]) -> Option<f64> {
    let p = scores.iter().filter(|(l, _)| *l).count();
    let n = scores.len() - p;
    if p == 0 || n == 0 {
        return None;
    }
    let mut v: Vec<(bool, f64)> = scores.to_vec();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut rank_sum_pos = 0.0;
    let mut i = 0;
    while i < v.len() {
        let mut j = i + 1;
        while j < v.len() && same(v[i].1, v[j].1) {
            j += 1;
        }
        // ranks i+1 ..= j share their average
        let avg = ((i + 1) + j) as f64 / 2.0;
        rank_sum_pos += avg * v[i..j].iter().filter(|(l, _)| *l).count() as f64;
        i = j;
    }
    let (pf, nf) = (p as f64, n as f64);
    Some((rank_sum_pos - pf * (pf + 1.0) / 2.0) / (pf * nf))
}

/// Scores each labelled party and summarises how well `score >= threshold` separates the classes.
pub fn evaluate(scores: &[(bool, f64)], opts: &CalibrationOptions) -> Evaluation {
    let positives = scores.iter().filter(|(l, _)| *l).count();
    let negatives = scores.len() - positives;
    let detected_positives = scores.iter().filter(|(l, s)| *l && *s > 0.0).count();
    let mut ev = Evaluation { positives, negatives, detected_positives, auc: auc(scores), ..Default::default() };

    let mut v: Vec<(bool, f64)> = scores.iter().copied().filter(|(_, s)| *s > 0.0).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (mut tp, mut fp) = (0usize, 0usize);
    let mut i = 0;
    while i < v.len() {
        let t = v[i].1;
        while i < v.len() && same(v[i].1, t) {
            if v[i].0 {
                tp += 1;
            } else {
                fp += 1;
            }
            i += 1;
        }
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if positives > 0 { tp as f64 / positives as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        ev.curve.push(Point { threshold: t, tp, fp, fn_: positives - tp, tn: negatives - fp, precision, recall, f1 });
    }
    ev.best_f1 = ev.curve.iter().copied().max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal));

    if positives < COMFORTABLE_PER_CLASS || negatives < COMFORTABLE_PER_CLASS {
        ev.warnings.push(format!(
            "small sample ({positives} linked, {negatives} not linked): below {COMFORTABLE_PER_CLASS} of either class every figure here is unstable; treat as indicative"
        ));
    }
    if positives > 0 && detected_positives < positives {
        ev.warnings.push(format!(
            "{} of {positives} linked parties have zero exposure (no network path from any seed), so no threshold can catch them; missing links in the data are the likely cause",
            positives - detected_positives
        ));
    }
    if positives < MIN_PER_CLASS_FOR_SUGGESTION || negatives < MIN_PER_CLASS_FOR_SUGGESTION {
        ev.warnings.push(format!("fewer than {MIN_PER_CLASS_FOR_SUGGESTION} of one class: no thresholds suggested"));
        return ev;
    }

    // recall-oriented: weakest observed score that still reaches the recall target, moved down into the
    // gap below it (no labelled party scores in between, so the labelled results are unchanged)
    ev.suggested_moderate = ev.curve.iter().position(|p| p.recall >= opts.recall_target).map(|i| {
        let s = ev.curve[i].threshold;
        let lower = ev.curve.get(i + 1).map(|p| p.threshold).unwrap_or(s / 2.0);
        (s * lower).sqrt()
    });
    if ev.suggested_moderate.is_none() {
        ev.warnings.push(format!(
            "recall {:.0}% is not reachable: the network only reaches {:.0}% of linked parties",
            opts.recall_target * 100.0,
            100.0 * detected_positives as f64 / positives as f64
        ));
    }
    // lowest threshold that is still precise enough, with enough true positives behind it
    ev.suggested_strong = ev.curve.iter().rev().find(|p| p.precision >= opts.precision_target && p.tp >= MIN_TRUE_POSITIVES_FOR_STRONG).map(|p| p.threshold);
    if ev.suggested_strong.is_none() {
        ev.warnings.push(format!(
            "precision {:.0}% with at least {MIN_TRUE_POSITIVES_FOR_STRONG} true positives is not reachable at any threshold",
            opts.precision_target * 100.0
        ));
    }
    if let (Some(m), Some(s)) = (ev.suggested_moderate, ev.suggested_strong) {
        if s < m {
            ev.warnings.push(format!(
                "the precise threshold ({s:.5}) is below the recall threshold ({m:.5}); the two targets pull in opposite directions on this data, so decide between them rather than applying both"
            ));
        }
    }
    ev
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auc_matches_hand_computed_values() {
        // perfectly separated
        assert_eq!(auc(&[(true, 0.9), (true, 0.8), (false, 0.2), (false, 0.0)]), Some(1.0));
        // perfectly inverted
        assert_eq!(auc(&[(true, 0.1), (false, 0.5), (false, 0.7)]), Some(0.0));
        // all tied: chance
        assert_eq!(auc(&[(true, 0.0), (false, 0.0), (true, 0.0), (false, 0.0)]), Some(0.5));
        // 3 pos {0.9,0.5,0.1}, 2 neg {0.6,0.2}: pos>neg pairs: 0.9>both(2) + 0.5>0.2(1) + 0.1>none(0) = 3 of 6
        let a = auc(&[(true, 0.9), (true, 0.5), (true, 0.1), (false, 0.6), (false, 0.2)]).unwrap();
        assert!((a - 0.5).abs() < 1e-12, "{a}");
        // one tie across classes counts half: pos {0.5}, neg {0.5, 0.1} -> (0.5 + 1) / 2
        let a = auc(&[(true, 0.5), (false, 0.5), (false, 0.1)]).unwrap();
        assert!((a - 0.75).abs() < 1e-12, "{a}");
        // one class empty
        assert_eq!(auc(&[(true, 0.5), (true, 0.1)]), None);
        assert_eq!(auc(&[]), None);
    }

    fn labelled(pos: &[f64], neg: &[f64]) -> Vec<(bool, f64)> {
        pos.iter().map(|s| (true, *s)).chain(neg.iter().map(|s| (false, *s))).collect()
    }

    #[test]
    fn curve_counts_are_exact() {
        // positives 0.9 0.5 0.0; negatives 0.7 0.0
        let ev = evaluate(&labelled(&[0.9, 0.5, 0.0], &[0.7, 0.0]), &CalibrationOptions::default());
        assert_eq!((ev.positives, ev.negatives, ev.detected_positives), (3, 2, 2));
        assert_eq!(ev.curve.len(), 3);
        let p = ev.curve[0]; // flag >= 0.9
        assert_eq!((p.threshold, p.tp, p.fp, p.fn_, p.tn), (0.9, 1, 0, 2, 2));
        let p = ev.curve[1]; // flag >= 0.7
        assert_eq!((p.tp, p.fp, p.fn_, p.tn), (1, 1, 2, 1));
        assert!((p.precision - 0.5).abs() < 1e-12 && (p.recall - 1.0 / 3.0).abs() < 1e-12);
        let p = ev.curve[2]; // flag >= 0.5
        assert_eq!((p.tp, p.fp, p.fn_, p.tn), (2, 1, 1, 1));
        assert!((p.f1 - 2.0 * (2.0 / 3.0) * (2.0 / 3.0) / (4.0 / 3.0)).abs() < 1e-12);
        // below the per-class minimum: warned, nothing suggested
        assert!(ev.suggested_moderate.is_none() && ev.suggested_strong.is_none());
        assert!(ev.warnings.iter().any(|w| w.contains("no thresholds suggested")));
        assert!(ev.warnings.iter().any(|w| w.contains("zero exposure")));
    }

    #[test]
    fn thresholds_are_suggested_from_the_targets() {
        // 20 linked at 0.011..0.030 (step .001), 20 unlinked at 0.0001..0.0020 (step .0001) plus one unlinked at 0.0125
        let pos: Vec<f64> = (11..=30).map(|i| i as f64 / 1000.0).collect();
        let mut neg: Vec<f64> = (1..=20).map(|i| i as f64 / 10_000.0).collect();
        neg.push(0.0125);
        let ev = evaluate(&labelled(&pos, &neg), &CalibrationOptions { recall_target: 0.9, precision_target: 0.95 });
        // 2 of the 20*21 linked/unlinked pairs are inverted (the unlinked 0.0125 beats 0.011 and 0.012)
        assert!((ev.auc.unwrap() - (1.0 - 2.0 / 420.0)).abs() < 1e-12, "{:?}", ev.auc);
        // recall 0.9 of 20 = 18 positives: the 18th highest positive is 0.013; the next lower observed score is the
        // unlinked 0.0125, so the threshold sits midway (geometric) between them rather than on 0.013
        let m = ev.suggested_moderate.unwrap();
        assert!(same(m, (0.013f64 * 0.0125).sqrt()), "{m}");
        assert!(m < 0.013 && m > 0.0125);
        // and it classifies the labelled parties exactly as a threshold of 0.013 would
        let flagged = |t: f64| labelled(&pos, &neg).iter().filter(|(_, s)| *s >= t).count();
        assert_eq!(flagged(m), flagged(0.013));
        // precision >= .95 needs the false positive at 0.0125 excluded or diluted:
        // at 0.012: tp=19,fp=1 -> .95 (meets) ; at 0.011: tp=20, fp=1 -> .952 (meets, lower). Lowest meeting is 0.011
        assert!(same(ev.suggested_strong.unwrap(), 0.011), "{:?}", ev.suggested_strong);
        // with a stricter target of 1.0 precision only thresholds above 0.0125 qualify; lowest such is 0.013
        let ev2 = evaluate(&labelled(&pos, &neg), &CalibrationOptions { recall_target: 0.9, precision_target: 1.0 });
        assert!(same(ev2.suggested_strong.unwrap(), 0.013), "{:?}", ev2.suggested_strong);
        // 20 linked parties is under the "comfortable" 30, so the sample-size warning is raised
        assert!(ev.warnings.iter().any(|w| w.contains("small sample")), "{:?}", ev.warnings);
    }

    #[test]
    fn unreachable_targets_say_so_instead_of_guessing() {
        // only half the linked parties are reachable at all
        let pos: Vec<f64> = (0..10).map(|i| if i < 5 { 0.01 * (i + 1) as f64 } else { 0.0 }).collect();
        let neg = vec![0.0; 12];
        let ev = evaluate(&labelled(&pos, &neg), &CalibrationOptions::default());
        assert!(ev.suggested_moderate.is_none());
        assert!(ev.warnings.iter().any(|w| w.contains("not reachable")), "{:?}", ev.warnings);
    }

    #[test]
    fn tied_scores_are_flagged_together() {
        let ev = evaluate(&labelled(&[0.5, 0.5, 0.5], &[0.5]), &CalibrationOptions::default());
        assert_eq!(ev.curve.len(), 1);
        assert_eq!((ev.curve[0].tp, ev.curve[0].fp), (3, 1));
    }
}
