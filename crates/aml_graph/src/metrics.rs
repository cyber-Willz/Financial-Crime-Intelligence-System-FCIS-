//! Ranking metrics for imbalanced transaction scoring.

use serde::Serialize;

/// Average precision (area under the precision-recall curve, step interpolation). Ties are broken
/// pessimistically (negatives first) so a constant score cannot look good.
pub fn average_precision(scored: &[(f32, bool)]) -> f64 {
    let p = scored.iter().filter(|s| s.1).count();
    if p == 0 {
        return 0.0;
    }
    let mut v: Vec<(f32, bool)> = scored.to_vec();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let (mut tp, mut ap) = (0usize, 0.0);
    for (i, s) in v.iter().enumerate() {
        if s.1 {
            tp += 1;
            ap += tp as f64 / (i + 1) as f64;
        }
    }
    ap / p as f64
}

/// ROC AUC via the rank-sum statistic (ties get half credit).
pub fn roc_auc(scored: &[(f32, bool)]) -> f64 {
    let (p, n) = (scored.iter().filter(|s| s.1).count(), scored.iter().filter(|s| !s.1).count());
    if p == 0 || n == 0 {
        return 0.5;
    }
    let mut v: Vec<(f32, bool)> = scored.to_vec();
    v.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let (mut rank_sum, mut i) = (0.0, 0usize);
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j].0 == v[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // ranks are 1-based
        rank_sum += avg_rank * v[i..j].iter().filter(|s| s.1).count() as f64;
        i = j;
    }
    (rank_sum - p as f64 * (p as f64 + 1.0) / 2.0) / (p as f64 * n as f64)
}

#[derive(Debug, Clone, Serialize)]
pub struct Operating {
    pub alerts: usize,
    pub tp: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

/// Precision/recall when the `k` highest-scored transactions are alerted.
pub fn at_top_k(scored: &[(f32, bool)], k: usize) -> Operating {
    let p = scored.iter().filter(|s| s.1).count();
    let mut v: Vec<(f32, bool)> = scored.to_vec();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let k = k.min(v.len());
    let tp = v[..k].iter().filter(|s| s.1).count();
    let precision = tp as f64 / k.max(1) as f64;
    let recall = tp as f64 / p.max(1) as f64;
    Operating { alerts: k, tp, precision, recall, f1: 2.0 * precision * recall / (precision + recall).max(1e-12) }
}

/// Best F1 over all thresholds.
pub fn best_f1(scored: &[(f32, bool)]) -> Operating {
    let p = scored.iter().filter(|s| s.1).count();
    let mut v: Vec<(f32, bool)> = scored.to_vec();
    v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal).then(a.1.cmp(&b.1)));
    let (mut tp, mut best) = (0usize, Operating { alerts: 0, tp: 0, precision: 0.0, recall: 0.0, f1: 0.0 });
    for (i, s) in v.iter().enumerate() {
        if s.1 {
            tp += 1;
        }
        // only evaluate at the end of a tie group
        if i + 1 < v.len() && v[i + 1].0 == s.0 {
            continue;
        }
        let precision = tp as f64 / (i + 1) as f64;
        let recall = tp as f64 / p.max(1) as f64;
        let f1 = 2.0 * precision * recall / (precision + recall).max(1e-12);
        if f1 > best.f1 {
            best = Operating { alerts: i + 1, tp, precision, recall, f1 };
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_and_random_rankings() {
        let perfect: Vec<(f32, bool)> = (0..100).map(|i| (i as f32, i >= 90)).collect();
        assert!((average_precision(&perfect) - 1.0).abs() < 1e-12);
        assert!((roc_auc(&perfect) - 1.0).abs() < 1e-12);
        let constant: Vec<(f32, bool)> = (0..100).map(|i| (0.5, i >= 90)).collect();
        assert!((roc_auc(&constant) - 0.5).abs() < 1e-12);
        assert!(average_precision(&constant) < 0.15, "constant score must not look good: {}", average_precision(&constant));
    }

    #[test]
    fn top_k_and_best_f1() {
        let v: Vec<(f32, bool)> = vec![(0.9, true), (0.8, true), (0.7, false), (0.6, true), (0.1, false), (0.05, false)];
        let o = at_top_k(&v, 3);
        assert_eq!((o.tp, o.alerts), (2, 3));
        let b = best_f1(&v);
        assert!((b.f1 - 0.857142857).abs() < 1e-6, "{b:?}"); // top-4: p=.75 r=1
    }
}
