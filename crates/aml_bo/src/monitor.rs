//! Streaming, causal transaction rules that use beneficial-ownership context.
//!
//! Each rule fires from information available when the transaction is processed (ownership as recorded in the
//! CDD engine, plus a few days of account history). Scores are combined by noisy-OR like the AML engine's.
//! Thresholds are untuned defaults.

use crate::analysis::OwnershipAnalysis;
use aml_graph::GTxn;
use cdd_sanctions::cdd::PartyKind;
use std::collections::{HashMap, VecDeque};

const DAY: i64 = 86_400;

#[derive(Clone, Debug)]
pub struct MonitorConfig {
    pub opaque_min: f64,
    pub opaque_amount: f64,
    pub pass_window_days: i64,
    pub pass_min_inflow: f64,
    pub pass_ratio: f64,
    pub group_window_days: i64,
    pub structuring_lo: f64,
    pub structuring_hi: f64,
    pub exposure_min: f64,
}
impl Default for MonitorConfig {
    fn default() -> Self {
        MonitorConfig { opaque_min: 0.55, opaque_amount: 3000.0, pass_window_days: 3, pass_min_inflow: 10_000.0, pass_ratio: 0.6, group_window_days: 7, structuring_lo: 7000.0, structuring_hi: 10_000.0, exposure_min: 0.2 }
    }
}

pub struct BoMonitor<'a> {
    an: &'a OwnershipAnalysis,
    cfg: MonitorConfig,
    inflow: HashMap<i64, VecDeque<(i64, f64)>>,
    group_events: HashMap<u32, VecDeque<i64>>,
    owner_events: HashMap<u64, VecDeque<i64>>,
}

fn or(scores: &[f64]) -> f64 {
    1.0 - scores.iter().fold(1.0, |a, s| a * (1.0 - s))
}

impl<'a> BoMonitor<'a> {
    pub fn new(an: &'a OwnershipAnalysis, cfg: MonitorConfig) -> Self {
        BoMonitor { an, cfg, inflow: HashMap::new(), group_events: HashMap::new(), owner_events: HashMap::new() }
    }

    /// Score one transaction (call in time order). Returns the combined score and the rules that fired.
    pub fn score(&mut self, t: &GTxn) -> (f32, Vec<&'static str>) {
        let an = self.an;
        let cfg = &self.cfg;
        let (ps, pr) = (an.registry.party_of(t.sender), an.registry.party_of(t.receiver));
        let (bs, br) = (ps.and_then(|p| an.parties.get(&p)), pr.and_then(|p| an.parties.get(&p)));
        let mut hits: Vec<(f64, &'static str)> = vec![];

        for b in [bs, br].into_iter().flatten() {
            if b.blocked_ownership >= an.cfg.sanctioned_ownership {
                hits.push((0.97, "blocked_ownership_50pct"));
            } else if b.blocked_ownership >= 0.25 {
                hits.push((0.55, "material_sanctioned_stake"));
            }
            if b.exposure >= cfg.exposure_min {
                hits.push((0.35, "network_exposure"));
            }
            if b.opacity >= cfg.opaque_min && t.amount >= cfg.opaque_amount {
                hits.push((0.35 + 0.4 * b.opacity, "opaque_owner"));
            }
            if b.fragmented && t.amount >= cfg.opaque_amount {
                hits.push((0.35, "fragmented_ownership"));
            }
        }
        // pass-through of an account whose ownership is opaque or part of a group
        if let Some(b) = bs {
            let q = self.inflow.entry(t.sender).or_default();
            while q.front().map_or(false, |(ts, _)| *ts < t.ts - cfg.pass_window_days * DAY) {
                q.pop_front();
            }
            let inflow: f64 = q.iter().map(|x| x.1).sum();
            if inflow >= cfg.pass_min_inflow && t.amount >= cfg.pass_ratio * inflow && (b.opacity >= 0.3 || b.no_owner_identified || (b.group != 0 && b.kind == PartyKind::Entity)) {
                hits.push((0.6, "ownership_pass_through"));
            }
        }
        // circulation inside one ownership group
        if let (Some(a), Some(b)) = (bs, br) {
            if a.party != b.party && a.group != 0 && a.group == b.group {
                let q = self.group_events.entry(a.group).or_default();
                while q.front().map_or(false, |ts| *ts < t.ts - cfg.group_window_days * DAY) {
                    q.pop_front();
                }
                q.push_back(t.ts);
                if q.len() >= 2 {
                    hits.push(((0.2 + 0.12 * (q.len() as f64 - 1.0)).min(0.8), "group_circulation"));
                }
            }
            // owners paying just below the reporting threshold into their own company
            if b.kind == PartyKind::Entity && b.related_owners.contains(&a.party) && t.amount >= cfg.structuring_lo && t.amount < cfg.structuring_hi {
                let q = self.owner_events.entry(b.party).or_default();
                while q.front().map_or(false, |ts| *ts < t.ts - cfg.group_window_days * DAY) {
                    q.pop_front();
                }
                q.push_back(t.ts);
                if q.len() >= 3 {
                    hits.push(((0.25 + 0.1 * q.len() as f64).min(0.75), "owner_structuring"));
                }
            }
        }
        // history is updated after scoring so only earlier transactions count
        self.inflow.entry(t.receiver).or_default().push_back((t.ts, t.amount));

        let s = or(&hits.iter().map(|h| h.0).collect::<Vec<_>>());
        let mut names: Vec<&'static str> = hits.iter().map(|h| h.1).collect();
        names.sort_unstable();
        names.dedup();
        (s as f32, names)
    }

    pub fn score_all(an: &'a OwnershipAnalysis, txns: &[GTxn], cfg: MonitorConfig) -> Vec<f32> {
        let mut m = Self::new(an, cfg);
        txns.iter().map(|t| m.score(t).0).collect()
    }
}

/// noisy-OR of two scores
pub fn fuse(a: f32, b: f32) -> f32 {
    1.0 - (1.0 - a) * (1.0 - b)
}
