//! Engine operations for the learned alert prioritiser (see [`crate::ml`]).
//!
//! Nothing here creates, closes, confirms or re-levels an alert. The model orders the queue and the
//! engine records which model produced which priority, so a later, prospective check is possible.

use super::*;
use crate::ml::{self, band, Example, MlMode, PriorityBand, ShadowReport, TrainOptions, TrainReport, PRIOR_VERSION};
use crate::screening::Subject;

/// One historical decision, e.g. exported from a previous screening tool, used to train before the
/// engine has accumulated enough of its own dispositions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HistoryRecord {
    pub subject: Subject,
    pub list: String,
    pub uid: String,
    /// `true` = confirmed true match; `false` = judged a false positive.
    pub confirmed: bool,
    /// Customer reference: records of one customer stay in one cross-validation fold. Defaults to a unique id.
    #[serde(default)]
    pub customer: Option<u64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HistoryImport {
    pub examples: Vec<ExampleOwned>,
    pub skipped_entry_missing: usize,
    pub skipped_below_review_threshold: usize,
}

/// Serializable twin of [`ml::Example`] (the training-time struct holds a fixed array).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExampleOwned {
    pub features: MatchFeatures,
    pub confirmed: bool,
    pub group: u64,
}

impl ExampleOwned {
    pub fn to_example(&self) -> Example {
        Example::new(&self.features, self.confirmed, self.group)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MlStatus {
    pub mode: MlMode,
    pub active_model: String,
    pub model_is_trained: bool,
    pub low_band_max_priority: f64,
    pub open_alerts: usize,
    pub labelled_alerts: usize,
    pub confirmed: usize,
    pub false_positives: usize,
    pub resolved_without_features: usize,
}

#[derive(Clone, Debug)]
pub struct QueueItem<'a> {
    pub alert: &'a Alert,
    pub band: PriorityBand,
}

impl Compliance {
    /// Priority and producing model for a hit under the current mode. `(None, None)` when the mode is `Off`.
    pub(super) fn priority_for(&self, f: &MatchFeatures) -> (Option<f64>, Option<String>) {
        if self.ml_cfg.mode == MlMode::Off {
            return (None, None);
        }
        match &self.ml_model {
            Some(m) => (Some(m.predict(f)), Some(m.version.clone())),
            None => {
                let m = LogisticModel::prior();
                (Some(m.predict(f)), Some(PRIOR_VERSION.to_string()))
            }
        }
    }

    pub fn ml_model(&self) -> Option<&LogisticModel> {
        self.ml_model.as_ref()
    }

    /// The model currently producing priorities (the unvalidated scorecard prior if none is activated).
    pub fn effective_model(&self) -> LogisticModel {
        self.ml_model.clone().unwrap_or_else(LogisticModel::prior)
    }

    // ---- training data -----------------------------------------------------------------
    /// Examples from this engine's own resolved alerts. `FalsePositive` = 0, `Confirmed` = 1; open and escalated
    /// alerts are not labels. Alerts raised before features were recorded cannot be used (second value).
    pub fn labelled_examples(&self) -> (Vec<Example>, usize) {
        let mut ex = Vec::new();
        let mut without = 0;
        for a in self.alerts.values() {
            let y = match a.status {
                AlertStatus::Confirmed => true,
                AlertStatus::FalsePositive => false,
                _ => continue,
            };
            match &a.features {
                Some(f) => ex.push(Example::new(f, y, a.root_party)),
                None => without += 1,
            }
        }
        (ex, without)
    }

    /// Recompute features for historical decisions against the **currently loaded** lists. Records whose list
    /// entry is gone, or that would not raise an alert today, are skipped and counted.
    pub fn import_history(&self, records: &[HistoryRecord]) -> HistoryImport {
        let mut out = HistoryImport::default();
        let cfg = ScreeningConfig { max_hits: usize::MAX, ..self.screening_cfg.clone() };
        for (i, r) in records.iter().enumerate() {
            if self.index.entry(&r.list, &r.uid).is_none() {
                out.skipped_entry_missing += 1;
                continue;
            }
            let res = self.index.screen(&r.subject, &cfg);
            match res.hits.into_iter().find(|h| h.list == r.list && h.uid == r.uid) {
                Some(h) => out.examples.push(ExampleOwned {
                    features: h.features,
                    confirmed: r.confirmed,
                    // offset keeps synthetic group ids clear of real party ids
                    group: r.customer.unwrap_or(1_000_000_000 + i as u64),
                }),
                None => out.skipped_below_review_threshold += 1,
            }
        }
        out
    }

    // ---- training and activation ------------------------------------------------------------
    /// Train and cross-validate on the engine's own labels plus any `extra` examples. If `activate` is set and **every
    /// gate passes**, the model is installed and open alerts are re-prioritised. Otherwise nothing changes except the
    /// audit entry. MLRO only.
    pub fn train_match_model(
        &mut self,
        extra: &[Example],
        opts: &TrainOptions,
        activate: bool,
        actor: &Actor,
        now: DateTime<Utc>,
    ) -> Result<(TrainReport, bool)> {
        actor.require(Role::Mlro, "training or activating the match model")?;
        let (mut ex, _) = self.labelled_examples();
        ex.extend(extra.iter().cloned());
        let report = ml::train(&ex, opts)?;
        let activated = activate && report.passed;
        self.log(
            now,
            &actor.name,
            "ml.train",
            &format!("model:{}", report.candidate.version),
            json!({
                "examples": report.examples, "positives": report.positives, "lambda": report.lambda,
                "cv_auc": report.model.auc, "rule_auc": report.rule.auc, "gain_ci": report.auc_gain_vs_rule,
                "gates_passed": report.passed, "activation_requested": activate, "activated": activated,
                "fingerprint": report.candidate.fingerprint(),
            }),
        );
        if activated {
            let m = report.candidate.clone();
            m.validate()?;
            self.ml_model = Some(m);
            if self.ml_cfg.mode == MlMode::Off {
                self.ml_cfg.mode = MlMode::Shadow;
            }
            self.rescore_open_alerts(now);
        }
        Ok((report, activated))
    }

    /// Re-run the active model on open alerts (history line added; resolved alerts keep the priority they had).
    pub fn rescore_open_alerts(&mut self, now: DateTime<Utc>) -> usize {
        let today = now.date_naive();
        let ids: Vec<u64> = self.alerts.values().filter(|a| a.status.is_unresolved() && a.features.is_some()).map(|a| a.id).collect();
        let mut n = 0;
        for id in ids {
            let f = self.alerts[&id].features.clone().expect("filtered");
            let (p, v) = self.priority_for(&f);
            let a = self.alerts.get_mut(&id).expect("exists");
            if a.model_version != v {
                a.history.push(format!("{today}: priority re-scored by {}", v.as_deref().unwrap_or("none")));
            }
            a.priority = p;
            a.model_version = v;
            n += 1;
        }
        n
    }

    /// Set the mode and the Low-band cut-off. `Active` requires an activated, validated model. MLRO only.
    pub fn set_ml_mode(&mut self, mode: MlMode, low_band_max_priority: Option<f64>, actor: &Actor, now: DateTime<Utc>) -> Result<()> {
        actor.require(Role::Mlro, "changing the prioritisation mode")?;
        if mode == MlMode::Active && self.ml_model.is_none() {
            return Err(Error::Invalid(
                "Active needs a trained model that passed the validation gates (cddctl ml-train --activate); the built-in scorecard is unvalidated".into(),
            ));
        }
        if let Some(c) = low_band_max_priority {
            if !(0.0..1.0).contains(&c) {
                return Err(Error::Invalid("low-band cut-off must be in [0, 1)".into()));
            }
            self.ml_cfg.low_band_max_priority = c;
        }
        self.ml_cfg.mode = mode;
        let low = self.ml_cfg.low_band_max_priority;
        self.log(now, &actor.name, "ml.mode", "ml", json!({"mode": mode, "low_band_max_priority": low}));
        if mode != MlMode::Off {
            self.rescore_open_alerts(now);
        }
        Ok(())
    }

    // ---- reading the queue -------------------------------------------------------------------
    pub fn alert_band(&self, a: &Alert) -> PriorityBand {
        match &a.features {
            Some(f) => band(a.level == HitLevel::High, f, a.priority, &self.ml_cfg),
            None => {
                if a.level == HitLevel::High {
                    PriorityBand::Critical
                } else {
                    PriorityBand::Standard
                }
            }
        }
    }

    /// Unresolved alerts in reading order. Bands first (Critical, Standard, Low); inside a band the learned priority
    /// in `Active` mode, the rule score otherwise. Alerts without a priority follow those with one.
    pub fn alert_queue(&self) -> Vec<QueueItem<'_>> {
        let active = self.ml_cfg.mode == MlMode::Active;
        let mut v: Vec<QueueItem<'_>> =
            self.alerts.values().filter(|a| a.status.is_unresolved()).map(|a| QueueItem { alert: a, band: self.alert_band(a) }).collect();
        v.sort_by(|x, y| {
            let key = |q: &QueueItem<'_>| -> (u8, f64) {
                if active {
                    match q.alert.priority {
                        Some(p) => (1, p),
                        None => (0, q.alert.score),
                    }
                } else {
                    (1, q.alert.score)
                }
            };
            let (kx, ky) = (key(x), key(y));
            x.band
                .cmp(&y.band)
                .then(ky.0.cmp(&kx.0))
                .then(ky.1.partial_cmp(&kx.1).unwrap_or(std::cmp::Ordering::Equal))
                .then(x.alert.id.cmp(&y.alert.id))
        });
        v
    }

    /// Why an alert has the priority it has: per-feature contributions to the logit under the model that scored it.
    pub fn explain_alert(&self, id: u64) -> Result<(Vec<(&'static str, f64)>, f64, String)> {
        let a = self.alerts.get(&id).ok_or_else(|| Error::NotFound(format!("alert {id}")))?;
        let f = a.features.as_ref().ok_or_else(|| Error::Invalid(format!("alert {id} predates feature capture")))?;
        let m = match (&self.ml_model, a.model_version.as_deref()) {
            (Some(m), Some(v)) if m.version == v => m.clone(),
            _ => LogisticModel::prior(),
        };
        Ok((m.explain(f), m.bias, m.version.clone()))
    }

    /// Prospective evaluation: priorities stored **at alert time** versus the rule score, on alerts resolved since.
    /// `version` restricts to alerts scored by one model.
    pub fn shadow_report(&self, version: Option<&str>) -> ShadowReport {
        let items: Vec<(bool, f64, f64)> = self
            .alerts
            .values()
            .filter_map(|a| {
                let y = match a.status {
                    AlertStatus::Confirmed => true,
                    AlertStatus::FalsePositive => false,
                    _ => return None,
                };
                let p = a.priority?;
                if version.map_or(false, |v| a.model_version.as_deref() != Some(v)) {
                    return None;
                }
                Some((y, p, a.score))
            })
            .collect();
        ml::shadow_eval(&items)
    }

    pub fn ml_status(&self) -> MlStatus {
        let (ex, without) = self.labelled_examples();
        MlStatus {
            mode: self.ml_cfg.mode,
            active_model: self.ml_model.as_ref().map_or_else(|| format!("{PRIOR_VERSION} (hand-set, unvalidated)"), |m| m.version.clone()),
            model_is_trained: self.ml_model.is_some(),
            low_band_max_priority: self.ml_cfg.low_band_max_priority,
            open_alerts: self.alerts.values().filter(|a| a.status.is_unresolved()).count(),
            labelled_alerts: ex.len(),
            confirmed: ex.iter().filter(|e| e.y).count(),
            false_positives: ex.iter().filter(|e| !e.y).count(),
            resolved_without_features: without,
        }
    }
}
