//! Analyst decisions turned into reinforcement signals.

use serde::{Deserialize, Serialize};

/// Terminal outcome of an alert, as decided by a human.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// A suspicious-activity report was filed (case or alert closed with `sar_filed`).
    SarFiled,
    /// The analyst confirmed the activity as suspicious but no SAR was filed (e.g. already reported).
    Confirmed,
    /// The analyst closed it as a false positive (alert `closed_false_positive`, case `no_action`).
    FalsePositive,
}

impl Decision {
    /// Map the alert workflow status; non-terminal statuses carry no signal.
    pub fn from_alert_status(s: &str) -> Option<Self> {
        match s {
            "sar_filed" => Some(Decision::SarFiled),
            "closed_confirmed" => Some(Decision::Confirmed),
            "closed_false_positive" => Some(Decision::FalsePositive),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Decision::SarFiled => "sar_filed",
            Decision::Confirmed => "confirmed",
            Decision::FalsePositive => "false_positive",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "sar_filed" => Some(Decision::SarFiled),
            "confirmed" => Some(Decision::Confirmed),
            "false_positive" => Some(Decision::FalsePositive),
            _ => None,
        }
    }
    pub fn is_positive(self) -> bool {
        !matches!(self, Decision::FalsePositive)
    }
    /// How much the label is trusted. A filed SAR passed a four-eyes review; a bare confirmation did not.
    pub fn reliability(self) -> f64 {
        match self {
            Decision::SarFiled => 1.0,
            Decision::Confirmed => 0.8,
            Decision::FalsePositive => 1.0,
        }
    }
    /// Reinforcement signal in `[-1, 1]` (positive = the system was right to alert).
    pub fn reward(self) -> f64 {
        match self {
            Decision::SarFiled => 1.0,
            Decision::Confirmed => 0.6,
            Decision::FalsePositive => -1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuleHit {
    pub rule: String,
    pub weight: f64,
}

/// One reviewed alert with everything the loop needs to learn from it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FeedbackEvent {
    pub alert_id: i64,
    pub account: i64,
    /// unix seconds of the decision
    pub decided_at: i64,
    /// engine score of the alert (0-100)
    pub score: u8,
    pub rules: Vec<RuleHit>,
    /// graph-model risk for the account at alert time, when a graph model was running
    pub graph_risk: Option<f32>,
    pub decision: Decision,
    pub decided_by: String,
    pub case_id: Option<i64>,
}

impl FeedbackEvent {
    pub fn label(&self) -> bool {
        self.decision.is_positive()
    }
    pub fn weight(&self) -> f64 {
        self.decision.reliability()
    }
    pub fn has_rule(&self, code: &str) -> bool {
        self.rules.iter().any(|r| r.rule == code)
    }
}
