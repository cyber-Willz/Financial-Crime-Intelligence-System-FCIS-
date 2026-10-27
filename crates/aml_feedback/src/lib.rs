//! # aml_feedback — the feedback half of the design guide
//!
//! ```text
//! Analyst Decision ─► Case Outcome ─► SAR Filed? ─► False Positive? ─► Reinforcement Signal
//!                                                                              │
//!        ┌───────────────────────┬─────────────────────┬───────────────────────┴──┐
//!        ▼                       ▼                     ▼                          ▼
//!   rules::analyse        calibrate::advise      memory::OutcomeMemory      tune::tune_head
//!   "Recommend Rule       "Calibrate Risk        per-account prior used     "Update Graph Attention"
//!    Tuning"               Thresholds"           for queue ordering         (guarded head fine-tune;
//!                                                ("Update Memory States")    RNN states stay valid)
//! ```
//!
//! Everything here is a *recommendation* or a *ranking aid*. Nothing changes a rule weight, an alert
//! threshold or an alert's status by itself; an administrator applies recommendations through the API,
//! and every application is audited. Outcomes exist only for alerts that were raised and reviewed, so
//! all statistics are conditional on "was alerted" (see each module for what that does and does not allow).

pub mod calibrate;
pub mod memory;
pub mod rules;
pub mod signal;
pub mod stats;
pub mod tune;

pub use calibrate::{advise, Calibrator, ThresholdAdvice, ThresholdConfig};
pub use memory::{Adjusted, MemoryConfig, OutcomeMemory};
pub use rules::{analyse, Action, RuleLearnerConfig, RuleRecommendation, RuleReport, RuleStat};
pub use signal::{Decision, FeedbackEvent, RuleHit};
pub use tune::{tune_head, TuneConfig, TuneReport};
