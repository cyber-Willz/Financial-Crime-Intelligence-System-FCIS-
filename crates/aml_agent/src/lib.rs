//! # aml_agent — the bidirectional LLM analyst agent
//!
//! ```text
//!  rule activations ┐                                        ┌─► alert explanation        ┌─► graph queries
//!  alert scores     │                                        │   typology identification  │   (whitelisted, bounded,
//!  graph risk       ├─► Evidence ─► FactBook [F1] [F2] … ─►  │   case prioritisation      │    read-only)
//!  ownership        │   (every claim has an id)              │   dynamic rule suggestions │         ▲
//!  sanctions        │                                        │   SAR / narrative drafting ▼         │
//!  analyst notes    │                                        └── Agent ◄──── optional local LLM ────┘
//!  SAR outcomes     ┘                                              │      (burn-llm sidecar)
//!                                                                  ▼
//!                                              grounding gate: unknown citation / number / accusation ⇒ rejected
//! ```
//!
//! **Down:** the system hands the agent an [`Evidence`] packet; it becomes a numbered fact list.
//! **Up:** the agent (the LLM, or the built-in planner when there is none) asks for more by emitting
//! `QUERY …` lines, which [`query`] parses against a closed grammar and runs read-only and bounded over the
//! transaction graph; the answers become new facts.
//!
//! The built-in [`agent::Agent`] works with no model at all — every function returns grounded text built from
//! the facts. A local model (the `llm_sidecar` process) may *choose queries* and *phrase the answer*; whatever it writes is
//! checked by [`grounding`] and replaced by the deterministic text if it cites a fact that does not exist,
//! uses a number that no fact contains, or states guilt. The agent never changes an alert, a case, a rule or
//! a threshold, and a SAR draft is always marked as a draft for MLRO review.

pub mod agent;
pub mod evidence;
pub mod grounding;
pub mod llm;
pub mod priority;
pub mod query;
pub mod suggest;
pub mod typology;

pub use agent::{risk_summary, RiskSummary, Agent, AgentConfig, CaseFacts, Explanation, Investigation, Provenance, SarDraft, SarSection};
pub use evidence::{Evidence, Fact, FactBook};
pub use grounding::{check as check_grounding, GroundingReport};
pub use llm::{Llm, ProcessLlm, ScriptedLlm};
pub use priority::{rank, PriorityConfig, QueueItem, Ranked};
pub use query::{parse_query, run_query, Flow, GraphSource, MemGraph, Query, QueryResult};
pub use suggest::{suggestions, Apply, Suggestion};
pub use typology::{identify, TypologyHit};
