//! `aml_engine` — a database-free, streaming transaction-monitoring engine.
//!
//! The engine keeps compact per-account rolling windows in memory and scores each
//! transaction against a configurable rule set (typologies from FATF/FinCEN-style
//! guidance and the SAML-D dataset). The same engine is used by the HTTP service,
//! the bulk importer and the offline evaluator, so what you measure is what you run.

pub mod config;
pub mod engine;
pub mod saml;
pub mod types;

pub use config::{default_config, EngineConfig, RuleCfg};
pub use engine::Engine;
pub use types::{is_trade_sector, Country, CustomerRisk, Evaluation, Hit, Linked, PaymentType, Profile, Severity, Side, TxnIn};
