//! Customer Due Diligence tracking and sanctions-list screening.
//!
//! * [`lists`] — parsers for OFAC CSV, UN XML, EU FSF XML and canonical JSONL.
//! * [`screening`] — indexed fuzzy screening with DOB/nationality/identifier corroboration.
//! * [`cdd`] — parties, ownership (incl. the 50% rule), risk rating, document rules.
//! * [`engine`] — workflows: alerts, four-eyes cases, list updates, persistence.
//! * [`hypernet`] — spectral hypergraph analysis of the customer network (indirect sanctions exposure, structure).
//! * [`links`] — shared addresses, phones, emails, bank accounts and registered agents (network only, never screening).
//! * [`calibrate`] — measure the network thresholds against labelled cases.
//! * [`ml`] — learned alert prioritisation (features, validated logistic model, shadow evaluation); orders alerts, never decides them.
//! * [`audit`] — tamper-evident hash-chained audit log.

pub mod audit;
pub mod calibrate;
pub mod cdd;
pub mod country;
pub mod engine;
pub mod error;
pub mod hypernet;
pub mod links;
pub mod lists;
pub mod ml;
pub mod model;
pub mod screening;
pub mod similarity;
pub mod text;
