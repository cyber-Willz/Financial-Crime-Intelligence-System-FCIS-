//! Beneficial-ownership analysis for the AML system, built on the vendored `cdd_sanctions` engine.
//!
//! * [`OwnershipAnalysis`]: UBOs, opacity, 50% rule, sanctions exposure, ownership groups per account holder.
//! * [`BoMonitor`]: causal transaction rules that use that context.
//! * [`OwnershipAnalysis::graph_map`]: the per-account context consumed by the `aml_graph` model.
//! * [`synth`]: a synthetic ownership world (not evidence about real laundering).
pub mod analysis;
pub mod monitor;
pub mod registry;
pub mod synth;
pub use analysis::*;
pub use monitor::{fuse, BoMonitor, MonitorConfig};
pub use registry::AccountRegistry;
