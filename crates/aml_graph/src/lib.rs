//! # aml_graph — connectome-informed recurrent graph neural network for AML
//!
//! ```text
//! Transaction Graph ─► Temporal Hypergraph ─► Recurrent Memory ─► Graph Neural Network ─► AML Risk
//!   (accounts, flows)    (daily snapshots,      (one connectome-RNN     (message passing over     (per-transaction
//!                         in/out-star            state per account,      the memory read-outs)     probability)
//!                         hyperedges)            Dale's law, delays)
//! ```
//!
//! * [`window`] — a time window as a directed graph plus star hyperedges; exact adjoint operators; the
//!   non-backtracking walk mass that exposes chains and cycles.
//! * [`features`] — causal node/transaction features and the window pipeline.
//! * [`memory`] — the connectome RNN stepped per account (event-driven) with exact BPTT.
//! * [`model`] — the four-stage network, loss, back-propagation and persistence.
//! * [`train`] — truncated-BPTT training, time-based split, comparison with the rule engine at equal alert volume.
//! * [`scorer`] — streaming inference (`GraphScorer`).
//! * [`ownership`] — beneficial-ownership context (who stands behind each account, ownership groups); supplied by `aml_bo`.
//! * [`rules`] — attaches the rule engine's causal and retro-linked scores to a stream.
//! * [`synth`] — a synthetic world for tests and demos (not evidence about real laundering).
//!
//! The vendored `human_connectome_rnn` crate supplies the wiring, delays, E/I structure, calibration and
//! stability control; this crate never modifies it.

pub mod features;
pub mod memory;
pub mod metrics;
pub mod model;
pub mod ownership;
pub mod rules;
pub mod scorer;
pub mod synth;
pub mod train;
pub mod types;
pub mod window;

pub use features::{FeatureConfig, Pipeline, RiskTable, WindowData, EDGE_FEATS, NODE_FEATS};
pub use model::{ChunkOut, CirGnn, CirGnnConfig, Grad, LossCfg, MemStore, Mode};
pub use scorer::{GraphScorer, WindowScores};
pub use train::{train_model, TrainConfig, TrainReport};
pub use types::GTxn;
pub use ownership::{OwnerInfo, OwnershipMap, Relation};
pub use window::WindowGraph;
