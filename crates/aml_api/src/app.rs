use crate::{auth::Keys, routes};
use aml_engine::Engine;
use axum::{extract::DefaultBodyLimit, middleware, Router};
use sea_orm::DatabaseConnection;
use std::sync::Arc;
use tokio::sync::Mutex;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

#[derive(Clone)]
pub struct AppState {
    pub db: DatabaseConnection,
    /// Rolling-window detection state. Held across "score + persist" so ingestion is serialised
    /// and the in-memory state can never run ahead of what the database has committed.
    pub engine: Arc<Mutex<Engine>>,
    pub keys: Arc<Keys>,
    pub hydrate_days: i64,
    /// Analyst agent: deterministic, or with a language-model sidecar attached (AML_LLM_CMD / AML_LLM_MODEL).
    pub agent: Arc<std::sync::Mutex<aml_agent::Agent>>,
    /// Optional beneficial-ownership records per account (AML_OWNERSHIP_FILE).
    pub ownership: Arc<std::collections::HashMap<i64, aml_agent::evidence::OwnershipFacts>>,
}

pub fn router(st: AppState) -> Router {
    Router::new()
        .merge(routes::router())
        .layer(middleware::from_fn_with_state(st.clone(), crate::auth::middleware))
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .layer(CorsLayer::new())
        .layer(TraceLayer::new_for_http())
        .with_state(st)
}
