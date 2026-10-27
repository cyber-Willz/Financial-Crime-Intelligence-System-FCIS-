use crate::{app::AppState, auth::Actor, error::ApiResult};
use axum::{
    routing::{delete, get, post, put},
    Extension, Json, Router,
};
use aml_entity::audit_log;
use axum::extract::{Query, State};
use chrono::Utc;
use sea_orm::{ActiveModelTrait, EntityTrait, QueryFilter, QueryOrder, QuerySelect, Set, ColumnTrait};
use serde::Deserialize;
use serde_json::{json, Value};

pub mod accounts;
pub mod alerts;
pub mod cases;
pub mod customers;
pub mod rules;
pub mod stats;
pub mod tx;
pub mod watchlist;

pub fn page(limit: Option<u64>, offset: Option<u64>) -> (u64, u64) {
    (limit.unwrap_or(50).clamp(1, 500), offset.unwrap_or(0))
}

pub async fn audit(st: &AppState, actor: &Actor, action: &str, entity: &str, id: impl ToString, detail: Value) {
    let r = audit_log::ActiveModel {
        actor: Set(actor.name.clone()),
        action: Set(action.into()),
        entity_type: Set(entity.into()),
        entity_id: Set(Some(id.to_string())),
        detail: Set(detail),
        at: Set(Utc::now()),
        ..Default::default()
    }
    .insert(&st.db)
    .await;
    if let Err(e) = r {
        tracing::error!(error = ?e, "audit write failed");
    }
}

#[derive(Deserialize)]
struct AuditQ {
    entity_type: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

async fn audit_list(State(st): State<AppState>, Query(q): Query<AuditQ>) -> ApiResult<Json<Vec<audit_log::Model>>> {
    let (limit, offset) = page(q.limit, q.offset);
    let mut s = audit_log::Entity::find();
    if let Some(t) = q.entity_type {
        s = s.filter(audit_log::Column::EntityType.eq(t));
    }
    Ok(Json(s.order_by_desc(audit_log::Column::Id).limit(limit).offset(offset).all(&st.db).await?))
}

async fn health(State(st): State<AppState>) -> (axum::http::StatusCode, Json<Value>) {
    let db_ok = st.db.ping().await.is_ok();
    let accounts = st.engine.lock().await.account_count();
    let code = if db_ok { axum::http::StatusCode::OK } else { axum::http::StatusCode::SERVICE_UNAVAILABLE };
    (code, Json(json!({"status": if db_ok {"ok"} else {"degraded"}, "database": db_ok, "engine_accounts": accounts})))
}

pub fn router() -> Router<AppState> {
    let api = Router::new()
        .route("/transactions", post(tx::create).get(tx::list))
        .route("/transactions/batch", post(tx::batch))
        .route("/transactions/{id}", get(tx::get_one))
        .route("/accounts", get(accounts::list))
        .route("/accounts/{id}", get(accounts::get_one))
        .route("/accounts/{id}/transactions", get(accounts::transactions))
        .route("/accounts/{id}/network", get(accounts::network))
        .route("/alerts", get(alerts::list))
        .route("/alerts/{id}", get(alerts::get_one).patch(alerts::update))
        .route("/cases", post(cases::create).get(cases::list))
        .route("/cases/{id}", get(cases::get_one).patch(cases::update))
        .route("/cases/{id}/notes", post(cases::add_note))
        .route("/cases/{id}/alerts", post(cases::add_alerts))
        .route("/cases/{id}/close", post(cases::close))
        .route("/customers", get(customers::list))
        .route("/customers/{cref}", get(customers::get_one).put(customers::put))
        .route("/rules", get(rules::list))
        .route("/rules/import", post(rules::import))
        .route("/rules/{code}", put(rules::update))
        .route("/settings", put(rules::settings))
        .route("/jurisdictions", get(rules::jurisdictions))
        .route("/jurisdictions/{cc}", put(rules::set_jurisdiction))
        .route("/watchlist", get(watchlist::list).post(watchlist::add))
        .route("/watchlist/{account}", delete(watchlist::remove))
        .route("/stats/overview", get(stats::overview))
        .route("/alerts/queue", get(crate::feedback::queue))
        .route("/feedback/summary", get(crate::feedback::summary))
        .route("/feedback/recompute", post(crate::feedback::recompute))
        .route("/feedback/recommendations", get(crate::feedback::list_recommendations))
        .route("/feedback/recommendations/{id}/apply", post(crate::feedback::apply))
        .route("/feedback/recommendations/{id}/reject", post(crate::feedback::reject))
        .route("/agent/status", get(crate::agent_api::status))
        .route("/agent/summary", get(crate::agent_api::summary))
        .route("/agent/investigate", post(crate::agent_api::investigate))
        .route("/agent/alerts/{id}/explain", post(crate::agent_api::explain))
        .route("/agent/cases/{id}/sar-draft", post(crate::agent_api::sar_draft))
        .route("/audit", get(audit_list));
    Router::new().route("/health", get(health)).nest("/api/v1", api)
}

#[allow(unused)]
fn _x(_: Extension<Actor>) {}
