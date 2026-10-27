use super::page;
use crate::{
    app::AppState,
    auth::Actor,
    error::{ApiError, ApiResult},
    ingest::{self, Decision, TxnRequest, MAX_BATCH},
};
use aml_entity::transaction as t;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Extension, Json,
};
use chrono::{DateTime, Utc};
use sea_orm::{ColumnTrait, Condition, EntityTrait, QueryFilter, QueryOrder, QuerySelect};
use serde::Deserialize;
use serde_json::json;

pub async fn create(State(st): State<AppState>, Extension(a): Extension<Actor>, Json(req): Json<TxnRequest>) -> ApiResult<(StatusCode, Json<Decision>)> {
    a.require_ingest()?;
    let mut d = ingest::ingest(&st, vec![req]).await?;
    Ok((StatusCode::CREATED, Json(d.remove(0))))
}

#[derive(Deserialize)]
pub struct BatchReq {
    transactions: Vec<TxnRequest>,
}

pub async fn batch(State(st): State<AppState>, Extension(a): Extension<Actor>, Json(b): Json<BatchReq>) -> ApiResult<(StatusCode, Json<serde_json::Value>)> {
    a.require_ingest()?;
    if b.transactions.len() > MAX_BATCH {
        return Err(ApiError::BadRequest(format!("max {MAX_BATCH} transactions per batch")));
    }
    let d = ingest::ingest(&st, b.transactions).await?;
    let flagged = d.iter().filter(|x| x.flagged).count();
    Ok((StatusCode::CREATED, Json(json!({"count": d.len(), "flagged": flagged, "decisions": d}))))
}

#[derive(Deserialize)]
pub struct Q {
    account: Option<i64>,
    flagged: Option<bool>,
    min_score: Option<i16>,
    from: Option<DateTime<Utc>>,
    to: Option<DateTime<Utc>>,
    payment_type: Option<String>,
    limit: Option<u64>,
    offset: Option<u64>,
}

pub async fn list(State(st): State<AppState>, Query(q): Query<Q>) -> ApiResult<Json<Vec<t::Model>>> {
    let (limit, offset) = page(q.limit, q.offset);
    let mut c = Condition::all();
    if let Some(a) = q.account {
        c = c.add(Condition::any().add(t::Column::SenderAccount.eq(a)).add(t::Column::ReceiverAccount.eq(a)));
    }
    if let Some(f) = q.flagged {
        c = c.add(t::Column::Flagged.eq(f));
    }
    if let Some(m) = q.min_score {
        c = c.add(t::Column::RiskScore.gte(m));
    }
    if let Some(f) = q.from {
        c = c.add(t::Column::OccurredAt.gte(f));
    }
    if let Some(x) = q.to {
        c = c.add(t::Column::OccurredAt.lt(x));
    }
    if let Some(p) = q.payment_type {
        c = c.add(t::Column::PaymentType.eq(p.to_ascii_uppercase()));
    }
    let rows = t::Entity::find()
        .filter(c)
        .order_by_desc(t::Column::OccurredAt)
        .order_by_desc(t::Column::Id)
        .limit(limit)
        .offset(offset)
        .all(&st.db)
        .await?;
    Ok(Json(rows))
}

pub async fn get_one(State(st): State<AppState>, Path(id): Path<i64>) -> ApiResult<Json<t::Model>> {
    t::Entity::find_by_id(id).one(&st.db).await?.map(Json).ok_or_else(|| ApiError::NotFound(format!("transaction {id}")))
}
