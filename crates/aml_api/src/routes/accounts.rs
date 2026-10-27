use super::page;
use crate::{app::AppState, error::{ApiError, ApiResult}};
use aml_entity::{account, alert, transaction as t};
use axum::{extract::{Path, Query, State}, Json};
use sea_orm::{
    sqlx::{self, Row},
    ColumnTrait, Condition, EntityTrait, QueryFilter, QueryOrder, QuerySelect,
};
use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Deserialize)]
pub struct ListQ {
    min_risk: Option<i16>,
    limit: Option<u64>,
    offset: Option<u64>,
}

pub async fn list(State(st): State<AppState>, Query(q): Query<ListQ>) -> ApiResult<Json<Vec<account::Model>>> {
    let (limit, offset) = page(q.limit, q.offset);
    let rows = account::Entity::find()
        .filter(account::Column::RiskScore.gte(q.min_risk.unwrap_or(0)))
        .order_by_desc(account::Column::RiskScore)
        .order_by_desc(account::Column::AlertCount)
        .limit(limit)
        .offset(offset)
        .all(&st.db)
        .await?;
    Ok(Json(rows))
}

pub async fn get_one(State(st): State<AppState>, Path(id): Path<i64>) -> ApiResult<Json<Value>> {
    let a = account::Entity::find_by_id(id).one(&st.db).await?.ok_or_else(|| ApiError::NotFound(format!("account {id}")))?;
    let alerts = alert::Entity::find()
        .filter(alert::Column::SubjectAccount.eq(id))
        .order_by_desc(alert::Column::Score)
        .limit(20)
        .all(&st.db)
        .await?;
    Ok(Json(json!({"account": a, "alerts": alerts})))
}

#[derive(Deserialize)]
pub struct TxQ {
    limit: Option<u64>,
    offset: Option<u64>,
}

pub async fn transactions(State(st): State<AppState>, Path(id): Path<i64>, Query(q): Query<TxQ>) -> ApiResult<Json<Vec<t::Model>>> {
    let (limit, offset) = page(q.limit, q.offset);
    let rows = t::Entity::find()
        .filter(Condition::any().add(t::Column::SenderAccount.eq(id)).add(t::Column::ReceiverAccount.eq(id)))
        .order_by_desc(t::Column::OccurredAt)
        .limit(limit)
        .offset(offset)
        .all(&st.db)
        .await?;
    Ok(Json(rows))
}

#[derive(Deserialize)]
pub struct NetQ {
    limit: Option<i64>,
}

/// One-hop counterparty graph: who this account pays / is paid by, with value and worst risk score.
pub async fn network(State(st): State<AppState>, Path(id): Path<i64>, Query(q): Query<NetQ>) -> ApiResult<Json<Value>> {
    let limit = q.limit.unwrap_or(25).clamp(1, 200);
    let rows = sqlx::query(
        "SELECT cp, SUM(out_n)::bigint AS out_n, SUM(in_n)::bigint AS in_n, SUM(out_amt)::float8 AS out_amt, \
                SUM(in_amt)::float8 AS in_amt, MAX(score)::int AS max_score \
         FROM (SELECT receiver_account AS cp, count(*) AS out_n, 0 AS in_n, sum(amount_gbp) AS out_amt, 0 AS in_amt, max(risk_score) AS score \
                 FROM transactions WHERE sender_account = $1 GROUP BY 1 \
               UNION ALL \
               SELECT sender_account, 0, count(*), 0, sum(amount_gbp), max(risk_score) \
                 FROM transactions WHERE receiver_account = $1 GROUP BY 1) x \
         GROUP BY cp ORDER BY SUM(out_amt) + SUM(in_amt) DESC LIMIT $2",
    )
    .bind(id)
    .bind(limit)
    .fetch_all(st.db.get_postgres_connection_pool())
    .await?;
    let edges: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "counterparty": r.get::<i64, _>("cp"),
                "sent_to": {"count": r.get::<i64, _>("out_n"), "total_gbp": r.get::<f64, _>("out_amt")},
                "received_from": {"count": r.get::<i64, _>("in_n"), "total_gbp": r.get::<f64, _>("in_amt")},
                "max_risk_score": r.get::<i32, _>("max_score"),
            })
        })
        .collect();
    Ok(Json(json!({"account": id, "counterparties": edges})))
}
