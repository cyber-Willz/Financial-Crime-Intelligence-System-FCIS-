use super::audit;
use crate::{app::AppState, auth::Actor, error::{ApiError, ApiResult}};
use aml_entity::watchlist as w;
use axum::{extract::{Path, State}, http::StatusCode, Extension, Json};
use chrono::Utc;
use sea_orm::{
    sqlx, ActiveModelTrait, EntityTrait, QueryOrder, Set,
};
use serde::Deserialize;
use serde_json::json;

pub async fn list(State(st): State<AppState>) -> ApiResult<Json<Vec<w::Model>>> {
    Ok(Json(w::Entity::find().order_by_desc(w::Column::AddedAt).all(&st.db).await?))
}

#[derive(Deserialize)]
pub struct Add {
    account_number: i64,
    list_name: String,
    reason: Option<String>,
}

async fn flag_account(st: &AppState, acct: i64, on: bool) -> ApiResult<()> {
    sqlx::query("UPDATE accounts SET watchlisted = $2, updated_at = now() WHERE account_number = $1")
        .bind(acct)
        .bind(on)
        .execute(st.db.get_postgres_connection_pool())
        .await?;
    st.engine.lock().await.set_watch(acct, on);
    Ok(())
}

pub async fn add(State(st): State<AppState>, Extension(a): Extension<Actor>, Json(r): Json<Add>) -> ApiResult<(StatusCode, Json<w::Model>)> {
    a.require_admin()?;
    if r.account_number <= 0 || r.list_name.trim().is_empty() {
        return Err(ApiError::Unprocessable(vec!["account_number must be positive and list_name non-empty".into()]));
    }
    let m = w::ActiveModel {
        account_number: Set(r.account_number),
        list_name: Set(r.list_name),
        reason: Set(r.reason),
        added_by: Set(a.name.clone()),
        added_at: Set(Utc::now()),
        ..Default::default()
    }
    .insert(&st.db)
    .await?;
    flag_account(&st, r.account_number, true).await?;
    audit(&st, &a, "watchlist.add", "account", r.account_number, json!({"list": m.list_name})).await;
    Ok((StatusCode::CREATED, Json(m)))
}

pub async fn remove(State(st): State<AppState>, Extension(a): Extension<Actor>, Path(acct): Path<i64>) -> ApiResult<StatusCode> {
    a.require_admin()?;
    let res = w::Entity::delete_many().filter_by_account(acct).exec(&st.db).await?;
    if res.rows_affected == 0 {
        return Err(ApiError::NotFound(format!("account {acct} is not on the watchlist")));
    }
    flag_account(&st, acct, false).await?;
    audit(&st, &a, "watchlist.remove", "account", acct, json!({})).await;
    Ok(StatusCode::NO_CONTENT)
}

trait ByAccount {
    fn filter_by_account(self, a: i64) -> Self;
}
impl ByAccount for sea_orm::DeleteMany<w::Entity> {
    fn filter_by_account(self, a: i64) -> Self {
        use sea_orm::{ColumnTrait, QueryFilter};
        self.filter(w::Column::AccountNumber.eq(a))
    }
}
