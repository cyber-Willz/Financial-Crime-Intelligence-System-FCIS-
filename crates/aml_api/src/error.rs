use axum::{http::StatusCode, response::IntoResponse, response::Response, Json};
use sea_orm::{DbErr, SqlErr};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    BadRequest(String),
    #[error("missing or invalid API key")]
    Unauthorized,
    #[error("{0}")]
    Forbidden(String),
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    Conflict(String),
    #[error("validation failed")]
    Unprocessable(Vec<String>),
    #[error("internal error")]
    Internal(#[source] anyhow::Error),
}

impl ApiError {
    pub fn code(&self) -> &'static str {
        match self {
            ApiError::BadRequest(_) => "bad_request",
            ApiError::Unauthorized => "unauthorized",
            ApiError::Forbidden(_) => "forbidden",
            ApiError::NotFound(_) => "not_found",
            ApiError::Conflict(_) => "conflict",
            ApiError::Unprocessable(_) => "validation_failed",
            ApiError::Internal(_) => "internal",
        }
    }
    pub fn status(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::Conflict(_) => StatusCode::CONFLICT,
            ApiError::Unprocessable(_) => StatusCode::UNPROCESSABLE_ENTITY,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = match &self {
            ApiError::Unprocessable(errs) => json!({"error": {"code": self.code(), "message": "validation failed", "details": errs}}),
            ApiError::Internal(e) => {
                tracing::error!(error = ?e, "internal error");
                json!({"error": {"code": self.code(), "message": "internal server error"}})
            }
            other => json!({"error": {"code": other.code(), "message": other.to_string()}}),
        };
        (status, Json(body)).into_response()
    }
}

impl From<DbErr> for ApiError {
    fn from(e: DbErr) -> Self {
        match e.sql_err() {
            Some(SqlErr::UniqueConstraintViolation(m)) => ApiError::Conflict(format!("already exists: {m}")),
            Some(SqlErr::ForeignKeyConstraintViolation(m)) => ApiError::Unprocessable(vec![format!("referenced record missing: {m}")]),
            _ => ApiError::Internal(e.into()),
        }
    }
}
impl From<sea_orm::sqlx::Error> for ApiError {
    fn from(e: sea_orm::sqlx::Error) -> Self {
        ApiError::Internal(e.into())
    }
}
impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        ApiError::Internal(e)
    }
}

pub type ApiResult<T> = Result<T, ApiError>;
