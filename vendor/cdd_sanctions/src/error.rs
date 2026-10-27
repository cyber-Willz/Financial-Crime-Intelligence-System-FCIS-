use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("csv: {0}")]
    Csv(#[from] csv::Error),
    #[error("xml: {0}")]
    Xml(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("not permitted: {0}")]
    Forbidden(String),
    #[error("invalid: {0}")]
    Invalid(String),
    #[error("not approvable: {}", .0.join("; "))]
    Blocked(Vec<String>),
    #[error("spectral network analysis: {0}")]
    Spectral(String),
}

impl From<spectral_hypergraph::HypergraphError> for Error {
    fn from(e: spectral_hypergraph::HypergraphError) -> Self {
        Error::Spectral(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
