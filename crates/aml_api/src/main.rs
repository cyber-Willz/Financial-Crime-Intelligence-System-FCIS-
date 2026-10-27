mod agent_api;
mod app;
mod auth;
mod error;
mod feedback;
mod graph_cli;
mod import;
mod ingest;
mod routes;

use anyhow::{Context, Result};
use aml_migration::{Migrator, MigratorTrait};
use clap::{Parser, Subcommand};
use sea_orm::{ConnectOptions, Database, DatabaseConnection};
use std::{path::PathBuf, sync::Arc, time::Duration};
use tokio::sync::Mutex;

#[derive(Parser)]
#[command(name = "aml", about = "Anti-money-laundering transaction monitoring service")]
struct Cli {
    #[arg(long, env = "DATABASE_URL")]
    database_url: String,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Apply schema migrations (use --fresh to drop everything first).
    Migrate {
        #[arg(long)]
        fresh: bool,
    },
    /// Run the HTTP API.
    Serve {
        #[arg(long, env = "AML_BIND", default_value = "0.0.0.0:8080")]
        bind: String,
        /// Apply pending migrations on start.
        #[arg(long)]
        migrate: bool,
        /// How much history (relative to the newest transaction) to load into the rolling windows.
        #[arg(long, env = "AML_HYDRATE_DAYS", default_value_t = 180)]
        hydrate_days: i64,
    },
    /// Bulk-load a supported dataset through the scoring pipeline.
    #[command(alias = "import-saml")]
    ImportDataset {
        #[arg(long)]
        file: PathBuf,
        /// saml-d | aml-suspicious | risk-prediction | fraudshield | kyc-risk | lending
        #[arg(long, default_value = "saml-d")]
        format: String,
        #[arg(long, default_value_t = 0)]
        skip: usize,
        #[arg(long)]
        limit: Option<usize>,
        #[arg(long, default_value_t = 5000)]
        batch: usize,
    },
    /// Score the last N days with a trained graph model and store each account's latest graph risk.
    GraphRescore {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 60)]
        days: i64,
    },
    /// Feedback loop: fine-tune the graph model head on analyst outcomes (kept only if a held-out period improves).
    TuneGraph {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 180)]
        days: i64,
    },
}

async fn connect(url: &str) -> Result<DatabaseConnection> {
    let mut o = ConnectOptions::new(url.to_string());
    o.max_connections(10).min_connections(1).connect_timeout(Duration::from_secs(10)).sqlx_logging(false);
    Database::connect(o).await.context("connecting to PostgreSQL")
}

async fn state(db: DatabaseConnection, hydrate_days: i64) -> Result<app::AppState> {
    let engine = ingest::load_engine(&db, hydrate_days).await.map_err(|e| anyhow::anyhow!("{e}"))?;
    let keys = auth::Keys::parse(&std::env::var("AML_API_KEYS").unwrap_or_default())?;
    if keys.is_empty() {
        tracing::warn!("AML_API_KEYS is not set: running in DEVELOPMENT mode, every caller is an admin");
    }
    let llm: Option<Box<dyn aml_agent::Llm>> = match (std::env::var("AML_LLM_CMD").ok(), std::env::var("AML_LLM_MODEL").ok()) {
        (Some(cmd), Some(model)) if !cmd.is_empty() && !model.is_empty() => {
            let mut p = aml_agent::ProcessLlm::new(cmd, vec![model]);
            if let Some(t) = std::env::var("AML_LLM_TIMEOUT_SECS").ok().and_then(|v| v.parse::<u64>().ok()) {
                p.request_timeout = Duration::from_secs(t.clamp(5, 900));
            }
            tracing::info!("language-model sidecar configured; it starts on first use, output is gated by grounding checks");
            Some(Box::new(p))
        }
        _ => None,
    };
    let ownership: std::collections::HashMap<i64, aml_agent::evidence::OwnershipFacts> = match std::env::var("AML_OWNERSHIP_FILE") {
        Ok(p) if !p.is_empty() => {
            let raw = std::fs::read_to_string(&p).with_context(|| format!("reading {p}"))?;
            let m: std::collections::HashMap<String, aml_agent::evidence::OwnershipFacts> = serde_json::from_str(&raw).with_context(|| format!("parsing {p}"))?;
            m.into_iter().filter_map(|(k, v)| Some((k.parse().ok()?, v))).collect()
        }
        _ => Default::default(),
    };
    let agent = aml_agent::Agent::new(llm, aml_agent::AgentConfig::default());
    Ok(app::AppState { db, engine: Arc::new(Mutex::new(engine)), keys: Arc::new(keys), hydrate_days, agent: Arc::new(std::sync::Mutex::new(agent)), ownership: Arc::new(ownership) })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,tower_http=info".into()))
        .init();
    let cli = Cli::parse();
    let db = connect(&cli.database_url).await?;
    match cli.cmd {
        Cmd::Migrate { fresh } => {
            if fresh {
                Migrator::fresh(&db).await?;
            } else {
                Migrator::up(&db, None).await?;
            }
            tracing::info!("migrations applied");
        }
        Cmd::Serve { bind, migrate, hydrate_days } => {
            if migrate {
                Migrator::up(&db, None).await?;
            }
            let st = state(db, hydrate_days).await?;
            let listener = tokio::net::TcpListener::bind(&bind).await.with_context(|| format!("binding {bind}"))?;
            tracing::info!(%bind, "aml api listening");
            axum::serve(listener, app::router(st))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
        }
        Cmd::ImportDataset { file, format, skip, limit, batch } => {
            let fmt = aml_datasets::Format::parse(&format)?;
            let st = state(db, 180).await?;
            import::import_dataset(&st, &file, fmt, skip, limit, batch.clamp(100, ingest::MAX_BATCH)).await?;
        }
        Cmd::GraphRescore { model, days } => graph_cli::rescore(&db, &model, days).await?,
        Cmd::TuneGraph { model, out, days } => graph_cli::tune(&db, &model, &out, days).await?,
    }
    Ok(())
}
