use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

const UP: &str = r#"
-- One row per reviewed alert: the analyst's decision plus what the system believed at the time.
-- This is the reinforcement signal of the connectome feedback loop. Reopening an alert removes its row.
CREATE TABLE feedback_events (
    alert_id    BIGINT PRIMARY KEY REFERENCES alerts(id) ON DELETE CASCADE,
    account     BIGINT NOT NULL,
    decision    VARCHAR(20) NOT NULL CHECK (decision IN ('sar_filed','confirmed','false_positive')),
    score       SMALLINT NOT NULL,
    rules       JSONB NOT NULL DEFAULT '[]',
    graph_risk  REAL,
    decided_by  TEXT NOT NULL,
    case_id     BIGINT,
    decided_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX feedback_events_account_idx ON feedback_events (account);
CREATE INDEX feedback_events_decided_idx ON feedback_events (decided_at);

-- Latest graph-model risk per account, written by `aml graph-rescore`.
CREATE TABLE graph_risk (
    account      BIGINT PRIMARY KEY,
    risk         REAL NOT NULL CHECK (risk >= 0 AND risk <= 1),
    window_start TIMESTAMPTZ,
    model        TEXT NOT NULL,
    scored_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Audit of every agent output: which evidence, which model (if any), whether it passed the grounding gate.
CREATE TABLE agent_runs (
    id              BIGSERIAL PRIMARY KEY,
    kind            VARCHAR(30) NOT NULL,
    subject         TEXT NOT NULL,
    actor           TEXT NOT NULL,
    evidence_digest TEXT,
    llm_model       TEXT,
    llm_used        BOOLEAN NOT NULL DEFAULT FALSE,
    grounding_ok    BOOLEAN NOT NULL,
    fallback_reason TEXT,
    output          JSONB NOT NULL,
    at              TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX agent_runs_subject_idx ON agent_runs (kind, subject);

-- Rule-weight / threshold suggestions from the feedback loop. Nothing is applied until an administrator does so.
CREATE TABLE rule_recommendations (
    id         BIGSERIAL PRIMARY KEY,
    kind       VARCHAR(20) NOT NULL CHECK (kind IN ('rule_weight','threshold','coverage')),
    title      TEXT NOT NULL,
    detail     TEXT NOT NULL,
    apply      JSONB,
    status     VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','applied','rejected','superseded')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_by TEXT,
    decided_at TIMESTAMPTZ
);
CREATE INDEX rule_recommendations_status_idx ON rule_recommendations (status);
"#;

const DOWN: &str = r#"
DROP TABLE IF EXISTS rule_recommendations;
DROP TABLE IF EXISTS agent_runs;
DROP TABLE IF EXISTS graph_risk;
DROP TABLE IF EXISTS feedback_events;
"#;

#[async_trait::async_trait]
impl MigrationTrait for Migration {
    async fn up(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager.get_connection().execute_unprepared(UP).await?;
        Ok(())
    }
    async fn down(&self, manager: &SchemaManager) -> Result<(), DbErr> {
        manager.get_connection().execute_unprepared(DOWN).await?;
        Ok(())
    }
}
