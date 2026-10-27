use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

const UP: &str = r#"
CREATE TABLE accounts (
    account_number      BIGINT PRIMARY KEY CHECK (account_number > 0),
    country             VARCHAR(2),
    first_seen          TIMESTAMPTZ NOT NULL,
    last_seen           TIMESTAMPTZ NOT NULL,
    txn_sent_count      BIGINT NOT NULL DEFAULT 0,
    txn_received_count  BIGINT NOT NULL DEFAULT 0,
    total_sent          NUMERIC(24,2) NOT NULL DEFAULT 0,
    total_received      NUMERIC(24,2) NOT NULL DEFAULT 0,
    risk_score          SMALLINT NOT NULL DEFAULT 0 CHECK (risk_score BETWEEN 0 AND 100),
    risk_level          VARCHAR(10) NOT NULL DEFAULT 'low' CHECK (risk_level IN ('low','medium','high','critical')),
    alert_count         INTEGER NOT NULL DEFAULT 0,
    watchlisted         BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_accounts_risk ON accounts (risk_score DESC) WHERE risk_score >= 40;

CREATE SEQUENCE transactions_id_seq;
CREATE TABLE transactions (
    id                BIGINT PRIMARY KEY DEFAULT nextval('transactions_id_seq'),
    occurred_at       TIMESTAMPTZ NOT NULL,
    sender_account    BIGINT NOT NULL REFERENCES accounts(account_number),
    receiver_account  BIGINT NOT NULL REFERENCES accounts(account_number),
    amount            NUMERIC(20,2) NOT NULL CHECK (amount > 0),
    payment_currency  VARCHAR(3) NOT NULL,
    received_currency VARCHAR(3) NOT NULL,
    amount_gbp        NUMERIC(20,2) NOT NULL CHECK (amount_gbp >= 0),
    sender_country    VARCHAR(2) NOT NULL,
    receiver_country  VARCHAR(2) NOT NULL,
    payment_type      VARCHAR(20) NOT NULL CHECK (payment_type IN
        ('CASH_DEPOSIT','CASH_WITHDRAWAL','CROSS_BORDER','ACH','CHEQUE','CREDIT_CARD','DEBIT_CARD')),
    risk_score        SMALLINT NOT NULL DEFAULT 0 CHECK (risk_score BETWEEN 0 AND 100),
    flagged           BOOLEAN NOT NULL DEFAULT FALSE,
    rule_codes        JSONB NOT NULL DEFAULT '[]'::jsonb,
    label_laundering  BOOLEAN,
    label_typology    VARCHAR(40),
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (sender_account <> receiver_account)
);
ALTER SEQUENCE transactions_id_seq OWNED BY transactions.id;
CREATE INDEX idx_txn_sender_time   ON transactions (sender_account, occurred_at DESC);
CREATE INDEX idx_txn_receiver_time ON transactions (receiver_account, occurred_at DESC);
CREATE INDEX idx_txn_time          ON transactions (occurred_at DESC, id DESC);
CREATE INDEX idx_txn_flagged       ON transactions (occurred_at DESC) WHERE flagged;

CREATE TABLE cases (
    id             BIGSERIAL PRIMARY KEY,
    title          TEXT NOT NULL,
    description    TEXT,
    status         VARCHAR(20) NOT NULL DEFAULT 'open' CHECK (status IN ('open','investigating','closed')),
    priority       VARCHAR(10) NOT NULL DEFAULT 'medium' CHECK (priority IN ('low','medium','high','critical')),
    assigned_to    TEXT,
    sar_reference  TEXT,
    outcome        VARCHAR(20) CHECK (outcome IN ('sar_filed','no_action')),
    created_by     TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at      TIMESTAMPTZ,
    CHECK ((outcome = 'sar_filed') = (sar_reference IS NOT NULL) OR outcome IS NULL)
);
CREATE INDEX idx_cases_status ON cases (status, created_at DESC);

CREATE TABLE case_notes (
    id         BIGSERIAL PRIMARY KEY,
    case_id    BIGINT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    author     TEXT NOT NULL,
    body       TEXT NOT NULL CHECK (length(body) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_case_notes_case ON case_notes (case_id, created_at);

CREATE TABLE alerts (
    id               BIGSERIAL PRIMARY KEY,
    dedupe_key       TEXT NOT NULL UNIQUE,
    subject_account  BIGINT NOT NULL REFERENCES accounts(account_number),
    first_txn_id     BIGINT NOT NULL,
    last_txn_id      BIGINT NOT NULL,
    txn_count        INTEGER NOT NULL DEFAULT 1,
    score            SMALLINT NOT NULL CHECK (score BETWEEN 0 AND 100),
    severity         VARCHAR(10) NOT NULL CHECK (severity IN ('low','medium','high','critical')),
    rule_codes       JSONB NOT NULL DEFAULT '[]'::jsonb,
    explanation      JSONB NOT NULL DEFAULT '{}'::jsonb,
    status           VARCHAR(24) NOT NULL DEFAULT 'open' CHECK (status IN
        ('open','under_review','escalated','closed_false_positive','closed_confirmed','sar_filed')),
    assigned_to      TEXT,
    resolution_note  TEXT,
    case_id          BIGINT REFERENCES cases(id) ON DELETE SET NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_alerts_status   ON alerts (status, severity, created_at DESC);
CREATE INDEX idx_alerts_subject  ON alerts (subject_account, created_at DESC);
CREATE INDEX idx_alerts_case     ON alerts (case_id) WHERE case_id IS NOT NULL;

CREATE TABLE rule_configs (
    code        VARCHAR(40) PRIMARY KEY,
    description TEXT NOT NULL DEFAULT '',
    enabled     BOOLEAN NOT NULL DEFAULT TRUE,
    weight      DOUBLE PRECISION NOT NULL CHECK (weight >= 0 AND weight < 1),
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_by  TEXT
);

CREATE TABLE jurisdictions (
    country_code VARCHAR(2) PRIMARY KEY,
    name         TEXT NOT NULL,
    risk_level   VARCHAR(10) NOT NULL CHECK (risk_level IN ('low','medium','high')),
    reason       TEXT
);

CREATE TABLE watchlist (
    id             BIGSERIAL PRIMARY KEY,
    account_number BIGINT NOT NULL UNIQUE CHECK (account_number > 0),
    list_name      TEXT NOT NULL,
    reason         TEXT,
    added_by       TEXT NOT NULL,
    added_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE audit_log (
    id          BIGSERIAL PRIMARY KEY,
    actor       TEXT NOT NULL,
    action      TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id   TEXT,
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb,
    at          TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_audit_at ON audit_log (at DESC);
-- The audit log is append-only.
CREATE OR REPLACE FUNCTION audit_log_immutable() RETURNS trigger AS $$
BEGIN RAISE EXCEPTION 'audit_log is append-only'; END; $$ LANGUAGE plpgsql;
CREATE TRIGGER audit_log_no_update BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_immutable();
"#;

const DOWN: &str = r#"
DROP TABLE IF EXISTS audit_log, watchlist, jurisdictions, rule_configs, alerts, case_notes, cases, transactions, accounts CASCADE;
DROP FUNCTION IF EXISTS audit_log_immutable();
DROP SEQUENCE IF EXISTS transactions_id_seq;
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
