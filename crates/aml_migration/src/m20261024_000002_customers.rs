use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

const UP: &str = r#"
CREATE TABLE customers (
    id                 BIGSERIAL PRIMARY KEY,
    external_ref       TEXT NOT NULL UNIQUE CHECK (length(external_ref) BETWEEN 1 AND 64),
    customer_type      VARCHAR(20) NOT NULL DEFAULT 'individual'
                       CHECK (customer_type IN ('individual','business','employee','other')),
    country            VARCHAR(2),
    risk_rating        VARCHAR(10) NOT NULL DEFAULT 'low' CHECK (risk_rating IN ('low','medium','high')),
    pep                BOOLEAN NOT NULL DEFAULT FALSE,
    annual_income_gbp  NUMERIC(20,2) CHECK (annual_income_gbp >= 0),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_customers_risk ON customers (risk_rating) WHERE risk_rating <> 'low' OR pep;

ALTER TABLE accounts ADD COLUMN customer_id BIGINT REFERENCES customers(id);
CREATE INDEX idx_accounts_customer ON accounts (customer_id) WHERE customer_id IS NOT NULL;

ALTER TABLE transactions DROP CONSTRAINT transactions_payment_type_check;
ALTER TABLE transactions ADD CONSTRAINT transactions_payment_type_check CHECK (payment_type IN
    ('CASH_DEPOSIT','CASH_WITHDRAWAL','CROSS_BORDER','ACH','CHEQUE','CREDIT_CARD','DEBIT_CARD',
     'WIRE','ONLINE_TRANSFER','POS','ATM'));

ALTER TABLE jurisdictions DROP CONSTRAINT jurisdictions_risk_level_check;
ALTER TABLE jurisdictions ADD CONSTRAINT jurisdictions_risk_level_check
    CHECK (risk_level IN ('low','medium','high','very_high'));
"#;

const DOWN: &str = r#"
DELETE FROM jurisdictions WHERE risk_level = 'very_high';
ALTER TABLE jurisdictions DROP CONSTRAINT jurisdictions_risk_level_check;
ALTER TABLE jurisdictions ADD CONSTRAINT jurisdictions_risk_level_check CHECK (risk_level IN ('low','medium','high'));
DELETE FROM transactions WHERE payment_type IN ('WIRE','ONLINE_TRANSFER','POS','ATM');
ALTER TABLE transactions DROP CONSTRAINT transactions_payment_type_check;
ALTER TABLE transactions ADD CONSTRAINT transactions_payment_type_check CHECK (payment_type IN
    ('CASH_DEPOSIT','CASH_WITHDRAWAL','CROSS_BORDER','ACH','CHEQUE','CREDIT_CARD','DEBIT_CARD'));
ALTER TABLE accounts DROP COLUMN customer_id;
DROP TABLE customers;
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
