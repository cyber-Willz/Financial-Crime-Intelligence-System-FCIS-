use sea_orm_migration::prelude::*;

#[derive(DeriveMigrationName)]
pub struct Migration;

const UP: &str = r#"
ALTER TABLE customers ADD COLUMN sector TEXT;

ALTER TABLE customers ALTER COLUMN customer_type TYPE VARCHAR(30);
ALTER TABLE customers DROP CONSTRAINT customers_customer_type_check;
ALTER TABLE customers ADD CONSTRAINT customers_customer_type_check CHECK (customer_type IN
    ('individual','business','employee','corporate','financial_institution','ngo','other'));

ALTER TABLE jurisdictions DROP CONSTRAINT jurisdictions_risk_level_check;
ALTER TABLE jurisdictions ADD CONSTRAINT jurisdictions_risk_level_check
    CHECK (risk_level IN ('low','medium','high','very_high','sanctioned'));

-- Countries whose OFAC-list match was determined purely by counterparty country in the KYC benchmark move up
-- to the new 'sanctioned' tier. Only untouched seed rows are migrated; an administrator's edits are preserved.
UPDATE jurisdictions
   SET risk_level = 'sanctioned',
       reason = 'Seed: OFAC-style sanctioned-jurisdiction list from the Synthetic KYC benchmark; replace with your own sanctions programme'
 WHERE country_code IN ('IR','KP','SY') AND risk_level = 'very_high' AND reason LIKE 'Seed:%';
"#;

const DOWN: &str = r#"
UPDATE jurisdictions SET risk_level = 'very_high' WHERE risk_level = 'sanctioned';
ALTER TABLE jurisdictions DROP CONSTRAINT jurisdictions_risk_level_check;
ALTER TABLE jurisdictions ADD CONSTRAINT jurisdictions_risk_level_check CHECK (risk_level IN ('low','medium','high','very_high'));
UPDATE customers SET customer_type = 'business' WHERE customer_type IN ('corporate','financial_institution','ngo');
ALTER TABLE customers DROP CONSTRAINT customers_customer_type_check;
ALTER TABLE customers ADD CONSTRAINT customers_customer_type_check CHECK (customer_type IN ('individual','business','employee','other'));
ALTER TABLE customers ALTER COLUMN customer_type TYPE VARCHAR(20);
ALTER TABLE customers DROP COLUMN sector;
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
