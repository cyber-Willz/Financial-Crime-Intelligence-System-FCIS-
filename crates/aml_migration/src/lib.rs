//! Schema migrations, run through SeaORM's migrator (`aml migrate` / on `serve --migrate`).
pub use sea_orm_migration::prelude::*;

mod m20261024_000001_init;
mod m20261024_000002_customers;
mod m20261025_000003_kyc_sanctions;
mod m20261026_000004_feedback_agent;

pub struct Migrator;

#[async_trait::async_trait]
impl MigratorTrait for Migrator {
    fn migrations() -> Vec<Box<dyn MigrationTrait>> {
        vec![
            Box::new(m20261024_000001_init::Migration),
            Box::new(m20261024_000002_customers::Migration),
            Box::new(m20261025_000003_kyc_sanctions::Migration),
            Box::new(m20261026_000004_feedback_agent::Migration),
        ]
    }
}
