use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "accounts")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub account_number: i64,
    pub country: Option<String>,
    pub first_seen: DateTimeUtc,
    pub last_seen: DateTimeUtc,
    pub txn_sent_count: i64,
    pub txn_received_count: i64,
    pub total_sent: Decimal,
    pub total_received: Decimal,
    pub risk_score: i16,
    pub risk_level: String,
    pub alert_count: i32,
    pub watchlisted: bool,
    pub customer_id: Option<i64>,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
