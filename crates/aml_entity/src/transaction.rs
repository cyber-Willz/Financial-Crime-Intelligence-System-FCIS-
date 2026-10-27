use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "transactions")]
pub struct Model {
    #[sea_orm(primary_key, auto_increment = false)]
    pub id: i64,
    pub occurred_at: DateTimeUtc,
    pub sender_account: i64,
    pub receiver_account: i64,
    pub amount: Decimal,
    pub payment_currency: String,
    pub received_currency: String,
    pub amount_gbp: Decimal,
    pub sender_country: String,
    pub receiver_country: String,
    pub payment_type: String,
    pub risk_score: i16,
    pub flagged: bool,
    pub rule_codes: Json,
    pub label_laundering: Option<bool>,
    pub label_typology: Option<String>,
    pub created_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {}

impl ActiveModelBehavior for ActiveModel {}
