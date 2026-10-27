use sea_orm::entity::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, DeriveEntityModel, Serialize, Deserialize)]
#[sea_orm(table_name = "alerts")]
pub struct Model {
    #[sea_orm(primary_key)]
    pub id: i64,
    pub dedupe_key: String,
    pub subject_account: i64,
    pub first_txn_id: i64,
    pub last_txn_id: i64,
    pub txn_count: i32,
    pub score: i16,
    pub severity: String,
    pub rule_codes: Json,
    pub explanation: Json,
    pub status: String,
    pub assigned_to: Option<String>,
    pub resolution_note: Option<String>,
    pub case_id: Option<i64>,
    pub created_at: DateTimeUtc,
    pub updated_at: DateTimeUtc,
}

#[derive(Copy, Clone, Debug, EnumIter, DeriveRelation)]
pub enum Relation {
    #[sea_orm(belongs_to = "super::case::Entity", from = "Column::CaseId", to = "super::case::Column::Id")]
    Case,
}

impl Related<super::case::Entity> for Entity {
    fn to() -> RelationDef {
        Relation::Case.def()
    }
}

impl ActiveModelBehavior for ActiveModel {}
