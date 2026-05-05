use sea_orm::DatabaseConnection;

pub struct Migrator;

impl Migrator {
    pub async fn up(_db: &DatabaseConnection) -> anyhow::Result<()> {
        Ok(())
    }
}
