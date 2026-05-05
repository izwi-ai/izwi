//! Persistent chat thread storage backed by SQLite.

use anyhow::{Context, anyhow};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseConnection, DbBackend, EntityTrait, QueryFilter,
    QueryResult, Set, Statement, TransactionTrait,
};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::db::StoreDatabase;
use crate::entity::{chat_messages, chat_threads};
use crate::ids::new_uuid;

const DEFAULT_THREAD_TITLE: &str = "New chat";

#[derive(Debug, Clone, Serialize)]
pub struct ChatThreadSummary {
    pub id: String,
    pub title: String,
    pub model_id: Option<String>,
    pub created_at: u64,
    pub updated_at: u64,
    pub last_message_preview: Option<String>,
    pub message_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatThreadMessage {
    pub id: String,
    pub thread_id: String,
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content_parts: Option<Vec<JsonValue>>,
    pub created_at: u64,
    pub tokens_generated: Option<usize>,
    pub generation_time_ms: Option<f64>,
}

#[derive(Clone)]
pub struct ChatStore {
    db: StoreDatabase,
}

impl ChatStore {
    pub fn initialize() -> anyhow::Result<Self> {
        Ok(Self {
            db: StoreDatabase::from_default_path()?,
        })
    }

    pub async fn list_threads(&self) -> anyhow::Result<Vec<ChatThreadSummary>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(Statement::from_string(
                DbBackend::Sqlite,
                THREAD_SUMMARY_LIST_SQL.to_string(),
            ))
            .await
            .context("Failed to list chat threads")?;
        rows.iter().map(map_thread_summary).collect()
    }

    pub async fn get_thread(&self, thread_id: String) -> anyhow::Result<Option<ChatThreadSummary>> {
        let db = self.db.connection().await?;
        fetch_thread_summary(db, &thread_id).await
    }

    pub async fn create_thread(
        &self,
        title: Option<String>,
        model_id: Option<String>,
    ) -> anyhow::Result<ChatThreadSummary> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let thread_id = new_uuid();
        let resolved_title = sanitize_thread_title(title.as_deref());

        chat_threads::Entity::insert(chat_threads::ActiveModel {
            id: Set(thread_id.clone()),
            title: Set(resolved_title.clone()),
            model_id: Set(model_id.clone()),
            created_at: Set(now),
            updated_at: Set(now),
        })
        .exec(db)
        .await
        .context("Failed to create chat thread")?;

        Ok(ChatThreadSummary {
            id: thread_id,
            title: resolved_title,
            model_id,
            created_at: now as u64,
            updated_at: now as u64,
            last_message_preview: None,
            message_count: 0,
        })
    }

    pub async fn delete_thread(&self, thread_id: String) -> anyhow::Result<bool> {
        let db = self.db.connection().await?;
        let result = chat_threads::Entity::delete_by_id(thread_id)
            .exec(db)
            .await
            .context("Failed to delete chat thread")?;
        Ok(result.rows_affected > 0)
    }

    pub async fn list_messages(&self, thread_id: String) -> anyhow::Result<Vec<ChatThreadMessage>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(Statement::from_sql_and_values(
                DbBackend::Sqlite,
                CHAT_MESSAGES_LIST_SQL,
                vec![thread_id.into()],
            ))
            .await
            .context("Failed to list chat messages")?;
        rows.iter().map(map_thread_message).collect()
    }

    pub async fn update_thread_title(
        &self,
        thread_id: String,
        title: String,
    ) -> anyhow::Result<Option<ChatThreadSummary>> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let resolved_title = sanitize_thread_title(Some(title.as_str()));

        let result = chat_threads::Entity::update_many()
            .col_expr(
                chat_threads::Column::Title,
                Expr::value(resolved_title.clone()),
            )
            .col_expr(chat_threads::Column::UpdatedAt, Expr::value(now))
            .filter(chat_threads::Column::Id.eq(thread_id.clone()))
            .exec(db)
            .await
            .context("Failed to update chat thread title")?;

        if result.rows_affected == 0 {
            return Ok(None);
        }

        fetch_thread_summary(db, &thread_id).await
    }

    pub async fn append_message(
        &self,
        thread_id: String,
        role: String,
        content: String,
        content_parts: Option<Vec<JsonValue>>,
        model_id: Option<String>,
        tokens_generated: Option<usize>,
        generation_time_ms: Option<f64>,
    ) -> anyhow::Result<ChatThreadMessage> {
        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start chat message transaction")?;

        let thread_exists = chat_threads::Entity::find_by_id(thread_id.clone())
            .one(&tx)
            .await
            .context("Failed to load chat thread")?
            .is_some();

        if !thread_exists {
            return Err(anyhow!("Thread not found"));
        }

        let now = now_unix_millis_i64();
        let message_id = new_uuid();
        let tokens_i64 = opt_usize_to_i64(tokens_generated)?;
        let serialized_content_parts = content_parts
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .context("Failed serializing chat message content_parts")?;

        chat_messages::Entity::insert(chat_messages::ActiveModel {
            id: Set(message_id.clone()),
            thread_id: Set(thread_id.clone()),
            role: Set(role.clone()),
            content: Set(content.clone()),
            content_parts: Set(serialized_content_parts),
            created_at: Set(now),
            tokens_generated: Set(tokens_i64),
            generation_time_ms: Set(generation_time_ms),
        })
        .exec(&tx)
        .await
        .context("Failed to append chat message")?;

        chat_threads::Entity::update_many()
            .col_expr(chat_threads::Column::UpdatedAt, Expr::value(now))
            .col_expr(chat_threads::Column::ModelId, Expr::value(model_id.clone()))
            .filter(chat_threads::Column::Id.eq(thread_id.clone()))
            .exec(&tx)
            .await
            .context("Failed to update chat thread metadata")?;

        tx.commit()
            .await
            .context("Failed to commit chat message transaction")?;

        Ok(ChatThreadMessage {
            id: message_id,
            thread_id,
            role,
            content,
            content_parts,
            created_at: now as u64,
            tokens_generated,
            generation_time_ms,
        })
    }
}

const THREAD_SUMMARY_LIST_SQL: &str = r#"
    SELECT
        t.id,
        t.title,
        t.model_id,
        t.created_at,
        t.updated_at,
        (
            SELECT m.content
            FROM chat_messages m
            WHERE m.thread_id = t.id
            ORDER BY m.created_at DESC, m.id DESC
            LIMIT 1
        ) AS last_message_preview,
        (
            SELECT COUNT(1)
            FROM chat_messages m
            WHERE m.thread_id = t.id
        ) AS message_count
    FROM chat_threads t
    ORDER BY t.updated_at DESC, t.created_at DESC
"#;

const THREAD_SUMMARY_BY_ID_SQL: &str = r#"
    SELECT
        t.id,
        t.title,
        t.model_id,
        t.created_at,
        t.updated_at,
        (
            SELECT m.content
            FROM chat_messages m
            WHERE m.thread_id = t.id
            ORDER BY m.created_at DESC, m.id DESC
            LIMIT 1
        ) AS last_message_preview,
        (
            SELECT COUNT(1)
            FROM chat_messages m
            WHERE m.thread_id = t.id
        ) AS message_count
    FROM chat_threads t
    WHERE t.id = ?1
"#;

const CHAT_MESSAGES_LIST_SQL: &str = r#"
    SELECT
        id,
        thread_id,
        role,
        content,
        content_parts,
        created_at,
        tokens_generated,
        generation_time_ms
    FROM chat_messages
    WHERE thread_id = ?1
    ORDER BY created_at ASC, id ASC
"#;

async fn fetch_thread_summary(
    db: &DatabaseConnection,
    thread_id: &str,
) -> anyhow::Result<Option<ChatThreadSummary>> {
    let row = db
        .query_one_raw(Statement::from_sql_and_values(
            DbBackend::Sqlite,
            THREAD_SUMMARY_BY_ID_SQL,
            vec![thread_id.into()],
        ))
        .await
        .context("Failed to load chat thread summary")?;
    row.as_ref().map(map_thread_summary).transpose()
}

fn map_thread_summary(row: &QueryResult) -> anyhow::Result<ChatThreadSummary> {
    let message_count_raw: i64 = row.try_get_by_index(6)?;
    Ok(ChatThreadSummary {
        id: row.try_get_by_index(0)?,
        title: row.try_get_by_index(1)?,
        model_id: row.try_get_by_index(2)?,
        created_at: i64_to_u64(row.try_get_by_index(3)?),
        updated_at: i64_to_u64(row.try_get_by_index(4)?),
        last_message_preview: row.try_get_by_index(5)?,
        message_count: i64_to_usize(message_count_raw),
    })
}

fn map_thread_message(row: &QueryResult) -> anyhow::Result<ChatThreadMessage> {
    let content_parts_raw: Option<String> = row.try_get_by_index(4)?;
    let content_parts = content_parts_raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Vec<JsonValue>>(raw).ok());
    let tokens_generated_raw: Option<i64> = row.try_get_by_index(6)?;
    Ok(ChatThreadMessage {
        id: row.try_get_by_index(0)?,
        thread_id: row.try_get_by_index(1)?,
        role: row.try_get_by_index(2)?,
        content: row.try_get_by_index(3)?,
        content_parts,
        created_at: i64_to_u64(row.try_get_by_index(5)?),
        tokens_generated: tokens_generated_raw.map(i64_to_usize),
        generation_time_ms: row.try_get_by_index(7)?,
    })
}

fn now_unix_millis_i64() -> i64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    duration.as_millis() as i64
}

fn sanitize_thread_title(raw: Option<&str>) -> String {
    let normalized = raw
        .unwrap_or("")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if normalized.is_empty() {
        DEFAULT_THREAD_TITLE.to_string()
    } else {
        truncate_string(&normalized, 80)
    }
}

fn truncate_string(input: &str, max_chars: usize) -> String {
    let mut result = String::new();
    for (idx, ch) in input.chars().enumerate() {
        if idx >= max_chars {
            break;
        }
        result.push(ch);
    }
    if input.chars().count() > max_chars {
        result.push_str("...");
    }
    result
}

fn opt_usize_to_i64(value: Option<usize>) -> anyhow::Result<Option<i64>> {
    match value {
        Some(number) => Ok(Some(
            i64::try_from(number).context("Numeric conversion overflow for tokens_generated")?,
        )),
        None => Ok(None),
    }
}

fn i64_to_u64(value: i64) -> u64 {
    if value.is_negative() { 0 } else { value as u64 }
}

fn i64_to_usize(value: i64) -> usize {
    if value.is_negative() {
        0
    } else {
        value as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::env_lock;
    use serde_json::json;
    use std::future::Future;
    use tempfile::TempDir;

    async fn with_env_lock<T>(action: impl Future<Output = T>) -> T {
        let _guard = env_lock();
        action.await
    }

    fn setup_store() -> (TempDir, ChatStore) {
        let temp_dir = tempfile::tempdir().expect("temp dir should create");
        let db_path = temp_dir.path().join("izwi.sqlite3");
        let media_root = temp_dir.path().join("media");

        std::env::set_var("IZWI_DB_PATH", &db_path);
        std::env::set_var("IZWI_MEDIA_DIR", &media_root);

        let store = ChatStore::initialize().expect("store should init");
        (temp_dir, store)
    }

    fn clear_env() {
        std::env::remove_var("IZWI_DB_PATH");
        std::env::remove_var("IZWI_MEDIA_DIR");
    }

    #[tokio::test]
    async fn persists_threads_messages_and_content_parts() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            assert!(
                store
                    .list_threads()
                    .await
                    .expect("threads should list")
                    .is_empty()
            );

            let thread = store
                .create_thread(
                    Some("  A   thread title  ".to_string()),
                    Some("qwen".to_string()),
                )
                .await
                .expect("thread should create");
            assert_eq!(thread.title, "A thread title");
            assert_eq!(thread.message_count, 0);

            let content_parts = vec![json!({"type": "text", "text": "hello"})];
            let user_message = store
                .append_message(
                    thread.id.clone(),
                    "user".to_string(),
                    "hello".to_string(),
                    Some(content_parts.clone()),
                    Some("qwen-updated".to_string()),
                    None,
                    None,
                )
                .await
                .expect("user message should append");
            assert_eq!(user_message.content_parts, Some(content_parts.clone()));

            let assistant_message = store
                .append_message(
                    thread.id.clone(),
                    "assistant".to_string(),
                    "world".to_string(),
                    None,
                    Some("qwen-updated".to_string()),
                    Some(42),
                    Some(12.5),
                )
                .await
                .expect("assistant message should append");
            assert_eq!(assistant_message.tokens_generated, Some(42));

            let messages = store
                .list_messages(thread.id.clone())
                .await
                .expect("messages should list");
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[0].id, user_message.id);
            assert_eq!(messages[0].content_parts, Some(content_parts));
            assert_eq!(messages[1].id, assistant_message.id);
            assert_eq!(messages[1].generation_time_ms, Some(12.5));

            let summary = store
                .get_thread(thread.id.clone())
                .await
                .expect("thread should load")
                .expect("thread should exist");
            assert_eq!(summary.model_id.as_deref(), Some("qwen-updated"));
            assert_eq!(summary.last_message_preview.as_deref(), Some("world"));
            assert_eq!(summary.message_count, 2);

            let updated = store
                .update_thread_title(thread.id.clone(), "Renamed".to_string())
                .await
                .expect("title should update")
                .expect("thread should exist");
            assert_eq!(updated.title, "Renamed");

            assert!(
                store
                    .delete_thread(thread.id.clone())
                    .await
                    .expect("thread should delete")
            );
            assert!(
                store
                    .list_messages(thread.id)
                    .await
                    .expect("messages should list")
                    .is_empty()
            );
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn append_message_requires_existing_thread() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let err = store
                .append_message(
                    "missing".to_string(),
                    "user".to_string(),
                    "hello".to_string(),
                    None,
                    None,
                    None,
                    None,
                )
                .await
                .expect_err("missing thread should fail");
            assert!(err.to_string().contains("Thread not found"));
            clear_env();
        })
        .await;
    }
}
