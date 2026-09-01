//! Persistent chat thread storage backed by SQLite.

use anyhow::{anyhow, Context};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseConnection, EntityTrait, QueryFilter, QueryOrder,
    QueryResult, Set, TransactionTrait,
};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, Weak};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

use crate::db::{raw, StoreDatabase};
use crate::entity::{chat_messages, chat_threads};
use crate::ids::new_uuid;

const DEFAULT_THREAD_TITLE: &str = "New chat";

#[derive(Debug, Clone, Serialize)]
pub struct ChatThreadSummary {
    pub id: String,
    pub title: String,
    pub model_id: Option<String>,
    pub system_prompt: Option<String>,
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
    turn_locks: Arc<AsyncMutex<HashMap<String, Weak<AsyncMutex<()>>>>>,
}

impl ChatStore {
    pub fn initialize() -> anyhow::Result<Self> {
        Ok(Self {
            db: StoreDatabase::from_default_path()?,
            turn_locks: Arc::new(AsyncMutex::new(HashMap::new())),
        })
    }

    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self {
            db,
            turn_locks: Arc::new(AsyncMutex::new(HashMap::new())),
        }
    }

    /// Serialize the complete read-generate-persist lifecycle for one thread.
    ///
    /// Locks are weakly retained so idle thread ids do not accumulate forever.
    /// The owned guard can move into a streaming response task and remains valid
    /// even when the `ChatStore` handle used to acquire it is dropped.
    pub async fn lock_turn(&self, thread_id: &str) -> OwnedMutexGuard<()> {
        let turn_lock = {
            let mut locks = self.turn_locks.lock().await;
            locks.retain(|_, lock| lock.strong_count() > 0);

            if let Some(lock) = locks.get(thread_id).and_then(Weak::upgrade) {
                lock
            } else {
                let lock = Arc::new(AsyncMutex::new(()));
                locks.insert(thread_id.to_string(), Arc::downgrade(&lock));
                lock
            }
        };

        turn_lock.lock_owned().await
    }

    /// Build the user half of a turn without writing it to storage. The exact
    /// id is returned to streaming clients immediately and is committed only
    /// if generation completes successfully.
    pub async fn prepare_user_message(
        &self,
        thread_id: String,
        content: String,
        content_parts: Option<Vec<JsonValue>>,
    ) -> anyhow::Result<ChatThreadMessage> {
        let db = self.db.connection().await?;
        let thread = chat_threads::Entity::find_by_id(thread_id.clone())
            .one(db)
            .await
            .context("Failed to load chat thread")?
            .ok_or_else(|| anyhow!("Thread not found"))?;
        let previous_timestamp = latest_thread_message_timestamp(db, &thread_id)
            .await?
            .unwrap_or(thread.updated_at)
            .max(thread.updated_at);
        let created_at = now_unix_millis_i64().max(previous_timestamp.saturating_add(1));

        Ok(ChatThreadMessage {
            id: new_uuid(),
            thread_id,
            role: "user".to_string(),
            content,
            content_parts,
            created_at: i64_to_u64(created_at),
            tokens_generated: None,
            generation_time_ms: None,
        })
    }

    pub async fn list_threads(&self) -> anyhow::Result<Vec<ChatThreadSummary>> {
        let db = self.db.connection().await?;
        let rows = db
            .query_all_raw(raw::statement_without_values(db, THREAD_SUMMARY_LIST_SQL))
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
        self.create_thread_with_system_prompt(title, model_id, None)
            .await
    }

    pub async fn create_thread_with_system_prompt(
        &self,
        title: Option<String>,
        model_id: Option<String>,
        system_prompt: Option<String>,
    ) -> anyhow::Result<ChatThreadSummary> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let thread_id = new_uuid();
        let resolved_title = sanitize_thread_title(title.as_deref());
        let system_prompt = sanitize_system_prompt(system_prompt.as_deref());

        chat_threads::Entity::insert(chat_threads::ActiveModel {
            id: Set(thread_id.clone()),
            title: Set(resolved_title.clone()),
            model_id: Set(model_id.clone()),
            system_prompt: Set(system_prompt.clone()),
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
            system_prompt,
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
            .query_all_raw(raw::statement(
                db,
                CHAT_MESSAGES_LIST_SQL,
                vec![thread_id.into()],
            )?)
            .await
            .context("Failed to list chat messages")?;
        rows.iter().map(map_thread_message).collect()
    }

    /// Update thread-owned settings. `Some("")` explicitly clears the system
    /// prompt, while `None` inherits the existing value.
    pub async fn update_thread_settings(
        &self,
        thread_id: String,
        title: Option<String>,
        system_prompt: Option<String>,
    ) -> anyhow::Result<Option<ChatThreadSummary>> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let mut update = chat_threads::Entity::update_many()
            .col_expr(chat_threads::Column::UpdatedAt, Expr::value(now));
        if let Some(title) = title {
            update = update.col_expr(
                chat_threads::Column::Title,
                Expr::value(sanitize_thread_title(Some(title.as_str()))),
            );
        }
        if let Some(system_prompt) = system_prompt {
            update = update.col_expr(
                chat_threads::Column::SystemPrompt,
                Expr::value(sanitize_system_prompt(Some(system_prompt.as_str()))),
            );
        }

        let result = update
            .filter(chat_threads::Column::Id.eq(thread_id.clone()))
            .exec(db)
            .await
            .context("Failed to update chat thread settings")?;

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

        let thread = chat_threads::Entity::find_by_id(thread_id.clone())
            .one(&tx)
            .await
            .context("Failed to load chat thread")?
            .ok_or_else(|| anyhow!("Thread not found"))?;

        let previous_timestamp = latest_thread_message_timestamp(&tx, &thread_id)
            .await?
            .unwrap_or(thread.updated_at)
            .max(thread.updated_at);
        let created_at = now_unix_millis_i64().max(previous_timestamp.saturating_add(1));
        if created_at == i64::MAX {
            return Err(anyhow!("Chat message timestamp space is exhausted"));
        }
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
            created_at: Set(created_at),
            tokens_generated: Set(tokens_i64),
            generation_time_ms: Set(generation_time_ms),
        })
        .exec(&tx)
        .await
        .context("Failed to append chat message")?;

        chat_threads::Entity::update_many()
            .col_expr(chat_threads::Column::UpdatedAt, Expr::value(created_at))
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
            created_at: i64_to_u64(created_at),
            tokens_generated,
            generation_time_ms,
        })
    }

    /// Atomically append a complete user/assistant turn. A generation failure
    /// never calls this method, so no orphan user message can become part of a
    /// future prompt. The assistant timestamp is forced after the pending user
    /// timestamp to make ordering deterministic even within one millisecond.
    /// Append a successful turn and an optional thread system-prompt change in
    /// one transaction. `Some("")` clears the prompt; `None` preserves it.
    pub async fn append_turn_with_system_prompt(
        &self,
        mut user_message: ChatThreadMessage,
        assistant_content: String,
        model_id: String,
        tokens_generated: usize,
        generation_time_ms: f64,
        system_prompt_update: Option<String>,
    ) -> anyhow::Result<(ChatThreadMessage, ChatThreadMessage)> {
        if user_message.role != "user" {
            return Err(anyhow!("Chat turn must start with a user message"));
        }
        if user_message.thread_id.is_empty() {
            return Err(anyhow!("Chat turn requires a thread id"));
        }

        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start chat turn transaction")?;

        let thread = chat_threads::Entity::find_by_id(user_message.thread_id.clone())
            .one(&tx)
            .await
            .context("Failed to load chat thread")?
            .ok_or_else(|| anyhow!("Thread not found"))?;
        let previous_timestamp = latest_thread_message_timestamp(&tx, &user_message.thread_id)
            .await?
            .unwrap_or(thread.updated_at)
            .max(thread.updated_at);
        let requested_user_timestamp = i64::try_from(user_message.created_at)
            .context("Pending chat user timestamp is out of range")?;
        let user_created_at = requested_user_timestamp.max(previous_timestamp.saturating_add(1));
        if user_created_at == i64::MAX {
            return Err(anyhow!("Chat message timestamp space is exhausted"));
        }
        user_message.created_at = i64_to_u64(user_created_at);
        let assistant_created_at = now_unix_millis_i64().max(user_created_at.saturating_add(1));
        let assistant_message = ChatThreadMessage {
            id: new_uuid(),
            thread_id: user_message.thread_id.clone(),
            role: "assistant".to_string(),
            content: assistant_content,
            content_parts: None,
            created_at: i64_to_u64(assistant_created_at),
            tokens_generated: Some(tokens_generated),
            generation_time_ms: Some(generation_time_ms),
        };

        insert_thread_message(&tx, &user_message).await?;
        insert_thread_message(&tx, &assistant_message).await?;

        let mut thread_update = chat_threads::Entity::update_many()
            .col_expr(
                chat_threads::Column::UpdatedAt,
                Expr::value(assistant_created_at),
            )
            .col_expr(chat_threads::Column::ModelId, Expr::value(Some(model_id)));
        if let Some(system_prompt_update) = system_prompt_update {
            thread_update = thread_update.col_expr(
                chat_threads::Column::SystemPrompt,
                Expr::value(sanitize_system_prompt(Some(system_prompt_update.as_str()))),
            );
        }
        thread_update
            .filter(chat_threads::Column::Id.eq(user_message.thread_id.clone()))
            .exec(&tx)
            .await
            .context("Failed to update chat thread metadata")?;

        tx.commit()
            .await
            .context("Failed to commit chat turn transaction")?;

        Ok((user_message, assistant_message))
    }
}

async fn latest_thread_message_timestamp<C>(db: &C, thread_id: &str) -> anyhow::Result<Option<i64>>
where
    C: ConnectionTrait,
{
    Ok(chat_messages::Entity::find()
        .filter(chat_messages::Column::ThreadId.eq(thread_id.to_string()))
        .order_by_desc(chat_messages::Column::CreatedAt)
        .order_by_desc(chat_messages::Column::Id)
        .one(db)
        .await
        .context("Failed to load latest chat message")?
        .map(|message| message.created_at))
}

async fn insert_thread_message<C>(db: &C, message: &ChatThreadMessage) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    let serialized_content_parts = message
        .content_parts
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .context("Failed serializing chat message content_parts")?;

    chat_messages::Entity::insert(chat_messages::ActiveModel {
        id: Set(message.id.clone()),
        thread_id: Set(message.thread_id.clone()),
        role: Set(message.role.clone()),
        content: Set(message.content.clone()),
        content_parts: Set(serialized_content_parts),
        created_at: Set(i64::try_from(message.created_at).unwrap_or(i64::MAX)),
        tokens_generated: Set(opt_usize_to_i64(message.tokens_generated)?),
        generation_time_ms: Set(message.generation_time_ms),
    })
    .exec(db)
    .await
    .context("Failed to append chat turn message")?;

    Ok(())
}

const THREAD_SUMMARY_LIST_SQL: &str = r#"
    SELECT
        t.id,
        t.title,
        t.model_id,
        t.system_prompt,
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
        t.system_prompt,
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
        .query_one_raw(raw::statement(
            db,
            THREAD_SUMMARY_BY_ID_SQL,
            vec![thread_id.into()],
        )?)
        .await
        .context("Failed to load chat thread summary")?;
    row.as_ref().map(map_thread_summary).transpose()
}

fn map_thread_summary(row: &QueryResult) -> anyhow::Result<ChatThreadSummary> {
    let message_count_raw: i64 = row.try_get_by_index(7)?;
    Ok(ChatThreadSummary {
        id: row.try_get_by_index(0)?,
        title: row.try_get_by_index(1)?,
        model_id: row.try_get_by_index(2)?,
        system_prompt: row.try_get_by_index(3)?,
        created_at: i64_to_u64(row.try_get_by_index(4)?),
        updated_at: i64_to_u64(row.try_get_by_index(5)?),
        last_message_preview: row.try_get_by_index(6)?,
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

pub(crate) fn sanitize_system_prompt(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|prompt| !prompt.is_empty())
        .map(|prompt| prompt.chars().take(65_536).collect())
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
    if value.is_negative() {
        0
    } else {
        value as u64
    }
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
            assert!(store
                .list_threads()
                .await
                .expect("threads should list")
                .is_empty());

            let thread = store
                .create_thread_with_system_prompt(
                    Some("  A   thread title  ".to_string()),
                    Some("qwen".to_string()),
                    Some("  Be concise.  ".to_string()),
                )
                .await
                .expect("thread should create");
            assert_eq!(thread.title, "A thread title");
            assert_eq!(thread.system_prompt.as_deref(), Some("Be concise."));
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

            // Force the predecessor ahead of wall clock so message ordering
            // cannot accidentally rely on UUID tie-breaking or clock progress.
            let forced_user_created_at = i64::try_from(user_message.created_at)
                .expect("message timestamp should fit i64")
                .saturating_add(60_000);
            let db = store
                .db
                .connection()
                .await
                .expect("database should connect");
            chat_messages::Entity::update_many()
                .col_expr(
                    chat_messages::Column::CreatedAt,
                    Expr::value(forced_user_created_at),
                )
                .filter(chat_messages::Column::Id.eq(user_message.id.clone()))
                .exec(db)
                .await
                .expect("message timestamp should update");

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
            assert_eq!(
                assistant_message.created_at,
                i64_to_u64(forced_user_created_at.saturating_add(1))
            );

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
                .update_thread_settings(
                    thread.id.clone(),
                    Some("Renamed".to_string()),
                    Some("".to_string()),
                )
                .await
                .expect("title should update")
                .expect("thread should exist");
            assert_eq!(updated.title, "Renamed");
            assert!(updated.system_prompt.is_none());

            assert!(store
                .delete_thread(thread.id.clone())
                .await
                .expect("thread should delete"));
            assert!(store
                .list_messages(thread.id)
                .await
                .expect("messages should list")
                .is_empty());
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

    #[tokio::test]
    async fn append_turn_publishes_both_messages_atomically_and_in_order() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread(None, None)
                .await
                .expect("thread should create");
            let content_parts = vec![json!({"type": "text", "text": "hello"})];
            let pending_user = store
                .prepare_user_message(
                    thread.id.clone(),
                    "hello".to_string(),
                    Some(content_parts.clone()),
                )
                .await
                .expect("pending user should prepare");

            assert!(store
                .list_messages(thread.id.clone())
                .await
                .expect("messages should list before commit")
                .is_empty());

            let (user, assistant) = store
                .append_turn_with_system_prompt(
                    pending_user.clone(),
                    "world".to_string(),
                    "Qwen3.5-4B".to_string(),
                    7,
                    12.5,
                    None,
                )
                .await
                .expect("turn should append");
            assert_eq!(user.id, pending_user.id);
            assert!(assistant.created_at > user.created_at);

            let messages = store
                .list_messages(thread.id.clone())
                .await
                .expect("messages should list after commit");
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[0].id, user.id);
            assert_eq!(messages[0].content_parts, Some(content_parts));
            assert_eq!(messages[1].id, assistant.id);
            assert_eq!(messages[1].tokens_generated, Some(7));

            let summary = store
                .get_thread(thread.id)
                .await
                .expect("thread should load")
                .expect("thread should exist");
            assert_eq!(summary.message_count, 2);
            assert_eq!(summary.last_message_preview.as_deref(), Some("world"));
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn consecutive_turns_receive_strictly_monotonic_timestamps() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread_with_system_prompt(None, None, Some("original prompt".to_string()))
                .await
                .expect("thread should create");
            let first_pending = store
                .prepare_user_message(thread.id.clone(), "first".to_string(), None)
                .await
                .expect("first pending user");
            let (_, first_assistant) = store
                .append_turn_with_system_prompt(
                    first_pending,
                    "first answer".to_string(),
                    "Qwen3.5-4B".to_string(),
                    2,
                    1.0,
                    Some("updated prompt".to_string()),
                )
                .await
                .expect("first turn");

            // Simulate a wall-clock tie or rollback. Transactional ordering
            // must advance past the already committed assistant timestamp.
            let mut second_pending = store
                .prepare_user_message(thread.id.clone(), "second".to_string(), None)
                .await
                .expect("second pending user");
            second_pending.created_at = first_assistant.created_at;
            let (second_user, second_assistant) = store
                .append_turn_with_system_prompt(
                    second_pending,
                    "second answer".to_string(),
                    "Qwen3.5-4B".to_string(),
                    2,
                    1.0,
                    None,
                )
                .await
                .expect("second turn");

            assert!(second_user.created_at > first_assistant.created_at);
            assert!(second_assistant.created_at > second_user.created_at);
            let messages = store
                .list_messages(thread.id.clone())
                .await
                .expect("messages");
            assert_eq!(
                messages
                    .windows(2)
                    .map(|pair| pair[0].created_at < pair[1].created_at)
                    .collect::<Vec<_>>(),
                vec![true, true, true]
            );
            let summary = store
                .get_thread(thread.id)
                .await
                .expect("thread lookup")
                .expect("thread exists");
            assert_eq!(summary.system_prompt.as_deref(), Some("updated prompt"));
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn failed_turn_rolls_back_messages_and_system_prompt_update() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread_with_system_prompt(None, None, Some("original prompt".to_string()))
                .await
                .expect("thread should create");
            let pending = store
                .prepare_user_message(thread.id.clone(), "hello".to_string(), None)
                .await
                .expect("pending user");

            store
                .append_turn_with_system_prompt(
                    pending,
                    "answer".to_string(),
                    "Qwen3.5-4B".to_string(),
                    usize::MAX,
                    1.0,
                    Some("must not persist".to_string()),
                )
                .await
                .expect_err("overflowing token metadata should fail the transaction");

            assert!(store
                .list_messages(thread.id.clone())
                .await
                .expect("messages")
                .is_empty());
            let summary = store
                .get_thread(thread.id)
                .await
                .expect("thread lookup")
                .expect("thread exists");
            assert_eq!(summary.system_prompt.as_deref(), Some("original prompt"));
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn turn_lock_serializes_work_for_the_same_thread() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let first_guard = store.lock_turn("thread-1").await;
            let waiter_store = store.clone();
            let waiter = tokio::spawn(async move {
                let _guard = waiter_store.lock_turn("thread-1").await;
            });

            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(20), async {
                    while !waiter.is_finished() {
                        tokio::task::yield_now().await;
                    }
                })
                .await
                .is_err()
            );

            drop(first_guard);
            tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
                .await
                .expect("waiter should acquire after release")
                .expect("waiter task should complete");
            clear_env();
        })
        .await;
    }
}
