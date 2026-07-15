//! Persistent chat thread storage backed by SQLite.

use anyhow::{anyhow, Context};
use sea_orm::sea_query::Expr;
use sea_orm::{
    ColumnTrait, ConnectionTrait, DatabaseConnection, EntityTrait, QueryFilter, QueryResult, Set,
    TransactionTrait,
};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex, Weak};
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
    turn_locks: Arc<StdMutex<HashMap<String, Weak<AsyncMutex<()>>>>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChatThreadSystemPromptUpdate {
    Retain,
    Replace(Option<String>),
}

impl ChatStore {
    pub fn initialize() -> anyhow::Result<Self> {
        Ok(Self {
            db: StoreDatabase::from_default_path()?,
            turn_locks: Arc::new(StdMutex::new(HashMap::new())),
        })
    }

    pub fn initialize_with_database(db: StoreDatabase) -> Self {
        Self {
            db,
            turn_locks: Arc::new(StdMutex::new(HashMap::new())),
        }
    }

    /// Serialize the complete history-read, generation, and persistence span
    /// for one thread while allowing unrelated threads to proceed independently.
    pub async fn acquire_turn(&self, thread_id: &str) -> OwnedMutexGuard<()> {
        let turn_lock = {
            let mut locks = self
                .turn_locks
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            locks.retain(|_, lock| lock.strong_count() > 0);
            match locks.get(thread_id).and_then(Weak::upgrade) {
                Some(lock) => lock,
                None => {
                    let lock = Arc::new(AsyncMutex::new(()));
                    locks.insert(thread_id.to_string(), Arc::downgrade(&lock));
                    lock
                }
            }
        };
        turn_lock.lock_owned().await
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

    pub async fn update_thread_title(
        &self,
        thread_id: String,
        title: String,
    ) -> anyhow::Result<Option<ChatThreadSummary>> {
        let db = self.db.connection().await?;
        let now = now_unix_millis_i64();
        let resolved_title = sanitize_thread_title(Some(title.as_str()));
        let tx = db
            .begin()
            .await
            .context("Failed to start chat thread title transaction")?;
        let Some(existing) = chat_threads::Entity::find_by_id(thread_id.clone())
            .one(&tx)
            .await
            .context("Failed to load chat thread for title update")?
        else {
            return Ok(None);
        };
        let updated_at = now.max(existing.updated_at);

        chat_threads::Entity::update_many()
            .col_expr(
                chat_threads::Column::Title,
                Expr::value(resolved_title.clone()),
            )
            .col_expr(chat_threads::Column::UpdatedAt, Expr::value(updated_at))
            .filter(chat_threads::Column::Id.eq(thread_id.clone()))
            .exec(&tx)
            .await
            .context("Failed to update chat thread title")?;
        tx.commit()
            .await
            .context("Failed to commit chat thread title transaction")?;

        fetch_thread_summary(db, &thread_id).await
    }

    /// Build the user half of a chat turn without making it durable yet.
    ///
    /// Streaming responses expose this record in their `start` event. The exact
    /// same id and timestamp are committed with the assistant response only
    /// after generation succeeds.
    pub fn prepare_user_message(
        &self,
        thread_id: String,
        content: String,
        content_parts: Option<Vec<JsonValue>>,
        after: u64,
    ) -> ChatThreadMessage {
        ChatThreadMessage {
            id: new_uuid(),
            thread_id,
            role: "user".to_string(),
            content,
            content_parts,
            created_at: now_unix_millis_u64().max(after.saturating_add(1)),
            tokens_generated: None,
            generation_time_ms: None,
        }
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

        let previous_updated_at = thread_updated_at(&tx, &thread_id).await?;

        let message = ChatThreadMessage {
            id: new_uuid(),
            thread_id: thread_id.clone(),
            role,
            content,
            content_parts,
            created_at: now_unix_millis_u64().max(previous_updated_at.saturating_add(1)),
            tokens_generated,
            generation_time_ms,
        };
        insert_message(&tx, &message).await?;
        update_thread_metadata(
            &tx,
            &thread_id,
            model_id,
            message.created_at,
            &ChatThreadSystemPromptUpdate::Retain,
        )
        .await?;

        tx.commit()
            .await
            .context("Failed to commit chat message transaction")?;

        Ok(message)
    }

    /// Atomically persist a completed turn and its effective thread system
    /// prompt update. Failed or cancelled generation never mutates the prompt.
    pub async fn append_turn_with_system_prompt_update(
        &self,
        user_message: ChatThreadMessage,
        assistant_content: String,
        model_id: Option<String>,
        tokens_generated: usize,
        generation_time_ms: f64,
        system_prompt_update: ChatThreadSystemPromptUpdate,
    ) -> anyhow::Result<(ChatThreadMessage, ChatThreadMessage)> {
        validate_pending_user_message(&user_message)?;

        let assistant_message = ChatThreadMessage {
            id: new_uuid(),
            thread_id: user_message.thread_id.clone(),
            role: "assistant".to_string(),
            content: assistant_content,
            content_parts: None,
            created_at: now_unix_millis_u64().max(user_message.created_at.saturating_add(1)),
            tokens_generated: Some(tokens_generated),
            generation_time_ms: Some(generation_time_ms),
        };

        let db = self.db.connection().await?;
        let tx = db
            .begin()
            .await
            .context("Failed to start chat turn transaction")?;
        ensure_thread_exists(&tx, &user_message.thread_id).await?;
        insert_message(&tx, &user_message)
            .await
            .context("Failed to append chat turn user message")?;
        insert_message(&tx, &assistant_message)
            .await
            .context("Failed to append chat turn assistant message")?;
        update_thread_metadata(
            &tx,
            &user_message.thread_id,
            model_id,
            assistant_message.created_at,
            &system_prompt_update,
        )
        .await?;
        tx.commit()
            .await
            .context("Failed to commit chat turn transaction")?;

        Ok((user_message, assistant_message))
    }
}

async fn ensure_thread_exists<C>(db: &C, thread_id: &str) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    thread_updated_at(db, thread_id).await.map(|_| ())
}

async fn thread_updated_at<C>(db: &C, thread_id: &str) -> anyhow::Result<u64>
where
    C: ConnectionTrait,
{
    let thread = chat_threads::Entity::find_by_id(thread_id.to_string())
        .one(db)
        .await
        .context("Failed to load chat thread")?
        .ok_or_else(|| anyhow!("Thread not found"))?;
    Ok(i64_to_u64(thread.updated_at))
}

async fn insert_message<C>(db: &C, message: &ChatThreadMessage) -> anyhow::Result<()>
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
        created_at: Set(u64_to_i64(message.created_at, "created_at")?),
        tokens_generated: Set(opt_usize_to_i64(message.tokens_generated)?),
        generation_time_ms: Set(message.generation_time_ms),
    })
    .exec(db)
    .await
    .context("Failed to append chat message")?;
    Ok(())
}

async fn update_thread_metadata<C>(
    db: &C,
    thread_id: &str,
    model_id: Option<String>,
    updated_at: u64,
    system_prompt_update: &ChatThreadSystemPromptUpdate,
) -> anyhow::Result<()>
where
    C: ConnectionTrait,
{
    let mut update = chat_threads::Entity::update_many()
        .col_expr(
            chat_threads::Column::UpdatedAt,
            Expr::value(u64_to_i64(updated_at, "updated_at")?),
        )
        .col_expr(chat_threads::Column::ModelId, Expr::value(model_id));
    if let ChatThreadSystemPromptUpdate::Replace(system_prompt) = system_prompt_update {
        update = update.col_expr(
            chat_threads::Column::SystemPrompt,
            Expr::value(system_prompt.clone()),
        );
    }
    update
        .filter(chat_threads::Column::Id.eq(thread_id.to_string()))
        .exec(db)
        .await
        .context("Failed to update chat thread metadata")?;
    Ok(())
}

fn validate_pending_user_message(message: &ChatThreadMessage) -> anyhow::Result<()> {
    if message.role != "user" {
        return Err(anyhow!("Pending chat turn message must have role=user"));
    }
    if message.tokens_generated.is_some() || message.generation_time_ms.is_some() {
        return Err(anyhow!(
            "Pending chat turn user message cannot contain generation statistics"
        ));
    }
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
    i64::try_from(now_unix_millis_u64()).unwrap_or(i64::MAX)
}

fn now_unix_millis_u64() -> u64 {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
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

fn u64_to_i64(value: u64, field: &str) -> anyhow::Result<i64> {
    i64::try_from(value).with_context(|| format!("Numeric conversion overflow for {field}"))
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
    async fn append_turn_persists_both_messages_with_stable_user_identity() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread(None, Some("old-model".to_string()))
                .await
                .expect("thread should create");
            let pending = store.prepare_user_message(
                thread.id.clone(),
                "hello".to_string(),
                Some(vec![json!({"type":"text","text":"hello"})]),
                thread.updated_at,
            );
            let pending_id = pending.id.clone();
            let pending_created_at = pending.created_at;

            let (user, assistant) = store
                .append_turn_with_system_prompt_update(
                    pending,
                    "world".to_string(),
                    Some("new-model".to_string()),
                    7,
                    12.5,
                    ChatThreadSystemPromptUpdate::Retain,
                )
                .await
                .expect("turn should append");

            assert_eq!(user.id, pending_id);
            assert_eq!(user.created_at, pending_created_at);
            assert!(assistant.created_at > user.created_at);
            let messages = store
                .list_messages(thread.id.clone())
                .await
                .expect("messages should list");
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[0].id, user.id);
            assert_eq!(messages[1].id, assistant.id);
            assert_eq!(messages[1].tokens_generated, Some(7));

            let summary = store
                .get_thread(thread.id)
                .await
                .expect("thread should load")
                .expect("thread should exist");
            assert_eq!(summary.model_id.as_deref(), Some("new-model"));
            assert_eq!(summary.last_message_preview.as_deref(), Some("world"));
            assert_eq!(summary.message_count, 2);
            assert_eq!(summary.updated_at, assistant.created_at);
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn system_prompt_round_trips_and_updates_with_completed_turn() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread_with_system_prompt(
                    None,
                    Some("Qwen3.5-4B".to_string()),
                    Some("Be concise.".to_string()),
                )
                .await
                .expect("thread should create");
            assert_eq!(thread.system_prompt.as_deref(), Some("Be concise."));

            let pending = store.prepare_user_message(
                thread.id.clone(),
                "hello".to_string(),
                None,
                thread.updated_at,
            );
            store
                .append_turn_with_system_prompt_update(
                    pending,
                    "world".to_string(),
                    Some("Qwen3.5-4B".to_string()),
                    1,
                    2.0,
                    ChatThreadSystemPromptUpdate::Replace(Some("Answer in one word.".to_string())),
                )
                .await
                .expect("turn should append");

            let summary = store
                .get_thread(thread.id)
                .await
                .expect("thread should load")
                .expect("thread should exist");
            assert_eq!(
                summary.system_prompt.as_deref(),
                Some("Answer in one word.")
            );
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn title_updates_never_move_the_thread_clock_behind_messages() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread(None, Some("Qwen3.5-4B".to_string()))
                .await
                .expect("thread should create");
            let pending = store.prepare_user_message(
                thread.id.clone(),
                "future turn".to_string(),
                None,
                thread.updated_at.saturating_add(10_000),
            );
            let (_, assistant) = store
                .append_turn_with_system_prompt_update(
                    pending,
                    "complete".to_string(),
                    Some("Qwen3.5-4B".to_string()),
                    1,
                    1.0,
                    ChatThreadSystemPromptUpdate::Retain,
                )
                .await
                .expect("turn should append");

            let updated = store
                .update_thread_title(thread.id, "Renamed".to_string())
                .await
                .expect("title should update")
                .expect("thread should exist");
            assert_eq!(updated.updated_at, assistant.created_at);
            clear_env();
        })
        .await;
    }

    #[tokio::test]
    async fn turn_locks_serialize_same_thread_but_not_different_threads() {
        let store = ChatStore::initialize_with_database(StoreDatabase::new(
            tempfile::tempdir()
                .expect("temp dir")
                .path()
                .join("chat.sqlite3"),
        ));
        let first = store.acquire_turn("thread-a").await;
        let other = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            store.acquire_turn("thread-b"),
        )
        .await
        .expect("different thread should not block");
        drop(other);

        let blocked_store = store.clone();
        let waiter = tokio::spawn(async move {
            let _guard = blocked_store.acquire_turn("thread-a").await;
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished());
        drop(first);
        tokio::time::timeout(std::time::Duration::from_millis(100), waiter)
            .await
            .expect("same-thread waiter should proceed after release")
            .expect("waiter should not panic");
    }

    #[tokio::test]
    async fn append_turn_rolls_back_user_when_assistant_insert_fails() {
        with_env_lock(async {
            let (_temp, store) = setup_store();
            let thread = store
                .create_thread_with_system_prompt(
                    None,
                    Some("old-model".to_string()),
                    Some("old prompt".to_string()),
                )
                .await
                .expect("thread should create");
            let original_updated_at = thread.updated_at;
            let db = store.db.connection().await.expect("database connection");
            db.execute_unprepared(
                r#"
                CREATE TRIGGER fail_chat_assistant_insert
                BEFORE INSERT ON chat_messages
                WHEN NEW.role = 'assistant'
                BEGIN
                    SELECT RAISE(ABORT, 'forced assistant insert failure');
                END
                "#,
            )
            .await
            .expect("failure trigger should create");

            let pending = store.prepare_user_message(
                thread.id.clone(),
                "must roll back".to_string(),
                None,
                thread.updated_at,
            );
            let err = store
                .append_turn_with_system_prompt_update(
                    pending,
                    "not persisted".to_string(),
                    Some("new-model".to_string()),
                    3,
                    4.0,
                    ChatThreadSystemPromptUpdate::Replace(Some("new prompt".to_string())),
                )
                .await
                .expect_err("assistant failure should abort the transaction");
            assert!(err.to_string().contains("assistant"));

            assert!(store
                .list_messages(thread.id.clone())
                .await
                .expect("messages should list")
                .is_empty());
            let summary = store
                .get_thread(thread.id)
                .await
                .expect("thread should load")
                .expect("thread should exist");
            assert_eq!(summary.model_id.as_deref(), Some("old-model"));
            assert_eq!(summary.system_prompt.as_deref(), Some("old prompt"));
            assert_eq!(summary.updated_at, original_updated_at);
            assert_eq!(summary.message_count, 0);
            assert!(summary.last_message_preview.is_none());
            clear_env();
        })
        .await;
    }
}
