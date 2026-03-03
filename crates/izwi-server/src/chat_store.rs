//! Persistent chat thread storage backed by SQLite.

use anyhow::{anyhow, Context};
use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;
use serde_json::Value as JsonValue;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::task;

use crate::storage_layout;

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
    db_path: PathBuf,
}

impl ChatStore {
    pub fn initialize() -> anyhow::Result<Self> {
        let db_path = storage_layout::resolve_db_path();
        let media_root = storage_layout::resolve_media_root();

        storage_layout::ensure_storage_dirs(&db_path, &media_root)
            .context("Failed to prepare chat storage layout")?;

        let conn = storage_layout::open_sqlite_connection(&db_path)
            .with_context(|| format!("Failed to open chat database: {}", db_path.display()))?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS chat_threads (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model_id TEXT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
                content TEXT NOT NULL,
                content_parts TEXT NULL,
                created_at INTEGER NOT NULL,
                tokens_generated INTEGER NULL,
                generation_time_ms REAL NULL,
                FOREIGN KEY(thread_id) REFERENCES chat_threads(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chat_threads_updated_at
                ON chat_threads(updated_at DESC, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_created_at
                ON chat_messages(thread_id, created_at, id);
            "#,
        )
        .context("Failed to initialize chat database schema")?;
        ensure_chat_messages_content_parts_column(&conn)?;

        Ok(Self { db_path })
    }

    pub async fn list_threads(&self) -> anyhow::Result<Vec<ChatThreadSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
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
                "#,
            )?;

            let rows = stmt.query_map([], map_thread_summary)?;
            let mut threads = Vec::new();
            for row in rows {
                threads.push(row?);
            }
            Ok(threads)
        })
        .await
    }

    pub async fn get_thread(&self, thread_id: String) -> anyhow::Result<Option<ChatThreadSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let thread = fetch_thread_summary(&conn, &thread_id)?;
            Ok(thread)
        })
        .await
    }

    pub async fn create_thread(
        &self,
        title: Option<String>,
        model_id: Option<String>,
    ) -> anyhow::Result<ChatThreadSummary> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let thread_id = format!("thread_{}", uuid::Uuid::new_v4().simple());
            let resolved_title = sanitize_thread_title(title.as_deref());

            conn.execute(
                r#"
                INSERT INTO chat_threads (id, title, model_id, created_at, updated_at)
                VALUES (?1, ?2, ?3, ?4, ?5)
                "#,
                params![thread_id, resolved_title, model_id, now, now],
            )?;

            Ok(ChatThreadSummary {
                id: thread_id,
                title: resolved_title,
                model_id,
                created_at: now as u64,
                updated_at: now as u64,
                last_message_preview: None,
                message_count: 0,
            })
        })
        .await
    }

    pub async fn delete_thread(&self, thread_id: String) -> anyhow::Result<bool> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let deleted_rows =
                conn.execute("DELETE FROM chat_threads WHERE id = ?1", params![thread_id])?;
            Ok(deleted_rows > 0)
        })
        .await
    }

    pub async fn list_messages(&self, thread_id: String) -> anyhow::Result<Vec<ChatThreadMessage>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let mut stmt = conn.prepare(
                r#"
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
                "#,
            )?;

            let rows = stmt.query_map(params![thread_id], map_thread_message)?;
            let mut messages = Vec::new();
            for row in rows {
                messages.push(row?);
            }
            Ok(messages)
        })
        .await
    }

    pub async fn update_thread_title(
        &self,
        thread_id: String,
        title: String,
    ) -> anyhow::Result<Option<ChatThreadSummary>> {
        self.run_blocking(move |db_path| {
            let conn = storage_layout::open_sqlite_connection(&db_path)?;
            let now = now_unix_millis_i64();
            let resolved_title = sanitize_thread_title(Some(title.as_str()));

            let updated_rows = conn.execute(
                "UPDATE chat_threads SET title = ?1, updated_at = ?2 WHERE id = ?3",
                params![resolved_title, now, &thread_id],
            )?;

            if updated_rows == 0 {
                return Ok(None);
            }

            fetch_thread_summary(&conn, &thread_id)
        })
        .await
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
        self.run_blocking(move |db_path| {
            let mut conn = storage_layout::open_sqlite_connection(&db_path)?;
            let tx = conn.transaction()?;

            let thread_exists = tx
                .query_row(
                    "SELECT 1 FROM chat_threads WHERE id = ?1 LIMIT 1",
                    params![thread_id],
                    |_| Ok(()),
                )
                .optional()?
                .is_some();

            if !thread_exists {
                return Err(anyhow!("Thread not found"));
            }

            let now = now_unix_millis_i64();
            let message_id = format!("msg_{}", uuid::Uuid::new_v4().simple());
            let tokens_i64 = opt_usize_to_i64(tokens_generated)?;
            let serialized_content_parts = content_parts
                .as_ref()
                .map(serde_json::to_string)
                .transpose()
                .context("Failed serializing chat message content_parts")?;

            tx.execute(
                r#"
                INSERT INTO chat_messages (
                    id,
                    thread_id,
                    role,
                    content,
                    content_parts,
                    created_at,
                    tokens_generated,
                    generation_time_ms
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                "#,
                params![
                    message_id,
                    thread_id,
                    role,
                    content,
                    serialized_content_parts,
                    now,
                    tokens_i64,
                    generation_time_ms
                ],
            )?;

            tx.execute(
                "UPDATE chat_threads SET updated_at = ?1, model_id = ?2 WHERE id = ?3",
                params![now, model_id, thread_id],
            )?;

            tx.commit()?;

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
        })
        .await
    }

    async fn run_blocking<F, T>(&self, task_fn: F) -> anyhow::Result<T>
    where
        F: FnOnce(PathBuf) -> anyhow::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let db_path = self.db_path.clone();
        task::spawn_blocking(move || task_fn(db_path))
            .await
            .map_err(|err| anyhow!("Chat storage worker failed: {err}"))?
    }
}

fn fetch_thread_summary(
    conn: &Connection,
    thread_id: &str,
) -> anyhow::Result<Option<ChatThreadSummary>> {
    let thread = conn
        .query_row(
            r#"
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
            "#,
            params![thread_id],
            map_thread_summary,
        )
        .optional()?;
    Ok(thread)
}

fn map_thread_summary(row: &Row<'_>) -> rusqlite::Result<ChatThreadSummary> {
    let message_count_raw: i64 = row.get(6)?;
    Ok(ChatThreadSummary {
        id: row.get(0)?,
        title: row.get(1)?,
        model_id: row.get(2)?,
        created_at: i64_to_u64(row.get(3)?),
        updated_at: i64_to_u64(row.get(4)?),
        last_message_preview: row.get(5)?,
        message_count: i64_to_usize(message_count_raw),
    })
}

fn map_thread_message(row: &Row<'_>) -> rusqlite::Result<ChatThreadMessage> {
    let content_parts_raw: Option<String> = row.get(4)?;
    let content_parts = content_parts_raw
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Vec<JsonValue>>(raw).ok());
    let tokens_generated_raw: Option<i64> = row.get(6)?;
    Ok(ChatThreadMessage {
        id: row.get(0)?,
        thread_id: row.get(1)?,
        role: row.get(2)?,
        content: row.get(3)?,
        content_parts,
        created_at: i64_to_u64(row.get(5)?),
        tokens_generated: tokens_generated_raw.map(i64_to_usize),
        generation_time_ms: row.get(7)?,
    })
}

fn ensure_chat_messages_content_parts_column(conn: &Connection) -> anyhow::Result<()> {
    let mut stmt = conn
        .prepare("PRAGMA table_info(chat_messages)")
        .context("Failed to inspect chat_messages schema")?;
    let mut rows = stmt
        .query([])
        .context("Failed to query chat_messages schema info")?;

    while let Some(row) = rows
        .next()
        .context("Failed reading chat_messages schema row")?
    {
        let name: String = row
            .get(1)
            .context("Failed reading chat_messages column name")?;
        if name == "content_parts" {
            return Ok(());
        }
    }

    conn.execute(
        "ALTER TABLE chat_messages ADD COLUMN content_parts TEXT NULL",
        [],
    )
    .context("Failed adding chat_messages.content_parts column")?;
    Ok(())
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
