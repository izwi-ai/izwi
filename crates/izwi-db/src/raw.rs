use anyhow::{bail, Context};
use sea_orm::{ConnectionTrait, DbBackend, Statement, Value};

pub fn statement<C>(db: &C, sql: impl Into<String>, values: Vec<Value>) -> anyhow::Result<Statement>
where
    C: ConnectionTrait,
{
    statement_for_backend(db.get_database_backend(), sql, values)
}

pub fn statement_for_backend(
    backend: DbBackend,
    sql: impl Into<String>,
    values: Vec<Value>,
) -> anyhow::Result<Statement> {
    let (sql, values) = rewrite_indexed_placeholders(backend, sql.into(), values)?;
    Ok(Statement::from_sql_and_values(backend, sql, values))
}

pub fn statement_without_values<C>(db: &C, sql: impl Into<String>) -> Statement
where
    C: ConnectionTrait,
{
    Statement::from_string(db.get_database_backend(), sql)
}

pub fn json_extract_text(
    backend: DbBackend,
    expression: &str,
    key: &str,
) -> anyhow::Result<String> {
    validate_json_key(key)?;
    Ok(match backend {
        DbBackend::Sqlite => format!("json_extract({expression}, '$.{key}')"),
        DbBackend::Postgres => format!("(({expression})::jsonb ->> '{key}')"),
        _ => bail!("Unsupported SeaORM database backend: {backend:?}"),
    })
}

pub fn greatest(backend: DbBackend, left: &str, right: &str) -> anyhow::Result<String> {
    Ok(match backend {
        DbBackend::Sqlite => format!("MAX({left}, {right})"),
        DbBackend::Postgres => format!("GREATEST({left}, {right})"),
        _ => bail!("Unsupported SeaORM database backend: {backend:?}"),
    })
}

fn rewrite_indexed_placeholders(
    backend: DbBackend,
    sql: String,
    values: Vec<Value>,
) -> anyhow::Result<(String, Vec<Value>)> {
    let placeholders = parse_placeholders(&sql)?;
    // Debug and test builds hold every statement to an exact match between
    // the placeholders and the bound values, so a misnumbered placeholder or
    // a value the SQL never reads fails the test suite on either backend.
    // Release builds skip the check: PostgreSQL declares each bound value's
    // type when preparing, so an unread trailing value is not an error there.
    #[cfg(debug_assertions)]
    ensure_placeholders_match_values(&placeholders, values.len())?;
    if values.is_empty() || placeholders.is_empty() {
        return Ok((sql, values));
    }

    match backend {
        DbBackend::Sqlite => Ok((sql, values)),
        DbBackend::Postgres => rewrite_postgres(sql, placeholders, values),
        _ => bail!("Unsupported SeaORM database backend: {backend:?}"),
    }
}

/// Every bound value is read by a placeholder, and every placeholder has a
/// value: the distinct placeholder indexes are exactly `1..=value_count`.
#[cfg(debug_assertions)]
fn ensure_placeholders_match_values(
    placeholders: &[Placeholder],
    value_count: usize,
) -> anyhow::Result<()> {
    let mut used = vec![false; value_count];
    for placeholder in placeholders {
        anyhow::ensure!(
            placeholder.index <= value_count,
            "SQL placeholder ?{} has no matching value; only {value_count} values were supplied",
            placeholder.index
        );
        used[placeholder.index - 1] = true;
    }
    if let Some(unused) = used.iter().position(|used| !used) {
        bail!(
            "SQL binds {value_count} values but no placeholder reads value ?{}",
            unused + 1
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Placeholder {
    start: usize,
    end: usize,
    index: usize,
}

fn parse_placeholders(sql: &str) -> anyhow::Result<Vec<Placeholder>> {
    let bytes = sql.as_bytes();
    let mut placeholders = Vec::new();
    let mut index = 0;
    let mut in_single_quote = false;

    while index < bytes.len() {
        let byte = bytes[index];

        if byte == b'\'' {
            if in_single_quote && bytes.get(index + 1) == Some(&b'\'') {
                index += 2;
                continue;
            }
            in_single_quote = !in_single_quote;
            index += 1;
            continue;
        }

        if in_single_quote || byte != b'?' {
            index += 1;
            continue;
        }

        let digits_start = index + 1;
        let mut digits_end = digits_start;
        while bytes
            .get(digits_end)
            .is_some_and(|candidate| candidate.is_ascii_digit())
        {
            digits_end += 1;
        }

        if digits_end == digits_start {
            index += 1;
            continue;
        }

        let placeholder_index = sql[digits_start..digits_end]
            .parse::<usize>()
            .context("Invalid SQL placeholder index")?;
        if placeholder_index == 0 {
            bail!("SQL placeholders are one-indexed: ?0 is invalid");
        }

        placeholders.push(Placeholder {
            start: index,
            end: digits_end,
            index: placeholder_index,
        });
        index = digits_end;
    }

    Ok(placeholders)
}

fn rewrite_postgres(
    sql: String,
    placeholders: Vec<Placeholder>,
    values: Vec<Value>,
) -> anyhow::Result<(String, Vec<Value>)> {
    let mut rewritten = String::with_capacity(sql.len() + placeholders.len());
    let mut cursor = 0;

    for placeholder in placeholders {
        ensure_value_exists(&values, placeholder.index)?;
        rewritten.push_str(&sql[cursor..placeholder.start]);
        rewritten.push('$');
        rewritten.push_str(&placeholder.index.to_string());
        cursor = placeholder.end;
    }

    rewritten.push_str(&sql[cursor..]);
    Ok((rewritten, values))
}

fn ensure_value_exists(values: &[Value], placeholder_index: usize) -> anyhow::Result<()> {
    if placeholder_index > values.len() {
        bail!(
            "SQL placeholder ?{placeholder_index} has no matching value; only {} values were supplied",
            values.len()
        );
    }
    Ok(())
}

fn validate_json_key(key: &str) -> anyhow::Result<()> {
    if key.is_empty()
        || !key
            .chars()
            .all(|candidate| candidate.is_ascii_alphanumeric() || candidate == '_')
    {
        bail!("JSON key is not safe for raw SQL fragment: {key}");
    }
    Ok(())
}

/// `LIKE` pattern matching `search` as a literal substring. Use with
/// `ESCAPE '\'`: the `%`, `_` and `\` characters in `search` are escaped.
pub fn like_contains_pattern(search: &str) -> String {
    let mut pattern = String::with_capacity(search.len() + 2);
    pattern.push('%');
    for character in search.chars() {
        if matches!(character, '%' | '_' | '\\') {
            pattern.push('\\');
        }
        pattern.push(character);
    }
    pattern.push('%');
    pattern
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postgres_rewrites_indexed_placeholders_without_duplication() {
        let values = vec!["first".into(), 2_i64.into()];
        let statement =
            statement_for_backend(DbBackend::Postgres, "SELECT ?1, ?2, ?1, '?'", values)
                .expect("statement should build");

        assert_eq!(statement.sql, "SELECT $1, $2, $1, '?'");
        assert_eq!(statement.values.expect("values").0.len(), 2);
    }

    #[test]
    fn sqlite_keeps_indexed_placeholders() {
        let values = vec!["first".into()];
        let statement = statement_for_backend(DbBackend::Sqlite, "SELECT ?1", values)
            .expect("statement should build");

        assert_eq!(statement.sql, "SELECT ?1");
    }

    #[test]
    fn values_no_placeholder_reads_are_rejected() {
        for backend in [DbBackend::Sqlite, DbBackend::Postgres] {
            let error = statement_for_backend(
                backend,
                "SELECT ?1, ?3",
                vec!["a".into(), "b".into(), "c".into()],
            )
            .expect_err("an unread value should fail");
            assert!(
                error.to_string().contains("no placeholder reads value ?2"),
                "{error}"
            );
            let error = statement_for_backend(backend, "SELECT 1 = 1", vec!["a".into()])
                .expect_err("a statement without placeholders binds nothing");
            assert!(
                error.to_string().contains("no placeholder reads value ?1"),
                "{error}"
            );
        }
        statement_for_backend(DbBackend::Sqlite, "SELECT ?2, ?1, ?2", vec![1.into(), 2.into()])
            .expect("reused and reordered placeholders cover every value");
    }

    #[test]
    fn missing_placeholder_values_are_rejected() {
        let error = statement_for_backend(DbBackend::Postgres, "SELECT ?2", vec!["first".into()])
            .expect_err("missing value should fail");

        assert!(
            error.to_string().contains("has no matching value"),
            "{error}"
        );
    }

    #[test]
    fn json_text_fragments_are_backend_specific() {
        assert_eq!(
            json_extract_text(DbBackend::Sqlite, "project_json", "name").expect("sqlite"),
            "json_extract(project_json, '$.name')"
        );
        assert_eq!(
            json_extract_text(DbBackend::Postgres, "project_json", "name").expect("postgres"),
            "((project_json)::jsonb ->> 'name')"
        );
    }

    #[test]
    fn greatest_fragments_are_backend_specific() {
        assert_eq!(
            greatest(DbBackend::Sqlite, "confidence", "?1").expect("sqlite"),
            "MAX(confidence, ?1)"
        );
        assert_eq!(
            greatest(DbBackend::Postgres, "confidence", "?1").expect("postgres"),
            "GREATEST(confidence, ?1)"
        );
    }
}
