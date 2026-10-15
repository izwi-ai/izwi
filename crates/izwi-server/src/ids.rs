pub fn new_uuid() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// An OpenAI-style object id: `prefix` followed by a random UUID in hex,
/// such as `call_<hex>`.
pub fn new_prefixed_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}
