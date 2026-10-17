//! Request execution: the chat pipeline and its remote (worker) dispatch,
//! prefix-aware routing, fleet coordination, and the realtime relay with its
//! wire protocol.

pub mod chat;
pub mod chat_content;
pub mod fleet_coordinator;
pub mod prefix_router;
pub mod realtime_protocol;
pub mod realtime_relay;
pub mod realtime_translate;
pub mod realtime_wire;
pub mod remote_chat_dispatch;
