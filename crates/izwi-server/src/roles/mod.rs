//! The two process roles `izwi-server` can run as.

pub(crate) mod gateway;
#[cfg(test)]
mod gateway_env;
pub(crate) mod gateway_workers;
pub(crate) mod local;
