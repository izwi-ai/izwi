//! Realtime ASR ingress and operations on a request.

use super::*;

impl EngineCoreRequest {
    pub(crate) fn enable_realtime_asr_ingress(&mut self) -> Result<()> {
        if self.task_type != TaskType::ASR || self.realtime_asr_ingress.is_some() {
            return Err(Error::InvalidInput(format!(
                "request {} cannot enable realtime ASR ingress",
                self.id
            )));
        }
        self.streaming = true;
        self.realtime_asr_ingress = Some(Arc::new(RealtimeAsrIngress::default()));
        Ok(())
    }

    pub(crate) fn is_realtime_asr_session(&self) -> bool {
        self.realtime_asr_ingress.is_some()
    }

    #[cfg(test)]
    pub(crate) fn install_realtime_asr_operation(
        &self,
        operation_id: RealtimeOperationId,
        payload: RealtimeAsrOperationPayload,
    ) -> Result<RealtimeAsrOperationWaiter> {
        let logical_units = match &payload {
            RealtimeAsrOperationPayload::Push { samples, .. } => {
                u64::try_from(samples.len()).unwrap_or(u64::MAX).max(1)
            }
            RealtimeAsrOperationPayload::Finish => 1,
        };
        self.install_realtime_asr_operation_with_cost(
            operation_id,
            payload,
            WorkCost::new(logical_units, logical_units, 0),
        )
    }

    pub(crate) fn install_realtime_asr_operation_with_cost(
        &self,
        operation_id: RealtimeOperationId,
        payload: RealtimeAsrOperationPayload,
        preparation_cost: WorkCost,
    ) -> Result<RealtimeAsrOperationWaiter> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        let mut state = ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if state.terminal.is_some() {
            return Err(Error::InferenceError(
                "realtime ASR session is already terminal".into(),
            ));
        }
        if state
            .highest_operation_id
            .is_some_and(|highest| operation_id <= highest)
        {
            return Err(Error::InferenceError(format!(
                "realtime ASR operation identity {} is duplicate or stale",
                operation_id.get()
            )));
        }
        let (completion, waiter) = oneshot::channel();
        match state.operations.entry(operation_id) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                entry.insert(RealtimeAsrIngressOperation {
                    payload,
                    preparation_cost,
                    completion,
                });
            }
            std::collections::hash_map::Entry::Occupied(_) => {
                return Err(Error::InferenceError(
                    "realtime ASR operation identity was installed twice".into(),
                ));
            }
        }
        state.highest_operation_id = Some(operation_id);
        Ok(waiter)
    }

    pub(crate) fn realtime_asr_preparation_cost(
        &self,
        operation_id: RealtimeOperationId,
    ) -> Result<WorkCost> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .operations
            .get(&operation_id)
            .map(|operation| operation.preparation_cost)
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "realtime ASR operation {} has no retained preparation cost",
                    operation_id.get()
                ))
            })
    }

    pub(crate) fn realtime_asr_operation(
        &self,
        operation_id: RealtimeOperationId,
    ) -> Result<RealtimeAsrOperationPayload> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .operations
            .get(&operation_id)
            .map(|operation| operation.payload.clone())
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "realtime ASR operation {} has no retained payload",
                    operation_id.get()
                ))
            })
    }

    /// Atomically retires one payload before publishing its acknowledgement.
    /// A repeated or unknown identity is rejected and cannot acknowledge a
    /// later operation that happens to have the same payload shape.
    pub(in crate::engine) fn complete_realtime_asr_operation(
        &self,
        operation_id: RealtimeOperationId,
    ) -> Result<()> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        let operation = ingress
            .state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .operations
            .remove(&operation_id)
            .ok_or_else(|| {
                Error::InferenceError(format!(
                    "realtime ASR operation {} is unknown, stale, or already completed",
                    operation_id.get()
                ))
            })?;
        let acknowledgement = RealtimeAsrOperationAck {
            operation_id,
            kind: operation.payload.kind(),
            accepted_samples: operation.payload.accepted_samples(),
        };
        // Sending may fail when an external waiter was cancelled. The payload
        // is still authoritatively retired and must not be resurrected.
        let _ = operation.completion.send(Ok(acknowledgement));
        Ok(())
    }

    /// Fences future installation, retires every retained payload, and wakes
    /// all operation waiters with the same authoritative terminal outcome.
    pub(in crate::engine) fn terminate_realtime_asr_ingress(
        &self,
        terminal: RealtimeAsrTerminalOutcome,
    ) -> Result<usize> {
        let ingress = self.realtime_asr_ingress.as_ref().ok_or_else(|| {
            Error::InvalidInput("request is not an engine-owned realtime ASR session".into())
        })?;
        let operations = {
            let mut state = ingress
                .state
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if state.terminal.is_some() {
                return Ok(0);
            }
            state.terminal = Some(terminal.clone());
            std::mem::take(&mut state.operations)
        };
        let retired = operations.len();
        for (_, operation) in operations {
            let _ = operation.completion.send(Err(terminal.clone()));
        }
        Ok(retired)
    }
}
