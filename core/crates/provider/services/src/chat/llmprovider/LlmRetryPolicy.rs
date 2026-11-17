use operit_host_api::HostManager::defaultHostRuntimeTaskSchedulerHost;
use operit_host_api::HostRuntimeTaskSchedulerHost;

pub struct LlmRetryPolicy;

impl LlmRetryPolicy {
    pub const MAX_RETRY_ATTEMPTS: i32 = 5;
    const RETRY_BASE_DELAY_MS: i64 = 1000;
    /// Upper bound for the random jitter added to every delay.
    const RETRY_JITTER_MS: i64 = 750;

    /// Calculates the exponential retry delay for an attempt.
    #[allow(non_snake_case)]
    pub fn nextDelayMs(retryAttempt: i32) -> i64 {
        let normalizedAttempt = retryAttempt.max(1);
        Self::RETRY_BASE_DELAY_MS * (1_i64 << (normalizedAttempt - 1))
    }

    /// Calculates the exponential retry delay including bounded jitter.
    ///
    /// Without jitter, several tasks that lost the same connection retry in
    /// lockstep and immediately re-create the same congestion. Spreading the
    /// retries keeps a flaky network from turning into a synchronized stall.
    #[allow(non_snake_case)]
    pub fn nextDelayMsWithJitter(retryAttempt: i32) -> i64 {
        let base = Self::nextDelayMs(retryAttempt);
        base + Self::jitterMs()
    }

    /// Produces a bounded non-negative jitter offset.
    fn jitterMs() -> i64 {
        use std::collections::hash_map::RandomState;
        use std::hash::{BuildHasher, Hasher};
        // RandomState is seeded per process, so consecutive calls differ
        // without pulling in an extra dependency.
        let mut hasher = RandomState::new().build_hasher();
        hasher.write_u64(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|elapsed| elapsed.as_nanos() as u64)
                .unwrap_or_default(),
        );
        let raw = hasher.finish();
        (raw % (Self::RETRY_JITTER_MS as u64 + 1)) as i64
    }
}

/// Waits for the retry delay selected by the provider implementation policy.
pub async fn delay_retry_ms(retry_attempt: i32) {
    let delay_ms = LlmRetryPolicy::nextDelayMsWithJitter(retry_attempt);
    defaultHostRuntimeTaskSchedulerHost()
        .waitForHostRuntimeDelay(delay_ms as u64)
        .await
        .expect("runtime retry delay must be provided by the Host");
}
