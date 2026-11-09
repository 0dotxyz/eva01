//! Lets an action happen for each key at most once per interval.

use std::{
    collections::HashMap,
    hash::Hash,
    time::{Duration, Instant},
};

pub struct RateLimiter<K> {
    interval: Duration,
    last: HashMap<K, Instant>,
}

impl<K: Eq + Hash> RateLimiter<K> {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last: HashMap::new(),
        }
    }

    /// Whether `key` may act at `now`; if so, the action is recorded.
    pub fn allow(&mut self, key: K, now: Instant) -> bool {
        match self.last.get(&key) {
            Some(at) if now.saturating_duration_since(*at) < self.interval => false,
            _ => {
                self.last.insert(key, now);
                true
            }
        }
    }

    /// Forgets the keys that may act again, so keys seen once do not accumulate.
    pub fn prune(&mut self, now: Instant) {
        let interval = self.interval;
        self.last
            .retain(|_, at| now.saturating_duration_since(*at) < interval);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_once_per_interval_per_key() {
        let start = Instant::now();
        let mut limiter = RateLimiter::new(Duration::from_secs(10));

        assert!(limiter.allow("a", start));
        assert!(!limiter.allow("a", start + Duration::from_secs(9)));
        assert!(limiter.allow("b", start + Duration::from_secs(9)));
        assert!(limiter.allow("a", start + Duration::from_secs(10)));
        assert!(!limiter.allow("a", start + Duration::from_secs(19)));
    }

    #[test]
    fn prune_drops_only_expired_keys() {
        let start = Instant::now();
        let mut limiter = RateLimiter::new(Duration::from_secs(10));
        limiter.allow("old", start);
        limiter.allow("new", start + Duration::from_secs(5));

        limiter.prune(start + Duration::from_secs(12));

        assert_eq!(limiter.last.len(), 1);
        assert!(!limiter.allow("new", start + Duration::from_secs(12)));
    }
}
