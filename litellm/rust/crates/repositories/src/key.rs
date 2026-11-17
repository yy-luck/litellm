//! The virtual key repository: the `combined_view` read and the rotation grace-period fallback with its memory
//! tier, each view stamped with when it was read.

use std::{
    collections::{HashMap, VecDeque},
    sync::{Mutex, PoisonError},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use chrono::{DateTime, NaiveDateTime, Utc};
use litellm_db::Call;
use serde::Serialize;

use crate::{Database, Side};

const KEY_VIEW: Call = Call {
    call_type: "get_data",
    table: Some("combined_view"),
    operation: None,
};

/// How long a found grace-period mapping is served from memory before the database is asked again.
const DEPRECATED_KEY_TTL: Duration = Duration::from_secs(60);
/// How many grace-period mappings are held in memory; the oldest goes first.
const DEPRECATED_KEY_CAPACITY: usize = 1000;

/// Reads virtual keys by hash. A hash that is not a key but was rotated away from one still within its grace
/// period resolves to the key it was rotated into.
pub struct KeyRepository {
    database: Database,
    deprecated: Mutex<DeprecatedKeys>,
}

impl KeyRepository {
    pub fn new(database: Database) -> Self {
        Self {
            database,
            deprecated: Mutex::new(DeprecatedKeys::default()),
        }
    }

    /// The view of the key whose hash is `token`, or `None` when neither a key nor a rotated-away hash in its
    /// grace period has it. `Side::Writer` reads the latest commit and skips the grace-period memory tier.
    pub async fn find(
        &self,
        token: &str,
        side: Side,
    ) -> Result<Option<FetchedKey>, litellm_db::Error> {
        let now = SystemTime::now();
        self.database
            .run(KEY_VIEW, || self.find_once(token, side, now))
            .await
    }

    async fn find_once(
        &self,
        token: &str,
        side: Side,
        now: SystemTime,
    ) -> Result<Option<FetchedKey>, litellm_db::Error> {
        let mut session = self.database.session(side).await?;
        if let Some(row) = litellm_db::find_key_view(session.executor(), token).await? {
            return Ok(Some(FetchedKey::new(row, now)));
        }
        let cached = match side {
            Side::Read => self.deprecated_keys().active_token(token, now),
            Side::Writer => None,
        };
        let active_token = match cached {
            Some(active_token) => active_token,
            None => {
                let at = DateTime::<Utc>::from(now).naive_utc();
                match litellm_db::find_deprecated_key(session.executor(), token, at).await {
                    Ok(Some(deprecated)) => {
                        self.deprecated_keys().insert(
                            token,
                            &deprecated.active_token_id,
                            now,
                            deprecated.revoke_at,
                        );
                        deprecated.active_token_id
                    }
                    Ok(None) => return Ok(None),
                    Err(error) => {
                        tracing::debug!(error = %litellm_error::Report::new(&error), "deprecated key lookup skipped");
                        return Ok(None);
                    }
                }
            }
        };
        Ok(litellm_db::find_key_view(session.executor(), &active_token)
            .await?
            .map(|row| FetchedKey::new(row, now)))
    }

    fn deprecated_keys(&self) -> std::sync::MutexGuard<'_, DeprecatedKeys> {
        self.deprecated
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// Grace-period mappings found in the database, each served until its memory TTL or its revoke time,
/// whichever comes first. Only found mappings are held, so a miss never evicts a real one.
#[derive(Default)]
struct DeprecatedKeys {
    entries: HashMap<String, DeprecatedEntry>,
    order: VecDeque<String>,
}

struct DeprecatedEntry {
    active_token: String,
    serve_until: SystemTime,
}

impl DeprecatedKeys {
    fn active_token(&mut self, token: &str, now: SystemTime) -> Option<String> {
        let entry = self.entries.get(token)?;
        if now < entry.serve_until {
            return Some(entry.active_token.clone());
        }
        self.remove(token);
        None
    }

    fn insert(
        &mut self,
        token: &str,
        active_token: &str,
        now: SystemTime,
        revoke_at: NaiveDateTime,
    ) {
        let revoke_at = SystemTime::from(revoke_at.and_utc());
        let serve_until = (now + DEPRECATED_KEY_TTL).min(revoke_at);
        if self.entries.contains_key(token) {
            self.remove(token);
        }
        while self.entries.len() >= DEPRECATED_KEY_CAPACITY {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
        self.order.push_back(token.to_owned());
        self.entries.insert(
            token.to_owned(),
            DeprecatedEntry {
                active_token: active_token.to_owned(),
                serve_until,
            },
        );
    }

    fn remove(&mut self, token: &str) {
        if self.entries.remove(token).is_some() {
            self.order.retain(|held| held != token);
        }
    }
}

/// A key view stamped with when it was read.
#[derive(Serialize)]
pub struct FetchedKey {
    #[serde(flatten)]
    pub view: litellm_db::KeyView,
    pub last_refreshed_at: f64,
}

impl FetchedKey {
    fn new(view: litellm_db::KeyView, now: SystemTime) -> Self {
        Self {
            view,
            last_refreshed_at: now
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use chrono::{DateTime, NaiveDateTime, Utc};
    use rstest::rstest;

    use super::{DEPRECATED_KEY_CAPACITY, DeprecatedKeys};

    fn at(seconds: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(1_800_000_000 + seconds)
    }

    fn naive(time: SystemTime) -> NaiveDateTime {
        DateTime::<Utc>::from(time).naive_utc()
    }

    #[rstest]
    #[case::fresh(at(30), Some("active"))]
    #[case::memory_ttl_elapsed(at(60), None)]
    fn deprecated_mapping_is_served_for_sixty_seconds(
        #[case] now: SystemTime,
        #[case] expected: Option<&str>,
    ) {
        let mut keys = DeprecatedKeys::default();
        keys.insert("old", "active", at(0), naive(at(3600)));

        assert_eq!(keys.active_token("old", now).as_deref(), expected);
    }

    #[rstest]
    fn deprecated_mapping_stops_at_revoke_time() {
        let mut keys = DeprecatedKeys::default();
        keys.insert("old", "active", at(0), naive(at(10)));

        assert_eq!(
            (
                keys.active_token("old", at(9)).as_deref(),
                keys.active_token("old", at(10))
            ),
            (Some("active"), None)
        );
    }

    #[rstest]
    fn oldest_deprecated_mapping_is_evicted_at_capacity() {
        let mut keys = DeprecatedKeys::default();
        for index in 0..=DEPRECATED_KEY_CAPACITY {
            keys.insert(&format!("old-{index}"), "active", at(0), naive(at(3600)));
        }

        assert_eq!(
            (
                keys.active_token("old-0", at(1)),
                keys.active_token(&format!("old-{DEPRECATED_KEY_CAPACITY}"), at(1))
                    .as_deref(),
                keys.entries.len(),
            ),
            (None, Some("active"), DEPRECATED_KEY_CAPACITY)
        );
    }
}
