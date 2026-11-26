//! Local project, multi-session continuity, and turn history persistence (`~/.spark/history.json`).

use crate::cli::{char_prefix, AgentKind};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Maximum number of turn records retained in `history.json`.
pub const MAX_HISTORY_TURNS: usize = 500;

/// Legacy v0.1 per-agent persisted session and option state.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentSessionState {
    pub session_id: Option<String>,
    pub model: String,
    pub mode: String,
    pub effort: String,
}

/// Project workspace grouping multiple concurrent AI sessions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectRecord {
    pub id: String,
    pub name: String,
    pub working_dir: Option<PathBuf>,
    pub created_at_epoch_secs: u64,
    #[serde(default)]
    pub collapsed: bool,
}

/// First-class AI session belonging to a project.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionRecord {
    pub uid: String,
    pub project_id: String,
    pub title: String,
    pub agent: AgentKind,
    pub cli_session_id: Option<String>,
    pub model: String,
    pub mode: String,
    pub effort: String,
    #[serde(default)]
    pub imported: bool,
    #[serde(default)]
    pub archived: bool,
    pub created_at_epoch_secs: u64,
    pub updated_at_epoch_secs: u64,
}

/// Single completed or cancelled agent turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TurnRecord {
    pub id: u64,
    pub timestamp_epoch_secs: u64,
    pub agent: AgentKind,
    pub prompt: String,
    pub output: String,
    pub stderr: String,
    pub success: bool,
    pub exit_code: Option<i32>,
    pub duration_ms: u64,
    pub session_id: Option<String>,
    pub model: Option<String>,
    pub mode: Option<String>,
    pub effort: Option<String>,
    #[serde(default)]
    pub project_id: Option<String>,
    #[serde(default)]
    pub session_uid: Option<String>,
}

/// Persistent store of projects, active sessions, legacy agent defaults, and turn history.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryStore {
    pub next_id: u64,
    #[serde(default)]
    pub sessions: HashMap<AgentKind, AgentSessionState>,
    #[serde(default)]
    pub turns: Vec<TurnRecord>,
    #[serde(default)]
    pub projects: Vec<ProjectRecord>,
    #[serde(default)]
    pub active_sessions: Vec<SessionRecord>,
    #[serde(default)]
    pub selected_session_uid: Option<String>,
}

impl HistoryStore {
    /// Return the default history file path (`~/.spark/history.json`).
    pub fn default_path() -> Option<PathBuf> {
        let home = std::env::var_os("USERPROFILE")
            .or_else(|| std::env::var_os("HOME"))
            .map(PathBuf::from)?;
        Some(home.join(".spark").join("history.json"))
    }

    /// Load history from `path`. Returns `Ok(Self::default())` if the file does not exist.
    pub fn load_from(path: &Path) -> Result<Self, String> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(path)
            .map_err(|err| format!("Failed to read history file {}: {err}", path.display()))?;
        serde_json::from_str(&raw)
            .map_err(|err| format!("Failed to parse history file {}: {err}", path.display()))
    }

    /// Load history at startup without ever destroying a file that failed to load.
    ///
    /// The app saves the store straight after loading it, so falling back to an
    /// empty store on a parse error would overwrite the user's history with
    /// nothing. A file that cannot be loaded is moved aside first (to
    /// `history.json.corrupt-<epoch secs>`), and the caller gets a message to show.
    pub fn load_for_startup(path: &Path) -> (Self, Option<String>) {
        if !path.exists() {
            return (Self::default(), None);
        }

        // A file that cannot be READ is not known to be bad: it may be locked, or
        // an offline cloud placeholder. Leave it exactly where it is. It is still
        // at `path`, which is how the caller knows not to save over it.
        let raw = match std::fs::read_to_string(path) {
            Ok(raw) => raw,
            Err(err) => {
                return (
                    Self::default(),
                    Some(format!(
                        "Failed to read history file {}: {err}. It was left untouched; \
                         history will not be saved this run",
                        path.display()
                    )),
                );
            }
        };

        match serde_json::from_str(&raw) {
            Ok(store) => (store, None),
            Err(err) => {
                let err = format!("Failed to parse history file {}: {err}", path.display());
                let backup = free_backup_path(path);
                let note = match std::fs::rename(path, &backup) {
                    Ok(()) => format!("{err}. The file was kept as {}", backup.display()),
                    Err(move_err) => format!(
                        "{err}. It could not be moved aside ({move_err}); history will not be saved this run"
                    ),
                };
                (Self::default(), Some(note))
            }
        }
    }

    /// Persist history to `path`, creating parent directories if needed.
    pub fn save_to(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|err| {
                format!(
                    "Failed to create history directory {}: {err}",
                    parent.display()
                )
            })?;
        }
        let json = serde_json::to_string_pretty(self)
            .map_err(|err| format!("Failed to serialize history: {err}"))?;
        let tmp_path = path.with_extension("json.tmp");
        std::fs::write(&tmp_path, json).map_err(|err| {
            format!(
                "Failed to write temp history file {}: {err}",
                tmp_path.display()
            )
        })?;
        std::fs::rename(&tmp_path, path)
            .map_err(|err| format!("Failed to commit history file {}: {err}", path.display()))
    }

    fn alloc_id(&mut self, prefix: &str) -> String {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);
        format!("{prefix}-{id}")
    }

    /// Ensure at least one project exists and return its `id`.
    pub fn ensure_default_project(&mut self, default_dir: Option<PathBuf>) -> String {
        if let Some(first) = self.projects.first() {
            return first.id.clone();
        }
        let name = default_dir
            .as_deref()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("General")
            .to_string();
        self.create_project(name, default_dir).id.clone()
    }

    /// Migrate v0.1 single-agent sessions & turns into v0.2 projects and multi-sessions if needed.
    pub fn migrate_v01_if_needed(&mut self, default_dir: Option<PathBuf>) {
        if !self.projects.is_empty() && !self.active_sessions.is_empty() {
            if self.selected_session_uid.is_none() {
                self.selected_session_uid = self
                    .active_sessions
                    .iter()
                    .find(|s| !s.archived)
                    .map(|s| s.uid.clone());
            }
            return;
        }

        let default_project_id = self.ensure_default_project(default_dir);

        if self.active_sessions.is_empty() {
            for agent in AgentKind::ALL {
                let legacy = self.sessions.get(&agent).cloned().unwrap_or_default();
                let uid = self.alloc_id("sess");
                let now = now_epoch_secs();
                self.active_sessions.push(SessionRecord {
                    uid: uid.clone(),
                    project_id: default_project_id.clone(),
                    title: format!("{} Session", agent.display_name()),
                    agent,
                    cli_session_id: legacy.session_id,
                    model: legacy.model,
                    mode: legacy.mode,
                    effort: legacy.effort,
                    imported: false,
                    archived: false,
                    created_at_epoch_secs: now,
                    updated_at_epoch_secs: now,
                });

                for turn in &mut self.turns {
                    if turn.agent == agent && turn.session_uid.is_none() {
                        turn.session_uid = Some(uid.clone());
                        turn.project_id = Some(default_project_id.clone());
                    }
                }
            }
        }

        if self.selected_session_uid.is_none() {
            self.selected_session_uid = self
                .active_sessions
                .iter()
                .find(|s| !s.archived)
                .map(|s| s.uid.clone());
        }
    }

    /// Create a new project and return a reference to it.
    pub fn create_project(
        &mut self,
        name: impl Into<String>,
        working_dir: Option<PathBuf>,
    ) -> &ProjectRecord {
        let id = self.alloc_id("proj");
        let name = name.into().trim().to_string();
        let display_name = if name.is_empty() {
            "Untitled Project".to_string()
        } else {
            name
        };
        self.projects.push(ProjectRecord {
            id,
            name: display_name,
            working_dir,
            created_at_epoch_secs: now_epoch_secs(),
            collapsed: false,
        });
        self.projects.last().expect("just pushed")
    }

    /// Look up a project by `id`.
    pub fn project(&self, project_id: &str) -> Option<&ProjectRecord> {
        self.projects.iter().find(|p| p.id == project_id)
    }

    /// Look up a mutable project by `id`.
    pub fn project_mut(&mut self, project_id: &str) -> Option<&mut ProjectRecord> {
        self.projects.iter_mut().find(|p| p.id == project_id)
    }

    /// Find an existing project matching `dir`, or create one named after the folder.
    pub fn find_or_create_project_for_dir(&mut self, dir: &Path) -> String {
        if let Some(existing) = self
            .projects
            .iter()
            .find(|p| p.working_dir.as_deref() == Some(dir))
        {
            return existing.id.clone();
        }
        let name = dir
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("Workspace")
            .to_string();
        self.create_project(name, Some(dir.to_path_buf()))
            .id
            .clone()
    }

    /// Create a new Spark session inside `project_id` and select it.
    pub fn create_session(
        &mut self,
        project_id: impl Into<String>,
        agent: AgentKind,
        model: impl Into<String>,
        title: impl Into<String>,
    ) -> &SessionRecord {
        let uid = self.alloc_id("sess");
        let now = now_epoch_secs();
        let raw_title = title.into();
        let clean_title = raw_title.trim();
        let final_title = if clean_title.is_empty() {
            format!("{} Session", agent.display_name())
        } else {
            clean_title.to_string()
        };

        self.active_sessions.push(SessionRecord {
            uid: uid.clone(),
            project_id: project_id.into(),
            title: final_title,
            agent,
            cli_session_id: None,
            model: model.into().trim().to_string(),
            mode: String::new(),
            effort: String::new(),
            imported: false,
            archived: false,
            created_at_epoch_secs: now,
            updated_at_epoch_secs: now,
        });
        self.selected_session_uid = Some(uid);
        self.active_sessions.last().expect("just pushed")
    }

    /// Import an external CLI session into `project_id`, deduplicating by `(agent, cli_session_id)`.
    pub fn import_external_session(
        &mut self,
        project_id: impl Into<String>,
        agent: AgentKind,
        cli_session_id: impl Into<String>,
        title: impl Into<String>,
        model: impl Into<String>,
    ) -> String {
        let cli_id = cli_session_id.into().trim().to_string();
        let target_project = project_id.into();

        if let Some(existing) = self
            .active_sessions
            .iter_mut()
            .find(|s| s.agent == agent && s.cli_session_id.as_deref() == Some(cli_id.as_str()))
        {
            existing.archived = false;
            existing.project_id = target_project;
            existing.updated_at_epoch_secs = now_epoch_secs();
            let uid = existing.uid.clone();
            self.selected_session_uid = Some(uid.clone());
            return uid;
        }

        let uid = self.alloc_id("sess");
        let now = now_epoch_secs();
        let clean_title = title.into().trim().to_string();
        let final_title = if clean_title.is_empty() {
            format!("{} ({})", agent.display_name(), char_prefix(&cli_id, 8))
        } else {
            clean_title
        };

        self.active_sessions.push(SessionRecord {
            uid: uid.clone(),
            project_id: target_project,
            title: final_title,
            agent,
            cli_session_id: Some(cli_id),
            model: model.into().trim().to_string(),
            mode: String::new(),
            effort: String::new(),
            imported: true,
            archived: false,
            created_at_epoch_secs: now,
            updated_at_epoch_secs: now,
        });
        self.selected_session_uid = Some(uid.clone());
        uid
    }

    /// Mark a session archived (closed) and advance selection if needed.
    pub fn archive_session(&mut self, uid: &str) {
        if let Some(sess) = self.active_sessions.iter_mut().find(|s| s.uid == uid) {
            sess.archived = true;
            sess.updated_at_epoch_secs = now_epoch_secs();
        }
        if self.selected_session_uid.as_deref() == Some(uid) {
            self.selected_session_uid = self
                .active_sessions
                .iter()
                .find(|s| !s.archived)
                .map(|s| s.uid.clone());
        }
    }

    /// Look up a session by Spark `uid`.
    pub fn session_by_uid(&self, uid: &str) -> Option<&SessionRecord> {
        self.active_sessions.iter().find(|s| s.uid == uid)
    }

    /// Look up a mutable session by Spark `uid`.
    pub fn session_by_uid_mut(&mut self, uid: &str) -> Option<&mut SessionRecord> {
        self.active_sessions.iter_mut().find(|s| s.uid == uid)
    }

    /// Remember the CLI session a Spark session's first run created, so the next
    /// prompt continues that conversation. Never replaces an ID already set (an
    /// imported session's, or one typed into Session Options). Returns `true` if
    /// the ID was stored.
    pub fn adopt_cli_session_id(&mut self, uid: &str, cli_session_id: String) -> bool {
        match self.session_by_uid_mut(uid) {
            Some(sess)
                if sess
                    .cli_session_id
                    .as_deref()
                    .is_none_or(|id| id.trim().is_empty()) =>
            {
                sess.cli_session_id = Some(cli_session_id);
                sess.updated_at_epoch_secs = now_epoch_secs();
                true
            }
            _ => false,
        }
    }

    /// Drop a CLI session ID that the CLI no longer has, so the next prompt
    /// starts a new conversation instead of failing the same way again. Only
    /// drops `stale_id` itself: an ID the user typed in meanwhile is kept.
    pub fn forget_cli_session_id(&mut self, uid: &str, stale_id: &str) -> bool {
        match self.session_by_uid_mut(uid) {
            Some(sess) if sess.cli_session_id.as_deref() == Some(stale_id) => {
                sess.cli_session_id = None;
                sess.updated_at_epoch_secs = now_epoch_secs();
                true
            }
            _ => false,
        }
    }

    /// Record a new turn (v0.1 compatible) and enforce the `MAX_HISTORY_TURNS` cap.
    #[allow(clippy::too_many_arguments)]
    pub fn record_turn(
        &mut self,
        agent: AgentKind,
        prompt: String,
        output: String,
        stderr: String,
        success: bool,
        exit_code: Option<i32>,
        duration_ms: u64,
        session_id: Option<String>,
        model: Option<String>,
        mode: Option<String>,
        effort: Option<String>,
    ) -> u64 {
        self.record_session_turn(
            None,
            None,
            agent,
            prompt,
            output,
            stderr,
            success,
            exit_code,
            duration_ms,
            session_id,
            model,
            mode,
            effort,
        )
    }

    /// Record a turn associated with a Spark `project_id` and `session_uid`.
    #[allow(clippy::too_many_arguments)]
    pub fn record_session_turn(
        &mut self,
        project_id: Option<String>,
        session_uid: Option<String>,
        agent: AgentKind,
        prompt: String,
        output: String,
        stderr: String,
        success: bool,
        exit_code: Option<i32>,
        duration_ms: u64,
        session_id: Option<String>,
        model: Option<String>,
        mode: Option<String>,
        effort: Option<String>,
    ) -> u64 {
        let id = self.next_id;
        self.next_id = self.next_id.saturating_add(1);

        self.turns.push(TurnRecord {
            id,
            timestamp_epoch_secs: now_epoch_secs(),
            agent,
            prompt,
            output,
            stderr,
            success,
            exit_code,
            duration_ms,
            session_id,
            model,
            mode,
            effort,
            project_id,
            session_uid,
        });

        if self.turns.len() > MAX_HISTORY_TURNS {
            let overflow = self.turns.len() - MAX_HISTORY_TURNS;
            self.turns.drain(0..overflow);
        }

        id
    }

    /// Return turns for `agent` in reverse chronological order (newest first).
    pub fn turns_for(&self, agent: AgentKind) -> Vec<&TurnRecord> {
        self.turns
            .iter()
            .rev()
            .filter(|turn| turn.agent == agent)
            .collect()
    }

    /// Return turns for a specific Spark `session_uid` in reverse chronological order.
    pub fn turns_for_session(&self, session_uid: &str) -> Vec<&TurnRecord> {
        self.turns
            .iter()
            .rev()
            .filter(|turn| turn.session_uid.as_deref() == Some(session_uid))
            .collect()
    }

    /// Get or initialize mutable legacy session state for `agent`.
    pub fn session_mut(&mut self, agent: AgentKind) -> &mut AgentSessionState {
        self.sessions.entry(agent).or_default()
    }

    /// Get legacy session state for `agent`, if present.
    pub fn session(&self, agent: AgentKind) -> Option<&AgentSessionState> {
        self.sessions.get(&agent)
    }
}

/// A name beside `path` for an unparseable history file, that nothing has yet.
/// `rename` replaces an existing file on Windows, and the timestamp only has
/// one-second resolution, so an earlier backup must not be the destination.
fn free_backup_path(path: &Path) -> PathBuf {
    let stamp = now_epoch_secs();
    let mut candidate = path.with_extension(format!("json.corrupt-{stamp}"));
    let mut n = 1;
    while candidate.exists() {
        candidate = path.with_extension(format!("json.corrupt-{stamp}-{n}"));
        n += 1;
    }
    candidate
}

fn now_epoch_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_history_path(name: &str) -> PathBuf {
        let pid = std::process::id();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir()
            .join(format!("spark_test_{name}_{pid}_{nanos}"))
            .join("history.json")
    }

    #[test]
    fn history_round_trip_preserves_sessions_and_turns() {
        let path = temp_history_path("round_trip");
        let mut store = HistoryStore::default();
        store.session_mut(AgentKind::Agy).session_id = Some("conv-42".to_string());
        store.session_mut(AgentKind::Claude).model = "sonnet".to_string();
        store.session_mut(AgentKind::OpenCode).session_id = Some("oc-9".to_string());

        let id = store.record_turn(
            AgentKind::Agy,
            "ping".to_string(),
            "pong".to_string(),
            String::new(),
            true,
            Some(0),
            120,
            Some("conv-42".to_string()),
            None,
            Some("auto".to_string()),
            None,
        );
        assert_eq!(id, 0);

        store.save_to(&path).expect("save should succeed");
        let loaded = HistoryStore::load_from(&path).expect("load should succeed");
        assert_eq!(loaded, store);
        assert_eq!(loaded.turns_for(AgentKind::Agy).len(), 1);
        assert_eq!(loaded.turns_for(AgentKind::Claude).len(), 0);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn load_missing_returns_default_and_corrupt_returns_error() {
        let missing = temp_history_path("missing");
        let loaded = HistoryStore::load_from(&missing).expect("missing file should return default");
        assert_eq!(loaded, HistoryStore::default());

        let corrupt = temp_history_path("corrupt");
        std::fs::create_dir_all(corrupt.parent().unwrap()).unwrap();
        std::fs::write(&corrupt, "{ not valid json").unwrap();
        let err = HistoryStore::load_from(&corrupt).expect_err("corrupt JSON must fail");
        assert!(err.contains("Failed to parse history file"));

        let _ = std::fs::remove_dir_all(corrupt.parent().unwrap());
    }

    #[test]
    fn startup_load_keeps_a_history_file_it_could_not_parse() {
        let path = temp_history_path("startup_corrupt");
        let dir = path.parent().unwrap().to_path_buf();
        std::fs::create_dir_all(&dir).unwrap();
        let original = "{ \"turns\": [ this file was cut off mid-write";
        std::fs::write(&path, original).unwrap();

        // What the app does on launch: load, then save straight away.
        let (store, warning) = HistoryStore::load_for_startup(&path);
        assert_eq!(store, HistoryStore::default());
        assert!(warning.is_some(), "a failed load must be reported");
        store.save_to(&path).expect("save should succeed");

        let kept: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .filter(|content| content == original)
            .collect();
        assert_eq!(
            kept.len(),
            1,
            "the unparseable history must survive the first save, somewhere beside it"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn startup_load_of_a_good_or_missing_file_reports_nothing() {
        let path = temp_history_path("startup_ok");
        let (store, warning) = HistoryStore::load_for_startup(&path);
        assert_eq!(store, HistoryStore::default());
        assert!(warning.is_none());

        let mut saved = HistoryStore::default();
        saved.create_project("Spark", None);
        saved.save_to(&path).unwrap();
        let (loaded, warning) = HistoryStore::load_for_startup(&path);
        assert_eq!(loaded, saved);
        assert!(warning.is_none());

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn importing_a_session_whose_id_is_not_ascii_does_not_panic() {
        let mut store = HistoryStore::default();
        let proj_id = store.create_project("Spark", None).id.clone();

        // Byte 8 of this ID falls inside an "\u{e9}".
        let uid = store.import_external_session(
            &proj_id,
            AgentKind::Agy,
            "a\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}",
            "",
            "",
        );
        let title = &store.session_by_uid(&uid).unwrap().title;
        assert_eq!(title, "Antigravity (a\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}\u{e9})");
    }

    #[test]
    fn adopt_cli_session_id_sets_an_id_once_and_never_replaces_one() {
        let mut store = HistoryStore::default();
        let proj_id = store.create_project("Spark", None).id.clone();
        let uid = store
            .create_session(&proj_id, AgentKind::Claude, "sonnet", "")
            .uid
            .clone();

        assert!(store.adopt_cli_session_id(&uid, "first".to_string()));
        assert!(!store.adopt_cli_session_id(&uid, "second".to_string()));
        assert_eq!(
            store
                .session_by_uid(&uid)
                .unwrap()
                .cli_session_id
                .as_deref(),
            Some("first")
        );
        assert!(!store.adopt_cli_session_id("no-such-session", "x".to_string()));
    }

    #[test]
    fn startup_load_leaves_a_file_it_could_not_read_where_it_is() {
        // A directory at the history path: `exists()` is true and reading fails,
        // which is what a locked file or an offline placeholder looks like too.
        let path = temp_history_path("startup_unreadable");
        std::fs::create_dir_all(&path).unwrap();

        let (store, warning) = HistoryStore::load_for_startup(&path);
        assert_eq!(store, HistoryStore::default());
        assert!(warning.is_some_and(|w| w.contains("left untouched")));
        assert!(path.is_dir(), "an unreadable history must not be moved");
        let beside: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .collect();
        assert_eq!(beside.len(), 1, "nothing may be created beside it");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn startup_load_never_reuses_a_backup_name() {
        let path = temp_history_path("startup_backup_names");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();

        // Two corrupt files in a row, well inside one second.
        for content in ["{ first", "{ second"] {
            std::fs::write(&path, content).unwrap();
            let (_, warning) = HistoryStore::load_for_startup(&path);
            assert!(warning.is_some());
        }

        let mut kept: Vec<String> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .filter_map(|e| std::fs::read_to_string(e.path()).ok())
            .collect();
        kept.sort();
        assert_eq!(kept, ["{ first", "{ second"], "both backups must survive");

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn a_blank_cli_session_id_counts_as_none_when_adopting() {
        let mut store = HistoryStore::default();
        let proj_id = store.create_project("Spark", None).id.clone();
        let uid = store
            .create_session(&proj_id, AgentKind::Claude, "sonnet", "")
            .uid
            .clone();
        // What a migrated v0.1 file or a hand edit can leave behind.
        store.session_by_uid_mut(&uid).unwrap().cli_session_id = Some("  ".to_string());

        assert!(store.adopt_cli_session_id(&uid, "real-id".to_string()));
        assert_eq!(
            store
                .session_by_uid(&uid)
                .unwrap()
                .cli_session_id
                .as_deref(),
            Some("real-id")
        );
    }

    #[test]
    fn forget_cli_session_id_only_drops_the_id_that_went_stale() {
        let mut store = HistoryStore::default();
        let proj_id = store.create_project("Spark", None).id.clone();
        let uid = store
            .create_session(&proj_id, AgentKind::Claude, "sonnet", "")
            .uid
            .clone();
        store.adopt_cli_session_id(&uid, "old-id".to_string());

        assert!(!store.forget_cli_session_id(&uid, "some-other-id"));
        assert!(store.forget_cli_session_id(&uid, "old-id"));
        assert_eq!(store.session_by_uid(&uid).unwrap().cli_session_id, None);
        assert!(!store.forget_cli_session_id(&uid, "old-id"));
    }

    #[test]
    fn record_turn_enforces_max_history_cap() {
        let mut store = HistoryStore::default();
        for i in 0..(MAX_HISTORY_TURNS + 25) {
            store.record_turn(
                AgentKind::Claude,
                format!("prompt-{i}"),
                "ok".to_string(),
                String::new(),
                true,
                Some(0),
                10,
                None,
                None,
                None,
                None,
            );
        }
        assert_eq!(store.turns.len(), MAX_HISTORY_TURNS);
        assert_eq!(store.turns.first().unwrap().prompt, "prompt-25");
        assert_eq!(
            store.turns.last().unwrap().prompt,
            format!("prompt-{}", MAX_HISTORY_TURNS + 24)
        );
    }

    #[test]
    fn v01_json_migrates_cleanly_to_projects_and_multi_sessions() {
        let path = temp_history_path("migrate_v01");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let v01_json = r#"{
            "next_id": 2,
            "sessions": {
                "Claude": { "session_id": "cl-99", "model": "sonnet", "mode": "", "effort": "high" }
            },
            "turns": [
                {
                    "id": 1,
                    "timestamp_epoch_secs": 1700000000,
                    "agent": "Claude",
                    "prompt": "old prompt",
                    "output": "old output",
                    "stderr": "",
                    "success": true,
                    "exit_code": 0,
                    "duration_ms": 200,
                    "session_id": "cl-99",
                    "model": "sonnet",
                    "mode": null,
                    "effort": "high"
                }
            ]
        }"#;
        std::fs::write(&path, v01_json).unwrap();

        let mut store = HistoryStore::load_from(&path).expect("v0.1 json should load");
        store.migrate_v01_if_needed(Some(PathBuf::from(r"D:\Projects\spark")));

        assert_eq!(store.projects.len(), 1);
        assert_eq!(store.projects[0].name, "spark");
        assert_eq!(store.active_sessions.len(), 3);
        let claude_sess = store
            .active_sessions
            .iter()
            .find(|s| s.agent == AgentKind::Claude)
            .unwrap();
        assert_eq!(claude_sess.cli_session_id.as_deref(), Some("cl-99"));
        assert_eq!(claude_sess.model, "sonnet");
        assert_eq!(
            store.turns_for_session(&claude_sess.uid).len(),
            1,
            "legacy turn should bind to migrated session"
        );

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[test]
    fn multiple_models_in_same_project_and_import_deduplication() {
        let mut store = HistoryStore::default();
        let proj_id = store
            .create_project("Spark", Some(PathBuf::from(r"D:\Projects\spark")))
            .id
            .clone();

        let s1 = store
            .create_session(&proj_id, AgentKind::Claude, "sonnet", "Claude Sonnet")
            .uid
            .clone();
        let s2 = store
            .create_session(&proj_id, AgentKind::Claude, "opus", "Claude Opus")
            .uid
            .clone();
        let s3 = store
            .create_session(&proj_id, AgentKind::Agy, "gemini-4-hydrogen-high", "Agy H4")
            .uid
            .clone();

        assert_ne!(s1, s2);
        assert_eq!(store.active_sessions.len(), 3);
        assert!(store
            .active_sessions
            .iter()
            .all(|s| s.project_id == proj_id));

        let imp1 = store.import_external_session(
            &proj_id,
            AgentKind::OpenCode,
            "ses_123",
            "External OC",
            "opencode/big-pickle",
        );
        let imp2 = store.import_external_session(
            &proj_id,
            AgentKind::OpenCode,
            "ses_123",
            "External OC",
            "opencode/big-pickle",
        );
        assert_eq!(imp1, imp2);
        assert_eq!(store.active_sessions.len(), 4);

        store.archive_session(&s3);
        assert!(store.session_by_uid(&s3).unwrap().archived);
    }
}
