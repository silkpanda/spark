//! Spark v0.2.0 GUI application state and Impeccable Obsidian & Mist Blue views.

use crate::cli::detectors::{self, CliStatus};
use crate::cli::models::{self, ModelOption};
use crate::cli::runner::{self, RunRequest, RunningProcess, StreamEvent};
use crate::cli::sessions::{self, DiscoveredSession};
use crate::cli::{char_prefix, AgentKind};
use crate::history::{HistoryStore, ProjectRecord, SessionRecord, TurnRecord};
use crate::markdown;
use crate::theme;
use eframe::egui::{
    self, Color32, CornerRadius, CursorIcon, Frame, Margin, RichText, Stroke, UiBuilder,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::mpsc as std_mpsc;
use std::time::Instant;
use tokio::runtime::Runtime;
use tokio::sync::mpsc as tokio_mpsc;

const MAX_OUTPUT_LINES: usize = 5_000;
const MAX_OUTPUT_BYTES: usize = 512 * 1024;
const READING_COL_MAX_WIDTH: f32 = 860.0;

/// Execution lifecycle state for an active session.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum RunState {
    #[default]
    Idle,
    Running {
        started_at: Instant,
        prompt: String,
    },
    Done {
        duration_ms: u64,
    },
    Failed {
        duration_ms: u64,
        exit_code: Option<i32>,
    },
}

/// Background messages delivered to the UI thread.
enum UiMessage {
    DetectionComplete {
        statuses: Vec<CliStatus>,
        models: HashMap<AgentKind, Vec<ModelOption>>,
    },
    ExternalSessionsDiscovered(Vec<DiscoveredSession>),
    Stream(StreamEvent),
}

/// Bounded line buffer for live CLI stdout/stderr.
#[derive(Debug, Default)]
pub struct OutputBuffer {
    pub lines: Vec<String>,
    pub total_bytes: usize,
    pub truncated: bool,
}

impl OutputBuffer {
    pub fn clear(&mut self) {
        self.lines.clear();
        self.total_bytes = 0;
        self.truncated = false;
    }

    pub fn push_line(&mut self, line: String) {
        self.total_bytes = self.total_bytes.saturating_add(line.len() + 1);
        self.lines.push(line);

        while self.lines.len() > MAX_OUTPUT_LINES || self.total_bytes > MAX_OUTPUT_BYTES {
            if let Some(removed) = self.lines.first().cloned() {
                self.lines.remove(0);
                self.total_bytes = self.total_bytes.saturating_sub(removed.len() + 1);
                self.truncated = true;
            } else {
                break;
            }
        }
    }

    pub fn joined(&self) -> String {
        self.lines.join("\n")
    }
}

/// Ephemeral runtime state for a Spark session (process handle + output buffers).
#[derive(Default)]
pub struct SessionRuntimeState {
    pub run_state: RunState,
    pub process: Option<RunningProcess>,
    pub stdout: OutputBuffer,
    pub stderr: OutputBuffer,
    pub live_turn_committed: bool,
    /// CLI session ID the running turn was told to create; kept on the session
    /// once that turn succeeds.
    pub pending_cli_session_id: Option<String>,
    /// CLI session ID the running turn was told to resume.
    pub resumed_cli_session_id: Option<String>,
}

impl SessionRuntimeState {
    pub fn is_running(&self) -> bool {
        matches!(self.run_state, RunState::Running { .. })
    }

    pub fn cancel(&mut self) {
        if let Some(proc) = self.process.as_mut() {
            proc.cancel();
        }
    }

    /// Terminate the session's process tree synchronously (used when the app closes).
    pub fn kill_now(&mut self) {
        if let Some(proc) = self.process.as_mut() {
            proc.kill_now();
        }
    }
}

struct NewProjectDraft {
    name: String,
    working_dir: String,
}

struct NewSessionDraft {
    project_id: String,
    agent: AgentKind,
    model: String,
    custom_model_mode: bool,
    title: String,
}

/// Return `true` when `session` is visible under the active top-bar AI filters.
pub fn session_matches_filters(
    session: &SessionRecord,
    ai_filters: &HashMap<AgentKind, bool>,
) -> bool {
    !session.archived && ai_filters.get(&session.agent).copied().unwrap_or(true)
}

/// Return `true` when `session` matches both the AI provider filters and the sidebar search query.
pub fn session_matches_filters_and_query(
    session: &SessionRecord,
    ai_filters: &HashMap<AgentKind, bool>,
    query: &str,
) -> bool {
    if !session_matches_filters(session, ai_filters) {
        return false;
    }
    let needle = query.trim().to_lowercase();
    if needle.is_empty() {
        return true;
    }
    session.title.to_lowercase().contains(&needle)
        || session.model.to_lowercase().contains(&needle)
        || session
            .agent
            .display_name()
            .to_lowercase()
            .contains(&needle)
}

/// Main Spark desktop application.
pub struct SparkApp {
    runtime: Runtime,
    ui_tx: std_mpsc::Sender<UiMessage>,
    ui_rx: std_mpsc::Receiver<UiMessage>,
    cli_statuses: HashMap<AgentKind, CliStatus>,
    model_catalog: HashMap<AgentKind, Vec<ModelOption>>,
    runtime_sessions: HashMap<String, SessionRuntimeState>,
    ai_filters: HashMap<AgentKind, bool>,
    custom_model_input_open: HashMap<String, bool>,
    prompt: String,
    history: HistoryStore,
    history_path: Option<PathBuf>,
    last_error: Option<String>,
    detecting: bool,
    sidebar_search: String,
    renaming_session_uid: Option<String>,
    rename_buffer: String,
    rename_just_started: bool,
    session_options_open: bool,
    raw_output_mode: bool,
    copied_feedback: Option<(String, Instant)>,
    new_project_draft: Option<NewProjectDraft>,
    new_session_draft: Option<NewSessionDraft>,
    import_window_open: bool,
    discovering_external: bool,
    discovered_external: Vec<DiscoveredSession>,
    import_target_project_id: Option<String>,
}

impl SparkApp {
    pub fn new(ctx: egui::Context) -> Self {
        Self::with_history_path(ctx, HistoryStore::default_path())
    }

    /// Build the app around the history file at `history_path` (`None`: keep
    /// history in memory only). `new` uses `~/.spark/history.json`.
    pub fn with_history_path(ctx: egui::Context, mut history_path: Option<PathBuf>) -> Self {
        let runtime = Runtime::new().expect("Tokio runtime should initialize");
        let (ui_tx, ui_rx) = std_mpsc::channel();

        let (mut history, load_warning) = match history_path.as_deref() {
            Some(path) => HistoryStore::load_for_startup(path),
            None => (HistoryStore::default(), None),
        };
        // An unreadable file that could not be moved aside is still at `path`.
        // Saving would overwrite it, so this run keeps its history in memory only.
        if load_warning.is_some() && history_path.as_deref().is_some_and(|p| p.exists()) {
            history_path = None;
        }

        let default_dir = std::env::current_dir().ok();
        history.migrate_v01_if_needed(default_dir);

        let mut model_catalog = HashMap::new();
        let mut ai_filters = HashMap::new();
        for agent in AgentKind::ALL {
            model_catalog.insert(agent, models::default_models_for(agent));
            ai_filters.insert(agent, true);
        }

        let mut app = Self {
            runtime,
            ui_tx,
            ui_rx,
            cli_statuses: HashMap::new(),
            model_catalog,
            runtime_sessions: HashMap::new(),
            ai_filters,
            custom_model_input_open: HashMap::new(),
            prompt: String::new(),
            history,
            history_path,
            last_error: load_warning,
            detecting: false,
            sidebar_search: String::new(),
            renaming_session_uid: None,
            rename_buffer: String::new(),
            rename_just_started: false,
            session_options_open: false,
            raw_output_mode: false,
            copied_feedback: None,
            new_project_draft: None,
            new_session_draft: None,
            import_window_open: false,
            discovering_external: false,
            discovered_external: Vec::new(),
            import_target_project_id: None,
        };

        app.persist_history();
        app.trigger_detection(ctx);
        app
    }

    fn trigger_detection(&mut self, ctx: egui::Context) {
        self.detecting = true;
        let tx = self.ui_tx.clone();
        self.runtime.spawn(async move {
            let statuses = detectors::detect_all().await;
            let models = models::fetch_all_models(&statuses).await;
            let _ = tx.send(UiMessage::DetectionComplete { statuses, models });
            ctx.request_repaint();
        });
    }

    fn trigger_external_discovery(&mut self, ctx: egui::Context) {
        self.discovering_external = true;
        let tx = self.ui_tx.clone();
        let statuses: Vec<CliStatus> = self.cli_statuses.values().cloned().collect();
        let working_dirs: Vec<PathBuf> = self
            .history
            .projects
            .iter()
            .filter_map(|p| p.working_dir.clone())
            .collect();

        self.runtime.spawn(async move {
            let found = sessions::discover_all_sessions(&statuses, &working_dirs).await;
            let _ = tx.send(UiMessage::ExternalSessionsDiscovered(found));
            ctx.request_repaint();
        });
    }

    fn persist_history(&mut self) {
        if let Some(path) = self.history_path.as_deref() {
            if let Err(err) = self.history.save_to(path) {
                self.last_error = Some(err);
            }
        }
    }

    fn drain_messages(&mut self) {
        while let Ok(msg) = self.ui_rx.try_recv() {
            match msg {
                UiMessage::DetectionComplete { statuses, models } => {
                    self.detecting = false;
                    for status in statuses {
                        self.cli_statuses.insert(status.agent, status);
                    }
                    self.model_catalog = models;
                }
                UiMessage::ExternalSessionsDiscovered(found) => {
                    self.discovering_external = false;
                    self.discovered_external = found;
                }
                UiMessage::Stream(ev) => self.handle_stream_event(ev),
            }
        }
    }

    fn handle_stream_event(&mut self, ev: StreamEvent) {
        match ev {
            StreamEvent::StdoutLine {
                session_uid, line, ..
            } => {
                self.runtime_sessions
                    .entry(session_uid)
                    .or_default()
                    .stdout
                    .push_line(line);
            }
            StreamEvent::StderrLine {
                session_uid, line, ..
            } => {
                self.runtime_sessions
                    .entry(session_uid)
                    .or_default()
                    .stderr
                    .push_line(line);
            }
            StreamEvent::Finished {
                session_uid,
                agent,
                exit_code,
                success,
            } => {
                let completed = {
                    let rt = self
                        .runtime_sessions
                        .entry(session_uid.clone())
                        .or_default();
                    rt.process = None;
                    let created = rt.pending_cli_session_id.take();
                    let resumed = rt.resumed_cli_session_id.take();
                    let (duration_ms, prompt) = match &rt.run_state {
                        RunState::Running { started_at, prompt } => {
                            (started_at.elapsed().as_millis() as u64, prompt.clone())
                        }
                        _ => (0, String::new()),
                    };

                    rt.run_state = if success {
                        RunState::Done { duration_ms }
                    } else {
                        RunState::Failed {
                            duration_ms,
                            exit_code,
                        }
                    };

                    (
                        prompt,
                        rt.stdout.joined(),
                        rt.stderr.joined(),
                        duration_ms,
                        created,
                        resumed,
                    )
                };

                let (prompt, output, stderr, duration_ms, created, resumed) = completed;
                let outcome = session_id_outcome(agent, success, created, resumed, &stderr, |id| {
                    sessions::user_home_dir()
                        .is_some_and(|home| sessions::claude_session_exists(&home, id))
                });
                let changed = match outcome {
                    SessionIdOutcome::Keep(id) => {
                        self.history.adopt_cli_session_id(&session_uid, id)
                    }
                    SessionIdOutcome::Forget(stale) => {
                        let forgotten = self.history.forget_cli_session_id(&session_uid, &stale);
                        if forgotten {
                            self.last_error = Some(
                                "Claude Code no longer has this session's conversation (it is \
                                 tied to the project folder). The next prompt starts a new one."
                                    .to_string(),
                            );
                        }
                        forgotten
                    }
                    SessionIdOutcome::Unchanged => false,
                };
                if changed {
                    self.persist_history();
                }
                if !prompt.is_empty() {
                    let meta = self.history.session_by_uid(&session_uid).map(|s| {
                        (
                            s.project_id.clone(),
                            s.cli_session_id.clone(),
                            non_empty(&s.model),
                            non_empty(&s.mode),
                            non_empty(&s.effort),
                        )
                    });
                    if let Some((project_id, cli_session_id, model, mode, effort)) = meta {
                        self.history.record_session_turn(
                            Some(project_id),
                            Some(session_uid.clone()),
                            agent,
                            prompt,
                            output,
                            stderr,
                            success,
                            exit_code,
                            duration_ms,
                            cli_session_id,
                            model,
                            mode,
                            effort,
                        );
                        if let Some(rt) = self.runtime_sessions.get_mut(&session_uid) {
                            rt.live_turn_committed = true;
                        }
                        self.persist_history();
                    }
                }
            }
            StreamEvent::Error {
                session_uid,
                message,
                ..
            } => {
                let rt = self.runtime_sessions.entry(session_uid).or_default();
                rt.process = None;
                rt.pending_cli_session_id = None;
                rt.resumed_cli_session_id = None;
                rt.stderr.push_line(message);
                rt.live_turn_committed = false;
                let duration_ms = match &rt.run_state {
                    RunState::Running { started_at, .. } => started_at.elapsed().as_millis() as u64,
                    _ => 0,
                };
                rt.run_state = RunState::Failed {
                    duration_ms,
                    exit_code: None,
                };
            }
        }
    }

    fn dispatch_to_session(&mut self, session_uid: &str, ctx: &egui::Context) {
        let prompt = self.prompt.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        if self
            .runtime_sessions
            .get(session_uid)
            .is_some_and(SessionRuntimeState::is_running)
        {
            return;
        }

        let Some(sess) = self.history.session_by_uid(session_uid).cloned() else {
            return;
        };
        let Some(binary_path) = self
            .cli_statuses
            .get(&sess.agent)
            .filter(|s| s.available)
            .and_then(|s| s.binary_path.clone())
        else {
            self.last_error = Some(format!(
                "{} CLI is not available on this machine",
                sess.agent.display_name()
            ));
            return;
        };

        let working_dir = self
            .history
            .project(&sess.project_id)
            .and_then(|p| p.working_dir.clone());

        self.last_error = None;

        // Auto-name default sessions from their first prompt for quick identification.
        if let Some(sess_mut) = self.history.session_by_uid_mut(session_uid) {
            if sess_mut.title == format!("{} Session", sess_mut.agent.display_name()) {
                sess_mut.title = compact_session_title(&prompt);
            }
        }

        let mut req = RunRequest::new(sess.agent, prompt.clone());
        req.working_dir = working_dir;
        req.session_id = sess.cli_session_id.filter(|s| !s.trim().is_empty());
        let new_cli_session_id = new_cli_session_id_for(sess.agent, req.session_id.as_deref());
        req.new_session_id = new_cli_session_id.clone();
        req.model = non_empty(&sess.model);
        req.mode = non_empty(&sess.mode);
        req.effort = non_empty(&sess.effort);

        let (tokio_tx, mut tokio_rx) = tokio_mpsc::unbounded_channel();
        let _guard = self.runtime.enter();
        match runner::spawn_agent(&binary_path, session_uid, &req, tokio_tx) {
            Ok(proc_handle) => {
                let rt = self
                    .runtime_sessions
                    .entry(session_uid.to_string())
                    .or_default();
                rt.stdout.clear();
                rt.stderr.clear();
                rt.live_turn_committed = false;
                rt.pending_cli_session_id = new_cli_session_id;
                rt.resumed_cli_session_id = req.session_id.clone();
                rt.process = Some(proc_handle);
                rt.run_state = RunState::Running {
                    started_at: Instant::now(),
                    prompt,
                };
                self.prompt.clear();

                let ui_tx = self.ui_tx.clone();
                let repaint_ctx = ctx.clone();
                self.runtime.spawn(async move {
                    while let Some(ev) = tokio_rx.recv().await {
                        let _ = ui_tx.send(UiMessage::Stream(ev));
                        repaint_ctx.request_repaint();
                    }
                });
                self.persist_history();
            }
            Err(err) => {
                let rt = self
                    .runtime_sessions
                    .entry(session_uid.to_string())
                    .or_default();
                rt.stderr.push_line(err.clone());
                rt.live_turn_committed = false;
                rt.run_state = RunState::Failed {
                    duration_ms: 0,
                    exit_code: None,
                };
                self.last_error = Some(err);
            }
        }
    }

    fn running_count(&self) -> usize {
        self.runtime_sessions
            .values()
            .filter(|rt| rt.is_running())
            .count()
    }

    fn first_running_session_uid(&self) -> Option<String> {
        self.history
            .active_sessions
            .iter()
            .find(|s| {
                !s.archived
                    && self
                        .runtime_sessions
                        .get(&s.uid)
                        .is_some_and(SessionRuntimeState::is_running)
            })
            .map(|s| s.uid.clone())
    }

    fn any_running(&self) -> bool {
        self.running_count() > 0
    }

    fn stop_all(&mut self) {
        for rt in self.runtime_sessions.values_mut() {
            rt.cancel();
        }
    }

    fn all_ai_filters_enabled(&self) -> bool {
        AgentKind::ALL
            .iter()
            .all(|a| self.ai_filters.get(a).copied().unwrap_or(true))
    }

    fn reset_ai_filters(&mut self) {
        for agent in AgentKind::ALL {
            self.ai_filters.insert(agent, true);
        }
    }

    fn default_target_project_id(&mut self) -> String {
        self.history
            .selected_session_uid
            .as_deref()
            .and_then(|uid| self.history.session_by_uid(uid))
            .map(|s| s.project_id.clone())
            .or_else(|| self.history.projects.first().map(|p| p.id.clone()))
            .unwrap_or_else(|| self.history.ensure_default_project(None))
    }

    fn open_new_session_draft(&mut self, project_id: String) {
        let agent = AgentKind::ALL
            .into_iter()
            .find(|a| self.cli_statuses.get(a).is_some_and(|s| s.available))
            .unwrap_or(AgentKind::Claude);
        let default_model = self
            .model_catalog
            .get(&agent)
            .and_then(|list| list.first())
            .map(|m| m.id.clone())
            .unwrap_or_default();

        self.new_session_draft = Some(NewSessionDraft {
            project_id,
            agent,
            model: default_model,
            custom_model_mode: false,
            title: String::new(),
        });
    }

    fn handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        let ctrl_n = ctx.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::N));
        let esc = ctx.input(|i| i.key_pressed(egui::Key::Escape));

        if esc {
            if self.new_session_draft.is_some() {
                self.new_session_draft = None;
            } else if self.import_window_open {
                self.import_window_open = false;
            } else if self.new_project_draft.is_some() {
                self.new_project_draft = None;
            } else if self.renaming_session_uid.is_some() {
                self.renaming_session_uid = None;
            }
        }

        if ctrl_n && self.new_session_draft.is_none() && !self.import_window_open {
            let target_project = self.default_target_project_id();
            self.open_new_session_draft(target_project);
        }
    }

    fn render_top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("spark_header")
            .frame(
                Frame::new()
                    .fill(theme::OBSIDIAN_RAISED)
                    .inner_margin(Margin::symmetric(16, 9))
                    .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    // Brand mark
                    ui.label(
                        RichText::new("◆")
                            .strong()
                            .size(16.0)
                            .color(theme::MIST_BLUE),
                    );
                    ui.label(
                        RichText::new("SPARK")
                            .strong()
                            .size(14.5)
                            .color(theme::TEXT_BRIGHT),
                    );
                    ui.label(RichText::new("0.2").size(11.0).color(theme::TEXT_FAINT));

                    ui.add_space(6.0);
                    ui.separator();
                    ui.add_space(6.0);

                    // Provider filter chips with live CLI availability dots
                    for agent in AgentKind::ALL {
                        let active = self.ai_filters.get(&agent).copied().unwrap_or(true);
                        let count = self
                            .history
                            .active_sessions
                            .iter()
                            .filter(|s| !s.archived && s.agent == agent)
                            .count();
                        let cli_status = self.cli_statuses.get(&agent);
                        let (dot_color, tooltip) = match cli_status {
                            None => (
                                theme::TEXT_FAINT,
                                format!("Probing {} CLI...", agent.display_name()),
                            ),
                            Some(s) if s.available => (
                                theme::VERDIGRIS_PATINA,
                                format!(
                                    "{} ready ({})",
                                    agent.display_name(),
                                    s.version.as_deref().unwrap_or("detected")
                                ),
                            ),
                            Some(s) => (
                                theme::ERROR_CORAL,
                                s.error.clone().unwrap_or_else(|| {
                                    format!("{} CLI not found", agent.display_name())
                                }),
                            ),
                        };

                        let resp = provider_filter_chip(
                            ui,
                            active,
                            agent.glyph(),
                            agent.display_name(),
                            count,
                            dot_color,
                        )
                        .on_hover_text(tooltip);

                        if resp.clicked() {
                            self.ai_filters.insert(agent, !active);
                        }
                    }

                    if !self.all_ai_filters_enabled() {
                        let reset_btn = egui::Button::new(
                            RichText::new("All").size(11.5).color(theme::MIST_BLUE),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::NONE);
                        if ui
                            .add(reset_btn)
                            .on_hover_text("Show all AI providers")
                            .clicked()
                        {
                            self.reset_ai_filters();
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let new_btn = egui::Button::new(
                            RichText::new("+ New Session")
                                .color(theme::ON_ACCENT)
                                .strong()
                                .size(12.0),
                        )
                        .fill(theme::MIST_BLUE)
                        .stroke(Stroke::NONE)
                        .corner_radius(CornerRadius::same(6));
                        if ui
                            .add(new_btn)
                            .on_hover_text("Create a new session (Ctrl+N)")
                            .clicked()
                        {
                            let target_project = self.default_target_project_id();
                            self.open_new_session_draft(target_project);
                        }

                        let import_btn = egui::Button::new(
                            RichText::new("Import").size(12.0).color(theme::TEXT_BODY),
                        )
                        .fill(theme::SLATE_CONTROL)
                        .stroke(Stroke::new(1.0_f32, theme::HAIRLINE))
                        .corner_radius(CornerRadius::same(6));
                        if ui
                            .add(import_btn)
                            .on_hover_text("Import or resume external CLI sessions")
                            .clicked()
                        {
                            self.import_window_open = true;
                            if self.import_target_project_id.is_none() {
                                self.import_target_project_id =
                                    self.history.projects.first().map(|p| p.id.clone());
                            }
                            self.trigger_external_discovery(ctx.clone());
                        }

                        let rescan_label = if self.detecting {
                            "Scanning..."
                        } else {
                            "Scan CLIs"
                        };
                        let scan_btn = egui::Button::new(
                            RichText::new(rescan_label)
                                .size(12.0)
                                .color(theme::TEXT_MUTED),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                        .corner_radius(CornerRadius::same(6));
                        if ui
                            .add_enabled(!self.detecting, scan_btn)
                            .on_hover_text("Re-probe installed CLIs and models")
                            .clicked()
                        {
                            self.trigger_detection(ctx.clone());
                        }

                        let running = self.running_count();
                        if running > 0 {
                            let label = format!("● {running} running");
                            let jump_btn = egui::Button::new(
                                RichText::new(label)
                                    .size(11.5)
                                    .strong()
                                    .color(theme::VERDIGRIS_PATINA),
                            )
                            .fill(theme::VERDIGRIS_SOFT_BG)
                            .stroke(Stroke::new(1.0_f32, theme::VERDIGRIS_PATINA))
                            .corner_radius(CornerRadius::same(10));
                            if ui
                                .add(jump_btn)
                                .on_hover_text("Jump to running session")
                                .clicked()
                            {
                                if let Some(uid) = self.first_running_session_uid() {
                                    self.history.selected_session_uid = Some(uid);
                                    self.persist_history();
                                }
                            }
                        }
                    });
                });
            });
    }

    fn render_sessions_sidebar(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("spark_sessions_sidebar")
            .resizable(true)
            .default_width(284.0)
            .min_width(230.0)
            .max_width(420.0)
            .frame(
                Frame::new()
                    .fill(theme::OBSIDIAN_RAISED)
                    .inner_margin(Margin::symmetric(10, 12))
                    .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("WORKSPACES")
                            .strong()
                            .size(11.0)
                            .color(theme::TEXT_FAINT),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let add_proj_btn = egui::Button::new(
                            RichText::new("+ Project")
                                .size(11.5)
                                .color(theme::TEXT_MUTED),
                        )
                        .fill(Color32::TRANSPARENT)
                        .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                        .corner_radius(CornerRadius::same(5));
                        if ui.add(add_proj_btn).clicked() {
                            self.new_project_draft = Some(NewProjectDraft {
                                name: String::new(),
                                working_dir: String::new(),
                            });
                        }
                    });
                });

                ui.add_space(6.0);

                // Sleek sidebar session search
                Frame::new()
                    .fill(theme::OBSIDIAN_DEEP)
                    .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                    .corner_radius(CornerRadius::same(6))
                    .inner_margin(Margin::symmetric(8, 4))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let clear_w = if self.sidebar_search.is_empty() {
                                0.0
                            } else {
                                20.0
                            };
                            let input_w = (ui.available_width() - clear_w).max(60.0);
                            ui.add(
                                egui::TextEdit::singleline(&mut self.sidebar_search)
                                    .frame(false)
                                    .hint_text("Filter sessions...")
                                    .desired_width(input_w),
                            );
                            if !self.sidebar_search.is_empty()
                                && ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("×").size(12.0).color(theme::TEXT_MUTED),
                                        )
                                        .frame(false),
                                    )
                                    .clicked()
                            {
                                self.sidebar_search.clear();
                            }
                        });
                    });

                self.render_new_project_inline(ui);
                ui.add_space(8.0);

                let projects_snapshot: Vec<ProjectRecord> = self.history.projects.clone();
                let sessions_snapshot: Vec<SessionRecord> = self.history.active_sessions.clone();
                let selected_uid = self.history.selected_session_uid.clone();
                let pulse = running_pulse(ctx);

                let mut toggle_collapse_id: Option<String> = None;
                let mut pick_folder_project_id: Option<String> = None;
                let mut add_session_project_id: Option<String> = None;
                let mut select_uid: Option<String> = None;
                let mut archive_uid: Option<String> = None;
                let mut stop_uid: Option<String> = None;
                let mut hidden_by_ai_count: usize = 0;

                egui::ScrollArea::vertical()
                    .id_salt("active_sessions_scroll")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        for (proj_idx, project) in projects_snapshot.iter().enumerate() {
                            if proj_idx > 0 {
                                ui.add_space(8.0);
                            }

                            let proj_sessions: Vec<&SessionRecord> = sessions_snapshot
                                .iter()
                                .filter(|s| s.project_id == project.id && !s.archived)
                                .collect();
                            let ai_visible: Vec<&SessionRecord> = proj_sessions
                                .iter()
                                .copied()
                                .filter(|s| session_matches_filters(s, &self.ai_filters))
                                .collect();
                            hidden_by_ai_count +=
                                proj_sessions.len().saturating_sub(ai_visible.len());

                            let visible_sessions: Vec<&SessionRecord> = ai_visible
                                .into_iter()
                                .filter(|s| {
                                    session_matches_filters_and_query(
                                        s,
                                        &self.ai_filters,
                                        &self.sidebar_search,
                                    )
                                })
                                .collect();

                            // Borderless project header row
                            ui.horizontal(|ui| {
                                let arrow = if project.collapsed { "▸" } else { "▾" };
                                let header_label = format!("{arrow}  {}", project.name);
                                let header_btn = egui::Button::new(
                                    RichText::new(header_label)
                                        .strong()
                                        .size(12.5)
                                        .color(theme::TEXT_BRIGHT),
                                )
                                .frame(false);
                                if ui
                                    .add(header_btn)
                                    .on_hover_cursor(CursorIcon::PointingHand)
                                    .clicked()
                                {
                                    toggle_collapse_id = Some(project.id.clone());
                                }

                                ui.label(
                                    RichText::new(format!("{}", proj_sessions.len()))
                                        .size(11.0)
                                        .color(theme::TEXT_FAINT),
                                );

                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        let plus_btn = egui::Button::new(
                                            RichText::new("+").size(13.0).color(theme::TEXT_MUTED),
                                        )
                                        .frame(false);
                                        if ui
                                            .add(plus_btn)
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text("New session in this project")
                                            .clicked()
                                        {
                                            add_session_project_id = Some(project.id.clone());
                                        }

                                        let dir_btn = egui::Button::new(
                                            RichText::new("Dir")
                                                .size(10.5)
                                                .color(theme::TEXT_FAINT),
                                        )
                                        .frame(false);
                                        if ui
                                            .add(dir_btn)
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text("Set project working directory")
                                            .clicked()
                                        {
                                            pick_folder_project_id = Some(project.id.clone());
                                        }
                                    },
                                );
                            });

                            if let Some(dir) = &project.working_dir {
                                let full_path = dir.display().to_string();
                                let short_path = truncate_path_left(&full_path, 36);
                                ui.horizontal(|ui| {
                                    ui.add_space(16.0);
                                    ui.label(
                                        RichText::new(short_path)
                                            .size(10.5)
                                            .color(theme::TEXT_FAINT),
                                    )
                                    .on_hover_text(full_path);
                                });
                            }

                            ui.add_space(3.0);

                            if !project.collapsed {
                                if visible_sessions.is_empty() {
                                    let msg = if proj_sessions.is_empty() {
                                        "No sessions yet"
                                    } else if !self.sidebar_search.trim().is_empty() {
                                        "No matching sessions"
                                    } else {
                                        "Hidden by top AI filter"
                                    };
                                    ui.horizontal(|ui| {
                                        ui.add_space(16.0);
                                        ui.label(
                                            RichText::new(msg).size(11.0).color(theme::TEXT_FAINT),
                                        );
                                    });
                                } else {
                                    for sess in visible_sessions {
                                        let is_selected =
                                            selected_uid.as_deref() == Some(sess.uid.as_str());
                                        let rt = self.runtime_sessions.get(&sess.uid);
                                        let is_running =
                                            rt.is_some_and(SessionRuntimeState::is_running);
                                        let turn_count =
                                            self.history.turns_for_session(&sess.uid).len();

                                        let outcome = render_sidebar_session_row(
                                            ui,
                                            sess,
                                            rt,
                                            turn_count,
                                            is_selected,
                                            is_running,
                                            pulse,
                                        );
                                        if outcome.select {
                                            select_uid = Some(sess.uid.clone());
                                        }
                                        if outcome.stop {
                                            stop_uid = Some(sess.uid.clone());
                                        }
                                        if outcome.archive {
                                            archive_uid = Some(sess.uid.clone());
                                        }
                                        ui.add_space(2.0);
                                    }
                                }
                            }
                        }

                        if hidden_by_ai_count > 0 {
                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(format!(
                                        "{hidden_by_ai_count} hidden by AI filter"
                                    ))
                                    .size(11.0)
                                    .color(theme::TEXT_FAINT),
                                );
                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Reset")
                                                .size(11.0)
                                                .color(theme::MIST_BLUE),
                                        )
                                        .frame(false),
                                    )
                                    .clicked()
                                {
                                    self.reset_ai_filters();
                                }
                            });
                        }
                    });

                let mut changed = false;
                if let Some(pid) = toggle_collapse_id {
                    if let Some(proj) = self.history.project_mut(&pid) {
                        proj.collapsed = !proj.collapsed;
                        changed = true;
                    }
                }
                if let Some(pid) = pick_folder_project_id {
                    let initial = self
                        .history
                        .project(&pid)
                        .and_then(|p| p.working_dir.clone());
                    if let Some(folder) = pick_folder_dialog(initial.as_deref()) {
                        if let Some(proj) = self.history.project_mut(&pid) {
                            proj.working_dir = Some(folder);
                            changed = true;
                        }
                    }
                }
                if let Some(pid) = add_session_project_id {
                    self.open_new_session_draft(pid);
                }
                if let Some(uid) = select_uid {
                    self.history.selected_session_uid = Some(uid);
                    self.renaming_session_uid = None;
                    changed = true;
                }
                if let Some(uid) = stop_uid {
                    if let Some(rt) = self.runtime_sessions.get_mut(&uid) {
                        rt.cancel();
                    }
                }
                if let Some(uid) = archive_uid {
                    self.history.archive_session(&uid);
                    changed = true;
                }
                if changed {
                    self.persist_history();
                }
            });
    }

    fn render_new_project_inline(&mut self, ui: &mut egui::Ui) {
        let Some(mut draft) = self.new_project_draft.take() else {
            return;
        };

        let mut keep_open = true;
        let mut create = false;
        ui.add_space(6.0);
        Frame::new()
            .fill(theme::USER_SURFACE)
            .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_HOVER))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::same(10))
            .show(ui, |ui| {
                ui.label(
                    RichText::new("New Project")
                        .strong()
                        .size(12.0)
                        .color(theme::MIST_BLUE),
                );
                ui.add_space(4.0);
                ui.add(
                    egui::TextEdit::singleline(&mut draft.name)
                        .hint_text("Project name")
                        .desired_width(f32::INFINITY),
                );
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.add(
                        egui::TextEdit::singleline(&mut draft.working_dir)
                            .hint_text("Working directory (optional)")
                            .desired_width((ui.available_width() - 78.0).max(80.0)),
                    );
                    if ui.button("Browse").clicked() {
                        if let Some(folder) = pick_folder_dialog(None) {
                            if draft.name.trim().is_empty() {
                                if let Some(name) = folder.file_name().and_then(|n| n.to_str()) {
                                    draft.name = name.to_string();
                                }
                            }
                            draft.working_dir = folder.display().to_string();
                        }
                    }
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let create_btn = egui::Button::new(
                        RichText::new("Create")
                            .strong()
                            .size(12.0)
                            .color(theme::ON_ACCENT),
                    )
                    .fill(theme::MIST_BLUE)
                    .stroke(Stroke::NONE)
                    .corner_radius(CornerRadius::same(5));
                    if ui.add(create_btn).clicked() {
                        create = true;
                        keep_open = false;
                    }
                    if ui.button("Cancel").clicked() {
                        keep_open = false;
                    }
                });
            });

        if create {
            let dir = non_empty(&draft.working_dir).map(PathBuf::from);
            self.history.create_project(draft.name, dir);
            self.persist_history();
        } else if keep_open {
            self.new_project_draft = Some(draft);
        }
    }

    fn render_modal_scrim(&self, ctx: &egui::Context) {
        egui::Area::new(egui::Id::new("spark_modal_scrim"))
            .order(egui::Order::Middle)
            .fixed_pos(egui::Pos2::ZERO)
            .interactable(false)
            .show(ctx, |ui| {
                let rect = ctx.screen_rect();
                ui.painter().rect_filled(rect, 0.0, theme::MODAL_BACKDROP);
            });
    }

    fn render_new_session_window(&mut self, ctx: &egui::Context) {
        let Some(mut draft) = self.new_session_draft.take() else {
            return;
        };

        self.render_modal_scrim(ctx);

        let mut open = true;
        let mut create = false;
        let mut cancel = false;

        egui::Window::new("New Session")
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(450.0)
            .show(ctx, |ui| {
                ui.add_space(2.0);
                ui.label(RichText::new("Project").size(12.0).color(theme::TEXT_MUTED));
                let selected_proj_name = self
                    .history
                    .project(&draft.project_id)
                    .map(|p| p.name.clone())
                    .unwrap_or_else(|| "Select project".to_string());
                egui::ComboBox::from_id_salt("new_sess_project")
                    .selected_text(selected_proj_name)
                    .width(410.0)
                    .show_ui(ui, |ui| {
                        for proj in &self.history.projects {
                            ui.selectable_value(&mut draft.project_id, proj.id.clone(), &proj.name);
                        }
                    });

                ui.add_space(10.0);
                ui.label(
                    RichText::new("AI Provider")
                        .size(12.0)
                        .color(theme::TEXT_MUTED),
                );
                ui.horizontal(|ui| {
                    let card_w = (410.0 - 16.0) / 3.0;
                    for agent in AgentKind::ALL {
                        let selected = draft.agent == agent;
                        let avail = self.cli_statuses.get(&agent).is_some_and(|s| s.available);
                        let fill = if selected {
                            theme::ACCENT_SOFT_BG
                        } else {
                            theme::SLATE_CONTROL
                        };
                        let stroke = if selected {
                            Stroke::new(1.0_f32, theme::MIST_BLUE)
                        } else {
                            Stroke::new(1.0_f32, theme::HAIRLINE)
                        };
                        let resp = Frame::new()
                            .fill(fill)
                            .stroke(stroke)
                            .corner_radius(CornerRadius::same(8))
                            .inner_margin(Margin::symmetric(10, 8))
                            .show(ui, |ui| {
                                ui.set_min_width(card_w - 20.0);
                                ui.set_max_width(card_w - 20.0);
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new(agent.glyph()).strong().size(14.0).color(
                                            if selected {
                                                theme::MIST_BLUE
                                            } else {
                                                theme::TEXT_MUTED
                                            },
                                        ),
                                    );
                                    ui.label(
                                        RichText::new(agent.display_name())
                                            .strong()
                                            .size(12.0)
                                            .color(theme::TEXT_BRIGHT),
                                    );
                                });
                                let (status_txt, status_col) = if avail {
                                    ("Ready", theme::VERDIGRIS_PATINA)
                                } else if self.cli_statuses.contains_key(&agent) {
                                    ("Not found", theme::TEXT_FAINT)
                                } else {
                                    ("Probing...", theme::TEXT_FAINT)
                                };
                                ui.label(RichText::new(status_txt).size(10.5).color(status_col));
                            })
                            .response
                            .interact(egui::Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand);

                        if resp.clicked() {
                            draft.agent = agent;
                            draft.custom_model_mode = false;
                            draft.model = self
                                .model_catalog
                                .get(&agent)
                                .and_then(|m| m.first())
                                .map(|m| m.id.clone())
                                .unwrap_or_default();
                        }
                    }
                });

                ui.add_space(10.0);
                ui.label(RichText::new("Model").size(12.0).color(theme::TEXT_MUTED));
                let options = self
                    .model_catalog
                    .get(&draft.agent)
                    .cloned()
                    .unwrap_or_else(|| models::default_models_for(draft.agent));
                render_model_selector_controls(
                    ui,
                    "new_sess_model",
                    &options,
                    &mut draft.model,
                    &mut draft.custom_model_mode,
                    410.0,
                );

                ui.add_space(10.0);
                ui.label(
                    RichText::new("Session Title (optional)")
                        .size(12.0)
                        .color(theme::TEXT_MUTED),
                );
                let title_resp = ui.add(
                    egui::TextEdit::singleline(&mut draft.title)
                        .hint_text("Auto-named from your first prompt if left blank")
                        .desired_width(410.0),
                );
                if title_resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    create = true;
                }

                ui.add_space(14.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let btn = egui::Button::new(
                        RichText::new("Create Session")
                            .color(theme::ON_ACCENT)
                            .strong()
                            .size(12.5),
                    )
                    .fill(theme::MIST_BLUE)
                    .stroke(Stroke::NONE)
                    .corner_radius(CornerRadius::same(6));
                    if ui.add(btn).clicked() {
                        create = true;
                    }
                    if ui.button("Cancel").clicked() {
                        cancel = true;
                    }
                });
            });

        if create {
            self.history
                .create_session(draft.project_id, draft.agent, draft.model, draft.title);
            self.persist_history();
        } else if open && !cancel {
            self.new_session_draft = Some(draft);
        }
    }

    fn render_import_window(&mut self, ctx: &egui::Context) {
        if !self.import_window_open {
            return;
        }

        self.render_modal_scrim(ctx);

        let mut open = true;
        let mut import_selected: Option<DiscoveredSession> = None;
        let mut refresh = false;

        egui::Window::new("Import / Resume CLI Session")
            .open(&mut open)
            .collapsible(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_size([580.0, 440.0])
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("Target Project:")
                            .size(12.0)
                            .color(theme::TEXT_MUTED),
                    );
                    let current_name = self
                        .import_target_project_id
                        .as_deref()
                        .and_then(|id| self.history.project(id))
                        .map(|p| p.name.clone())
                        .unwrap_or_else(|| "Auto (match folder)".to_string());

                    egui::ComboBox::from_id_salt("import_target_proj")
                        .selected_text(current_name)
                        .width(220.0)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.import_target_project_id,
                                None,
                                "Auto (match folder)",
                            );
                            for proj in &self.history.projects {
                                ui.selectable_value(
                                    &mut self.import_target_project_id,
                                    Some(proj.id.clone()),
                                    &proj.name,
                                );
                            }
                        });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let label = if self.discovering_external {
                            "Scanning..."
                        } else {
                            "Refresh"
                        };
                        if ui
                            .add_enabled(!self.discovering_external, egui::Button::new(label))
                            .clicked()
                        {
                            refresh = true;
                        }
                    });
                });

                ui.add_space(6.0);
                ui.separator();
                ui.add_space(4.0);

                egui::ScrollArea::vertical()
                    .id_salt("import_external_list")
                    .max_height(340.0)
                    .show(ui, |ui| {
                        if self.discovering_external && self.discovered_external.is_empty() {
                            ui.add_space(20.0);
                            ui.label(
                                RichText::new("Scanning local CLI sessions...")
                                    .color(theme::TEXT_MUTED),
                            );
                            return;
                        }
                        if self.discovered_external.is_empty() {
                            ui.add_space(20.0);
                            ui.label(
                                RichText::new(
                                    "No external sessions found in OpenCode, Claude Code, or Antigravity.",
                                )
                                .color(theme::TEXT_FAINT),
                            );
                            return;
                        }

                        for item in &self.discovered_external {
                            Frame::new()
                                .fill(theme::USER_SURFACE)
                                .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                                .corner_radius(CornerRadius::same(7))
                                .inner_margin(Margin::symmetric(10, 8))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(item.agent.glyph())
                                                .strong()
                                                .size(15.0)
                                                .color(theme::MIST_BLUE),
                                        );
                                        ui.vertical(|ui| {
                                            ui.label(
                                                RichText::new(&item.title)
                                                    .strong()
                                                    .size(12.5)
                                                    .color(theme::TEXT_BRIGHT),
                                            );
                                            let detail = match &item.working_dir {
                                                Some(dir) => {
                                                    format!(
                                                        "{} • {}",
                                                        item.cli_session_id,
                                                        dir.display()
                                                    )
                                                }
                                                None => item.cli_session_id.clone(),
                                            };
                                            ui.label(
                                                RichText::new(detail)
                                                    .size(10.5)
                                                    .color(theme::TEXT_FAINT),
                                            );
                                        });
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                let btn = egui::Button::new(
                                                    RichText::new("Import")
                                                        .strong()
                                                        .size(11.5)
                                                        .color(theme::ON_ACCENT),
                                                )
                                                .fill(theme::MIST_BLUE)
                                                .stroke(Stroke::NONE)
                                                .corner_radius(CornerRadius::same(5));
                                                if ui.add(btn).clicked() {
                                                    import_selected = Some(item.clone());
                                                }
                                            },
                                        );
                                    });
                                });
                            ui.add_space(4.0);
                        }
                    });
            });

        if !open {
            self.import_window_open = false;
        }
        if refresh {
            self.trigger_external_discovery(ctx.clone());
        }
        if let Some(item) = import_selected {
            let project_id = match &self.import_target_project_id {
                Some(pid) => pid.clone(),
                None => match &item.working_dir {
                    Some(dir) if dir.is_dir() => self.history.find_or_create_project_for_dir(dir),
                    _ => self.history.ensure_default_project(None),
                },
            };
            self.history.import_external_session(
                project_id,
                item.agent,
                item.cli_session_id,
                item.title,
                String::new(),
            );
            self.persist_history();
            self.import_window_open = false;
        }
    }

    fn render_prompt_bar(&mut self, ctx: &egui::Context, focused_uid: Option<&str>) {
        let Some(uid) = focused_uid.map(str::to_string) else {
            return;
        };
        let Some(mut sess) = self.history.session_by_uid(&uid).cloned() else {
            return;
        };

        let model_options = self
            .model_catalog
            .get(&sess.agent)
            .cloned()
            .unwrap_or_else(|| models::default_models_for(sess.agent));

        let is_running = self
            .runtime_sessions
            .get(&uid)
            .is_some_and(SessionRuntimeState::is_running);
        let cli_ok = self
            .cli_statuses
            .get(&sess.agent)
            .is_some_and(|st| st.available);

        let mut settings_changed = false;
        let mut clear_error = false;
        let mut transmit_now = false;
        let mut stop_all_now = false;

        egui::TopBottomPanel::bottom("spark_prompt_bar")
            .frame(
                Frame::new()
                    .fill(theme::OBSIDIAN_BG)
                    .inner_margin(Margin::symmetric(16, 10)),
            )
            .show(ctx, |ui| {
                let avail = ui.available_width();
                let col_w = avail.min(READING_COL_MAX_WIDTH);
                let side_pad = ((avail - col_w) * 0.5).max(0.0);

                ui.horizontal(|ui| {
                    ui.add_space(side_pad);
                    ui.vertical(|ui| {
                        ui.set_min_width(col_w);
                        ui.set_max_width(col_w);

                        if let Some(err) = &self.last_error {
                            Frame::new()
                                .fill(theme::CORAL_SOFT_BG)
                                .stroke(Stroke::new(1.0_f32, theme::ERROR_CORAL))
                                .corner_radius(CornerRadius::same(8))
                                .inner_margin(Margin::symmetric(10, 6))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            RichText::new(err).color(theme::ERROR_CORAL).size(12.0),
                                        );
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if ui
                                                    .add(
                                                        egui::Button::new(
                                                            RichText::new("×")
                                                                .size(12.0)
                                                                .color(theme::ERROR_CORAL),
                                                        )
                                                        .frame(false),
                                                    )
                                                    .clicked()
                                                {
                                                    clear_error = true;
                                                }
                                            },
                                        );
                                    });
                                });
                            ui.add_space(6.0);
                        }

                        let ctrl_enter =
                            ui.input(|i| i.modifiers.ctrl && i.key_pressed(egui::Key::Enter));

                        // Unified rounded composer card (Claude Code / Antigravity style)
                        let border_id = ui.make_persistent_id(("composer_focus", &uid));
                        let had_focus = ui.data(|d| d.get_temp::<bool>(border_id).unwrap_or(false));
                        let card_stroke = if had_focus {
                            Stroke::new(1.0_f32, theme::ACCENT_SOFT_BORDER)
                        } else {
                            Stroke::new(1.0_f32, theme::HAIRLINE)
                        };

                        Frame::new()
                            .fill(theme::OBSIDIAN_RAISED)
                            .stroke(card_stroke)
                            .corner_radius(CornerRadius::same(12))
                            .inner_margin(Margin::symmetric(14, 10))
                            .show(ui, |ui| {
                                let hint = format!(
                                    "Message {}...  (Ctrl+Enter to transmit)",
                                    sess.agent.display_name()
                                );
                                let edit_resp = egui::ScrollArea::vertical()
                                    .id_salt(("composer_scroll", &uid))
                                    .max_height(160.0)
                                    .show(ui, |ui| {
                                        ui.add(
                                            egui::TextEdit::multiline(&mut self.prompt)
                                                .frame(false)
                                                .hint_text(hint)
                                                .desired_rows(2)
                                                .desired_width(f32::INFINITY),
                                        )
                                    })
                                    .inner;
                                // Read the focus state BEFORE taking the data lock:
                                // `has_focus()` locks the same `Context`, and doing
                                // it inside `data_mut` deadlocks the UI thread.
                                let has_focus = edit_resp.has_focus();
                                ui.data_mut(|d| d.insert_temp(border_id, has_focus));

                                ui.add_space(6.0);
                                ui.separator();
                                ui.add_space(6.0);

                                // Integrated bottom toolbar: Model / Mode / Effort on left, Transmit / Stop on right
                                ui.horizontal(|ui| {
                                    let custom_open = self
                                        .custom_model_input_open
                                        .entry(uid.clone())
                                        .or_insert(false);
                                    if render_model_selector_controls(
                                        ui,
                                        ("composer_model", &uid),
                                        &model_options,
                                        &mut sess.model,
                                        custom_open,
                                        195.0,
                                    ) {
                                        settings_changed = true;
                                    }

                                    if sess.agent == AgentKind::Agy {
                                        egui::ComboBox::from_id_salt(("composer_mode", &uid))
                                            .selected_text(if sess.mode.is_empty() {
                                                "mode: default".to_string()
                                            } else {
                                                format!("mode: {}", sess.mode)
                                            })
                                            .width(120.0)
                                            .show_ui(ui, |ui| {
                                                for opt in ["", "auto", "accept-edits", "plan"] {
                                                    let label = if opt.is_empty() {
                                                        "default"
                                                    } else {
                                                        opt
                                                    };
                                                    if ui
                                                        .selectable_value(
                                                            &mut sess.mode,
                                                            opt.to_string(),
                                                            label,
                                                        )
                                                        .clicked()
                                                    {
                                                        settings_changed = true;
                                                    }
                                                }
                                            });
                                    }

                                    if matches!(sess.agent, AgentKind::Agy | AgentKind::Claude) {
                                        egui::ComboBox::from_id_salt(("composer_effort", &uid))
                                            .selected_text(if sess.effort.is_empty() {
                                                "effort: default".to_string()
                                            } else {
                                                format!("effort: {}", sess.effort)
                                            })
                                            .width(115.0)
                                            .show_ui(ui, |ui| {
                                                for opt in ["", "low", "medium", "high", "max"] {
                                                    let label = if opt.is_empty() {
                                                        "default"
                                                    } else {
                                                        opt
                                                    };
                                                    if ui
                                                        .selectable_value(
                                                            &mut sess.effort,
                                                            opt.to_string(),
                                                            label,
                                                        )
                                                        .clicked()
                                                    {
                                                        settings_changed = true;
                                                    }
                                                }
                                            });
                                    }

                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            let can_send = !self.prompt.trim().is_empty()
                                                && !is_running
                                                && cli_ok;

                                            let transmit_fill = if can_send {
                                                theme::MIST_BLUE
                                            } else {
                                                theme::SLATE_CONTROL
                                            };
                                            let transmit_fg = if can_send {
                                                theme::ON_ACCENT
                                            } else {
                                                theme::TEXT_FAINT
                                            };
                                            let transmit_btn = egui::Button::new(
                                                RichText::new("Transmit  ↑")
                                                    .color(transmit_fg)
                                                    .strong()
                                                    .size(12.5),
                                            )
                                            .fill(transmit_fill)
                                            .stroke(Stroke::NONE)
                                            .corner_radius(CornerRadius::same(7))
                                            .min_size(egui::vec2(104.0, 28.0));

                                            let disabled_hint = if !cli_ok {
                                                format!(
                                                    "{} CLI not detected",
                                                    sess.agent.display_name()
                                                )
                                            } else if is_running {
                                                "Wait for current run to finish".to_string()
                                            } else {
                                                "Transmit prompt (Ctrl+Enter)".to_string()
                                            };

                                            let resp = ui
                                                .add_enabled(can_send, transmit_btn)
                                                .on_hover_text(disabled_hint);

                                            if (resp.clicked() || ctrl_enter) && can_send {
                                                transmit_now = true;
                                            }

                                            if self.any_running() {
                                                let stop_label = if self.running_count() > 1 {
                                                    format!("Stop All ({})", self.running_count())
                                                } else {
                                                    "Stop".to_string()
                                                };
                                                let stop_btn = egui::Button::new(
                                                    RichText::new(stop_label)
                                                        .strong()
                                                        .color(theme::ERROR_CORAL)
                                                        .size(12.0),
                                                )
                                                .fill(theme::CORAL_SOFT_BG)
                                                .stroke(Stroke::new(1.0_f32, theme::ERROR_CORAL))
                                                .corner_radius(CornerRadius::same(7))
                                                .min_size(egui::vec2(72.0, 28.0));
                                                if ui.add(stop_btn).clicked() {
                                                    stop_all_now = true;
                                                }
                                            }
                                        },
                                    );
                                });
                            });
                    });
                });
            });

        if clear_error {
            self.last_error = None;
        }
        if settings_changed {
            if let Some(target) = self.history.session_by_uid_mut(&uid) {
                *target = sess;
            }
            self.persist_history();
        }
        if stop_all_now {
            self.stop_all();
        }
        if transmit_now {
            self.dispatch_to_session(&uid, ctx);
        }
    }

    fn render_focused_session(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(uid) = self.history.selected_session_uid.clone() else {
            self.render_no_session_selected(ui, ctx);
            return;
        };

        let Some(mut sess) = self.history.session_by_uid(&uid).cloned() else {
            return;
        };

        let projects_list: Vec<(String, String)> = self
            .history
            .projects
            .iter()
            .map(|p| (p.id.clone(), p.name.clone()))
            .collect();
        let current_project_name = self
            .history
            .project(&sess.project_id)
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "General".to_string());
        let project_dir = self
            .history
            .project(&sess.project_id)
            .and_then(|p| p.working_dir.clone());
        let cli_status = self.cli_statuses.get(&sess.agent).cloned();
        let turns: Vec<TurnRecord> = self
            .history
            .turns_for_session(&uid)
            .into_iter()
            .cloned()
            .collect();

        let mut settings_changed = false;
        let mut stop_clicked = false;
        let mut archive_clicked = false;
        let mut pick_folder_for_project: Option<String> = None;
        let mut rescan_clicked = false;

        let pulse = running_pulse(ctx);
        let rt = self.runtime_sessions.entry(uid.clone()).or_default();
        let is_running = rt.is_running();

        // 1. Clean Session Header Strip
        Frame::new()
            .fill(theme::OBSIDIAN_BG)
            .inner_margin(Margin::symmetric(20, 10))
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(sess.agent.glyph()).strong().size(17.0).color(
                        if is_running {
                            theme::VERDIGRIS_PATINA
                        } else {
                            theme::MIST_BLUE
                        },
                    ));

                    // Click-to-rename title (clean text by default, TextEdit only while renaming)
                    if self.renaming_session_uid.as_deref() == Some(uid.as_str()) {
                        let edit_resp = ui.add(
                            egui::TextEdit::singleline(&mut self.rename_buffer)
                                .desired_width(260.0)
                                .font(egui::TextStyle::Heading),
                        );
                        if self.rename_just_started {
                            edit_resp.request_focus();
                            self.rename_just_started = false;
                        }
                        let commit = (edit_resp.lost_focus()
                            && ui.input(|i| !i.key_pressed(egui::Key::Escape)))
                            || ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if commit {
                            if let Some(clean) = non_empty(&self.rename_buffer) {
                                sess.title = clean;
                                settings_changed = true;
                            }
                            self.renaming_session_uid = None;
                        }
                    } else {
                        let title_btn = egui::Button::new(
                            RichText::new(&sess.title)
                                .strong()
                                .size(16.5)
                                .color(theme::TEXT_BRIGHT),
                        )
                        .frame(false);
                        if ui
                            .add(title_btn)
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("Click to rename session")
                            .clicked()
                        {
                            self.renaming_session_uid = Some(uid.clone());
                            self.rename_buffer = sess.title.clone();
                            self.rename_just_started = true;
                        }
                    }

                    ui.add_space(4.0);
                    render_status_pill(ui, &rt.run_state, cli_status.as_ref(), pulse);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if is_running {
                            let stop_btn = egui::Button::new(
                                RichText::new("Stop")
                                    .strong()
                                    .size(11.5)
                                    .color(theme::ERROR_CORAL),
                            )
                            .fill(theme::CORAL_SOFT_BG)
                            .stroke(Stroke::new(1.0_f32, theme::ERROR_CORAL))
                            .corner_radius(CornerRadius::same(6));
                            if ui.add(stop_btn).clicked() {
                                stop_clicked = true;
                            }
                        }

                        // Options drawer toggle
                        let opt_fill = if self.session_options_open {
                            theme::SLATE_HOVER
                        } else {
                            Color32::TRANSPARENT
                        };
                        let opt_stroke = if self.session_options_open {
                            Stroke::new(1.0_f32, theme::ACCENT_SOFT_BORDER)
                        } else {
                            Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE)
                        };
                        let opt_label = if self.session_options_open {
                            "Options ▴"
                        } else {
                            "Options ▾"
                        };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(opt_label).size(11.5).color(theme::TEXT_MUTED),
                                )
                                .fill(opt_fill)
                                .stroke(opt_stroke)
                                .corner_radius(CornerRadius::same(6)),
                            )
                            .clicked()
                        {
                            self.session_options_open = !self.session_options_open;
                        }

                        // Formatted / Raw output toggle
                        let view_label = if self.raw_output_mode {
                            "Raw"
                        } else {
                            "Formatted"
                        };
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(view_label)
                                        .size(11.5)
                                        .color(theme::TEXT_MUTED),
                                )
                                .fill(Color32::TRANSPARENT)
                                .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                                .corner_radius(CornerRadius::same(6)),
                            )
                            .on_hover_text("Toggle formatted markdown vs raw CLI output")
                            .clicked()
                        {
                            self.raw_output_mode = !self.raw_output_mode;
                        }
                    });
                });

                ui.add_space(2.0);
                // Quiet subtitle metadata line
                ui.horizontal_wrapped(|ui| {
                    ui.label(
                        RichText::new(sess.agent.display_name())
                            .size(11.5)
                            .color(theme::TEXT_MUTED),
                    );
                    if let Some(ver) = cli_status.as_ref().and_then(|s| s.version.as_deref()) {
                        ui.label(
                            RichText::new(format!("· {ver}"))
                                .size(11.5)
                                .color(theme::TEXT_FAINT),
                        );
                    }
                    ui.label(
                        RichText::new(format!("· {current_project_name}"))
                            .size(11.5)
                            .color(theme::TEXT_MUTED),
                    );
                    if let Some(dir) = &project_dir {
                        let full = dir.display().to_string();
                        let short = truncate_path_left(&full, 48);
                        ui.label(
                            RichText::new(format!("({short})"))
                                .size(11.0)
                                .color(theme::TEXT_FAINT),
                        )
                        .on_hover_text(full);
                    }
                    if let Some(cli_id) = &sess.cli_session_id {
                        let short_id = resume_id_label(cli_id);
                        ui.label(
                            RichText::new(format!("· resume: {short_id}"))
                                .monospace()
                                .size(11.0)
                                .color(theme::TEXT_FAINT),
                        )
                        .on_hover_text(cli_id);
                    }
                });
            });

        // Optional collapsible Session Options drawer (CLI resume ID, project move, archive)
        if self.session_options_open {
            Frame::new()
                .fill(theme::USER_SURFACE)
                .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                .inner_margin(Margin::symmetric(20, 10))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new("Project:")
                                .size(12.0)
                                .color(theme::TEXT_MUTED),
                        );
                        egui::ComboBox::from_id_salt(("sess_proj", &uid))
                            .selected_text(&current_project_name)
                            .show_ui(ui, |ui| {
                                for (pid, pname) in &projects_list {
                                    if ui
                                        .selectable_value(&mut sess.project_id, pid.clone(), pname)
                                        .changed()
                                    {
                                        settings_changed = true;
                                    }
                                }
                            });

                        if ui.small_button("Change Folder...").clicked() {
                            pick_folder_for_project = Some(sess.project_id.clone());
                        }

                        ui.add_space(12.0);
                        let session_flag = match sess.agent {
                            AgentKind::Agy => "CLI --conversation:",
                            AgentKind::Claude => "CLI --resume:",
                            AgentKind::OpenCode => "CLI --session:",
                        };
                        ui.label(
                            RichText::new(session_flag)
                                .size(12.0)
                                .color(theme::TEXT_MUTED),
                        );
                        let mut cli_id_buf = sess.cli_session_id.clone().unwrap_or_default();
                        if ui
                            .add(
                                egui::TextEdit::singleline(&mut cli_id_buf)
                                    .hint_text("optional session ID")
                                    .desired_width(180.0),
                            )
                            .changed()
                        {
                            sess.cli_session_id = non_empty(&cli_id_buf);
                            settings_changed = true;
                        }
                        if sess.cli_session_id.is_some() && ui.small_button("Clear").clicked() {
                            sess.cli_session_id = None;
                            settings_changed = true;
                        }

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if !is_running
                                && ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Archive Session")
                                                .size(11.5)
                                                .color(theme::ERROR_CORAL),
                                        )
                                        .fill(Color32::TRANSPARENT)
                                        .stroke(Stroke::new(1.0_f32, theme::HAIRLINE)),
                                    )
                                    .clicked()
                            {
                                archive_clicked = true;
                            }
                        });
                    });
                });
        }

        ui.separator();

        // 2. Unified Conversation Timeline
        let show_live_block = rt.is_running()
            || (!rt.live_turn_committed
                && (!rt.stdout.lines.is_empty() || !rt.stderr.lines.is_empty()));

        egui::ScrollArea::vertical()
            .id_salt(("conversation_timeline", &uid))
            .stick_to_bottom(true)
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let avail = ui.available_width();
                let col_w = (avail - 24.0).clamp(240.0, READING_COL_MAX_WIDTH);
                let side_pad = ((avail - col_w) * 0.5).max(12.0);

                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    ui.add_space(side_pad);
                    ui.vertical(|ui| {
                        ui.set_min_width(col_w);
                        ui.set_max_width(col_w);

                        if turns.is_empty() && !show_live_block {
                            rescan_clicked = render_empty_session_canvas(
                                ui,
                                &sess,
                                project_dir.as_deref(),
                                cli_status.as_ref(),
                            );
                        } else {
                            // Chronological order: oldest turn at top -> newest at bottom
                            for turn in turns.iter().rev() {
                                render_committed_turn(
                                    ui,
                                    turn,
                                    self.raw_output_mode,
                                    ctx,
                                    &mut self.copied_feedback,
                                );
                                ui.add_space(16.0);
                            }

                            if show_live_block {
                                render_live_turn(
                                    ui,
                                    &uid,
                                    &sess,
                                    rt,
                                    self.raw_output_mode,
                                    pulse,
                                    ctx,
                                    &mut self.copied_feedback,
                                );
                            }
                        }

                        ui.add_space(18.0);
                    });
                });
            });

        if stop_clicked {
            rt.cancel();
        }
        if archive_clicked {
            self.history.archive_session(&uid);
            self.session_options_open = false;
            self.persist_history();
            return;
        }
        if let Some(pid) = pick_folder_for_project {
            let initial = self
                .history
                .project(&pid)
                .and_then(|p| p.working_dir.clone());
            if let Some(folder) = pick_folder_dialog(initial.as_deref()) {
                if let Some(proj) = self.history.project_mut(&pid) {
                    proj.working_dir = Some(folder);
                    settings_changed = true;
                }
            }
        }
        if settings_changed {
            if let Some(target) = self.history.session_by_uid_mut(&uid) {
                *target = sess;
            }
            self.persist_history();
        }
        if rescan_clicked {
            self.trigger_detection(ctx.clone());
        }
    }

    fn render_no_session_selected(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.centered_and_justified(|ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(ui.available_height() * 0.28);
                ui.label(RichText::new("◆").size(32.0).color(theme::MIST_BLUE));
                ui.add_space(8.0);
                ui.label(
                    RichText::new("Spark Command Deck")
                        .strong()
                        .size(20.0)
                        .color(theme::TEXT_BRIGHT),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new("Select a session on the left, or start a new conversation.")
                        .size(13.5)
                        .color(theme::TEXT_MUTED),
                );
                ui.add_space(16.0);
                ui.horizontal(|ui| {
                    let avail = ui.available_width();
                    ui.add_space(((avail - 240.0) * 0.5).max(0.0));
                    let new_btn = egui::Button::new(
                        RichText::new("+ New Session")
                            .strong()
                            .size(13.0)
                            .color(theme::ON_ACCENT),
                    )
                    .fill(theme::MIST_BLUE)
                    .stroke(Stroke::NONE)
                    .corner_radius(CornerRadius::same(7));
                    if ui.add(new_btn).clicked() {
                        let target = self.default_target_project_id();
                        self.open_new_session_draft(target);
                    }
                    if ui.button("Import CLI Session").clicked() {
                        self.import_window_open = true;
                        self.trigger_external_discovery(ctx.clone());
                    }
                });
            });
        });
    }
}

impl eframe::App for SparkApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.render_frame(ctx);
    }

    /// Closing the window must not leave agent CLIs working unattended. The
    /// runtime is dropped right after this, so the kill cannot be left to a task.
    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        for rt in self.runtime_sessions.values_mut() {
            rt.kill_now();
        }
    }
}

impl SparkApp {
    /// Draw one frame. Separate from `eframe::App::update` so a test can drive it.
    pub fn render_frame(&mut self, ctx: &egui::Context) {
        self.drain_messages();
        self.handle_global_shortcuts(ctx);

        if self.any_running() {
            ctx.request_repaint_after(std::time::Duration::from_millis(60));
        }

        self.render_top_bar(ctx);
        let focused_uid = self.history.selected_session_uid.clone();
        self.render_prompt_bar(ctx, focused_uid.as_deref());
        self.render_sessions_sidebar(ctx);

        egui::CentralPanel::default()
            .frame(
                Frame::new()
                    .fill(theme::OBSIDIAN_BG)
                    .inner_margin(Margin::ZERO),
            )
            .show(ctx, |ui| {
                self.render_focused_session(ui, ctx);
            });

        self.render_new_session_window(ctx);
        self.render_import_window(ctx);
    }
}

struct SidebarRowOutcome {
    select: bool,
    stop: bool,
    archive: bool,
}

fn render_sidebar_session_row(
    ui: &mut egui::Ui,
    sess: &SessionRecord,
    rt: Option<&SessionRuntimeState>,
    turn_count: usize,
    is_selected: bool,
    is_running: bool,
    pulse: f32,
) -> SidebarRowOutcome {
    let mut outcome = SidebarRowOutcome {
        select: false,
        stop: false,
        archive: false,
    };

    let bg_idx = ui.painter().add(egui::Shape::Noop);
    let accent_idx = ui.painter().add(egui::Shape::Noop);

    let row_resp = Frame::new()
        .inner_margin(Margin::symmetric(10, 7))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let action_w = if is_running { 42.0 } else { 22.0 };
                let content_w = (ui.available_width() - action_w).max(60.0);

                let (content_rect, content_resp) =
                    ui.allocate_exact_size(egui::vec2(content_w, 38.0), egui::Sense::click());
                if content_resp
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .clicked()
                {
                    outcome.select = true;
                }

                let mut child_ui = ui.new_child(
                    UiBuilder::new()
                        .max_rect(content_rect)
                        .layout(egui::Layout::top_down(egui::Align::LEFT)),
                );
                child_ui.spacing_mut().item_spacing.y = 2.0;

                child_ui.horizontal(|ui| {
                    let glyph_color =
                        session_status_color(rt.map(|r| &r.run_state), is_selected, pulse);
                    ui.label(
                        RichText::new(sess.agent.glyph())
                            .strong()
                            .size(13.0)
                            .color(glyph_color),
                    );
                    ui.add(
                        egui::Label::new(RichText::new(&sess.title).size(12.5).strong().color(
                            if is_selected {
                                theme::TEXT_BRIGHT
                            } else {
                                theme::TEXT_BODY
                            },
                        ))
                        .truncate()
                        .selectable(false),
                    );
                });

                child_ui.horizontal(|ui| {
                    ui.add_space(18.0);
                    let model_label = if sess.model.is_empty() {
                        "default".to_string()
                    } else {
                        sess.model.clone()
                    };
                    let meta = if let Some(elapsed) = rt.and_then(|r| format_elapsed(&r.run_state))
                    {
                        format!("{model_label} · {elapsed}")
                    } else if turn_count > 0 {
                        let unit = if turn_count == 1 { "turn" } else { "turns" };
                        format!("{model_label} · {turn_count} {unit}")
                    } else {
                        model_label
                    };
                    let meta_color = if is_running {
                        theme::VERDIGRIS_PATINA
                    } else {
                        theme::TEXT_FAINT
                    };
                    ui.add(
                        egui::Label::new(RichText::new(meta).size(10.8).color(meta_color))
                            .truncate()
                            .selectable(false),
                    );
                });

                // Right-aligned action button (completely disjoint rect from content_rect)
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if is_running {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Stop")
                                        .size(10.5)
                                        .strong()
                                        .color(theme::ERROR_CORAL),
                                )
                                .fill(theme::CORAL_SOFT_BG)
                                .stroke(Stroke::NONE)
                                .corner_radius(CornerRadius::same(4)),
                            )
                            .clicked()
                        {
                            outcome.stop = true;
                        }
                    } else if ui
                        .add(
                            egui::Button::new(
                                RichText::new("×").size(12.5).color(theme::TEXT_FAINT),
                            )
                            .frame(false),
                        )
                        .on_hover_text("Archive session")
                        .clicked()
                    {
                        outcome.archive = true;
                    }
                });
            });
        });

    let rect = row_resp.response.rect;
    let hovered = ui.rect_contains_pointer(rect);
    let fill = if is_selected {
        theme::SLATE_CONTROL
    } else if hovered {
        theme::USER_SURFACE
    } else {
        Color32::TRANSPARENT
    };

    if fill != Color32::TRANSPARENT {
        ui.painter().set(
            bg_idx,
            egui::Shape::rect_filled(rect, CornerRadius::same(7), fill),
        );
    }

    if is_selected {
        let bar = egui::Rect::from_min_max(
            egui::pos2(rect.left() + 1.0, rect.top() + 6.0),
            egui::pos2(rect.left() + 4.0, rect.bottom() - 6.0),
        );
        ui.painter().set(
            accent_idx,
            egui::Shape::rect_filled(bar, CornerRadius::same(2), theme::MIST_BLUE),
        );
    }

    outcome
}

fn provider_filter_chip(
    ui: &mut egui::Ui,
    active: bool,
    glyph: &str,
    name: &str,
    count: usize,
    dot_color: Color32,
) -> egui::Response {
    let fill = if active {
        theme::SLATE_HOVER
    } else {
        Color32::TRANSPARENT
    };
    let stroke = if active {
        Stroke::new(1.0_f32, theme::ACCENT_SOFT_BORDER)
    } else {
        Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE)
    };
    let text_col = if active {
        theme::TEXT_BRIGHT
    } else {
        theme::TEXT_FAINT
    };
    let glyph_col = if active {
        theme::MIST_BLUE
    } else {
        theme::TEXT_FAINT
    };

    let resp = Frame::new()
        .fill(fill)
        .stroke(stroke)
        .corner_radius(CornerRadius::same(7))
        .inner_margin(Margin::symmetric(10, 5))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                let (dot_rect, _) =
                    ui.allocate_exact_size(egui::vec2(6.0, 6.0), egui::Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 3.0, dot_color);
                ui.label(RichText::new(glyph).strong().size(12.0).color(glyph_col));
                ui.label(RichText::new(name).size(12.0).color(text_col));
                ui.label(
                    RichText::new(format!("{count}"))
                        .size(11.0)
                        .color(theme::TEXT_FAINT),
                );
            });
        })
        .response
        .interact(egui::Sense::click())
        .on_hover_cursor(CursorIcon::PointingHand);

    resp
}

fn render_empty_session_canvas(
    ui: &mut egui::Ui,
    sess: &SessionRecord,
    project_dir: Option<&std::path::Path>,
    cli_status: Option<&CliStatus>,
) -> bool {
    let mut rescan = false;
    ui.add_space(48.0);
    ui.vertical_centered(|ui| {
        ui.label(
            RichText::new(sess.agent.glyph())
                .size(30.0)
                .color(theme::MIST_BLUE),
        );
        ui.add_space(8.0);
        ui.label(
            RichText::new(format!("{} Workspace", sess.agent.display_name()))
                .strong()
                .size(18.0)
                .color(theme::TEXT_BRIGHT),
        );
        ui.add_space(4.0);

        let model_desc = if sess.model.is_empty() {
            "CLI default model"
        } else {
            sess.model.as_str()
        };
        let dir_desc = project_dir
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "Current directory".to_string());
        ui.label(
            RichText::new(format!("{model_desc}  ·  {dir_desc}"))
                .size(12.5)
                .color(theme::TEXT_MUTED),
        );

        if let Some(status) = cli_status {
            if !status.available {
                ui.add_space(16.0);
                Frame::new()
                    .fill(theme::CORAL_SOFT_BG)
                    .stroke(Stroke::new(1.0_f32, theme::ERROR_CORAL))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        let msg = status.error.as_deref().unwrap_or("CLI binary not found.");
                        ui.label(
                            RichText::new(format!(
                                "{} CLI unavailable: {msg}",
                                sess.agent.display_name()
                            ))
                            .size(12.5)
                            .color(theme::ERROR_CORAL),
                        );
                        ui.add_space(6.0);
                        if ui.button("Re-scan CLIs").clicked() {
                            rescan = true;
                        }
                    });
            }
        }

        ui.add_space(20.0);
        ui.label(
            RichText::new(
                "Ctrl+Enter to transmit  ·  Ctrl+N for new session  ·  Click title above to rename",
            )
            .size(11.5)
            .color(theme::TEXT_FAINT),
        );
    });
    rescan
}

fn render_committed_turn(
    ui: &mut egui::Ui,
    turn: &TurnRecord,
    raw_mode: bool,
    ctx: &egui::Context,
    copied_feedback: &mut Option<(String, Instant)>,
) {
    // User Prompt Bubble
    Frame::new()
        .fill(theme::USER_SURFACE)
        .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(14, 10))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("You")
                        .strong()
                        .size(11.5)
                        .color(theme::MIST_BLUE),
                );
                if let Some(model) = &turn.model {
                    ui.label(
                        RichText::new(format!("· {model}"))
                            .size(11.0)
                            .color(theme::TEXT_FAINT),
                    );
                }
            });
            ui.add_space(4.0);
            ui.add(
                egui::Label::new(
                    RichText::new(&turn.prompt)
                        .size(13.5)
                        .color(theme::TEXT_BRIGHT),
                )
                .selectable(true)
                .wrap(),
            );
        });

    ui.add_space(10.0);

    // Agent Response Block (clean typography directly on lacquer ground)
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(turn.agent.glyph())
                .strong()
                .size(13.5)
                .color(if turn.success {
                    theme::VERDIGRIS_PATINA
                } else {
                    theme::ERROR_CORAL
                }),
        );
        ui.label(
            RichText::new(turn.agent.display_name())
                .strong()
                .size(12.5)
                .color(theme::TEXT_BRIGHT),
        );
        let duration_label = format!("{:.2}s", turn.duration_ms as f32 / 1000.0);
        let status_txt = if turn.success {
            format!("· {duration_label}")
        } else if let Some(code) = turn.exit_code {
            format!("· Failed (exit {code}) · {duration_label}")
        } else {
            format!("· Failed · {duration_label}")
        };
        let status_col = if turn.success {
            theme::TEXT_FAINT
        } else {
            theme::ERROR_CORAL
        };
        ui.label(RichText::new(status_txt).size(11.5).color(status_col));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if !turn.output.is_empty() {
                let key = format!("turn_copy_{}", turn.id);
                let copied = copied_feedback
                    .as_ref()
                    .is_some_and(|(k, at)| k == &key && at.elapsed().as_secs_f32() < 1.5);
                let label = if copied { "Copied" } else { "Copy" };
                let color = if copied {
                    theme::VERDIGRIS_PATINA
                } else {
                    theme::TEXT_FAINT
                };
                if ui
                    .add(
                        egui::Button::new(RichText::new(label).size(11.0).color(color))
                            .frame(false),
                    )
                    .on_hover_cursor(CursorIcon::PointingHand)
                    .clicked()
                {
                    ctx.copy_text(turn.output.clone());
                    *copied_feedback = Some((key, Instant::now()));
                }
            }
        });
    });

    ui.add_space(4.0);

    if turn.output.trim().is_empty() {
        ui.label(
            RichText::new("(no standard output)")
                .size(12.0)
                .color(theme::TEXT_FAINT),
        );
    } else if raw_mode {
        render_raw_output_block(ui, &turn.output, theme::TEXT_BODY);
    } else {
        markdown::render_markdown(ui, &turn.output, ("turn_md", turn.id), ctx, copied_feedback);
    }

    if !turn.stderr.trim().is_empty() {
        ui.add_space(6.0);
        let err_lines = turn.stderr.lines().count();
        egui::CollapsingHeader::new(
            RichText::new(format!("Stderr ({err_lines} lines)"))
                .size(11.5)
                .color(theme::ERROR_CORAL),
        )
        .id_salt(("turn_stderr", turn.id))
        .default_open(!turn.success)
        .show(ui, |ui| {
            render_raw_output_block(ui, &turn.stderr, theme::ERROR_CORAL);
        });
    }

    ui.add_space(8.0);
    ui.separator();
}

#[allow(clippy::too_many_arguments)]
fn render_live_turn(
    ui: &mut egui::Ui,
    uid: &str,
    sess: &SessionRecord,
    rt: &SessionRuntimeState,
    raw_mode: bool,
    pulse: f32,
    ctx: &egui::Context,
    copied_feedback: &mut Option<(String, Instant)>,
) {
    if let RunState::Running { prompt, .. } = &rt.run_state {
        Frame::new()
            .fill(theme::USER_SURFACE)
            .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::symmetric(14, 10))
            .show(ui, |ui| {
                ui.label(
                    RichText::new("You")
                        .strong()
                        .size(11.5)
                        .color(theme::MIST_BLUE),
                );
                ui.add_space(4.0);
                ui.add(
                    egui::Label::new(RichText::new(prompt).size(13.5).color(theme::TEXT_BRIGHT))
                        .selectable(true)
                        .wrap(),
                );
            });
        ui.add_space(10.0);
    }

    let is_running = rt.is_running();
    let header_col = if is_running {
        pulse_verdigris(pulse)
    } else {
        theme::ERROR_CORAL
    };

    ui.horizontal(|ui| {
        ui.label(
            RichText::new(sess.agent.glyph())
                .strong()
                .size(13.5)
                .color(header_col),
        );
        ui.label(
            RichText::new(sess.agent.display_name())
                .strong()
                .size(12.5)
                .color(theme::TEXT_BRIGHT),
        );
        let status_label = if is_running {
            let elapsed = format_elapsed(&rt.run_state).unwrap_or_default();
            format!("· Responding… {elapsed}")
        } else {
            "· Error".to_string()
        };
        ui.label(RichText::new(status_label).size(11.5).color(header_col));

        if !rt.stdout.lines.is_empty() {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let key = format!("live_copy_{uid}");
                let copied = copied_feedback
                    .as_ref()
                    .is_some_and(|(k, at)| k == &key && at.elapsed().as_secs_f32() < 1.5);
                let label = if copied { "Copied" } else { "Copy" };
                if ui
                    .add(
                        egui::Button::new(RichText::new(label).size(11.0).color(if copied {
                            theme::VERDIGRIS_PATINA
                        } else {
                            theme::TEXT_FAINT
                        }))
                        .frame(false),
                    )
                    .clicked()
                {
                    ctx.copy_text(rt.stdout.joined());
                    *copied_feedback = Some((key, Instant::now()));
                }
            });
        }
    });

    ui.add_space(6.0);

    if rt.stdout.truncated || rt.stderr.truncated {
        ui.label(
            RichText::new("[older stream lines truncated to preserve memory]")
                .monospace()
                .size(11.0)
                .color(theme::TEXT_FAINT),
        );
    }

    if rt.stdout.lines.is_empty() && rt.stderr.lines.is_empty() {
        let dots = match ((pulse * 3.0).round() as usize) % 3 {
            0 => ".",
            1 => "..",
            _ => "...",
        };
        ui.label(
            RichText::new(format!("Thinking{dots}"))
                .size(13.0)
                .color(theme::TEXT_MUTED),
        );
    } else if !rt.stdout.lines.is_empty() {
        let joined = rt.stdout.joined();
        if raw_mode {
            render_raw_output_block(ui, &joined, theme::TEXT_BODY);
        } else {
            markdown::render_markdown(ui, &joined, ("live_md", uid), ctx, copied_feedback);
        }
    }

    if !rt.stderr.lines.is_empty() {
        ui.add_space(6.0);
        render_raw_output_block(ui, &rt.stderr.joined(), theme::ERROR_CORAL);
    }
}

fn render_raw_output_block(ui: &mut egui::Ui, text: &str, color: Color32) {
    Frame::new()
        .fill(theme::OBSIDIAN_DEEP)
        .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::symmetric(12, 8))
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(RichText::new(text).monospace().size(12.2).color(color))
                    .selectable(true)
                    .wrap(),
            );
        });
}

fn render_model_selector_controls(
    ui: &mut egui::Ui,
    id_salt: impl std::hash::Hash,
    options: &[ModelOption],
    current_model: &mut String,
    custom_open: &mut bool,
    combo_width: f32,
) -> bool {
    let mut changed = false;
    let selected_label = if current_model.is_empty() {
        "CLI default".to_string()
    } else {
        options
            .iter()
            .find(|opt| opt.id == *current_model)
            .map(|opt| opt.label.clone())
            .unwrap_or_else(|| current_model.clone())
    };

    egui::ComboBox::from_id_salt(id_salt)
        .selected_text(selected_label)
        .width(combo_width)
        .show_ui(ui, |ui| {
            if ui
                .selectable_label(current_model.is_empty() && !*custom_open, "CLI default")
                .clicked()
            {
                current_model.clear();
                *custom_open = false;
                changed = true;
            }
            for opt in options {
                let is_sel = current_model == &opt.id && !*custom_open;
                if ui.selectable_label(is_sel, &opt.label).clicked() {
                    *current_model = opt.id.clone();
                    *custom_open = false;
                    changed = true;
                }
            }
            if ui.selectable_label(*custom_open, "Custom...").clicked() {
                *custom_open = true;
            }
        });

    let is_unlisted =
        !current_model.is_empty() && !options.iter().any(|opt| opt.id == *current_model);
    if (*custom_open || is_unlisted)
        && ui
            .add(
                egui::TextEdit::singleline(current_model)
                    .hint_text("custom model ID")
                    .desired_width(145.0),
            )
            .changed()
    {
        changed = true;
    }

    changed
}

fn pick_folder_dialog(initial_dir: Option<&std::path::Path>) -> Option<PathBuf> {
    let mut dialog = rfd::FileDialog::new();
    if let Some(dir) = initial_dir.filter(|d| d.is_dir()) {
        dialog = dialog.set_directory(dir);
    }
    dialog.pick_folder()
}

fn non_empty(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// The ID to create a new CLI session under, when `agent` lets Spark choose one
/// and the session has none to resume.
///
/// Without this a session Spark created never had a CLI session ID, so every
/// prompt after the first started a new conversation. Claude Code takes the ID
/// up front (`--session-id <uuid>`); `agy` and `opencode` have no such flag.
fn new_cli_session_id_for(agent: AgentKind, existing: Option<&str>) -> Option<String> {
    (agent == AgentKind::Claude && existing.is_none()).then(|| uuid::Uuid::new_v4().to_string())
}

/// What a finished turn means for the session's CLI session ID.
#[derive(Debug, PartialEq, Eq)]
enum SessionIdOutcome {
    /// The turn created this session: keep the ID so the next prompt resumes it.
    Keep(String),
    /// The CLI no longer has this ID: drop it so the next prompt starts fresh.
    Forget(String),
    Unchanged,
}

/// The CLI's message when `--resume` names a session it does not have (checked
/// against claude 2.1.331: `No conversation found with session ID: <id>`, exit 1).
const CLAUDE_UNKNOWN_SESSION: &str = "No conversation found with session ID";

/// Decide what to do with a session's CLI ID once a turn has finished.
///
/// `created` is the ID a first turn was told to create. A turn can create the
/// session and still exit non-zero (the user pressed Stop, an API error), so a
/// failed turn keeps the ID when the CLI's transcript for it exists; otherwise
/// the conversation so far would be dropped. `resumed` is the ID a later turn
/// resumed. Claude Code ties a session to its working directory and prunes old
/// ones, so a stored ID can stop resolving; without `Forget` every prompt after
/// that fails the same way until the user clears the ID by hand.
fn session_id_outcome(
    agent: AgentKind,
    success: bool,
    created: Option<String>,
    resumed: Option<String>,
    stderr: &str,
    transcript_exists: impl FnOnce(&str) -> bool,
) -> SessionIdOutcome {
    if let Some(id) = created {
        return if success || transcript_exists(&id) {
            SessionIdOutcome::Keep(id)
        } else {
            SessionIdOutcome::Unchanged
        };
    }
    match resumed {
        Some(id)
            if agent == AgentKind::Claude
                && !success
                && stderr.contains(CLAUDE_UNKNOWN_SESSION) =>
        {
            SessionIdOutcome::Forget(id)
        }
        _ => SessionIdOutcome::Unchanged,
    }
}

/// Shorten a CLI session ID for the session header (14 characters, then `...`).
fn resume_id_label(cli_id: &str) -> String {
    let head = char_prefix(cli_id, 14);
    if head.len() < cli_id.len() {
        format!("{head}...")
    } else {
        cli_id.to_string()
    }
}

fn compact_session_title(prompt: &str) -> String {
    let first = prompt.lines().next().unwrap_or("").trim();
    let mut out = String::new();
    for ch in first.chars().take(36) {
        out.push(ch);
    }
    if first.chars().count() > 36 {
        out.push_str("...");
    }
    if out.is_empty() {
        "Session".to_string()
    } else {
        out
    }
}

fn truncate_path_left(path: &str, max_chars: usize) -> String {
    let count = path.chars().count();
    if count <= max_chars || max_chars <= 4 {
        return path.to_string();
    }
    let skip = count - (max_chars - 3);
    let tail: String = path.chars().skip(skip).collect();
    format!("...{tail}")
}

fn running_pulse(ctx: &egui::Context) -> f32 {
    let t = ctx.input(|i| i.time);
    ((t * 3.6).sin() * 0.5 + 0.5) as f32
}

fn pulse_verdigris(pulse: f32) -> Color32 {
    let r = (0x38 as f32 + pulse * 0x22 as f32) as u8;
    let g = (0x92 as f32 + pulse * 0x2E as f32) as u8;
    let b = (0x88 as f32 + pulse * 0x2A as f32) as u8;
    Color32::from_rgb(r, g, b)
}

fn session_status_color(state: Option<&RunState>, is_selected: bool, pulse: f32) -> Color32 {
    match state {
        Some(RunState::Running { .. }) => pulse_verdigris(pulse),
        Some(RunState::Done { .. }) => theme::VERDIGRIS_PATINA,
        Some(RunState::Failed { .. }) => theme::ERROR_CORAL,
        _ if is_selected => theme::MIST_BLUE,
        _ => theme::TEXT_MUTED,
    }
}

fn render_status_pill(
    ui: &mut egui::Ui,
    run_state: &RunState,
    cli_status: Option<&CliStatus>,
    pulse: f32,
) {
    let (label, fg, bg, border) = match run_state {
        RunState::Running { .. } => {
            let elapsed = format_elapsed(run_state).unwrap_or_default();
            (
                format!("RUNNING {elapsed}"),
                pulse_verdigris(pulse),
                theme::VERDIGRIS_SOFT_BG,
                theme::VERDIGRIS_PATINA,
            )
        }
        RunState::Done { .. } => {
            let elapsed = format_elapsed(run_state).unwrap_or_default();
            (
                format!("DONE {elapsed}"),
                theme::VERDIGRIS_PATINA,
                theme::VERDIGRIS_SOFT_BG,
                theme::HAIRLINE,
            )
        }
        RunState::Failed { .. } => (
            "FAILED".to_string(),
            theme::ERROR_CORAL,
            theme::CORAL_SOFT_BG,
            theme::ERROR_CORAL,
        ),
        RunState::Idle => match cli_status {
            None => (
                "PROBING".to_string(),
                theme::TEXT_FAINT,
                theme::SLATE_CONTROL,
                theme::HAIRLINE_SUBTLE,
            ),
            Some(s) if s.available => (
                "READY".to_string(),
                theme::TEXT_MUTED,
                theme::SLATE_CONTROL,
                theme::HAIRLINE,
            ),
            Some(_) => (
                "UNAVAILABLE".to_string(),
                theme::ERROR_CORAL,
                theme::CORAL_SOFT_BG,
                theme::ERROR_CORAL,
            ),
        },
    };

    Frame::new()
        .fill(bg)
        .stroke(Stroke::new(1.0_f32, border))
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::symmetric(8, 2))
        .show(ui, |ui| {
            ui.label(RichText::new(label).strong().size(10.5).color(fg));
        });
}

fn format_elapsed(state: &RunState) -> Option<String> {
    match state {
        RunState::Running { started_at, .. } => {
            Some(format!("{:.1}s", started_at.elapsed().as_secs_f32()))
        }
        RunState::Done { duration_ms } | RunState::Failed { duration_ms, .. } => {
            Some(format!("{:.2}s", *duration_ms as f32 / 1000.0))
        }
        RunState::Idle => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_buffer_enforces_line_and_byte_caps() {
        let mut buf = OutputBuffer::default();
        for i in 0..(MAX_OUTPUT_LINES + 50) {
            buf.push_line(format!("line-{i}"));
        }
        assert!(buf.truncated);
        assert!(buf.lines.len() <= MAX_OUTPUT_LINES);
        assert!(buf.total_bytes <= MAX_OUTPUT_BYTES);
        assert_eq!(
            buf.lines.last().unwrap(),
            &format!("line-{}", MAX_OUTPUT_LINES + 49)
        );
    }

    #[test]
    fn resume_id_label_shortens_without_splitting_a_character() {
        assert_eq!(resume_id_label("short-id"), "short-id");
        assert_eq!(resume_id_label("0123456789abcdef"), "0123456789abcd...");
        // The session ID field is free text. Byte 14 of this one is inside an "\u{e9}".
        let pasted = format!("a{}", "\u{e9}".repeat(14));
        assert_eq!(
            resume_id_label(&pasted),
            format!("a{}...", "\u{e9}".repeat(13))
        );
    }

    #[test]
    fn second_prompt_in_a_new_claude_session_resumes_the_first() {
        let mut store = HistoryStore::default();
        let proj_id = store.create_project("Spark", None).id.clone();
        let uid = store
            .create_session(&proj_id, AgentKind::Claude, "sonnet", "")
            .uid
            .clone();

        // First prompt: nothing to resume, so Spark names the session.
        let created = new_cli_session_id_for(AgentKind::Claude, None)
            .expect("a new Claude session must be given an ID");
        assert!(uuid::Uuid::parse_str(&created).is_ok(), "{created}");
        let mut first = RunRequest::new(AgentKind::Claude, "remember papaya");
        first.new_session_id = Some(created.clone());
        assert!(runner::build_args(&first).contains(&"--session-id".to_string()));

        // The turn succeeds and the ID is kept on the session.
        assert!(store.adopt_cli_session_id(&uid, created.clone()));

        // Second prompt: built the way `dispatch_to_session` builds it.
        let sess = store.session_by_uid(&uid).unwrap().clone();
        let mut second = RunRequest::new(AgentKind::Claude, "which word?");
        second.session_id = sess.cli_session_id;
        second.new_session_id =
            new_cli_session_id_for(AgentKind::Claude, second.session_id.as_deref());
        let args = runner::build_args(&second);
        let at = args
            .iter()
            .position(|a| a == "--resume")
            .expect("the second prompt must resume the first");
        assert_eq!(args[at + 1], created);
    }

    #[test]
    fn a_stopped_first_turn_keeps_the_session_it_created() {
        let id = || Some("new-id".to_string());
        // Succeeded: kept without looking.
        assert_eq!(
            session_id_outcome(AgentKind::Claude, true, id(), None, "", |_| false),
            SessionIdOutcome::Keep("new-id".to_string())
        );
        // Stopped or failed after the CLI wrote its transcript: still kept.
        assert_eq!(
            session_id_outcome(AgentKind::Claude, false, id(), None, "", |_| true),
            SessionIdOutcome::Keep("new-id".to_string())
        );
        // Failed before the CLI created anything: nothing to resume.
        assert_eq!(
            session_id_outcome(AgentKind::Claude, false, id(), None, "", |_| false),
            SessionIdOutcome::Unchanged
        );
    }

    #[test]
    fn a_resume_the_cli_no_longer_knows_is_forgotten_and_nothing_else_is() {
        let resumed = || Some("old-id".to_string());
        let gone = "No conversation found with session ID: old-id";
        let never = |_: &str| -> bool { panic!("only a created ID is looked up") };

        assert_eq!(
            session_id_outcome(AgentKind::Claude, false, None, resumed(), gone, never),
            SessionIdOutcome::Forget("old-id".to_string())
        );
        // Any other failure leaves the conversation resumable.
        assert_eq!(
            session_id_outcome(
                AgentKind::Claude,
                false,
                None,
                resumed(),
                "API Error: 529",
                never
            ),
            SessionIdOutcome::Unchanged
        );
        // The text appearing in a turn that worked is just output.
        assert_eq!(
            session_id_outcome(AgentKind::Claude, true, None, resumed(), gone, never),
            SessionIdOutcome::Unchanged
        );
        // Imported agy and opencode sessions are never touched.
        assert_eq!(
            session_id_outcome(AgentKind::OpenCode, false, None, resumed(), gone, never),
            SessionIdOutcome::Unchanged
        );
    }

    #[test]
    fn agy_and_opencode_sessions_are_not_given_an_id() {
        assert_eq!(new_cli_session_id_for(AgentKind::Agy, None), None);
        assert_eq!(new_cli_session_id_for(AgentKind::OpenCode, None), None);
        assert_eq!(new_cli_session_id_for(AgentKind::Claude, Some("x")), None);
    }

    /// The whole UI, drawn headlessly for a few frames with a session selected.
    /// egui's `Context` is one lock: asking it a question from inside one of its
    /// own `data_mut`/`memory_mut` closures deadlocks the UI thread on the first
    /// frame, the window never appears, and nothing else notices.
    #[test]
    fn the_first_frames_render_without_deadlocking() {
        let dir = std::env::temp_dir().join(format!(
            "spark_test_first_frame_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        let history_path = dir.join("history.json");

        let (done_tx, done_rx) = std_mpsc::channel();
        std::thread::spawn(move || {
            let ctx = egui::Context::default();
            let mut app = SparkApp::with_history_path(ctx.clone(), Some(history_path));
            assert!(
                app.history.selected_session_uid.is_some(),
                "a fresh install starts with a session selected"
            );
            for _ in 0..3 {
                let _ = ctx.run(egui::RawInput::default(), |ctx| app.render_frame(ctx));
            }
            let _ = done_tx.send(());
        });

        let rendered = done_rx.recv_timeout(std::time::Duration::from_secs(30));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            rendered.is_ok(),
            "drawing the UI hung or panicked: the app would never show its window"
        );
    }

    #[test]
    fn non_empty_trims_whitespace() {
        assert_eq!(non_empty("   "), None);
        assert_eq!(non_empty("  sonnet  "), Some("sonnet".to_string()));
    }

    #[test]
    fn top_bar_ai_filters_filter_sessions_accurately() {
        let mut filters = HashMap::new();
        filters.insert(AgentKind::Agy, true);
        filters.insert(AgentKind::Claude, false);
        filters.insert(AgentKind::OpenCode, true);

        let agy_sess = SessionRecord {
            uid: "s1".to_string(),
            project_id: "p1".to_string(),
            title: "Agy".to_string(),
            agent: AgentKind::Agy,
            cli_session_id: None,
            model: "gemini-4-hydrogen-high".to_string(),
            mode: String::new(),
            effort: String::new(),
            imported: false,
            archived: false,
            created_at_epoch_secs: 0,
            updated_at_epoch_secs: 0,
        };
        let mut claude_sess = agy_sess.clone();
        claude_sess.uid = "s2".to_string();
        claude_sess.agent = AgentKind::Claude;

        let mut archived_agy = agy_sess.clone();
        archived_agy.uid = "s3".to_string();
        archived_agy.archived = true;

        assert!(session_matches_filters(&agy_sess, &filters));
        assert!(!session_matches_filters(&claude_sess, &filters));
        assert!(!session_matches_filters(&archived_agy, &filters));

        assert!(session_matches_filters_and_query(
            &agy_sess, &filters, "hydrogen"
        ));
        assert!(!session_matches_filters_and_query(
            &agy_sess, &filters, "opus"
        ));
    }

    #[test]
    fn truncate_path_left_preserves_tail_segments() {
        let p = r"D:\Projects\very-long-workspace-folder\spark";
        let short = truncate_path_left(p, 20);
        assert!(short.starts_with("..."));
        assert!(short.ends_with("spark"));
        assert_eq!(short.chars().count(), 20);
    }
}
