//! Read-only discovery of external CLI sessions (`opencode`, `claude`, `agy`).

use crate::cli::detectors::CliStatus;
use crate::cli::{char_prefix, AgentKind};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Duration, UNIX_EPOCH};
use tokio::process::Command;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const SESSION_QUERY_TIMEOUT: Duration = Duration::from_secs(6);
const MAX_DISCOVERED_PER_AGENT: usize = 30;

/// External CLI session discovered on disk or via CLI query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredSession {
    pub agent: AgentKind,
    pub cli_session_id: String,
    pub title: String,
    pub working_dir: Option<PathBuf>,
    pub updated_epoch_secs: u64,
}

/// Resolve the user's home directory (`USERPROFILE` on Windows, `HOME` on Unix).
pub fn user_home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

/// Parse JSON output from `opencode session list --format json`.
pub fn parse_opencode_sessions_json(raw: &str) -> Vec<DiscoveredSession> {
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(raw.trim()) else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for item in items {
        let Some(id) = item
            .get("id")
            .or_else(|| item.get("sessionId"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            continue;
        };

        let title = item
            .get("title")
            .or_else(|| item.get("slug"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(id)
            .to_string();

        let working_dir = item
            .get("directory")
            .or_else(|| item.get("cwd"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let updated_epoch_secs = item
            .get("time")
            .and_then(|t| t.get("updated"))
            .or_else(|| item.get("updated"))
            .and_then(Value::as_u64)
            .map(normalize_epoch_secs)
            .unwrap_or(0);

        out.push(DiscoveredSession {
            agent: AgentKind::OpenCode,
            cli_session_id: id.to_string(),
            title,
            working_dir,
            updated_epoch_secs,
        });
    }
    out
}

/// Extract `sessionId`, `cwd`, and preview title from the initial lines of a Claude `.jsonl` log.
pub fn parse_claude_jsonl_lines<'a>(
    lines: impl Iterator<Item = &'a str>,
) -> Option<(String, Option<PathBuf>, Option<String>)> {
    let mut session_id: Option<String> = None;
    let mut cwd: Option<PathBuf> = None;
    let mut title: Option<String> = None;

    for raw in lines.take(25) {
        let Ok(val) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        if session_id.is_none() {
            session_id = val
                .get("sessionId")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string);
        }
        if cwd.is_none() {
            cwd = val
                .get("cwd")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from);
        }
        if title.is_none() {
            let role = val.get("type").and_then(Value::as_str).unwrap_or("");
            if role == "user" || role == "custom-title" {
                let candidate = val
                    .get("customTitle")
                    .or_else(|| val.get("content"))
                    .or_else(|| val.get("message").and_then(|m| m.get("content")))
                    .and_then(Value::as_str)
                    .map(compact_title);
                if let Some(t) = candidate.filter(|s| !s.is_empty()) {
                    title = Some(t);
                }
            }
        }
        if session_id.is_some() && cwd.is_some() && title.is_some() {
            break;
        }
    }

    session_id.map(|id| (id, cwd, title))
}

/// Scan `~/.claude/projects/**/*.jsonl` for recent resumable Claude Code sessions.
pub fn discover_claude_sessions(home: &Path, limit: usize) -> Vec<DiscoveredSession> {
    let projects_root = home.join(".claude").join("projects");
    if !projects_root.is_dir() {
        return Vec::new();
    }

    let mut files: Vec<(u64, PathBuf)> = Vec::new();
    let Ok(project_dirs) = std::fs::read_dir(&projects_root) else {
        return Vec::new();
    };
    for proj in project_dirs.flatten() {
        let proj_path = proj.path();
        if !proj_path.is_dir() {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&proj_path) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push((file_mtime_secs(&path), path));
            }
        }
    }

    files.sort_by_key(|item| std::cmp::Reverse(item.0));
    files.truncate(limit);

    let mut sessions = Vec::new();
    for (mtime, path) in files {
        let fallback_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        if fallback_id.is_empty() {
            continue;
        }

        let mut head_lines = Vec::new();
        if let Ok(file) = File::open(&path) {
            let reader = BufReader::new(file);
            for line in reader.lines().take(25).flatten() {
                head_lines.push(line);
            }
        }

        let parsed = parse_claude_jsonl_lines(head_lines.iter().map(String::as_str));
        let (cli_session_id, working_dir, parsed_title) = match parsed {
            Some((id, cwd, title)) => (id, cwd, title),
            None => (fallback_id.clone(), None, None),
        };

        let title = parsed_title.unwrap_or_else(|| format!("Claude {}", short_id(&cli_session_id)));
        sessions.push(DiscoveredSession {
            agent: AgentKind::Claude,
            cli_session_id,
            title,
            working_dir,
            updated_epoch_secs: mtime,
        });
    }
    sessions
}

/// Scan `~/.gemini/antigravity/conversations/*.pb` for resumable Antigravity conversations.
pub fn discover_agy_sessions(home: &Path, limit: usize) -> Vec<DiscoveredSession> {
    let conv_dir = home
        .join(".gemini")
        .join("antigravity")
        .join("conversations");
    if !conv_dir.is_dir() {
        return Vec::new();
    }

    let Ok(entries) = std::fs::read_dir(&conv_dir) else {
        return Vec::new();
    };

    let mut files: Vec<(u64, String)> = entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path();
            if path.extension().is_none_or(|ext| ext != "pb") {
                return None;
            }
            let stem = path.file_stem()?.to_str()?.trim().to_string();
            if stem.is_empty() {
                return None;
            }
            Some((file_mtime_secs(&path), stem))
        })
        .collect();

    files.sort_by_key(|item| std::cmp::Reverse(item.0));
    files.truncate(limit);

    let brain_root = home.join(".gemini").join("antigravity").join("brain");
    files
        .into_iter()
        .map(|(mtime, uuid)| {
            let title = read_agy_brain_title(&brain_root, &uuid)
                .unwrap_or_else(|| format!("Agy {}", short_id(&uuid)));
            DiscoveredSession {
                agent: AgentKind::Agy,
                cli_session_id: uuid,
                title,
                working_dir: None,
                updated_epoch_secs: mtime,
            }
        })
        .collect()
}

/// Query `opencode session list --format json` inside `working_dir` (if provided).
pub async fn discover_opencode_sessions(
    binary_path: Option<&Path>,
    working_dir: Option<&Path>,
    limit: usize,
) -> Vec<DiscoveredSession> {
    let Some(bin) = binary_path else {
        return Vec::new();
    };

    let mut cmd = Command::new(bin);
    cmd.args(["session", "list", "--format", "json", "-n"])
        .arg(limit.to_string())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(dir) = working_dir.filter(|d| d.is_dir()) {
        cmd.current_dir(dir);
    }

    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let Ok(Ok(output)) = tokio::time::timeout(SESSION_QUERY_TIMEOUT, cmd.output()).await else {
        return Vec::new();
    };

    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    parse_opencode_sessions_json(&stdout)
}

/// Discover resumable external sessions across `agy`, `claude`, and `opencode`.
pub async fn discover_all_sessions(
    statuses: &[CliStatus],
    working_dirs: &[PathBuf],
) -> Vec<DiscoveredSession> {
    let mut all = Vec::new();

    if let Some(home) = user_home_dir() {
        all.extend(discover_agy_sessions(&home, MAX_DISCOVERED_PER_AGENT));
        all.extend(discover_claude_sessions(&home, MAX_DISCOVERED_PER_AGENT));
    }

    let opencode_bin = statuses
        .iter()
        .find(|s| s.agent == AgentKind::OpenCode && s.available)
        .and_then(|s| s.binary_path.as_deref());

    if let Some(bin) = opencode_bin {
        if working_dirs.is_empty() {
            all.extend(discover_opencode_sessions(Some(bin), None, MAX_DISCOVERED_PER_AGENT).await);
        } else {
            for dir in working_dirs {
                let found =
                    discover_opencode_sessions(Some(bin), Some(dir), MAX_DISCOVERED_PER_AGENT)
                        .await;
                for item in found {
                    if !all
                        .iter()
                        .any(|x| x.agent == item.agent && x.cli_session_id == item.cli_session_id)
                    {
                        all.push(item);
                    }
                }
            }
        }
    }

    all.sort_by_key(|item| std::cmp::Reverse(item.updated_epoch_secs));
    all
}

fn read_agy_brain_title(brain_root: &Path, uuid: &str) -> Option<String> {
    let task_md = brain_root.join(uuid).join("task.md");
    let raw = std::fs::read_to_string(task_md).ok()?;
    raw.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(|line| compact_title(line.trim_start_matches('#').trim()))
        .filter(|s| !s.is_empty())
}

fn file_mtime_secs(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn normalize_epoch_secs(raw: u64) -> u64 {
    if raw > 10_000_000_000 {
        raw / 1000
    } else {
        raw
    }
}

fn short_id(id: &str) -> &str {
    char_prefix(id, 8)
}

fn compact_title(s: &str) -> String {
    let first_line = s.lines().next().unwrap_or("").trim();
    let mut out = String::new();
    for ch in first_line.chars().take(48) {
        out.push(ch);
    }
    if first_line.chars().count() > 48 {
        out.push_str("...");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_home(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "spark_sess_test_{name}_{}_{nanos}",
            std::process::id()
        ))
    }

    #[test]
    fn parse_opencode_sessions_json_extracts_fields() {
        let raw = r#"[
            {
                "id": "ses_abc123",
                "title": "Refactor sidebar",
                "directory": "D:\\Projects\\spark",
                "time": { "updated": 1731510000000 }
            }
        ]"#;
        let parsed = parse_opencode_sessions_json(raw);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].agent, AgentKind::OpenCode);
        assert_eq!(parsed[0].cli_session_id, "ses_abc123");
        assert_eq!(parsed[0].title, "Refactor sidebar");
        assert_eq!(
            parsed[0].working_dir,
            Some(PathBuf::from(r"D:\Projects\spark"))
        );
        assert_eq!(parsed[0].updated_epoch_secs, 1_731_510_000);
    }

    #[test]
    fn discover_claude_and_agy_sessions_from_temp_home() {
        let home = temp_home("claude_agy");
        let claude_proj = home
            .join(".claude")
            .join("projects")
            .join("D--Projects-spark");
        std::fs::create_dir_all(&claude_proj).unwrap();
        let jsonl = claude_proj.join("11111111-2222-3333-4444-555555555555.jsonl");
        std::fs::write(
            &jsonl,
            "{\"type\":\"user\",\"cwd\":\"D:\\\\Projects\\\\spark\",\"sessionId\":\"11111111-2222-3333-4444-555555555555\",\"content\":\"Fix model selector\"}\n",
        )
        .unwrap();

        let agy_conv = home
            .join(".gemini")
            .join("antigravity")
            .join("conversations");
        std::fs::create_dir_all(&agy_conv).unwrap();
        std::fs::write(agy_conv.join("aaaa-bbbb.pb"), b"pb").unwrap();
        let agy_brain = home
            .join(".gemini")
            .join("antigravity")
            .join("brain")
            .join("aaaa-bbbb");
        std::fs::create_dir_all(&agy_brain).unwrap();
        std::fs::write(agy_brain.join("task.md"), "# Implement v0.2 Sidebar\n").unwrap();

        let claude_found = discover_claude_sessions(&home, 10);
        assert_eq!(claude_found.len(), 1);
        assert_eq!(
            claude_found[0].cli_session_id,
            "11111111-2222-3333-4444-555555555555"
        );
        assert_eq!(claude_found[0].title, "Fix model selector");
        assert_eq!(
            claude_found[0].working_dir,
            Some(PathBuf::from(r"D:\Projects\spark"))
        );

        let agy_found = discover_agy_sessions(&home, 10);
        assert_eq!(agy_found.len(), 1);
        assert_eq!(agy_found[0].cli_session_id, "aaaa-bbbb");
        assert_eq!(agy_found[0].title, "Implement v0.2 Sidebar");

        let _ = std::fs::remove_dir_all(&home);
    }
}
