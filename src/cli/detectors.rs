//! Locate `agy`, `claude`, and `opencode` binaries and query their versions.

use crate::cli::AgentKind;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;
use tokio::process::Command;

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

const VERSION_TIMEOUT: Duration = Duration::from_secs(5);

/// Result of probing a single agent CLI on the local machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliStatus {
    pub agent: AgentKind,
    pub available: bool,
    pub binary_path: Option<PathBuf>,
    pub version: Option<String>,
    pub error: Option<String>,
}

impl CliStatus {
    fn unavailable(agent: AgentKind, error: impl Into<String>) -> Self {
        Self {
            agent,
            available: false,
            binary_path: None,
            version: None,
            error: Some(error.into()),
        }
    }
}

/// Build Windows fallback paths from standard environment directories.
///
/// Separated from `std::env` lookups so path construction can be unit-tested
/// deterministically.
pub fn fallback_paths_from_env(
    agent: AgentKind,
    local_app_data: Option<&Path>,
    user_profile: Option<&Path>,
    app_data: Option<&Path>,
) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    let bin = agent.binary_name();

    match agent {
        AgentKind::Agy => {
            if let Some(local) = local_app_data {
                paths.push(
                    local
                        .join("Microsoft")
                        .join("WinGet")
                        .join("Links")
                        .join("agy.exe"),
                );
                let packages_dir = local.join("Microsoft").join("WinGet").join("Packages");
                if let Ok(entries) = std::fs::read_dir(packages_dir) {
                    for entry in entries.flatten() {
                        let name = entry.file_name();
                        if name.to_string_lossy().starts_with("Google.AntigravityCLI") {
                            paths.push(entry.path().join("agy.exe"));
                        }
                    }
                }
            }
            if let Some(profile) = user_profile {
                paths.push(profile.join(".local").join("bin").join("agy.exe"));
            }
        }
        AgentKind::Claude => {
            if let Some(profile) = user_profile {
                paths.push(profile.join(".local").join("bin").join("claude.exe"));
            }
            if let Some(roaming) = app_data {
                paths.push(roaming.join("npm").join("claude.cmd"));
            }
            if let Some(local) = local_app_data {
                paths.push(
                    local
                        .join("Microsoft")
                        .join("WinGet")
                        .join("Links")
                        .join("claude.exe"),
                );
            }
        }
        AgentKind::OpenCode => {
            if let Some(roaming) = app_data {
                paths.push(roaming.join("npm").join("opencode.cmd"));
                paths.push(roaming.join("npm").join("opencode.exe"));
            }
            if let Some(profile) = user_profile {
                paths.push(profile.join(".local").join("bin").join("opencode.exe"));
            }
            if let Some(local) = local_app_data {
                paths.push(
                    local
                        .join("Microsoft")
                        .join("WinGet")
                        .join("Links")
                        .join("opencode.exe"),
                );
            }
        }
    }

    // Generic non-Windows / fallback executable name if needed.
    if let Some(profile) = user_profile {
        paths.push(profile.join(".local").join("bin").join(bin));
    }

    paths
}

/// Return candidate fallback paths for `agent` using the current process environment.
pub fn candidate_fallback_paths(agent: AgentKind) -> Vec<PathBuf> {
    let local_app_data = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    let user_profile = std::env::var_os("USERPROFILE").map(PathBuf::from);
    let app_data = std::env::var_os("APPDATA").map(PathBuf::from);

    fallback_paths_from_env(
        agent,
        local_app_data.as_deref(),
        user_profile.as_deref(),
        app_data.as_deref(),
    )
}

/// Resolve the executable path for `agent` via `PATH` or known Windows fallbacks.
pub fn resolve_binary(agent: AgentKind) -> Option<PathBuf> {
    if let Ok(found) = which::which(agent.binary_name()) {
        return Some(found);
    }

    candidate_fallback_paths(agent)
        .into_iter()
        .find(|candidate| candidate.is_file())
}

/// Parse the first non-empty trimmed line from `--version` output.
pub fn parse_version_output(stdout: &str, stderr: &str) -> Option<String> {
    stdout
        .lines()
        .chain(stderr.lines())
        .map(str::trim)
        .find(|line| !line.is_empty())
        .map(ToString::to_string)
}

/// Probe a single CLI binary and query its `--version`.
pub async fn detect_cli(agent: AgentKind) -> CliStatus {
    let Some(binary_path) = resolve_binary(agent) else {
        return CliStatus::unavailable(
            agent,
            format!(
                "'{}' not found on PATH or standard install locations",
                agent.binary_name()
            ),
        );
    };

    let mut cmd = Command::new(&binary_path);
    cmd.arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    cmd.creation_flags(CREATE_NO_WINDOW);

    let output = match tokio::time::timeout(VERSION_TIMEOUT, cmd.output()).await {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            return CliStatus {
                agent,
                available: false,
                binary_path: Some(binary_path),
                version: None,
                error: Some(format!("Failed to execute --version: {err}")),
            };
        }
        Err(_) => {
            return CliStatus {
                agent,
                available: false,
                binary_path: Some(binary_path),
                version: None,
                error: Some("Timed out querying --version".to_string()),
            };
        }
    };

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let version = parse_version_output(&stdout, &stderr);

    if output.status.success() {
        CliStatus {
            agent,
            available: true,
            binary_path: Some(binary_path),
            version,
            error: None,
        }
    } else {
        CliStatus {
            agent,
            available: false,
            binary_path: Some(binary_path),
            version,
            error: Some(format!(
                "--version exited with status {}",
                output
                    .status
                    .code()
                    .map_or_else(|| "unknown".to_string(), |c| c.to_string())
            )),
        }
    }
}

/// What to run in place of an npm `.cmd` shim: the program the shim would start,
/// plus any arguments that must come before the caller's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShimTarget {
    pub program: PathBuf,
    pub leading_args: Vec<String>,
}

/// `true` when `path` is a `.cmd` or `.bat` file.
pub fn is_batch_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("cmd") || e.eq_ignore_ascii_case("bat"))
}

/// What an npm `.cmd` shim launches: a path relative to the shim's folder, and
/// whether the shim runs it with `node`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShimLaunch {
    pub target: String,
    pub via_node: bool,
}

/// Read what an npm `.cmd` shim launches, or `None` for anything that is not
/// exactly one of npm's own layouts.
///
/// Strict on purpose. Running the target directly skips the batch file, so any
/// shim that does more than launch one thing must be left alone: extra arguments
/// on the launch line (`node --no-warnings script`), another interpreter (`sh`),
/// environment set up first (pnpm's `NODE_PATH`), or a hand-written wrapper.
pub fn parse_cmd_shim(contents: &str) -> Option<ShimLaunch> {
    if contents.contains("NODE_PATH") {
        return None;
    }
    let mut found: Option<ShimLaunch> = None;
    for line in contents.lines().filter(|l| l.contains("%*")) {
        let launch = parse_shim_launch_line(line)?;
        match &found {
            None => found = Some(launch),
            // Older npm writes the launch twice (bundled node.exe, else `node`).
            Some(prev) if prev.target == launch.target => {}
            Some(_) => return None,
        }
    }
    found
}

/// One launch line: `"<target>" %*`, or `<node> "<target>" %*` where `<node>` is
/// `"%_prog%"`, `node`, or the `node.exe` beside the shim. Nothing else.
fn parse_shim_launch_line(line: &str) -> Option<ShimLaunch> {
    const NPM_PREAMBLE: &str = "endLocal & goto #_undefined_# 2>NUL || title %COMSPEC% &";
    fn beside_shim(token: &str) -> Option<&str> {
        token
            .strip_prefix("%dp0%\\")
            .or_else(|| token.strip_prefix("%~dp0\\"))
            .filter(|rel| !rel.is_empty())
    }

    let line = line.trim();
    let line = line.strip_prefix(NPM_PREAMBLE).unwrap_or(line);

    // Split into quoted and bare tokens.
    let mut tokens: Vec<(&str, bool)> = Vec::new();
    let mut rest = line.trim_start();
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix('"') {
            let end = after.find('"')?;
            tokens.push((&after[..end], true));
            rest = after[end + 1..].trim_start();
        } else {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            tokens.push((&rest[..end], false));
            rest = rest[end..].trim_start();
        }
    }

    match tokens.as_slice() {
        [(target, true), ("%*", false)] => Some(ShimLaunch {
            target: beside_shim(target)?.to_string(),
            via_node: false,
        }),
        [(interp, quoted), (target, true), ("%*", false)] => {
            let is_node = match (*interp, *quoted) {
                ("%_prog%", true) => true,
                ("node", false) => true,
                (other, true) => beside_shim(other).is_some_and(|r| r == "node.exe"),
                _ => false,
            };
            is_node
                .then(|| beside_shim(target))
                .flatten()
                .map(|rel| ShimLaunch {
                    target: rel.to_string(),
                    via_node: true,
                })
        }
        _ => None,
    }
}

/// Resolve the program behind the npm shim at `path`, if `path` is one and its
/// target exists. `None` means "this is not a shim Spark can safely see through".
pub fn resolve_cmd_shim(path: &Path) -> Option<ShimTarget> {
    if !is_batch_file(path) {
        return None;
    }
    let dir = path.parent()?;
    let contents = std::fs::read_to_string(path).ok()?;
    let launch = parse_cmd_shim(&contents)?;
    let target = dir.join(&launch.target);
    if !target.is_file() {
        return None;
    }

    let is_exe = target
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("exe"));
    if !launch.via_node {
        // Run directly by the shim, so it has to be something Windows can run.
        return is_exe.then_some(ShimTarget {
            program: target,
            leading_args: Vec::new(),
        });
    }

    // A script: the shim runs it with the `node.exe` beside it, else `node` on PATH.
    let bundled_node = dir.join("node.exe");
    let node = if bundled_node.is_file() {
        bundled_node
    } else {
        which::which("node").ok()?
    };
    Some(ShimTarget {
        program: node,
        leading_args: vec![target.to_string_lossy().into_owned()],
    })
}

/// Concurrently detect all three supported CLIs.
pub async fn detect_all() -> Vec<CliStatus> {
    let (agy, claude, opencode) = tokio::join!(
        detect_cli(AgentKind::Agy),
        detect_cli(AgentKind::Claude),
        detect_cli(AgentKind::OpenCode),
    );
    vec![agy, claude, opencode]
}

#[cfg(test)]
mod tests {
    use super::*;

    const NPM_NODE_SHIM: &str = "@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n:start\r\nSETLOCAL\r\nCALL :find_dp0\r\n\r\nIF EXIST \"%dp0%\\node.exe\" (\r\n  SET \"_prog=%dp0%\\node.exe\"\r\n) ELSE (\r\n  SET \"_prog=node\"\r\n  SET PATHEXT=%PATHEXT:;.JS;=;%\r\n)\r\n\r\nendLocal & goto #_undefined_# 2>NUL || title %COMSPEC% & \"%_prog%\"  \"%dp0%\\node_modules\\opencode-ai\\bin\\opencode\" %*\r\n";

    fn launch(target: &str, via_node: bool) -> Option<ShimLaunch> {
        Some(ShimLaunch {
            target: target.to_string(),
            via_node,
        })
    }

    #[test]
    fn parse_cmd_shim_finds_a_native_binary_target() {
        // What npm writes for @anthropic-ai/claude-code 2.x.
        let shim = "@ECHO off\r\nGOTO start\r\n:find_dp0\r\nSET dp0=%~dp0\r\nEXIT /b\r\n:start\r\nSETLOCAL\r\nCALL :find_dp0\r\n\"%dp0%\\node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe\"   %*\r\n";
        assert_eq!(
            parse_cmd_shim(shim),
            launch(
                "node_modules\\@anthropic-ai\\claude-code\\bin\\claude.exe",
                false
            )
        );
    }

    #[test]
    fn parse_cmd_shim_finds_a_node_script_target() {
        assert_eq!(
            parse_cmd_shim(NPM_NODE_SHIM),
            launch("node_modules\\opencode-ai\\bin\\opencode", true)
        );
    }

    #[test]
    fn parse_cmd_shim_reads_the_older_npm_layout_and_ignores_other_batch_files() {
        let old = "@IF EXIST \"%~dp0\\node.exe\" (\r\n  \"%~dp0\\node.exe\"  \"%~dp0\\node_modules\\tool\\cli.js\" %*\r\n) ELSE (\r\n  node  \"%~dp0\\node_modules\\tool\\cli.js\" %*\r\n)\r\n";
        assert_eq!(
            parse_cmd_shim(old),
            launch("node_modules\\tool\\cli.js", true)
        );
        assert_eq!(parse_cmd_shim("@echo off\r\necho hello %*\r\n"), None);
        assert_eq!(parse_cmd_shim(""), None);
    }

    #[test]
    fn parse_cmd_shim_refuses_a_shim_that_does_more_than_launch_one_thing() {
        // Running the target directly would silently drop each of these.
        let node_flags =
            NPM_NODE_SHIM.replace("\"%_prog%\"  \"%dp0%", "\"%_prog%\" --no-warnings \"%dp0%");
        assert_ne!(node_flags, NPM_NODE_SHIM);
        assert_eq!(parse_cmd_shim(&node_flags), None, "extra node arguments");

        let extra_arg =
            "@ECHO off\r\n\"%dp0%\\node_modules\\tool\\bin\\tool.exe\" --profile work %*\r\n";
        assert_eq!(parse_cmd_shim(extra_arg), None, "a wrapper adding flags");

        let sh = NPM_NODE_SHIM.replace("\"%_prog%\"", "\"%dp0%\\sh.exe\"");
        assert_eq!(parse_cmd_shim(&sh), None, "another interpreter");

        let pnpm = format!("@SET NODE_PATH=C:\\store\\node_modules\r\n{NPM_NODE_SHIM}");
        assert_eq!(parse_cmd_shim(&pnpm), None, "environment set up first");

        let two_targets = "\"%dp0%\\a.exe\" %*\r\n\"%dp0%\\b.exe\" %*\r\n";
        assert_eq!(parse_cmd_shim(two_targets), None, "two different targets");
    }

    #[test]
    fn resolve_cmd_shim_leaves_real_executables_alone() {
        assert!(!is_batch_file(Path::new(r"C:\tools\claude.exe")));
        assert!(is_batch_file(Path::new(r"C:\tools\claude.CMD")));
        assert_eq!(resolve_cmd_shim(Path::new(r"C:\tools\claude.exe")), None);
        assert_eq!(resolve_cmd_shim(Path::new(r"C:\no\such\shim.cmd")), None);
    }

    #[test]
    fn fallback_paths_include_expected_windows_locations() {
        let local = Path::new(r"C:\Users\test\AppData\Local");
        let profile = Path::new(r"C:\Users\test");
        let roaming = Path::new(r"C:\Users\test\AppData\Roaming");

        let agy_paths =
            fallback_paths_from_env(AgentKind::Agy, Some(local), Some(profile), Some(roaming));
        assert!(agy_paths
            .iter()
            .any(|p| p.ends_with(Path::new(r"Microsoft\WinGet\Links\agy.exe"))));
        assert!(agy_paths
            .iter()
            .any(|p| p.ends_with(Path::new(r".local\bin\agy.exe"))));

        let claude_paths =
            fallback_paths_from_env(AgentKind::Claude, Some(local), Some(profile), Some(roaming));
        assert!(claude_paths
            .iter()
            .any(|p| p.ends_with(Path::new(r".local\bin\claude.exe"))));
        assert!(claude_paths
            .iter()
            .any(|p| p.ends_with(Path::new(r"npm\claude.cmd"))));

        let opencode_paths = fallback_paths_from_env(
            AgentKind::OpenCode,
            Some(local),
            Some(profile),
            Some(roaming),
        );
        assert!(opencode_paths
            .iter()
            .any(|p| p.ends_with(Path::new(r"npm\opencode.cmd"))));
    }

    #[test]
    fn parse_version_output_trims_and_falls_back_to_stderr() {
        assert_eq!(
            parse_version_output("\n  1.5.5  \n", ""),
            Some("1.5.5".to_string())
        );
        assert_eq!(
            parse_version_output("", "  opencode v2.1.4\n"),
            Some("opencode v2.1.4".to_string())
        );
        assert_eq!(parse_version_output("   \n", "\t\n"), None);
    }

    #[tokio::test]
    async fn detect_all_returns_three_statuses_in_order() {
        let statuses = detect_all().await;
        assert_eq!(statuses.len(), 3);
        assert_eq!(statuses[0].agent, AgentKind::Agy);
        assert_eq!(statuses[1].agent, AgentKind::Claude);
        assert_eq!(statuses[2].agent, AgentKind::OpenCode);
    }
}
