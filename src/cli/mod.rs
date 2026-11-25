//! CLI discovery and async subprocess streaming for Spark.

#![allow(dead_code)]

pub mod detectors;
pub mod models;
pub mod runner;
pub mod sessions;

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported local coding-agent CLIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgentKind {
    Agy,
    Claude,
    OpenCode,
}

impl AgentKind {
    /// All supported agents in display order.
    pub const ALL: [AgentKind; 3] = [AgentKind::Agy, AgentKind::Claude, AgentKind::OpenCode];

    /// Executable name searched on `PATH`.
    pub const fn binary_name(self) -> &'static str {
        match self {
            AgentKind::Agy => "agy",
            AgentKind::Claude => "claude",
            AgentKind::OpenCode => "opencode",
        }
    }

    /// Human-readable agent name for UI headers.
    pub const fn display_name(self) -> &'static str {
        match self {
            AgentKind::Agy => "Antigravity",
            AgentKind::Claude => "Claude Code",
            AgentKind::OpenCode => "OpenCode",
        }
    }

    /// Distinguishing Impeccable glyph for each agent.
    pub const fn glyph(self) -> &'static str {
        match self {
            AgentKind::Agy => "◆",
            AgentKind::Claude => "●",
            AgentKind::OpenCode => "▲",
        }
    }
}

/// The first `max_chars` characters of `s`, cut on a character boundary.
///
/// Session IDs come from files on disk and from a text field, so they are not
/// guaranteed to be ASCII; slicing them by byte count panics when the cut lands
/// inside a multi-byte character.
pub fn char_prefix(s: &str, max_chars: usize) -> &str {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => &s[..byte_idx],
        None => s,
    }
}

impl fmt::Display for AgentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.display_name(), self.binary_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn char_prefix_cuts_on_character_boundaries() {
        assert_eq!(char_prefix("abcdef", 3), "abc");
        assert_eq!(char_prefix("ab", 3), "ab");
        assert_eq!(char_prefix("", 3), "");
        assert_eq!(char_prefix("a\u{e9}\u{e9}\u{e9}", 2), "a\u{e9}");
    }

    #[test]
    fn agent_kind_metadata_is_consistent() {
        assert_eq!(AgentKind::ALL.len(), 3);
        assert_eq!(AgentKind::Agy.glyph(), "◆");
        assert_eq!(AgentKind::Claude.glyph(), "●");
        assert_eq!(AgentKind::OpenCode.glyph(), "▲");
        assert_eq!(AgentKind::Agy.to_string(), "Antigravity (agy)");
        assert_eq!(AgentKind::Claude.to_string(), "Claude Code (claude)");
        assert_eq!(AgentKind::OpenCode.to_string(), "OpenCode (opencode)");
    }
}
