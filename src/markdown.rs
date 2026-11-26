//! Lightweight pure-Rust Markdown block and inline renderer for Spark agent responses.

use crate::theme;
use eframe::egui::{self, CornerRadius, FontId, Frame, Margin, RichText, Stroke, TextFormat};
use std::time::Instant;

/// Parsed top-level block from agent CLI output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MarkdownBlock {
    Heading { level: u8, text: String },
    Paragraph { text: String },
    ListItem { marker: String, text: String },
    Quote { text: String },
    Code { lang: String, code: String },
    Rule,
}

/// Parsed inline span within prose text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InlineSpan {
    Text(String),
    Bold(String),
    Code(String),
}

/// Parse markdown-formatted CLI output into display blocks.
/// Handles unclosed code fences gracefully so live streaming output renders cleanly.
pub fn parse_blocks(input: &str) -> Vec<MarkdownBlock> {
    let mut blocks = Vec::new();
    let mut para_lines: Vec<String> = Vec::new();
    let mut in_code = false;
    let mut code_fence = String::new();
    let mut code_lang = String::new();
    let mut code_lines: Vec<String> = Vec::new();

    let flush_para = |blocks: &mut Vec<MarkdownBlock>, para_lines: &mut Vec<String>| {
        if !para_lines.is_empty() {
            blocks.push(MarkdownBlock::Paragraph {
                text: para_lines.join("\n"),
            });
            para_lines.clear();
        }
    };

    for raw_line in input.lines() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim_start();

        if in_code {
            if trimmed.starts_with(&code_fence) && trimmed[code_fence.len()..].trim().is_empty() {
                blocks.push(MarkdownBlock::Code {
                    lang: std::mem::take(&mut code_lang),
                    code: code_lines.join("\n"),
                });
                code_lines.clear();
                code_fence.clear();
                in_code = false;
            } else {
                code_lines.push(line.to_string());
            }
            continue;
        }

        if let Some((fence, rest)) = detect_fence_open(trimmed) {
            flush_para(&mut blocks, &mut para_lines);
            in_code = true;
            code_fence = fence;
            code_lang = rest.trim().to_string();
            code_lines.clear();
            continue;
        }

        if trimmed.is_empty() {
            flush_para(&mut blocks, &mut para_lines);
            continue;
        }

        if is_horizontal_rule(trimmed) {
            flush_para(&mut blocks, &mut para_lines);
            blocks.push(MarkdownBlock::Rule);
            continue;
        }

        if let Some((level, heading_text)) = parse_heading(trimmed) {
            flush_para(&mut blocks, &mut para_lines);
            blocks.push(MarkdownBlock::Heading {
                level,
                text: heading_text.to_string(),
            });
            continue;
        }

        if let Some(quote_text) = trimmed.strip_prefix("> ") {
            flush_para(&mut blocks, &mut para_lines);
            blocks.push(MarkdownBlock::Quote {
                text: quote_text.to_string(),
            });
            continue;
        }

        if let Some((marker, item_text)) = parse_list_item(trimmed) {
            flush_para(&mut blocks, &mut para_lines);
            blocks.push(MarkdownBlock::ListItem {
                marker,
                text: item_text.to_string(),
            });
            continue;
        }

        para_lines.push(line.to_string());
    }

    if in_code {
        blocks.push(MarkdownBlock::Code {
            lang: code_lang,
            code: code_lines.join("\n"),
        });
    } else {
        flush_para(&mut blocks, &mut para_lines);
    }

    blocks
}

fn detect_fence_open(trimmed: &str) -> Option<(String, &str)> {
    for ch in ['`', '~'] {
        let count = trimmed.chars().take_while(|&c| c == ch).count();
        if count >= 3 {
            let byte_len = count * ch.len_utf8();
            let fence = trimmed[..byte_len].to_string();
            let rest = &trimmed[byte_len..];
            // Opening info string cannot contain backticks for backtick fences.
            if ch == '`' && rest.contains('`') {
                return None;
            }
            return Some((fence, rest));
        }
    }
    None
}

fn is_horizontal_rule(trimmed: &str) -> bool {
    let no_spaces: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
    if no_spaces.len() < 3 {
        return false;
    }
    let first = no_spaces.chars().next().unwrap_or(' ');
    matches!(first, '-' | '*' | '_') && no_spaces.chars().all(|c| c == first)
}

fn parse_heading(trimmed: &str) -> Option<(u8, &str)> {
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    if !(1..=4).contains(&hashes) {
        return None;
    }
    let rest = &trimmed[hashes..];
    rest.strip_prefix(' ').map(|t| (hashes as u8, t.trim()))
}

fn parse_list_item(trimmed: &str) -> Option<(String, &str)> {
    for prefix in ["- ", "* ", "+ "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            return Some(("•".to_string(), rest));
        }
    }

    let digit_count = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();
    if digit_count > 0 && digit_count <= 3 {
        let after_digits = &trimmed[digit_count..];
        if let Some(rest) = after_digits.strip_prefix(". ") {
            return Some((format!("{}.", &trimmed[..digit_count]), rest));
        }
    }
    None
}

/// Parse `**bold**` and `` `inline code` `` spans safely without panicking on unmatched markers.
pub fn parse_inline_spans(text: &str) -> Vec<InlineSpan> {
    let mut spans = Vec::new();
    let mut plain = String::new();
    let mut idx = 0;

    let flush_plain = |spans: &mut Vec<InlineSpan>, plain: &mut String| {
        if !plain.is_empty() {
            spans.push(InlineSpan::Text(std::mem::take(plain)));
        }
    };

    while idx < text.len() {
        let rest = &text[idx..];
        if let Some(after_tick) = rest.strip_prefix('`') {
            if let Some(end_rel) = after_tick.find('`') {
                if end_rel > 0 {
                    flush_plain(&mut spans, &mut plain);
                    spans.push(InlineSpan::Code(after_tick[..end_rel].to_string()));
                    idx += 1 + end_rel + 1;
                    continue;
                }
            }
        }

        if let Some(after_bold) = rest.strip_prefix("**") {
            if let Some(end_rel) = after_bold.find("**") {
                if end_rel > 0 && !after_bold[..end_rel].contains('\n') {
                    flush_plain(&mut spans, &mut plain);
                    spans.push(InlineSpan::Bold(after_bold[..end_rel].to_string()));
                    idx += 2 + end_rel + 2;
                    continue;
                }
            }
        }

        if let Some(ch) = rest.chars().next() {
            plain.push(ch);
            idx += ch.len_utf8();
        } else {
            break;
        }
    }

    flush_plain(&mut spans, &mut plain);
    spans
}

/// Build an `egui::text::LayoutJob` for inline markdown spans.
pub fn build_inline_layout_job(
    text: &str,
    base_color: egui::Color32,
    size: f32,
) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let normal_font = FontId::proportional(size);
    let mono_font = FontId::monospace((size - 1.0).max(11.0));

    for span in parse_inline_spans(text) {
        match span {
            InlineSpan::Text(s) => {
                job.append(
                    &s,
                    0.0,
                    TextFormat {
                        font_id: normal_font.clone(),
                        color: base_color,
                        ..Default::default()
                    },
                );
            }
            InlineSpan::Bold(s) => {
                job.append(
                    &s,
                    0.0,
                    TextFormat {
                        font_id: normal_font.clone(),
                        color: theme::TEXT_BRIGHT,
                        ..Default::default()
                    },
                );
            }
            InlineSpan::Code(s) => {
                job.append(
                    &format!(" {s} "),
                    0.0,
                    TextFormat {
                        font_id: mono_font.clone(),
                        color: theme::MIST_BRIGHT,
                        background: theme::SLATE_CONTROL,
                        ..Default::default()
                    },
                );
            }
        }
    }

    job
}

/// Render markdown blocks into `ui` with Impeccable Neo Kinpaku styling and per-code-block copy buttons.
pub fn render_markdown(
    ui: &mut egui::Ui,
    text: &str,
    id_salt: impl std::hash::Hash,
    ctx: &egui::Context,
    copied_feedback: &mut Option<(String, Instant)>,
) {
    let blocks = parse_blocks(text);
    let base_id = egui::Id::new(id_salt);

    for (idx, block) in blocks.into_iter().enumerate() {
        match block {
            MarkdownBlock::Heading { level, text } => {
                let size = match level {
                    1 => 16.5,
                    2 => 15.0,
                    _ => 14.0,
                };
                if idx > 0 {
                    ui.add_space(6.0);
                }
                ui.add(
                    egui::Label::new(
                        RichText::new(text)
                            .strong()
                            .size(size)
                            .color(theme::TEXT_BRIGHT),
                    )
                    .selectable(true)
                    .wrap(),
                );
                ui.add_space(2.0);
            }
            MarkdownBlock::Paragraph { text } => {
                let job = build_inline_layout_job(&text, theme::TEXT_BODY, 13.5);
                ui.add(egui::Label::new(job).selectable(true).wrap());
                ui.add_space(4.0);
            }
            MarkdownBlock::ListItem { marker, text } => {
                ui.horizontal_top(|ui| {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(marker)
                            .strong()
                            .size(13.0)
                            .color(theme::MIST_BLUE),
                    );
                    let job = build_inline_layout_job(&text, theme::TEXT_BODY, 13.5);
                    ui.add(egui::Label::new(job).selectable(true).wrap());
                });
            }
            MarkdownBlock::Quote { text } => {
                Frame::new()
                    .fill(theme::USER_SURFACE)
                    .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                    .corner_radius(CornerRadius::same(6))
                    .inner_margin(Margin::symmetric(10, 6))
                    .show(ui, |ui| {
                        let job = build_inline_layout_job(&text, theme::TEXT_MUTED, 13.0);
                        ui.add(egui::Label::new(job).selectable(true).wrap());
                    });
                ui.add_space(4.0);
            }
            MarkdownBlock::Rule => {
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);
            }
            MarkdownBlock::Code { lang, code } => {
                ui.add_space(3.0);
                let copy_key = format!("{base_id:?}_code_{idx}");
                let just_copied = copied_feedback
                    .as_ref()
                    .is_some_and(|(k, at)| k == &copy_key && at.elapsed().as_secs_f32() < 1.5);

                Frame::new()
                    .fill(theme::OBSIDIAN_DEEP)
                    .stroke(Stroke::new(1.0_f32, theme::HAIRLINE))
                    .corner_radius(CornerRadius::same(8))
                    .inner_margin(Margin::symmetric(12, 8))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let label = if lang.is_empty() {
                                "code"
                            } else {
                                lang.as_str()
                            };
                            ui.label(
                                RichText::new(label)
                                    .monospace()
                                    .size(11.0)
                                    .color(theme::TEXT_FAINT),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    let btn_text = if just_copied { "Copied" } else { "Copy" };
                                    let btn_color = if just_copied {
                                        theme::VERDIGRIS_PATINA
                                    } else {
                                        theme::TEXT_MUTED
                                    };
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new(btn_text).size(11.0).color(btn_color),
                                            )
                                            .fill(theme::SLATE_CONTROL)
                                            .stroke(Stroke::new(1.0_f32, theme::HAIRLINE_SUBTLE))
                                            .corner_radius(CornerRadius::same(4)),
                                        )
                                        .clicked()
                                    {
                                        ctx.copy_text(code.clone());
                                        *copied_feedback = Some((copy_key, Instant::now()));
                                    }
                                },
                            );
                        });
                        ui.add_space(4.0);
                        egui::ScrollArea::horizontal()
                            .id_salt(("code_scroll", base_id, idx))
                            .show(ui, |ui| {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(&code)
                                            .monospace()
                                            .size(12.2)
                                            .color(theme::TEXT_BODY),
                                    )
                                    .selectable(true),
                                );
                            });
                    });
                ui.add_space(5.0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_headings_lists_and_fenced_code_blocks() {
        let src = "# Summary\nHere is **important** `code`.\n- First item\n2. Second item\n\n```rust\nfn main() {}\n```";
        let blocks = parse_blocks(src);
        assert_eq!(
            blocks,
            vec![
                MarkdownBlock::Heading {
                    level: 1,
                    text: "Summary".to_string()
                },
                MarkdownBlock::Paragraph {
                    text: "Here is **important** `code`.".to_string()
                },
                MarkdownBlock::ListItem {
                    marker: "•".to_string(),
                    text: "First item".to_string()
                },
                MarkdownBlock::ListItem {
                    marker: "2.".to_string(),
                    text: "Second item".to_string()
                },
                MarkdownBlock::Code {
                    lang: "rust".to_string(),
                    code: "fn main() {}".to_string()
                }
            ]
        );
    }

    #[test]
    fn unclosed_fence_during_streaming_still_produces_code_block() {
        let src = "Streaming:\n```python\nprint('hi')";
        let blocks = parse_blocks(src);
        assert_eq!(blocks.len(), 2);
        assert_eq!(
            blocks[1],
            MarkdownBlock::Code {
                lang: "python".to_string(),
                code: "print('hi')".to_string()
            }
        );
    }

    #[test]
    fn inline_spans_handle_unmatched_markers_and_utf8() {
        let spans = parse_inline_spans("Café **bold** and `code` with *unmatched and `dangling");
        assert_eq!(
            spans,
            vec![
                InlineSpan::Text("Café ".to_string()),
                InlineSpan::Bold("bold".to_string()),
                InlineSpan::Text(" and ".to_string()),
                InlineSpan::Code("code".to_string()),
                InlineSpan::Text(" with *unmatched and `dangling".to_string()),
            ]
        );
    }
}
