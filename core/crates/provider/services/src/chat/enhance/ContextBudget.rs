use operit_model::PromptTurn::{PromptTurn, PromptTurnKind};

/// Character budget used for one prepared request window.
///
/// The budget is expressed in characters rather than provider tokens so the
/// trimming decision stays local and does not require a tokenizer round-trip.
const DEFAULT_CONTEXT_CHARACTER_BUDGET: usize = 120_000;

/// Minimum characters kept from the head of an oversized tool result.
const TOOL_RESULT_HEAD_CHARS: usize = 2_000;

/// Minimum characters kept from the tail of an oversized tool result.
const TOOL_RESULT_TAIL_CHARS: usize = 1_000;

/// Marker inserted where the middle of a tool result was removed.
const TRUNCATION_MARKER: &str =
    "\n\n[... context trimmed: middle of this tool result was omitted ...]\n\n";

/// Trims prepared prompt history so a request stays inside a workable budget.
///
/// Operit historically kept every historical turn verbatim and only failed the
/// request once the accumulated size crossed the model limit. The result was a
/// hard stall plus a full retry. Trimming oldest tool payloads first keeps the
/// conversation usable instead of aborting it, and mirrors the "fetch what is
/// needed, discard the rest" behaviour of lightweight agents.
pub fn trim_prompt_history(history: Vec<PromptTurn>) -> Vec<PromptTurn> {
    trim_prompt_history_with_budget(history, DEFAULT_CONTEXT_CHARACTER_BUDGET)
}

/// Trims prepared prompt history using an explicit character budget.
pub fn trim_prompt_history_with_budget(
    history: Vec<PromptTurn>,
    budget: usize,
) -> Vec<PromptTurn> {
    if budget == 0 || history.is_empty() {
        return history;
    }

    let mut total = total_characters(&history);
    if total <= budget {
        return history;
    }

    let mut trimmed = history;
    // Oldest tool results carry the most stale payload, so they are released
    // first. System and user turns are never removed because they carry the
    // live instruction and the pending question.
    for index in 0..trimmed.len() {
        if total <= budget {
            break;
        }
        if trimmed[index].kind != PromptTurnKind::TOOL_RESULT {
            continue;
        }
        let original = trimmed[index].content.len();
        let reduced = compact_tool_result(&trimmed[index].content, budget.min(TOOL_RESULT_HEAD_CHARS));
        if reduced.len() < original {
            total = total - original + reduced.len();
            trimmed[index].content = reduced;
        }
    }

    if total <= budget {
        return trimmed;
    }

    // Still oversized: drop the oldest tool results entirely before touching
    // conversational turns.
    let mut index = 0;
    while total > budget && index < trimmed.len() {
        if trimmed[index].kind == PromptTurnKind::TOOL_RESULT {
            let original = trimmed[index].content.len();
            total = total.saturating_sub(original);
            trimmed[index].content.clear();
            trimmed[index].metadata.insert(
                "context_trimmed".to_string(),
                "true".into(),
            );
        }
        index += 1;
    }

    trimmed
}

/// Sums the character weight of every turn in the prepared history.
fn total_characters(history: &[PromptTurn]) -> usize {
    history
        .iter()
        .map(|turn| turn.content.len())
        .fold(0usize, |acc, len| acc.saturating_add(len))
}

/// Keeps the head and tail of a tool result and drops the bulky middle.
fn compact_tool_result(content: &str, head_chars: usize) -> String {
    let total = content.len();
    if total <= head_chars.saturating_add(TOOL_RESULT_TAIL_CHARS) {
        return content.to_string();
    }
    let tail_chars = TOOL_RESULT_TAIL_CHARS.min(head_chars);
    let head = &content[..head_chars.min(total)];
    let tail_start = total.saturating_sub(tail_chars);
    let tail = &content[tail_start..];
    format!("{head}{TRUNCATION_MARKER}{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(kind: PromptTurnKind, content: &str) -> PromptTurn {
        PromptTurn::new(kind, content.to_string())
    }

    /// History that already fits the budget is returned untouched.
    #[test]
    fn history_within_budget_is_unchanged() {
        let history = vec![
            turn(PromptTurnKind::SYSTEM, "system prompt"),
            turn(PromptTurnKind::USER, "question"),
        ];
        let trimmed = trim_prompt_history_with_budget(history.clone(), 10_000);
        assert_eq!(trimmed.len(), history.len());
        assert_eq!(trimmed[0].content, "system prompt");
        assert_eq!(trimmed[1].content, "question");
    }

    /// An oversized tool result keeps its head and tail so the conclusion stays visible.
    #[test]
    fn oversized_tool_result_keeps_head_and_tail() {
        let payload = format!("HEAD_MARKER{}", "x".repeat(50_000));
        let history = vec![turn(PromptTurnKind::TOOL_RESULT, &payload)];
        let trimmed = trim_prompt_history_with_budget(history, 5_000);
        assert_eq!(trimmed.len(), 1);
        assert!(trimmed[0].content.contains("HEAD_MARKER"));
        assert!(trimmed[0].content.contains(TRUNCATION_MARKER));
        assert!(trimmed[0].content.len() < payload.len());
    }

    /// System and user turns are never dropped while trimming.
    #[test]
    fn system_and_user_turns_survive_trimming() {
        let history = vec![
            turn(PromptTurnKind::SYSTEM, "SYSTEM_MARKER"),
            turn(PromptTurnKind::TOOL_RESULT, &"y".repeat(40_000)),
            turn(PromptTurnKind::USER, "USER_MARKER"),
        ];
        let trimmed = trim_prompt_history_with_budget(history, 2_000);
        assert!(trimmed
            .iter()
            .any(|turn| turn.content.contains("SYSTEM_MARKER")));
        assert!(trimmed
            .iter()
            .any(|turn| turn.content.contains("USER_MARKER")));
    }

    /// A budget of zero disables trimming entirely.
    #[test]
    fn zero_budget_disables_trimming() {
        let history = vec![turn(PromptTurnKind::TOOL_RESULT, &"z".repeat(10_000))];
        let trimmed = trim_prompt_history_with_budget(history.clone(), 0);
        assert_eq!(trimmed[0].content, history[0].content);
    }
}
