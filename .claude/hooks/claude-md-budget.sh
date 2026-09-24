#!/bin/bash
# PostToolUse(Edit|Write): report when CLAUDE.md exceeds its line budget.
#
# The budget is the remove path for CLAUDE.md. Without it every addition is
# individually justified and the file grew to 1,381 lines. Exit 2 feeds the
# message back to Claude, which then evicts something in the same change.
set -u
CAP=350

# Hooks run in a non-login shell, so Homebrew is not on PATH -- the same trap
# that hides /Library/TeX/texbin from the Bash tool.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

INPUT=$(cat)

if command -v jq >/dev/null 2>&1; then
  FILE=$(printf '%s' "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
else
  FILE=$(printf '%s' "$INPUT" \
    | sed -n 's/.*"file_path"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)
fi

FILE="${FILE//\\//}"
case "$FILE" in
  */CLAUDE.md|CLAUDE.md) ;;
  *) exit 0 ;;
esac

[ -f "$FILE" ] || exit 0
n=$(wc -l < "$FILE" | tr -d ' ')

if [ "$n" -gt "$CAP" ]; then
  cat >&2 <<EOF
CLAUDE.md is now $n lines, over its $CAP-line budget.

Evict something in this same change. Apply the routing test in the
"Maintaining this file" section: most additions belong in .claude/rules/
(loaded only when the matching files are read) or in docs/, with at most one
imperative line and a pointer left here.
EOF
  exit 2
fi
exit 0
