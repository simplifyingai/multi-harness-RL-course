#!/usr/bin/env bash
# Run Claude Code headless on the chapter's task, against the stand-in model.
#
#   ./run_claude.sh                          our setup, the "solve" script
#   ./run_claude.sh --script long-output     another script (stand_in.py lists them)
#   ./run_claude.sh --setup yours            Claude Code's defaults, plus a CLAUDE.md,
#                                            a skill and an MCP tool
#   ./run_claude.sh --setup bare             our setup plus --bare (a minimal prompt)
#   ./run_claude.sh --folder extras          our setup on the folder "yours" uses
#   ./run_claude.sh --bash-max default       our setup without the 4,000-character cap
#   ./run_claude.sh --name NAME              the run folder's name (default: the script's)
#
# Each run gets a fresh folder, COURSE_HOME/16-inside-claude-code/runs/NAME/:
#   task/         the throwaway task folder (data.db; Claude Code's working folder)
#   config/       CLAUDE_CONFIG_DIR: Claude Code's private settings and history
#   home/, tmp/   HOME and TMPDIR, so nothing reaches your own dotfiles
#   calls.jsonl   every request Claude Code sent (written by the stand-in)
#   stream.jsonl  what Claude Code printed (--output-format stream-json --verbose)
#
# Safety: the only "model" Claude Code can reach is the stand-in on 127.0.0.1.
# No real API key is used, your own ~/.claude is never read, and Bash runs
# without asking (bypassPermissions) because every command comes from a
# script we wrote. Chapter 19 adds a sandbox for when a real model types.
set -euo pipefail

CLAUDE_VERSION="2.1.290"
CLAUDE_PACKAGE="@anthropic-ai/claude-code@${CLAUDE_VERSION}"
NPM_REGISTRY="https://registry.npmjs.org/"
PORT="${STAND_IN_PORT:-8016}"
BASE_URL="http://127.0.0.1:${PORT}"
API_KEY="any-string-works"
MODEL="stand-in"
RUN_TIMEOUT_S=120
READY_TIMEOUT_S=60

HERE="$(cd "$(dirname "$0")" && pwd)"
COURSE_HOME="${COURSE_HOME:-$HOME/.cache/mhrl-course}"
CHAPTER_HOME="${COURSE_HOME}/16-inside-claude-code"
INSTALL_DIR="${CHAPTER_HOME}/claude-code"

script="solve"
setup="ours"
folder=""
name=""
bash_max="4000"
while [ $# -gt 0 ]; do
  case "$1" in
    --script) script="$2"; shift 2 ;;
    --setup) setup="$2"; shift 2 ;;
    --folder) folder="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    --bash-max) bash_max="$2"; shift 2 ;;
    -h | --help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done
case "$setup" in ours | yours | bare) ;; *) echo "--setup is ours, yours or bare" >&2; exit 2 ;; esac
if [ -z "$folder" ]; then
  if [ "$setup" = "yours" ]; then folder="extras"; else folder="plain"; fi
fi
case "$folder" in plain | extras) ;; *) echo "--folder is plain or extras" >&2; exit 2 ;; esac
name="${name:-$script}"

say() { printf '%s\n' "$*"; }
# Paths are printed with your home folder as ~.
tidy() { printf '%s' "${1/#$HOME/~}"; }

# 1. The pinned Claude Code, installed with npx into the chapter's folder.
#    A private npm cache and an empty npm config: your own npm settings and
#    registry are never read. The binary's path is found once, then run directly.
install_claude() {
  mkdir -p "$INSTALL_DIR"
  : >"$INSTALL_DIR/npmrc"
  env npm_config_userconfig="$INSTALL_DIR/npmrc" \
    npm_config_cache="$INSTALL_DIR/npm-cache" \
    npm_config_registry="$NPM_REGISTRY" \
    npm_config_update_notifier=false \
    npm_config_fund=false \
    npx --yes --package "$CLAUDE_PACKAGE" -c 'command -v claude' >"$INSTALL_DIR/path.txt"
}
if [ ! -s "$INSTALL_DIR/path.txt" ] || [ ! -x "$(cat "$INSTALL_DIR/path.txt")" ]; then
  say "installing ${CLAUDE_PACKAGE} with npx into $(tidy "$INSTALL_DIR")"
  install_claude
fi
CLAUDE="$(cat "$INSTALL_DIR/path.txt")"

# 2. A fresh run folder.
RUN_DIR="${CHAPTER_HOME}/runs/${name}"
say "run       ${name} · script ${script} · setup ${setup} · folder ${folder} · bash cap ${bash_max}"
say "folder    $(tidy "$RUN_DIR")"
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR/task" "$RUN_DIR/config" "$RUN_DIR/home" "$RUN_DIR/tmp"
TASK_DIR="$RUN_DIR/task"
cp "$HERE/data/clinic.db" "$TASK_DIR/data.db"
if [ "$folder" = "extras" ]; then
  # What a typical setup holds: a CLAUDE.md and a skill in the project folder.
  cp "$HERE/data/your-setup/CLAUDE.md" "$TASK_DIR/CLAUDE.md"
  mkdir -p "$TASK_DIR/.claude/skills/sqlite-basics"
  cp "$HERE/data/your-setup/sqlite-basics/SKILL.md" "$TASK_DIR/.claude/skills/sqlite-basics/"
fi

# 3. The clean environment: nothing is inherited from your shell.
clean_env=(
  env -i
  PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  HOME="$RUN_DIR/home"
  TMPDIR="$RUN_DIR/tmp"
  USER="$(id -un)"
  LOGNAME="$(id -un)"
  SHELL="/bin/bash"
  LANG="en_US.UTF-8"
  TERM="dumb"
  NO_COLOR="1"
  ANTHROPIC_BASE_URL="$BASE_URL"
  ANTHROPIC_API_KEY="$API_KEY"
  CLAUDE_CONFIG_DIR="$RUN_DIR/config"
  CLAUDE_CODE_TMPDIR="$RUN_DIR/tmp"
  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"
  DISABLE_TELEMETRY="1"
  DISABLE_ERROR_REPORTING="1"
  DISABLE_AUTOUPDATER="1"
  # 2.1.x otherwise adds a <total_tokens> line to the system prompt on every
  # call, so no two calls would share it (the project turns it off too).
  CLAUDE_CODE_TOTAL_TOKENS_REMINDER="off"
)

version="$("${clean_env[@]}" "$CLAUDE" --version </dev/null)"
if [ "${version%% *}" != "$CLAUDE_VERSION" ]; then
  say "claude reports '${version}', this chapter is pinned to ${CLAUDE_VERSION}" >&2
  exit 1
fi
say "claude    ${version} · sha256 $(shasum -a 256 <"$CLAUDE" | cut -c1-16)"

# 4. The stand-in, in the background, logging to this run's calls.jsonl.
uv run --quiet "$HERE/stand_in.py" --script "$script" --log "$RUN_DIR/calls.jsonl" \
  --port "$PORT" >"$RUN_DIR/stand-in.txt" 2>&1 &
stand_in_pid=$!
stop_stand_in() {
  kill "$stand_in_pid" 2>/dev/null || true
  wait "$stand_in_pid" 2>/dev/null || true
}
trap stop_stand_in EXIT
waited=0
until curl -fsS "$BASE_URL/health" >/dev/null 2>&1; do
  if ! kill -0 "$stand_in_pid" 2>/dev/null || [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
    say "the stand-in did not start; its output:" >&2
    cat "$RUN_DIR/stand-in.txt" >&2
    exit 1
  fi
  sleep 1
  waited=$((waited + 1))
done
say "stand-in  ${BASE_URL} · script ${script} · logging every request to calls.jsonl"

# 5. Claude Code, headless, in the task folder.
instruction="$(uv run --quiet "$HERE/tasks.py")"
argv=("$CLAUDE" -p "$instruction" --output-format stream-json --verbose
  --model "$MODEL" --permission-mode bypassPermissions)
if [ "$setup" = "ours" ] || [ "$setup" = "bare" ]; then
  # The project's configuration (mhrl/harnesses/claude_code.py): one tool, no
  # MCP servers, no settings files, Bash output capped at 4,000 characters.
  settings='{"attribution": {"commit": "", "pr": ""}, "includeGitInstructions": false'
  if [ "$bash_max" != "default" ]; then
    settings="${settings}, \"bashOutputMaxChars\": ${bash_max}"
  fi
  settings="${settings}}"
  argv+=(--tools Bash --strict-mcp-config --setting-sources "" --settings "$settings"
    --disable-slash-commands --no-session-persistence)
  if [ "$setup" = "bare" ]; then
    # The project's opt-in: a much shorter system prompt and tool description.
    argv+=(--bare)
  fi
else
  # Claude Code's defaults (every tool, every settings source), plus one MCP
  # server: notes_mcp.py, run by the Python that uv uses for this chapter.
  python="$(uv python find '>=3.12')"
  printf '{"mcpServers": {"notes": {"command": "%s", "args": ["%s"]}}}\n' \
    "$python" "$HERE/notes_mcp.py" >"$RUN_DIR/mcp.json"
  argv+=(--mcp-config "$RUN_DIR/mcp.json")
fi

say "prompt    (claude -p's argument, printed by tasks.py)"
printf '%s\n' "$instruction" | sed 's/^/          | /'
started=$(date +%s)
set +e
(cd "$TASK_DIR" && exec "${clean_env[@]}" "${argv[@]}" </dev/null \
  >"$RUN_DIR/stream.jsonl" 2>"$RUN_DIR/stderr.txt") &
claude_pid=$!
# A watchdog: no run may take longer than RUN_TIMEOUT_S. Its output goes to
# /dev/null and its sleep is stopped with it, so nothing outlives the run.
(sleep "$RUN_TIMEOUT_S" && kill "$claude_pid") >/dev/null 2>&1 &
watchdog_pid=$!
wait "$claude_pid"
claude_exit=$?
pkill -P "$watchdog_pid" 2>/dev/null
kill "$watchdog_pid" 2>/dev/null
wait "$watchdog_pid" 2>/dev/null
set -e
say "claude    exit ${claude_exit} after $(($(date +%s) - started)) s"
stop_stand_in
trap - EXIT

# 6. What Claude Code printed, what it sent, and the grade.
uv run --quiet "$HERE/show_stream.py" "$RUN_DIR"
uv run --quiet "$HERE/grade.py" "$TASK_DIR"
