#!/usr/bin/env bash
# The capstone: the project's whole pipeline, from a fresh clone to the final
# table, on this laptop. Every stage is the project's own command:
#
#   clone     git clone of the project into $COURSE_HOME/28-capstone/
#   setup     uv sync, scripts/setup.sh, and the two Node harnesses at their
#             pins (Claude Code 2.1.290, Pi 1.0.4); mini-swe-agent 2.4.6 comes
#             with uv sync
#   tasks     python -m mhrl.tasks.generate: 230 tasks (180 train, 50 test)
#   tests     the project's pytest
#   start     the engine, the teacher and the proxy (./stack.sh start)
#   smoke     python -m mhrl.harnesses.smoke: one real captured rollout
#   pipeline  scripts/train_from_scratch.sh's five stages, command for command,
#             with `python -m mhrl.cli steps` watching tokens per call during
#             GRPO, then the project's report and read_results.py
#   stop      stop the servers this chapter started
#   all       every step above, in that order
#   plan      print every command `all` would run, and run nothing
#
# Why the pipeline is not scripts/train_from_scratch.sh itself: that script
# collects demonstrations from the teacher on port 8101 (`mhrl.cli demos`
# without --teacher-port), and 8101 is not ours to take on this machine. So
# this script runs the same commands in the same order, adding only
# `--teacher-port $MHRL_TEACHER_PORT`; tests/test_28_capstone.py checks that
# the two lists of commands stay the same. For the same reason the servers
# start with ./stack.sh (chapter 27's), never with scripts/serve.sh, whose
# stop kills every engine on the machine.
#
# Settings (environment):
#   FAST=1            EVAL_LIMIT=6 DEMO_LIMIT=10 GRPO_STEPS=2 (run capstone-fast)
#   GRPO_STEPS        6 unless FAST: the project's recorded run trained 6 steps,
#                     while train_from_scratch.sh defaults to 16 and the CLI to 20
#   EVAL_LIMIT DEMO_LIMIT SFT_EPOCHS GRPO_TASKS GRPO_GROUP   as in the project
#   RUN               the run's name (capstone, or capstone-fast with FAST=1)
#   MHRL_MODEL_PATH   the model every server and trainer loads; default
#                     $COURSE_HOME/models/lfm2.5-2.6b-4bit-ours (chapter 11's
#                     build, byte-identical weights to Liquid's export)
#   MHRL_ENGINE_PORT MHRL_TEACHER_PORT MHRL_PROXY_PORT   8149, 8159, 8219
#   MHRL_SOURCE       the repository to clone (default ~/sai_work/multi-harness-rl)
#   MHRL_PROJECT      run the steps in another checkout instead of the clone;
#                     only `pipeline` and the stack accept one, and only with
#                     MHRL_RUNS_DIR outside it, so nothing is written into it
#   STACK=scripted    the teacher answers as the engine (no 2.6B): the tests
#   MHRL_CLI          the command that stands for `python -m mhrl.cli` (tests)
#
# Usage: ./capstone.sh [all|plan|clone|setup|tasks|tests|start|smoke|pipeline|stop|status]
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
COURSE_HOME_DIR=${COURSE_HOME:-$HOME/.cache/mhrl-course}
COURSE_HOME_DIR=${COURSE_HOME_DIR/#\~/$HOME}
CHAPTER_HOME="$COURSE_HOME_DIR/28-capstone"
CLONE="$CHAPTER_HOME/multi-harness-rl"
SOURCE=${MHRL_SOURCE:-$HOME/sai_work/multi-harness-rl}
PROJECT=${MHRL_PROJECT:-$CLONE}
PROJECT=${PROJECT/#\~/$HOME}
for named in "$PROJECT" "${MHRL_RUNS_DIR:-/}"; do # compared as text below
  case "$named" in
    /*) ;;
    *) printf 'capstone: %s must be an absolute path\n' "$named" >&2 && exit 1 ;;
  esac
  case "/$named/" in
    */../* | */./*) printf 'capstone: %s must not hold . or ..\n' "$named" >&2 && exit 1 ;;
  esac
done
if [ -d "$PROJECT" ]; then # the real folder, so a symlink cannot hide where it is
  PROJECT=$(cd "$PROJECT" && pwd -P)
fi
PRIVATE="$CHAPTER_HOME/installers" # the installers' own settings and caches
STACK_LOGS="$CHAPTER_HOME/stack"
PY="$PROJECT/.venv/bin/python"

if [ -n "${FAST:-}" ]; then
  EVAL_LIMIT=${EVAL_LIMIT:-6}
  DEMO_LIMIT=${DEMO_LIMIT:-10}
  GRPO_STEPS=${GRPO_STEPS:-2}
  RUN=${RUN:-capstone-fast}
else
  EVAL_LIMIT=${EVAL_LIMIT:-24}
  DEMO_LIMIT=${DEMO_LIMIT:-60}
  GRPO_STEPS=${GRPO_STEPS:-6} # set explicitly: the project's script says 16
  RUN=${RUN:-capstone}
fi
SFT_EPOCHS=${SFT_EPOCHS:-1}
GRPO_TASKS=${GRPO_TASKS:-2}
GRPO_GROUP=${GRPO_GROUP:-4}
TRAIN_HARNESSES=${TRAIN_HARNESSES:-mini-swe-agent,claude-code}
EVAL_HARNESSES=${EVAL_HARNESSES:-mini-swe-agent,claude-code,pi}

export MHRL_ENGINE_PORT=${MHRL_ENGINE_PORT:-8149}
export MHRL_TEACHER_PORT=${MHRL_TEACHER_PORT:-8159}
export MHRL_PROXY_PORT=${MHRL_PROXY_PORT:-8219}
export MHRL_MODEL_PATH=${MHRL_MODEL_PATH:-$COURSE_HOME_DIR/models/lfm2.5-2.6b-4bit-ours}
OFFICIAL_MODEL="$COURSE_HOME_DIR/models/LFM2.5-2.6B-MLX-4bit"
PROJECT_MODEL_DIR="models/LFM2.5-2.6B-MLX-4bit" # where setup.sh looks
export PYTHONUNBUFFERED=1
STACK_MODE=${STACK:-start}
if [ -n "${MHRL_RUNS_DIR:-}" ]; then
  if [ -d "$MHRL_RUNS_DIR" ]; then
    MHRL_RUNS_DIR=$(cd "$MHRL_RUNS_DIR" && pwd -P)
  fi
  export MHRL_RUNS_DIR
  RUNS_DIR=$MHRL_RUNS_DIR
else
  RUNS_DIR="$PROJECT/runs"
fi
OUT="$RUNS_DIR/$RUN"
if [ -n "${MHRL_CLI:-}" ]; then
  read -r -a CLI <<<"$MHRL_CLI"
else
  CLI=("$PY" -m mhrl.cli)
fi

CLAUDE_PIN="2.1.290"
PI_PIN="1.0.4"
NPM_REGISTRY="https://registry.npmjs.org/"
TASKS_EXPECTED=230 # 3 train databases x 60 + 2 test databases x 25
SAMPLE_S=${MEMORY_EVERY_S:-15}   # memory: one look every 15 s
NETWORK_S=${NETWORK_EVERY_S:-10} # network: one look every 10 s
STEPS_POLL_S=20    # how often the steps watcher looks for a new GRPO step
PLAN=""

say() { printf '%s\n' "$*"; }

tilde() { # paths as a reader wants them: ./ for this folder, $COURSE_HOME, ~
  sed -e "s#$HERE/#./#g" -e "s#$COURSE_HOME_DIR#\$COURSE_HOME#g" -e "s#$HOME#~#g"
}

quote() { # one argument, quoted only when the shell needs it
  if [[ $1 =~ ^[A-Za-z0-9_./:@%+=,-]+$ ]]; then
    printf '%s' "$1"
  elif [[ $1 != *"'"* ]]; then
    printf "'%s'" "$1"
  else
    printf '%q' "$1"
  fi
}

show() { # print a command as it will run; BASE (a folder) is written relative
  local text="" tag="" arg
  case "$1" in
    uv_private) tag="   (uv: private settings)" && shift ;;
    npm_private) tag="   (npm: private settings)" && shift ;;
  esac
  for arg in "$@"; do
    [ -n "${BASE:-}" ] && arg=${arg#"$BASE"/}
    text+="$(quote "$arg") "
  done
  say "  \$ ${text% }$tag" | tilde
}

run() { # print a command, then run it (plan mode only prints)
  show "$@"
  [ -n "$PLAN" ] || "$@"
}

in_project() { # print and run a command from the project folder
  BASE="$PROJECT" show "$@"
  [ -n "$PLAN" ] || (cd "$PROJECT" && "$@")
}

refuse() { say "capstone: $*" >&2; exit 1; }

banner() { # the project's step(): a banner and the time, kept in stages.txt
  local now
  say ""
  say "=== $* ==="
  [ -n "$PLAN" ] && return 0
  now=$(date '+%Y-%m-%d %H:%M:%S')
  say "$now"
  printf '=== %s ===\n%s\n' "$*" "$now" >>"$OUT/stages.txt"
  printf '%s\n' "$*" >"$OUT/.stage"
}

inside_chapter() { # path -> 0 if it is inside this chapter's folder
  local home=$CHAPTER_HOME
  [ -d "$home" ] && home=$(cd "$home" && pwd -P)
  case "$1/" in "$home"/* | "$CHAPTER_HOME"/*) return 0 ;; *) return 1 ;; esac
}

writable_project() { # the steps that write into the project need our clone
  inside_chapter "$PROJECT" ||
    refuse "$1 writes into the project; it runs only in the clone ($CLONE), not $PROJECT"
}

# ---------------------------------------------------------------- clone, setup

step_clone() {
  say "━━ clone: a fresh copy of the project, under COURSE_HOME"
  [ -e "$CLONE" ] && [ -z "$PLAN" ] &&
    refuse "$CLONE already exists; remove it for a fresh clone"
  run mkdir -p "$CHAPTER_HOME"
  run git clone --quiet "$SOURCE" "$CLONE"
  run git -C "$CLONE" log -1 --format='%h %s'
}

uv_private() { # uv with no settings of yours: no uv.toml, no ~/.netrc, no git config
  local cache python
  cache=$(UV_NO_CONFIG=1 uv cache dir)
  python=$(UV_NO_CONFIG=1 uv python dir)
  env HOME="$PRIVATE/home" UV_NO_CONFIG=1 UV_CACHE_DIR="$cache" \
    UV_PYTHON_INSTALL_DIR="$python" GIT_CONFIG_GLOBAL=/dev/null \
    HF_HOME="$PRIVATE/huggingface" HF_HUB_DISABLE_TELEMETRY=1 "$@"
}

npm_private() { # npm with no settings of yours: an empty npmrc, its own cache
  env HOME="$PRIVATE/npm" npm_config_userconfig="$PRIVATE/npm/npmrc" \
    npm_config_globalconfig="$PRIVATE/npm/npmrc-global" \
    npm_config_cache="$PRIVATE/npm/cache" npm_config_registry="$NPM_REGISTRY" \
    npm_config_update_notifier=false npm_config_fund=false \
    npm_config_audit=false "$@"
}

step_setup() {
  writable_project setup
  say "━━ setup: uv sync, scripts/setup.sh, the Node harnesses at their pins"
  say "  uv and npm run with private settings ($PRIVATE): your uv.toml, ~/.netrc," | tilde
  say "  ~/.npmrc, git config and Hugging Face token are never read" | tilde
  if [ -z "$PLAN" ]; then
    mkdir -p "$PRIVATE/home" "$PRIVATE/npm" "$PRIVATE/huggingface"
    : >"$PRIVATE/npm/npmrc"
    : >"$PRIVATE/npm/npmrc-global"
  fi
  if [ -f "$OFFICIAL_MODEL/config.json" ]; then
    say "  setup.sh fetches Liquid's export when $PROJECT_MODEL_DIR is missing;"
    say "  chapter 11 already fetched it, so link it there and setup.sh fetches no weights"
    run mkdir -p "$PROJECT/models"
    run ln -sfn "$OFFICIAL_MODEL" "$PROJECT/$PROJECT_MODEL_DIR"
  else
    say "  no $OFFICIAL_MODEL: setup.sh will fetch LiquidAI/LFM2.5-2.6B-MLX-4bit (~1.5 GB)" | tilde
  fi
  in_project uv_private uv sync
  in_project uv_private scripts/setup.sh
  say "  the model every stage loads (MHRL_MODEL_PATH): ours, against Liquid's export"
  show cmp "$MHRL_MODEL_PATH/model.safetensors" "$OFFICIAL_MODEL/model.safetensors"
  if [ -z "$PLAN" ]; then
    [ -f "$MHRL_MODEL_PATH/model.safetensors" ] ||
      refuse "no model at $MHRL_MODEL_PATH (chapter 11's convert.py builds it)"
    if [ -f "$OFFICIAL_MODEL/model.safetensors" ]; then
      cmp -s "$MHRL_MODEL_PATH/model.safetensors" "$OFFICIAL_MODEL/model.safetensors" ||
        refuse "ours and Liquid's weights differ; chapter 11's compare.py says where"
      say "  same bytes: the weights are Liquid's export (chapter 11's compare.py)"
    fi
  fi
  say "  Claude Code $CLAUDE_PIN into tools/claude-code/, where mhrl/harnesses/claude_code.py looks"
  in_project npm_private npm install --prefix tools/claude-code/npm --save-exact \
    --no-audit --no-fund "@anthropic-ai/claude-code@$CLAUDE_PIN"
  run mkdir -p "$PROJECT/tools/claude-code/$CLAUDE_PIN"
  run ln -sfn "$PROJECT/tools/claude-code/npm/node_modules/@anthropic-ai/claude-code/bin/claude.exe" \
    "$PROJECT/tools/claude-code/$CLAUDE_PIN/claude"
  say "  Pi $PI_PIN into tools/pi/, where mhrl/harnesses/pi.py looks"
  in_project npm_private npm install --prefix tools/pi --save-exact --no-audit \
    --no-fund "@earendil-works/pi-coding-agent@$PI_PIN"
  say "  the versions, as each harness reports itself (a private HOME, so none writes yours)"
  in_project env HOME="$PRIVATE/probe" CLAUDE_CONFIG_DIR="$PRIVATE/probe/claude" \
    DISABLE_AUTOUPDATER=1 "tools/claude-code/$CLAUDE_PIN/claude" --version
  in_project env HOME="$PRIVATE/probe" node \
    tools/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js --version
  in_project "$PY" -c 'import importlib.metadata as m; print("mini-swe-agent", m.version("mini-swe-agent"))'
}

# ---------------------------------------------------------------- tasks, tests

step_tasks() {
  writable_project tasks
  say "━━ tasks: the databases and the train/test task files"
  in_project "$PY" -m mhrl.tasks.generate
  [ -n "$PLAN" ] && return 0
  local count changed
  count=$(cat "$PROJECT/data/tasks/train.jsonl" "$PROJECT/data/tasks/test.jsonl" | grep -c .)
  say "  $count tasks (train.jsonl + test.jsonl)"
  [ "$count" -eq "$TASKS_EXPECTED" ] || refuse "expected $TASKS_EXPECTED tasks, found $count"
  changed=$(git -C "$PROJECT" status --porcelain -- data/tasks)
  if [ -z "$changed" ]; then
    say "  git status: data/tasks unchanged, so the same seed rebuilt the committed files byte for byte"
  else
    say "  git status: these differ from the committed files:"
    say "$changed" | sed 's/^/    /'
  fi
}

step_tests() {
  writable_project tests
  say "━━ tests: the project's own"
  in_project "$PY" -m pytest -q
}

# ---------------------------------------------------------------- the servers

stack() { # start|scripted|stop|status
  show env MHRL_PROJECT="$PROJECT" "$HERE/stack.sh" "$1"
  [ -n "$PLAN" ] || MHRL_PROJECT="$PROJECT" COURSE_HOME="$COURSE_HOME_DIR" "$HERE/stack.sh" "$1"
}

step_start() {
  say "━━ the servers: engine $MHRL_ENGINE_PORT, teacher $MHRL_TEACHER_PORT, proxy $MHRL_PROXY_PORT"
  [ "$STACK_MODE" = scripted ] && say "  STACK=scripted: the teacher answers as the engine; no model is loaded"
  stack "$STACK_MODE"
}

step_smoke() {
  say "━━ smoke: one real captured rollout through the proxy"
  in_project "$PY" -m mhrl.harnesses.smoke --harness claude-code
}

# ---------------------------------------------------------------- watchers

mem_mb() { # pids -> "pid<TAB>MB" from top's MEM (it counts MLX's GPU memory; ps's RSS does not)
  local args=() pid
  for pid in "$@"; do args+=(-pid "$pid"); done
  top -l 1 -stats pid,mem "${args[@]}" 2>/dev/null | awk '
    $1 ~ /^[0-9]+$/ {
      size = $2; sub(/[+-]$/, "", size)
      unit = substr(size, length(size)); value = size + 0
      if (unit == "K") value /= 1024; else if (unit == "G") value *= 1024
      else if (unit == "B") value /= 1048576
      printf "%s\t%.0f\n", $1, value
    }'
}

children() { # pid -> every descendant pid, depth first
  local child
  for child in $(pgrep -P "$1" 2>/dev/null || true); do
    say "$child"
    children "$child"
  done
}

server_pids() { # "role pid" for each server ./stack.sh started
  local role
  for role in engine teacher proxy; do
    [ -f "$STACK_LOGS/$role.pid" ] && say "$role $(cat "$STACK_LOGS/$role.pid")"
  done
  return 0
}

stage_pids() { # "stage pid" for each descendant of $1 running a pipeline stage
  local pid
  for pid in $(children "$1"); do
    ps -p "$pid" -o command= 2>/dev/null |
      grep -qE ' (eval|demos|sft|grpo|report) --run ' && say "stage $pid"
  done
  return 0
}

memory_watch() { # parent pid; one look every SAMPLE_S s while .sampling exists
  local parent=$1 now stage list
  printf 'time\tstage\trole\tpid\tmb\n' >"$OUT/memory.tsv"
  while [ -f "$OUT/.sampling" ]; do
    now=$(date +%H:%M:%S)
    stage=$(cat "$OUT/.stage" 2>/dev/null || say "-")
    list=$({
      server_pids
      stage_pids "$parent"
    })
    if [ -n "$list" ]; then
      # shellcheck disable=SC2046  # one pid per word
      mem_mb $(awk '{print $2}' <<<"$list") |
        LIST="$list" NOW="$now" STAGE="${stage%% *}" awk -F '\t' '
          BEGIN {
            n = split(ENVIRON["LIST"], rows, "\n")
            for (i = 1; i <= n; i++) { split(rows[i], f, " "); role[f[2]] = f[1] }
          }
          { printf "%s\t%s\t%s\t%s\t%s\n", ENVIRON["NOW"], ENVIRON["STAGE"], role[$1], $1, $2 }
        ' >>"$OUT/memory.tsv" || true
    fi
    sleep "$SAMPLE_S"
  done
}

# A listener on every address (*:port) is printed too: it is not a connection,
# but anything on the network could reach it, so the check names it.
external_lines() { # lsof lines on stdin -> only those with an end off this machine
  awk 'NR > 1 {
    name = $9
    n = split(name, ends, "->")
    for (i = 1; i <= n; i++) {
      host = ends[i]; sub(/:[^:]*$/, "", host)
      if (host !~ /^(127\.|\[::1\]|localhost)/) { print; break }
    }
  }'
}

network_watch() { # parent pid; one look every NETWORK_S s while .watching exists
  local parent=$1 pids looks=0
  : >"$OUT/.network-found"
  while [ -f "$OUT/.watching" ]; do
    pids=$({
      server_pids | awk '{print $2}'
      children "$parent"
    } | sort -u | paste -sd, -)
    if [ -n "$pids" ]; then
      lsof -nP -a -p "$pids" -i 2>/dev/null | external_lines >>"$OUT/.network-found" || true
      looks=$((looks + 1))
      say "$looks" >"$OUT/.network-looks"
    fi
    sleep "$NETWORK_S"
  done
}

write_network_check() { # start time -> network_check.txt, in the project's layout
  local looks
  looks=$(cat "$OUT/.network-looks" 2>/dev/null || say 0)
  {
    say "# Network check during the RL evaluation, $1 to $(date '+%H:%M'), one look every ${NETWORK_S} s: $looks looks"
    say "# Non-loopback TCP/UDP of the engine (:$MHRL_ENGINE_PORT), teacher (:$MHRL_TEACHER_PORT), capture proxy (:$MHRL_PROXY_PORT) and harness processes:"
    sort -u "$OUT/.network-found"
    say "# (no lines above = no external connections)"
  } >"$OUT/network_check.txt"
  rm -f "$OUT/.network-found" "$OUT/.network-looks"
}

steps_watch() { # print `mhrl.cli steps` each time GRPO records a new step
  local seen=0 now
  while [ -f "$OUT/.watching-steps" ]; do
    now=$(grep -c . "$OUT/grpo/metrics.jsonl" 2>/dev/null || true)
    if [ "${now:-0}" -gt "$seen" ]; then
      seen=$now
      (cd "$PROJECT" && "${CLI[@]}" steps --run "$RUN") 2>&1 | sed 's/^/[steps] /' || true
    fi
    sleep "$STEPS_POLL_S"
  done
}

stop_watchers() {
  rm -f "$OUT/.sampling" "$OUT/.watching" "$OUT/.watching-steps" "$OUT/.stage"
}

# ---------------------------------------------------------------- the pipeline

step_pipeline() {
  if ! inside_chapter "$PROJECT"; then # another checkout: never write into it
    [ -n "${MHRL_RUNS_DIR:-}" ] ||
      refuse "pipeline in $PROJECT needs MHRL_RUNS_DIR outside it"
    case "$RUNS_DIR/" in "$PROJECT"/*) refuse "MHRL_RUNS_DIR must be outside $PROJECT" ;; esac
  fi
  say "━━ pipeline: train_from_scratch.sh's stages, run $RUN"
  say "  EVAL_LIMIT=$EVAL_LIMIT DEMO_LIMIT=$DEMO_LIMIT SFT_EPOCHS=$SFT_EPOCHS GRPO_STEPS=$GRPO_STEPS GRPO_TASKS=$GRPO_TASKS GRPO_GROUP=$GRPO_GROUP"
  say "  model $MHRL_MODEL_PATH" | tilde
  local eval_args=(--run "$RUN" --harnesses "$EVAL_HARNESSES" --limit "$EVAL_LIMIT")
  if [ -z "$PLAN" ]; then
    compgen -G "$OUT/eval_*.jsonl" >/dev/null &&
      refuse "$OUT already holds evaluation records; choose another RUN"
    mkdir -p "$OUT"
    : >"$OUT/stages.txt"
    touch "$OUT/.sampling"
    memory_watch $$ &
  fi

  banner "1/5 baseline: the base model in every harness"
  in_project "${CLI[@]}" eval "${eval_args[@]}" --label base

  banner "2/5 answer-key demonstrations, then supervised fine-tuning"
  in_project "${CLI[@]}" demos --run "$RUN" --harnesses "$TRAIN_HARNESSES" \
    --limit "$DEMO_LIMIT" --teacher-port "$MHRL_TEACHER_PORT"
  in_project "${CLI[@]}" sft --run "$RUN" --epochs "$SFT_EPOCHS"

  banner "3/5 evaluate the SFT model"
  in_project "${CLI[@]}" eval "${eval_args[@]}" --label sft --adapter "$OUT/sft/adapter.safetensors"

  banner "4/5 GRPO through the training harnesses"
  if [ -z "$PLAN" ]; then
    touch "$OUT/.watching-steps"
    steps_watch &
  fi
  in_project "${CLI[@]}" grpo --run "$RUN" --harnesses "$TRAIN_HARNESSES" \
    --init "$OUT/sft/adapter.safetensors" \
    --steps "$GRPO_STEPS" --tasks-per-step "$GRPO_TASKS" --group "$GRPO_GROUP"
  [ -n "$PLAN" ] || rm -f "$OUT/.watching-steps"
  in_project "${CLI[@]}" steps --run "$RUN"

  banner "5/5 evaluate the RL model"
  local started
  started=$(date '+%Y-%m-%d %H:%M')
  if [ -z "$PLAN" ]; then
    touch "$OUT/.watching"
    network_watch $$ &
  fi
  in_project "${CLI[@]}" eval "${eval_args[@]}" --label rl --adapter "$OUT/grpo/adapter_final.safetensors"
  if [ -z "$PLAN" ]; then
    rm -f "$OUT/.watching"
    sleep 1
    write_network_check "$started"
  fi

  banner "results"
  in_project "${CLI[@]}" report --run "$RUN" --harnesses "$EVAL_HARNESSES"
  banner "done"
  [ -n "$PLAN" ] || stop_watchers
  wait 2>/dev/null || true
  run "$PY" -B -s "$HERE/read_results.py" --run-dir "$OUT" # -B: no __pycache__ here
}

step_stop() {
  say "━━ stop: only the servers this chapter started"
  stack stop
}

all() {
  step_clone
  step_setup
  step_tasks
  step_tests
  step_start
  [ -n "$PLAN" ] || trap 'stop_watchers; stack stop' EXIT
  step_smoke
  step_pipeline
  step_stop
  [ -n "$PLAN" ] || trap - EXIT
}

case "${1:-all}" in
  all) all ;;
  plan)
    PLAN=1
    all
    ;;
  clone) step_clone ;;
  setup) step_setup ;;
  tasks) step_tasks ;;
  tests) step_tests ;;
  start) step_start ;;
  smoke) step_smoke ;;
  pipeline)
    trap 'stop_watchers' EXIT
    step_pipeline
    ;;
  stop) step_stop ;;
  status) stack status ;;
  *)
    say "usage: $0 [all|plan|clone|setup|tasks|tests|start|smoke|pipeline|stop|status]"
    exit 2
    ;;
esac
