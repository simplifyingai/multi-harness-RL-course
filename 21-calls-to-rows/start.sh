#!/usr/bin/env bash
# Chapter 21: an engine, and OpenEnv's capture proxy in front of it (chapter 20's
# stack), with a second engine to choose from: the scripted one.
#
#   ENGINE_PORT (8100)  the engine: our model, or the scripted model, behind
#                       OpenAI's chat API
#   PROXY_PORT  (8200)  the capture proxy, started as the project starts it:
#                       --llm-url http://127.0.0.1:$ENGINE_PORT
#                       --model lfm2.5-2.6b --max-output-tokens 1024
#
# Usage: ./start.sh [start|scripted|proxy|stop|status|install]
#   start     install the proxy if needed, start the engine (LFM2.5), then the proxy
#   scripted  the same with the scripted engine (scripted_engine.py): no model,
#             no MLX, replies queued by the chapter's scripts
#   proxy     start only the proxy, in front of an engine already on ENGINE_PORT
#   stop      stop what this script started, and nothing else
#   status    ask both for /health
#   install   install the capture proxy into its own environment
#
# Settings, as environment variables:
#   ENGINE_PORT   default 8100, the project's port; the proxy's --llm-url follows
#                 it. If another program holds 8100, use ENGINE_PORT=8119.
#   PROXY_PORT    default 8200, the project's port.
#   ENGINE_FLAGS  passed to engine.py: --no-token-fields, --raw-logprobs, --fast
#   COURSE_HOME   default ~/.cache/mhrl-course: the proxy's environment, the
#                 logs and the process ids go under COURSE_HOME/21-calls-to-rows
#   PROXY_ENV     where the proxy's environment lives (default
#                 COURSE_HOME/21-calls-to-rows/proxy-env)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
ENGINE_PORT=${ENGINE_PORT:-8100}
PROXY_PORT=${PROXY_PORT:-8200}
ENGINE_FLAGS=${ENGINE_FLAGS:-}
COURSE_HOME=${COURSE_HOME:-$HOME/.cache/mhrl-course}

STATE="$COURSE_HOME/21-calls-to-rows"
PROXY_ENV=${PROXY_ENV:-$STATE/proxy-env}  # the proxy's own Python environment
LOGS="$STATE/logs"
RUN="$STATE/run"              # each process this script started: pid, command
JUST_STARTED=""               # the processes this run started
ENGINE_SCRIPT=engine.py       # scripted_engine.py with ./start.sh scripted

HOST=127.0.0.1
MODEL_NAME=lfm2.5-2.6b        # the name the engine serves
MAX_OUTPUT_TOKENS=1024        # the project's cap on one reply (OUTPUT_CAP in lfm_files.py)
PYTHON_VERSION=3.13           # the project's Python
PROXY_COMMIT=3f14355631513f51c4b3211b32a069ead0faf837  # the project's pin
PROXY_MODULE=openenv.core.harness.capture.server
ENGINE_WAIT_S=600             # the first run builds engine.py's environment
PROXY_WAIT_S=300              # the proxy probes the engine before it listens
STOP_WAIT_S=30
RELEASE_WAIT_S=60             # macOS keeps closed connections about 30 s
ENGINE_URL="http://$HOST:$ENGINE_PORT"
PROXY_URL="http://$HOST:$PROXY_PORT"

# Run every background process in its own process group, so stopping one
# stops all of it (uv and the Python it started), and nothing else.
set -m

installed_commit() {  # prints the OpenEnv commit installed, or nothing
  "$PROXY_ENV/bin/python" - 2>/dev/null <<'PY' || true
import json
from importlib.metadata import distribution
print(json.loads(distribution("openenv").read_text("direct_url.json"))["vcs_info"]["commit_id"])
PY
}

install() {
  [ -n "${INSTALL_CHECKED:-}" ] && return 0  # checked once per run
  INSTALL_CHECKED=1
  if [ "$(installed_commit)" = "$PROXY_COMMIT" ] \
    && "$PROXY_ENV/bin/python" -c "import $PROXY_MODULE" 2>/dev/null; then
    echo "capture proxy ready: OpenEnv $PROXY_COMMIT in ${PROXY_ENV/#$HOME/~}"
    return 0
  fi
  echo "installing OpenEnv's capture proxy (commit ${PROXY_COMMIT:0:7}) into ${PROXY_ENV/#$HOME/~}"
  if [ -e "$PROXY_ENV" ] && [ ! -f "$PROXY_ENV/pyvenv.cfg" ]; then
    # PROXY_ENV is a setting: never delete a folder that is not an environment.
    echo "  $PROXY_ENV exists and is not a Python environment: choose another" \
      "PROXY_ENV" >&2
    return 1
  fi
  rm -rf "$PROXY_ENV"
  echo "  uv venv --python $PYTHON_VERSION ${PROXY_ENV/#$HOME/~}"
  uv venv --quiet --python "$PYTHON_VERSION" "$PROXY_ENV"
  echo "  uv pip install --python ${PROXY_ENV/#$HOME/~}/bin/python --no-deps -r proxy-requirements.txt"
  uv pip install --quiet --python "$PROXY_ENV/bin/python" --no-deps \
    -r "$HERE/proxy-requirements.txt"
  "$PROXY_ENV/bin/python" -c "import $PROXY_MODULE"  # fails loudly if anything is missing
  echo "  installed: OpenEnv $(installed_commit)"
}

listening() {  # port: true when some program listens there
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

lingering() {  # port: true while connections of a stopped server still hold it
  # macOS's netstat writes a local address as 127.0.0.1.8100 (the course's Mac).
  netstat -an -p tcp 2>/dev/null \
    | awk -v local="$HOST.$1" '$4 == local && $6 == "TIME_WAIT" { found = 1 } END { exit !found }'
}

wait_released() {  # port: wait until a stopped engine's old connections let go
  # The engine binds without SO_REUSEADDR (see engine.py's claim_port), so for
  # about 30 s after it stops, its closed connections keep the port.
  local waited=0
  if lingering "$1"; then
    echo "  port $1 is still held by the last engine's closed connections; waiting"
  fi
  while lingering "$1" && [ "$waited" -lt "$RELEASE_WAIT_S" ]; do
    sleep 1
    waited=$((waited + 1))
  done
}

record() {  # name: note what the process looks like, to know it again later
  ps -o command= -p "$(cat "$RUN/$1.pid")" >"$RUN/$1.cmd" 2>/dev/null || true
}

ours() {  # name: true when the pid file names a live process this script started
  [ -f "$RUN/$1.pid" ] || return 1
  local pid
  pid=$(cat "$RUN/$1.pid")
  kill -0 "$pid" 2>/dev/null || return 1
  # set -m made it a process-group leader...
  [ "$(ps -o pgid= -p "$pid" | tr -d ' ')" = "$pid" ] || return 1
  # ...running the command it ran when it started: after a reboot or a crash,
  # a recycled pid belongs to some other program. A process started by this
  # very run is checked once its command is recorded.
  case " $JUST_STARTED " in *" $1 "*) return 0 ;; esac
  [ -f "$RUN/$1.cmd" ] && [ "$(ps -o command= -p "$pid")" = "$(cat "$RUN/$1.cmd")" ]
}

wait_for() {  # url, name, pid, seconds
  local waited=0
  while [ "$waited" -lt "$4" ]; do
    if curl -sf -m 2 "$1" >/dev/null; then return 0; fi
    if ! kill -0 "$3" 2>/dev/null; then
      echo "  the $2 stopped before it answered; its log ends:" >&2
      tail -n 5 "$LOGS/$2.log" >&2
      rm -f "$RUN/$2.pid"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "  the $2 did not answer $1 within $4 s; see $LOGS/$2.log" >&2
  return 1
}

start_engine() {
  if listening "$ENGINE_PORT"; then
    echo "port $ENGINE_PORT is taken by another program: choose another, for" \
      "example ENGINE_PORT=8119 ./start.sh $COMMAND" >&2
    return 1
  fi
  wait_released "$ENGINE_PORT"
  local flags=""
  # The scripted engine has no model, so none of engine.py's flags.
  if [ "$ENGINE_SCRIPT" = engine.py ]; then flags=$ENGINE_FLAGS; fi
  echo "starting the engine ($ENGINE_SCRIPT) on $ENGINE_URL${flags:+ ($flags)}"
  # The flags are split into words on purpose: there may be several.
  # shellcheck disable=SC2086
  (cd "$HERE" && exec uv run "$ENGINE_SCRIPT" --port "$ENGINE_PORT" $flags) \
    >"$LOGS/engine.log" 2>&1 &
  echo $! >"$RUN/engine.pid"
  JUST_STARTED="$JUST_STARTED engine"
  wait_for "$ENGINE_URL/health" engine "$!" "$ENGINE_WAIT_S" || return 1
  record engine
  # The engine's own start-up lines: from "engine ready" to its first request.
  sed -n '/^engine ready/,/^chat ·/p' "$LOGS/engine.log" | grep -v '^chat ·' \
    | sed 's/^/  /' || true
}

start_proxy() {
  install || return 1
  if listening "$PROXY_PORT"; then
    echo "port $PROXY_PORT is taken by another program: stop it, or choose" \
      "another with PROXY_PORT" >&2
    return 1
  fi
  if ! curl -sf -m 2 "$ENGINE_URL/health" >/dev/null; then
    echo "no engine answers on $ENGINE_URL: start one first" >&2
    return 1
  fi
  echo "starting the capture proxy on $PROXY_URL, in front of $ENGINE_URL"
  # Unbuffered, so its "capture level" line reaches the log as it is printed.
  PYTHONUNBUFFERED=1 "$PROXY_ENV/bin/python" -m "$PROXY_MODULE" --llm-url "$ENGINE_URL" \
    --model "$MODEL_NAME" --port "$PROXY_PORT" \
    --max-output-tokens "$MAX_OUTPUT_TOKENS" >"$LOGS/proxy.log" 2>&1 &
  echo $! >"$RUN/proxy.pid"
  JUST_STARTED="$JUST_STARTED proxy"
  wait_for "$PROXY_URL/health" proxy "$!" "$PROXY_WAIT_S" || return 1
  record proxy
  # A port answering is not proof it is OUR proxy: check it names our engine.
  local upstream
  upstream=$(curl -sf -m 2 "$PROXY_URL/health" | "$PROXY_ENV/bin/python" -c \
    'import json, sys; print(json.load(sys.stdin).get("upstream", ""))' || true)
  if [ "$upstream" != "$ENGINE_URL" ]; then
    echo "  $PROXY_URL answers, but not for $ENGINE_URL: another proxy?" >&2
    return 1
  fi
  grep -E "^capture level|upstream compat" "$LOGS/proxy.log" | sed 's/^/  /' || true
}

stop_one() {  # name
  if ! ours "$1"; then
    if [ -f "$RUN/$1.pid" ]; then
      echo "the $1's pid file names no process this script started: removed it"
    fi
    rm -f "$RUN/$1.pid" "$RUN/$1.cmd"
    return 0
  fi
  local pid waited=0
  pid=$(cat "$RUN/$1.pid")
  # Ctrl-C to the group's leader only: uv passes it on to the engine's Python
  # (and the proxy IS its leader). Sent to the whole group, the engine would
  # get it twice, the second time while it shuts down.
  kill -INT "$pid" 2>/dev/null || true
  # Wait for the whole group (uv and its Python), not just its leader.
  while kill -0 -- "-$pid" 2>/dev/null && [ "$waited" -lt "$STOP_WAIT_S" ]; do
    sleep 1
    waited=$((waited + 1))
  done
  if kill -0 -- "-$pid" 2>/dev/null; then
    kill -KILL -- "-$pid" 2>/dev/null || true
  fi
  rm -f "$RUN/$1.pid" "$RUN/$1.cmd"
  echo "stopped the $1"
}

stop() {
  local engine_ran=0
  ours engine && engine_ran=1
  stop_one proxy
  stop_one engine
  if [ "$engine_ran" = 1 ]; then
    # The engine's own last line: its request count and peak memory.
    grep -h "^engine stopped" "$LOGS/engine.log" 2>/dev/null | tail -n 1 \
      | sed 's/^/  /' || true
  fi
}

refuse_if_running() {  # names: stop here if this script already started one
  for name in "$@"; do
    if ours "$name"; then
      echo "the $name this script started is still running: ./start.sh stop" \
        "first" >&2
      exit 1
    fi
  done
}

status() {
  for url in "$ENGINE_URL/health" "$PROXY_URL/health"; do
    printf '%-34s ' "$url"
    curl -s -m 2 "$url" || printf 'no answer'
    echo
  done
}

mkdir -p "$LOGS" "$RUN"
COMMAND=${1:-start}
case "$COMMAND" in
  start | scripted)
    refuse_if_running proxy engine
    install
    if [ "$COMMAND" = scripted ]; then ENGINE_SCRIPT=scripted_engine.py; fi
    start_engine || { stop_one engine; exit 1; }
    start_proxy || { stop_one proxy; stop_one engine; exit 1; }
    ;;
  proxy)
    refuse_if_running proxy
    start_proxy || { stop_one proxy; exit 1; }
    ;;
  stop) stop ;;
  status) status ;;
  install) install ;;
  *) echo "usage: $0 [start|scripted|proxy|stop|status|install]" >&2; exit 2 ;;
esac
