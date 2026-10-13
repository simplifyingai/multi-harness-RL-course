#!/usr/bin/env bash
# Start the project's own three servers, as its scripts/serve.sh does, on
# ports you choose:
#   engine   python -m mhrl.engine.server --port $MHRL_ENGINE_PORT       (default 8100)
#   teacher  python -m mhrl.engine.server --port $MHRL_TEACHER_PORT --oracle (8101)
#   proxy    python -m openenv.core.harness.capture.server in front of the engine
#            --port $MHRL_PROXY_PORT (8200) --max-output-tokens 1024
#
# Why not scripts/serve.sh itself: it writes 8100, 8101 and 8200 into its
# commands, so it cannot move to free ports without editing it; its health
# check (curl -s) calls ANY answer on the port "up", even another program's;
# and its stop kills every process named mhrl.engine.server or
# openenv.core.harness.capture.server on the machine, not only its own. The
# commands below are serve.sh's, the project unchanged; this script only
# chooses the ports, refuses a port already in use, checks that each answer
# is the right server's, and stops only the processes it started.
#
# Usage: ./stack.sh [start|scripted|stop|status]
#   start     the engine (the 2.6B model), the teacher and the proxy
#   scripted  the proxy in front of the teacher on the ENGINE port: the
#             project's answer-key teacher loads no weights, so the whole
#             pipeline runs without the 2.6B (the tests use this)
#   stop      stop what this script started (and print the logs' last lines)
#   status    each server's /health
# Needs MHRL_PROJECT (the project's checkout, with its .venv from `uv sync`).
# Logs and pid files go to $COURSE_HOME/27-how-it-combines/stack/.
set -euo pipefail

PROJECT=${MHRL_PROJECT:?set MHRL_PROJECT to the project checkout}
PY="$PROJECT/.venv/bin/python"
ENGINE_PORT=${MHRL_ENGINE_PORT:-8100}
TEACHER_PORT=${MHRL_TEACHER_PORT:-8101}
PROXY_PORT=${MHRL_PROXY_PORT:-8200}
HOST=127.0.0.1
MODEL_NAME=${MHRL_SERVED_MODEL_NAME:-lfm2.5-2.6b}
MAX_OUTPUT_TOKENS=1024 # the project's cap (Settings.max_completion_tokens)
HOME_DIR=${COURSE_HOME:-$HOME/.cache/mhrl-course}/27-how-it-combines
LOGS="$HOME_DIR/stack"
WAIT_S=180
SERVER_NAMES="mhrl.engine.server\|openenv.core.harness.capture.server"

say() { printf '%s\n' "$*"; }

port_busy() { # port -> 0 if something listens on it
  lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

health_has() { # url, key -> 0 if the url answers JSON holding that key
  curl -s -m 2 "$1" 2>/dev/null | "$PY" -c \
    'import json,sys; sys.exit(0 if sys.argv[1] in json.load(sys.stdin) else 1)' \
    "$2" 2>/dev/null
}

wait_for() { # name, url, key
  for _ in $(seq 1 "$WAIT_S"); do
    if health_has "$2" "$3"; then
      say "  $1 is up ($2)"
      return 0
    fi
    sleep 1
  done
  say "  $1 did not come up; see $LOGS/$1.log" >&2
  return 1
}

launch() { # name, port, then the command; runs it from the project, as serve.sh does
  local name=$1
  echo "$2" >"$LOGS/$name.port"
  shift 2
  # nohup execs the command, so $! is the server's own pid: stop kills it.
  (
    cd "$PROJECT"
    nohup "$@" >"$LOGS/$name.log" 2>&1 &
    echo $! >"$LOGS/$name.pid"
  )
}

refuse_busy() { # ports...
  local port
  for port in "$@"; do
    if port_busy "$port"; then
      say "port $port is already in use; choose another (MHRL_*_PORT)" >&2
      exit 1
    fi
  done
}

start_proxy() { # the engine port it forwards to
  launch proxy "$PROXY_PORT" "$PY" -m openenv.core.harness.capture.server \
    --llm-url "http://$HOST:$1" --model "$MODEL_NAME" \
    --port "$PROXY_PORT" --max-output-tokens "$MAX_OUTPUT_TOKENS"
  wait_for proxy "http://$HOST:$PROXY_PORT/health" capture_level
  grep -m1 "capture level" "$LOGS/proxy.log" || true
}

start() {
  [ -x "$PY" ] || { say "no $PY: run uv sync in the project" >&2; exit 1; }
  refuse_busy "$ENGINE_PORT" "$TEACHER_PORT" "$PROXY_PORT"
  mkdir -p "$LOGS"
  say "starting the project's servers (engine $ENGINE_PORT, teacher $TEACHER_PORT, proxy $PROXY_PORT)..."
  launch engine "$ENGINE_PORT" "$PY" -m mhrl.engine.server --port "$ENGINE_PORT"
  launch teacher "$TEACHER_PORT" "$PY" -m mhrl.engine.server --port "$TEACHER_PORT" --oracle
  wait_for engine "http://$HOST:$ENGINE_PORT/health" adapter
  wait_for teacher "http://$HOST:$TEACHER_PORT/health" adapter
  start_proxy "$ENGINE_PORT"
}

scripted() {
  [ -x "$PY" ] || { say "no $PY: run uv sync in the project" >&2; exit 1; }
  refuse_busy "$ENGINE_PORT" "$PROXY_PORT"
  mkdir -p "$LOGS"
  say "starting the answer-key teacher as the engine (no model weights) on $ENGINE_PORT, proxy $PROXY_PORT..."
  launch engine "$ENGINE_PORT" "$PY" -m mhrl.engine.server --port "$ENGINE_PORT" --oracle
  wait_for engine "http://$HOST:$ENGINE_PORT/health" adapter
  start_proxy "$ENGINE_PORT"
}

stop() {
  local name pid
  for name in proxy teacher engine; do
    [ -f "$LOGS/$name.pid" ] || continue
    pid=$(cat "$LOGS/$name.pid")
    # Only a server this script started: a reused pid belongs to someone else.
    if ps -p "$pid" -o command= 2>/dev/null | grep -q -e "$SERVER_NAMES"; then
      kill "$pid"
      for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      say "stopped $name (pid $pid)"
    fi
    rm -f "$LOGS/$name.pid" "$LOGS/$name.port"
    if [ -f "$LOGS/$name.log" ]; then
      tail -n 3 "$LOGS/$name.log" | sed "s/^/  $name: /"
    fi
  done
}

status() { # only the servers this script started
  local name port
  for name in engine teacher proxy; do
    [ -f "$LOGS/$name.port" ] || continue
    port=$(cat "$LOGS/$name.port")
    printf '%-8s %-6s ' "$name" "$port"
    curl -s -m 2 "http://$HOST:$port/health" | head -c 160 || printf 'not answering'
    echo
  done
}

case "${1:-start}" in
  start) start ;;
  scripted) scripted ;;
  stop) stop ;;
  status) status ;;
  *) say "usage: $0 [start|scripted|stop|status]"; exit 2 ;;
esac
