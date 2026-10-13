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
# Logs and pid files go to $COURSE_HOME/27-how-it-combines/stack/. The proxy's
# log names every session in its URLs, and a session id is a harness's API
# key, so stop prints the logs' tails with session ids redacted (the files
# themselves keep them).
set -euo pipefail

PROJECT=${MHRL_PROJECT:?set MHRL_PROJECT to the project checkout}
PY="$PROJECT/.venv/bin/python"
ENGINE_PORT=${MHRL_ENGINE_PORT:-8100}
TEACHER_PORT=${MHRL_TEACHER_PORT:-8101}
PROXY_PORT=${MHRL_PROXY_PORT:-8200}
HOST=127.0.0.1
MODEL_NAME=${MHRL_SERVED_MODEL_NAME:-lfm2.5-2.6b}
MAX_OUTPUT_TOKENS=1024 # the project's cap (Settings.max_completion_tokens)
COURSE_HOME_DIR=${COURSE_HOME:-$HOME/.cache/mhrl-course}
COURSE_HOME_DIR=${COURSE_HOME_DIR/#\~/$HOME} # a "~" no shell expanded
LOGS="$COURSE_HOME_DIR/27-how-it-combines/stack"
WAIT_S=180
STOP_WAIT_S=30
TAIL_LINES=3
# The proxy's access log names each session in its URLs (/sessions/<id>/...),
# and a session id is a harness's API key: never print one.
REDACT_SESSIONS='s#/sessions/[^/ ?"]+#/sessions/<redacted>#g'

say() { printf '%s\n' "$*"; }

port_busy() { # port -> 0 if something listens on it (lsof, then a connect)
  if lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; then
    return 0
  fi
  # lsof does not see other users' listeners; a connect does.
  "$PY" -c 'import socket, sys
s = socket.socket()
s.settimeout(1)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)' "$1"
}

health_has() { # url, key -> 0 if the url answers JSON holding that key
  curl -s -m 2 "$1" 2>/dev/null | "$PY" -c \
    'import json,sys; sys.exit(0 if sys.argv[1] in json.load(sys.stdin) else 1)' \
    "$2" 2>/dev/null
}

running() { # pid -> 0 if it runs (a zombie, exited but not yet reaped, does not)
  local state
  state=$(ps -p "$1" -o stat= 2>/dev/null) && [[ $state != Z* ]]
}

alive() { # name -> 0 if the server this script started under that name runs
  [ -f "$LOGS/$1.pid" ] && running "$(cat "$LOGS/$1.pid")"
}

started_at() { # pid -> its start time, as ps prints it
  ps -p "$1" -o lstart= 2>/dev/null || true
}

wait_for() { # name, url, key
  for _ in $(seq 1 "$WAIT_S"); do
    if health_has "$2" "$3"; then
      say "  $1 is up ($2)"
      return 0
    fi
    if ! alive "$1"; then
      say "  $1 exited; its log ends:" >&2
      tail -n "$TAIL_LINES" "$LOGS/$1.log" | sed -E "$REDACT_SESSIONS" >&2
      return 1
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
  # Its start time too: a later process given the same pid starts later.
  started_at "$(cat "$LOGS/$name.pid")" >"$LOGS/$name.start"
}

refuse_busy() { # ports...
  local port name
  for name in engine teacher proxy; do
    if [ -f "$LOGS/$name.pid" ] && ours "$name" "$(cat "$LOGS/$name.pid")"; then
      say "this script's $name is still running; ./stack.sh stop first" >&2
      exit 1
    fi
  done
  for port in "$@"; do
    if port_busy "$port"; then
      say "port $port is already in use; choose another (MHRL_*_PORT)" >&2
      exit 1
    fi
  done
}

failed() { # stop whatever did start, and fail
  say "a server did not start; stopping the ones that did" >&2
  stop
  exit 1
}

start_proxy() { # the engine port it forwards to
  launch proxy "$PROXY_PORT" "$PY" -m openenv.core.harness.capture.server \
    --llm-url "http://$HOST:$1" --model "$MODEL_NAME" \
    --port "$PROXY_PORT" --max-output-tokens "$MAX_OUTPUT_TOKENS"
  wait_for proxy "http://$HOST:$PROXY_PORT/health" capture_level || failed
  grep -m1 "capture level" "$LOGS/proxy.log" || true
}

start() {
  [ -x "$PY" ] || { say "no $PY: run uv sync in the project" >&2; exit 1; }
  refuse_busy "$ENGINE_PORT" "$TEACHER_PORT" "$PROXY_PORT"
  mkdir -p "$LOGS"
  say "starting the project's servers (engine $ENGINE_PORT, teacher $TEACHER_PORT, proxy $PROXY_PORT)..."
  launch engine "$ENGINE_PORT" "$PY" -m mhrl.engine.server --port "$ENGINE_PORT"
  launch teacher "$TEACHER_PORT" "$PY" -m mhrl.engine.server --port "$TEACHER_PORT" --oracle
  wait_for engine "http://$HOST:$ENGINE_PORT/health" adapter || failed
  wait_for teacher "http://$HOST:$TEACHER_PORT/health" adapter || failed
  start_proxy "$ENGINE_PORT"
}

scripted() {
  [ -x "$PY" ] || { say "no $PY: run uv sync in the project" >&2; exit 1; }
  refuse_busy "$ENGINE_PORT" "$PROXY_PORT"
  mkdir -p "$LOGS"
  say "starting the answer-key teacher as the engine (no model weights) on $ENGINE_PORT, proxy $PROXY_PORT..."
  launch engine "$ENGINE_PORT" "$PY" -m mhrl.engine.server --port "$ENGINE_PORT" --oracle
  wait_for engine "http://$HOST:$ENGINE_PORT/health" adapter || failed
  start_proxy "$ENGINE_PORT"
}

ours() { # name, pid -> 0 only if that pid is the server this script started
  local port start command
  port=$(cat "$LOGS/$1.port" 2>/dev/null || true)
  start=$(cat "$LOGS/$1.start" 2>/dev/null || true)
  command=$(ps -p "$2" -o command= 2>/dev/null || true)
  # Same start time (a reused pid starts later) and its own --port.
  [ -n "$port" ] && [ -n "$start" ] && [ "$(started_at "$2")" = "$start" ] &&
    [[ $command == *" -m "*"--port $port "* || $command == *" -m "*"--port $port" ]]
}

stop() {
  local name pid
  for name in proxy teacher engine; do
    [ -f "$LOGS/$name.pid" ] || continue
    pid=$(cat "$LOGS/$name.pid")
    if ours "$name" "$pid"; then
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 "$STOP_WAIT_S"); do running "$pid" || break; sleep 1; done
      if running "$pid"; then
        kill -9 "$pid" 2>/dev/null || true
        say "killed $name (pid $pid): it did not stop in $STOP_WAIT_S s"
      else
        say "stopped $name (pid $pid)"
      fi
    fi
    rm -f "$LOGS/$name.pid" "$LOGS/$name.port" "$LOGS/$name.start"
    if [ -f "$LOGS/$name.log" ]; then
      tail -n "$TAIL_LINES" "$LOGS/$name.log" | sed -E "$REDACT_SESSIONS" | sed "s/^/  $name: /"
    fi
  done
}

status() { # only the servers this script started
  local name port answer
  for name in engine teacher proxy; do
    [ -f "$LOGS/$name.port" ] || continue
    port=$(cat "$LOGS/$name.port")
    if answer=$(curl -s -m 2 "http://$HOST:$port/health"); then
      printf '%-8s %-6s %s\n' "$name" "$port" "${answer:0:160}"
    else
      printf '%-8s %-6s not answering\n' "$name" "$port"
    fi
  done
}

case "${1:-start}" in
  start) start ;;
  scripted) scripted ;;
  stop) stop ;;
  status) status ;;
  *) say "usage: $0 [start|scripted|stop|status]"; exit 2 ;;
esac
