#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

# Expose a local Ollama API through an authenticated zrok public share.
# Defaults can be overridden with environment variables:
#   OLLAMA_URL=http://127.0.0.1:11434
#   ZROK_BASIC_USER=ollama
#   ZROK_API_KEY_LENGTH=48
#   START_OLLAMA=auto|never
#   ZROK_ENABLE_TOKEN=<token>
#   OLLAMA_ZROK_STATE_DIR=<state-and-log-directory>

OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"
BASIC_USER="${ZROK_BASIC_USER:-ollama}"
API_KEY_LENGTH="${ZROK_API_KEY_LENGTH:-48}"
START_OLLAMA="${START_OLLAMA:-auto}"
ZROK_CMD="${ZROK_CMD:-zrok}"
OLLAMA_CMD="${OLLAMA_CMD:-ollama}"
STATE_DIR="${OLLAMA_ZROK_STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/ollama-zrok}"
LOG_DIR="$STATE_DIR/logs"

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 1
}

info() {
  printf '\n[%s] %s\n' "ollama-zrok" "$*" >&2
}

warn() {
  printf '\nWARNING: %s\n' "$*" >&2
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

print_ollama_install_help() {
  cat <<'EOF'

Ollama is not installed or is not on PATH.

Install Ollama:
  Linux:
    curl -fsSL https://ollama.com/install.sh | sh

  macOS:
    brew install ollama
    # or download from https://ollama.com/download

  Windows:
    Download from https://ollama.com/download and run this PowerShell script instead.

After installing, pull at least one model:
  ollama pull gemma3:1b

EOF
}

print_zrok_install_help() {
  cat <<'EOF'

zrok is not installed or is not on PATH.

Install zrok:
  Linux packages:
    curl -sSf https://get.openziti.io/install.bash | sudo bash -s zrok

  macOS Homebrew:
    brew install zrok

  Other installers:
    https://docs.zrok.io/docs/guides/install/

Then create or open a zrok account, copy your enable token, and rerun this script.

EOF
}

ensure_private_dirs() {
  mkdir -p "$LOG_DIR" || die "Cannot create state directory: $LOG_DIR"
  chmod 700 "$STATE_DIR" "$LOG_DIR" 2>/dev/null || true
}

http_ok() {
  local url="${OLLAMA_URL%/}/api/tags"
  if command_exists curl; then
    curl -fsS --max-time 5 "$url" >/dev/null 2>&1
  elif command_exists wget; then
    wget -q -T 5 -O - "$url" >/dev/null 2>&1
  else
    return 2
  fi
}

wait_for_ollama() {
  local timeout_seconds="${1:-30}"
  local start now
  start="$(date +%s)"
  while true; do
    if http_ok; then
      return 0
    fi
    now="$(date +%s)"
    if [ "$((now - start))" -ge "$timeout_seconds" ]; then
      return 1
    fi
    sleep 1
  done
}

ensure_ollama() {
  if ! command_exists "$OLLAMA_CMD"; then
    print_ollama_install_help
    die "Install Ollama, make sure 'ollama' is on PATH, then rerun."
  fi

  if http_ok; then
    info "Ollama is reachable at ${OLLAMA_URL%/}."
    return 0
  fi

  if [ "$START_OLLAMA" = "never" ]; then
    print_ollama_install_help
    die "Ollama is installed but not reachable at ${OLLAMA_URL%/}."
  fi

  info "Ollama is installed but not reachable. Starting 'ollama serve' in the background..."
  local log_file="$LOG_DIR/ollama-serve-$(date +%Y%m%d-%H%M%S).log"
  nohup "$OLLAMA_CMD" serve >"$log_file" 2>&1 &
  local ollama_pid=$!
  printf '%s\n' "$ollama_pid" >"$STATE_DIR/ollama.pid"
  chmod 600 "$STATE_DIR/ollama.pid" 2>/dev/null || true

  if wait_for_ollama 30; then
    info "Ollama started. Log: $log_file"
    return 0
  fi

  warn "Ollama did not become reachable within 30 seconds."
  warn "Ollama log: $log_file"
  if command_exists tail; then
    tail -n 20 "$log_file" >&2 || true
  fi
  die "Start Ollama manually with 'ollama serve', verify ${OLLAMA_URL%/}/api/tags, then rerun."
}

zrok_status_enabled() {
  local output status_file
  status_file="$STATE_DIR/zrok-status.txt"
  if "$ZROK_CMD" status >"$status_file" 2>&1; then
    output="$(cat "$status_file")"
    printf '%s\n' "$output" | grep -Eiq 'Account Token[[:space:]]+<<SET>>|Secret Token[[:space:]]+<<SET>>|Ziti Identity[[:space:]]+<<SET>>'
  else
    return 1
  fi
}

ensure_zrok() {
  if ! command_exists "$ZROK_CMD"; then
    print_zrok_install_help
    die "Install zrok, make sure 'zrok' is on PATH, then rerun."
  fi

  if zrok_status_enabled; then
    info "zrok environment is enabled."
    return 0
  fi

  info "zrok is installed, but this shell is not enabled yet."

  local token
  token="${ZROK_ENABLE_TOKEN:-}"
  if [ -z "$token" ]; then
    printf 'Paste your zrok enable token. Input is hidden: '
    stty -echo 2>/dev/null || true
    read -r token
    stty echo 2>/dev/null || true
    printf '\n'
  else
    info "Using zrok enable token from ZROK_ENABLE_TOKEN."
  fi

  if [ -z "$token" ]; then
    die "No zrok enable token provided."
  fi

  "$ZROK_CMD" enable "$token" >/dev/null
  unset token

  if ! zrok_status_enabled; then
    die "zrok enable finished, but 'zrok status' does not look enabled."
  fi
  info "zrok environment enabled."
}

generate_api_key() {
  local key="" chunk
  while [ "${#key}" -lt "$API_KEY_LENGTH" ]; do
    if command_exists openssl; then
      chunk="$(openssl rand -hex 32)"
    elif command_exists od && [ -r /dev/urandom ]; then
      chunk="$(od -An -N64 -tx1 /dev/urandom | tr -d ' \n')"
    else
      die "Could not generate a strong key. Install openssl or use a system with /dev/urandom."
    fi
    key="${key}${chunk}"
  done
  printf '%s' "${key:0:$API_KEY_LENGTH}"
}

extract_first_url() {
  local file="$1"
  grep -Eo 'https?://[^[:space:]")<>]+' "$file" 2>/dev/null | head -n 1 || true
}

start_zrok_share() {
  local api_key="$1"
  local log_file="$LOG_DIR/zrok-share-$(date +%Y%m%d-%H%M%S).log"
  local err_file="$LOG_DIR/zrok-share-$(date +%Y%m%d-%H%M%S).err.log"

  info "Starting zrok tunnel to ${OLLAMA_URL%/}..."
  nohup "$ZROK_CMD" share public "${OLLAMA_URL%/}" \
    --backend-mode proxy \
    --headless \
    --basic-auth "${BASIC_USER}:${api_key}" \
    >"$log_file" 2>"$err_file" &

  local zrok_pid=$!
  printf '%s\n' "$zrok_pid" >"$STATE_DIR/zrok.pid"
  chmod 600 "$STATE_DIR/zrok.pid" 2>/dev/null || true

  local public_url="" combined_log="$LOG_DIR/zrok-share-combined-${zrok_pid}.log"
  local waited=0
  while [ "$waited" -lt 60 ]; do
    cat "$log_file" "$err_file" >"$combined_log" 2>/dev/null || true
    public_url="$(extract_first_url "$combined_log")"
    if [ -n "$public_url" ]; then
      printf '%s\n' "$public_url" >"$STATE_DIR/zrok.url"
      chmod 600 "$STATE_DIR/zrok.url" 2>/dev/null || true
      printf '%s|%s|%s\n' "$public_url" "$zrok_pid" "$combined_log"
      return 0
    fi

    if ! kill -0 "$zrok_pid" >/dev/null 2>&1; then
      warn "zrok exited before a public URL was printed."
      cat "$combined_log" >&2 || true
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done

  warn "Timed out waiting for zrok to print a public URL."
  warn "zrok is still running as PID $zrok_pid. Log: $combined_log"
  return 1
}

print_success() {
  local public_url="$1"
  local api_key="$2"
  local zrok_pid="$3"
  local log_file="$4"

  cat <<EOF

========================================================================
  LOCAL OLLAMA API IS LIVE THROUGH ZROK
========================================================================

Public URL:
  $public_url

Basic Auth username:
  $BASIC_USER

API Key / Basic Auth password:
  $api_key

List installed models:
  curl -u "$BASIC_USER:$api_key" "$public_url/api/tags"

Sample generate request:
  curl -u "$BASIC_USER:$api_key" "$public_url/api/generate" \\
    -H "Content-Type: application/json" \\
    -d '{"model":"gemma3:1b","prompt":"Say hello from Ollama through zrok.","stream":false}'

Tunnel process:
  PID $zrok_pid

Stop the tunnel:
  kill $zrok_pid

Logs:
  $log_file

Security notes:
  - Do not paste this API key into public source code.
  - Anyone with the URL and API key can use your local models while the
    tunnel is running.
  - On shared machines, local admins or same-user processes may be able to
    inspect running command arguments. Use a host account you control.

========================================================================

EOF
}

main() {
  case "$START_OLLAMA" in
    auto|never) ;;
    *) die "START_OLLAMA must be 'auto' or 'never'." ;;
  esac

  case "$API_KEY_LENGTH" in
    ''|*[!0-9]*) die "ZROK_API_KEY_LENGTH must be a number." ;;
  esac
  if [ "$API_KEY_LENGTH" -lt 32 ]; then
    die "ZROK_API_KEY_LENGTH must be at least 32."
  fi

  ensure_private_dirs
  ensure_ollama
  ensure_zrok

  local api_key result public_url zrok_pid log_file
  api_key="$(generate_api_key)"
  result="$(start_zrok_share "$api_key")" || die "Unable to start zrok tunnel."
  public_url="${result%%|*}"
  result="${result#*|}"
  zrok_pid="${result%%|*}"
  log_file="${result#*|}"

  print_success "$public_url" "$api_key" "$zrok_pid" "$log_file"
}

main "$@"
