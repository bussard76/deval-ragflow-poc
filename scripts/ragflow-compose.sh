#!/usr/bin/env bash
# Stage and run the official RAGFlow v0.27.1 compose directory with a generated runtime env.
set -euo pipefail

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
STATE_DIR=${RAGFLOW_COMPOSE_STATE_DIR:-"$ROOT/.data/ragflow-v0.27.1"}
SOURCE_DIR="$STATE_DIR/source"
DOCKER_DIR="$STATE_DIR/docker"
ENV_FILE="$DOCKER_DIR/.env"
UPSTREAM=https://github.com/infiniflow/ragflow.git
COMMIT=b9df87c4c75a5b0d35c90d15329fc0f6f91cb73e
IMAGE=infiniflow/ragflow:v0.27.1

fail() {
  printf 'ragflow-compose: %s\n' "$*" >&2
  exit 1
}
need() { command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"; }

stage() {
  need git
  mkdir -p "$STATE_DIR"
  local canonical_state
  canonical_state=$(CDPATH= cd -P "$STATE_DIR" && pwd)
  [ "$canonical_state" != "/" ] || fail "RAGFLOW_COMPOSE_STATE_DIR must not be the filesystem root"
  [ "$canonical_state" != "$ROOT" ] || fail "RAGFLOW_COMPOSE_STATE_DIR must not be the repository root"
  if [ ! -d "$SOURCE_DIR/.git" ]; then
    mkdir -p "$SOURCE_DIR"
    git -C "$SOURCE_DIR" init -q
    git -C "$SOURCE_DIR" remote add origin "$UPSTREAM"
  fi
  git -C "$SOURCE_DIR" fetch --depth=1 origin "$COMMIT" >/dev/null
  git -C "$SOURCE_DIR" checkout -q --detach "$COMMIT"
  [ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$COMMIT" ] || fail "upstream checkout is not the pinned commit"
  local saved_env="$STATE_DIR/.env.runtime.saved"
  if [ -e "$DOCKER_DIR" ] && [ ! -f "$DOCKER_DIR/.deval-staged" ]; then
    fail "refusing to replace an unmanaged directory: $DOCKER_DIR"
  fi
  if [ -f "$ENV_FILE" ]; then cp "$ENV_FILE" "$saved_env"; fi
  rm -rf "$DOCKER_DIR"
  mkdir -p "$DOCKER_DIR"
  git -C "$SOURCE_DIR" archive "$COMMIT" docker | tar -x -C "$STATE_DIR"
  : >"$DOCKER_DIR/.deval-staged"
  [ -f "$DOCKER_DIR/docker-compose.yml" ] || fail "official docker-compose.yml was not staged"
  [ -f "$DOCKER_DIR/docker-compose-base.yml" ] || fail "official docker-compose-base.yml was not staged"
  if [ -f "$saved_env" ]; then
    mv "$saved_env" "$ENV_FILE"
  fi
  if ! grep -Fq '# Generated runtime values;' "$ENV_FILE"; then
    python3 - "$ENV_FILE" <<'PY'
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
values = {
    "DOC_ENGINE": "elasticsearch",
    "DB_TYPE": "mysql",
    "DEVICE": "cpu",
    "METADATA_DB_PROFILE": "mysql",
    "COMPOSE_PROFILES": "elasticsearch,cpu,metadata-mysql,tei-cpu",
    "RAGFLOW_IMAGE": "infiniflow/ragflow:v0.27.1",
    "TEI_MODEL": "BAAI/bge-small-en-v1.5",
    "LLM_TIMEOUT_SECONDS": "360",
    "ALLOW_ANY_HOST": "0",
    # These are host-side published ports; service-to-service ports remain unchanged.
    "ES_PORT": "127.0.0.1:1200",
    "OS_PORT": "127.0.0.1:1201",
    "KIBANA_PORT": "127.0.0.1:6601",
    "EXPOSE_MYSQL_PORT": "127.0.0.1:3306",
    "MINIO_PORT": "127.0.0.1:9000",
    "MINIO_CONSOLE_PORT": "127.0.0.1:9001",
    "REDIS_PORT": "127.0.0.1:6379",
    "SVR_WEB_HTTP_PORT": "127.0.0.1:80",
    "SVR_WEB_HTTPS_PORT": "127.0.0.1:443",
    "SVR_HTTP_PORT": "127.0.0.1:9380",
    "ADMIN_SVR_HTTP_PORT": "127.0.0.1:9381",
    "SVR_MCP_PORT": "127.0.0.1:9382",
    "GO_HTTP_PORT": "127.0.0.1:9384",
    "GO_ADMIN_PORT": "127.0.0.1:9383",
    "INFINITY_THRIFT_PORT": "127.0.0.1:23817",
    "INFINITY_HTTP_PORT": "127.0.0.1:23820",
    "INFINITY_PSQL_PORT": "127.0.0.1:5432",
    "SERENEDB_PORT": "127.0.0.1:7890",
    "OCEANBASE_PORT": "127.0.0.1:2881",
    "SEEKDB_PORT": "127.0.0.1:2881",
    "NATS_PORT": "127.0.0.1:4222",
    "EXPOSE_NATS_PORT": "127.0.0.1:4222",
    "EXPOSE_CLICKHOUSE_TCP_PORT": "127.0.0.1:9900",
    "CLICKHOUSE_HTTP_PORT": "127.0.0.1:8123",
    "JAEGER_OTLP_GRPC_PORT": "127.0.0.1:4317",
    "JAEGER_OTLP_HTTP_PORT": "127.0.0.1:4318",
    "JAEGER_UI_PORT": "127.0.0.1:16686",
    "TEI_PORT": "127.0.0.1:6380",
}
for key in ("ELASTIC_PASSWORD", "OPENSEARCH_PASSWORD", "SERENEDB_PASSWORD", "OCEANBASE_PASSWORD", "SEEKDB_PASSWORD", "MYSQL_PASSWORD", "MYSQL_ROOT_PASSWORD", "MINIO_PASSWORD", "MINIO_ROOT_PASSWORD", "REDIS_PASSWORD", "CLICKHOUSE_PASSWORD"):
    values[key] = secrets.token_hex(24)
# Do not carry any credential-looking defaults from a future upstream .env.
for line in lines:
    if "=" not in line:
        continue
    key, raw_value = line.split("=", 1)
    if raw_value.strip() and any(marker in key.upper() for marker in ("PASSWORD", "SECRET", "TOKEN", "API_KEY")):
        values[key] = secrets.token_hex(24)
seen = set()
out = ["# Generated runtime values; this file is ignored by the PoC repository."]
for line in lines:
    key = line.split("=", 1)[0] if "=" in line else ""
    if key in values:
        out.append(key + "=" + values[key])
        seen.add(key)
    else:
        out.append(line)
for key, value in values.items():
    if key not in seen:
        out.append(key + "=" + value)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
  fi
  # Keep older generated runtime files aligned with the local TEI defaults.
  python3 - "$ENV_FILE" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
wanted = {
    "COMPOSE_PROFILES": "elasticsearch,cpu,metadata-mysql,tei-cpu",
    "TEI_MODEL": "BAAI/bge-small-en-v1.5",
    "LLM_TIMEOUT_SECONDS": "360",
}
lines = path.read_text(encoding="utf-8").splitlines()
out = []
seen = set()
for line in lines:
    key = line.split("=", 1)[0].strip() if "=" in line else ""
    if key in wanted:
        if key not in seen:
            out.append(key + "=" + wanted[key])
            seen.add(key)
        continue
    out.append(line)
for key, value in wanted.items():
    if key not in seen:
        out.append(key + "=" + value)
path.write_text("\n".join(out) + "\n", encoding="utf-8")
PY
  rm -f "$saved_env"
}

compose() {
  (cd "$DOCKER_DIR" &&
    COMPOSE_PROFILES=elasticsearch,cpu,metadata-mysql,tei-cpu \
      RAGFLOW_IMAGE="$IMAGE" DOC_ENGINE=elasticsearch DB_TYPE=mysql DEVICE=cpu METADATA_DB_PROFILE=mysql ALLOW_ANY_HOST=0 \
      ES_PORT=127.0.0.1:1200 OS_PORT=127.0.0.1:1201 KIBANA_PORT=127.0.0.1:6601 \
      EXPOSE_MYSQL_PORT=127.0.0.1:3306 MINIO_PORT=127.0.0.1:9000 MINIO_CONSOLE_PORT=127.0.0.1:9001 \
      REDIS_PORT=127.0.0.1:6379 SVR_WEB_HTTP_PORT=127.0.0.1:80 SVR_WEB_HTTPS_PORT=127.0.0.1:443 \
      SVR_HTTP_PORT=127.0.0.1:9380 ADMIN_SVR_HTTP_PORT=127.0.0.1:9381 SVR_MCP_PORT=127.0.0.1:9382 \
      GO_HTTP_PORT=127.0.0.1:9384 GO_ADMIN_PORT=127.0.0.1:9383 \
      INFINITY_THRIFT_PORT=127.0.0.1:23817 INFINITY_HTTP_PORT=127.0.0.1:23820 INFINITY_PSQL_PORT=127.0.0.1:5432 \
      SERENEDB_PORT=127.0.0.1:7890 OCEANBASE_PORT=127.0.0.1:2881 SEEKDB_PORT=127.0.0.1:2881 \
      NATS_PORT=127.0.0.1:4222 EXPOSE_NATS_PORT=127.0.0.1:4222 EXPOSE_CLICKHOUSE_TCP_PORT=127.0.0.1:9900 \
      CLICKHOUSE_HTTP_PORT=127.0.0.1:8123 JAEGER_OTLP_GRPC_PORT=127.0.0.1:4317 JAEGER_OTLP_HTTP_PORT=127.0.0.1:4318 \
      JAEGER_UI_PORT=127.0.0.1:16686 TEI_PORT=127.0.0.1:6380 TEI_MODEL=BAAI/bge-small-en-v1.5 \
      docker compose --env-file .env -f docker-compose.yml "$@")
}

verified_config() {
  local rendered
  rendered=$(compose config)
  printf '%s\n' "$rendered" | grep -Fq "image: $IMAGE" || fail "compose image is not $IMAGE"
  printf '%s\n' "$rendered" | grep -Eq 'esdata01:|mysql_data:|minio_data:|redis_data:' || fail "official persistent volumes are missing"
  printf '%s\n' "$rendered" | grep -Fq 'tei-cpu:' || fail "TEI CPU embedding service is missing"
  printf '%s\n' "$rendered" | grep -Fq 'BAAI/bge-small-en-v1.5' || fail "TEI model is not the reproducible CPU default"
  local socket_pattern
  socket_pattern='docker.'$(printf 'sock')
  if printf '%s\n' "$rendered" | grep -Eiq "sandbox-executor-manager|${socket_pattern}|privileged"; then
    fail "sandbox, privileged mode, or Docker socket is active"
  fi
  local published_count loopback_count
  published_count=$(printf '%s\n' "$rendered" | grep -c 'published:' || true)
  loopback_count=$(printf '%s\n' "$rendered" | grep -c 'host_ip: 127.0.0.1' || true)
  [ "$published_count" -eq "$loopback_count" ] || fail "every published port must be explicitly loopback-bound"
  printf '%s\n' "$rendered"
}

verify() {
  stage
  need docker
  [ "$(git -C "$SOURCE_DIR" rev-parse HEAD)" = "$COMMIT" ] || fail "wrong upstream commit"
  grep -Eq '^RAGFLOW_IMAGE=infiniflow/ragflow:v0\.27\.1$' "$ENV_FILE" || fail "runtime image tag is wrong"
  verified_config >/dev/null
  printf 'verified official RAGFlow docker directory at %s\n' "$COMMIT"
}

redact_config() {
  # Compose also interpolates credentials into healthcheck/command strings;
  # redact values, not only YAML keys, before anything reaches stdout.
  python3 -c '
import os
import sys
from pathlib import Path

values = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if "=" not in line:
        continue
    key, value = line.split("=", 1)
    if value and any(marker in key.upper() for marker in ("PASSWORD", "SECRET", "TOKEN", "API_KEY", "ACCESS_KEY")):
        values.append(value)
for key, value in os.environ.items():
    if value and any(marker in key.upper() for marker in ("PASSWORD", "SECRET", "TOKEN", "API_KEY", "ACCESS_KEY")):
        values.append(value)
text = sys.stdin.read()
for value in sorted(set(values), key=len, reverse=True):
    if len(value) >= 4:
        text = text.replace(value, "<redacted>")
print(text, end="")
' "$ENV_FILE" | sed -E \
    -e 's/^([[:space:]-]*[A-Za-z0-9_]*(PASSWORD|SECRET|TOKEN|API_KEY|ACCESS_KEY)[^:]*: ).*$/\1<redacted>/I' \
    -e 's/^([[:space:]-]*[A-Za-z0-9_]*(PASSWORD|SECRET|TOKEN|API_KEY|ACCESS_KEY)[[:space:]]*=).*$/\1<redacted>/I'
}

config() {
  stage
  need docker
  verified_config | redact_config
}

up() {
  stage
  need docker
  verified_config >/dev/null
  compose up -d
}
wait_ready() {
  need curl
  local deadline=$(($(date +%s) + ${RAGFLOW_WAIT_SECONDS:-1800}))
  until curl --fail --silent --show-error --connect-timeout 5 --max-time 10 http://127.0.0.1:9380/api/v1/system/healthz >/dev/null; do
    [ "$(date +%s)" -lt "$deadline" ] || fail "RAGFlow healthz did not become ready"
    sleep 5
  done
  printf 'RAGFlow healthz is ready at http://127.0.0.1:9380/api/v1/system/healthz\n'
}
down() {
  stage
  need docker
  compose down --remove-orphans
}
ps() {
  stage
  need docker
  compose ps
}

command=${1:-help}
case "$command" in
verify) verify ;;
config) config ;;
up) up ;;
wait) wait_ready ;;
down) down ;;
ps) ps ;;
logs)
  stage
  need docker
  shift
  compose logs "$@"
  ;;
*)
  printf '%s\n' "usage: $0 {verify|config|up|wait|down|ps|logs [service...]}" >&2
  exit 2
  ;;
esac
