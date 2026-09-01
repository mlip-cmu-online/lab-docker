#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'Docker preflight: FAIL - %s\n' "$1" >&2
  exit 1
}

command -v docker >/dev/null 2>&1 \
  || fail "the Docker CLI is not installed; use the repository Codespace or a native local Docker installation"
docker info >/dev/null 2>&1 \
  || fail "the Docker daemon is not reachable; wait for the Codespace to finish starting, then rerun this script"
docker compose version >/dev/null 2>&1 \
  || fail "Docker Compose v2 is not available"

test_dir="$(mktemp -d "$PWD/.mlip-docker-preflight.XXXXXX")"
volume_name="mlip_docker_preflight_${RANDOM}_$$"

cleanup() {
  docker volume rm -f "$volume_name" >/dev/null 2>&1 || true
  rm -f "$test_dir/from-container"
  rmdir "$test_dir" 2>/dev/null || true
}
trap cleanup EXIT

docker volume create "$volume_name" >/dev/null
docker run --rm \
  --mount "type=volume,source=$volume_name,target=/volume" \
  alpine:3.20 sh -c 'printf named-volume-ok > /volume/check'

named_value="$(docker run --rm \
  --mount "type=volume,source=$volume_name,target=/volume,readonly" \
  alpine:3.20 cat /volume/check)"
[[ "$named_value" == "named-volume-ok" ]] \
  || fail "a second container could not read data from a named volume"

docker run --rm \
  --mount "type=bind,source=$test_dir,target=/bind" \
  alpine:3.20 sh -c 'printf bind-mount-ok > /bind/from-container'
[[ "$(<"$test_dir/from-container")" == "bind-mount-ok" ]] \
  || fail "a container write did not appear through a host bind mount"

if docker ps --format '{{.Ports}}' | grep -Eq '(^|[^0-9])8081->|:8081->'; then
  fail "host port 8081 is already published by another container"
fi

printf '%s\n' \
  'Docker preflight: PASS' \
  '- Docker daemon and Compose v2 are available.' \
  '- Named-volume data survived its creating container.' \
  '- A bind-mounted file appeared in the host workspace.' \
  '- Host port 8081 is available.'
