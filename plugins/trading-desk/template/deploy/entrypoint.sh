#!/usr/bin/env bash
# Thin launcher so docker-compose can start any bot by name without a custom
# image per service. Restarts on crash are handled by compose, not here: if a
# bot dies we want the container to exit so the restart policy is visible in
# `docker ps`, rather than a silent internal loop hiding a broken bot.
set -euo pipefail
case "${BOT:-}" in
  orb)       exec python3 -u /app/orb/live.py "$@" ;;
  keepalive) exec python3 -u /app/deploy/keepalive.py ;;
  *) echo "set BOT to one of: orb keepalive"; exit 64 ;;
esac
