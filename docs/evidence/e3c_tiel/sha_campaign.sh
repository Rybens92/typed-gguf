#!/bin/bash
# E3c-Tiel: SHA-256 of both pinned models, before the campaign.
set -u
cd /var/home/rybens/workspace/ggufone || exit 126
T=/var/home/rybens/.hermes/models/Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
O=/var/home/rybens/.hermes/models/Accio-Lab_occamy-1.0-Q4_K_L.gguf
{
  echo "== before: $(date -Iseconds)"
  echo "-- tiel   $(stat -c '%s %Y' "$T")"
  sha256sum "$T"
  echo "-- occamy $(stat -c '%s %Y' "$O")"
  sha256sum "$O"
  echo "== done: $(date -Iseconds)"
} > .e3c_tiel/sha256_before.txt 2>&1
echo "exit=$?"
