#!/bin/bash

set -u

source .venv/bin/activate

failed_commands=0

while IFS= read -r command; do
  if [[ -z "$command" ]]; then
    continue
  fi

  printf 'Running: %s\n' "$command"
  if ! eval "$command"; then
    printf 'Command failed: %s\n' "$command" >&2
    failed_commands=$((failed_commands + 1))
  fi
done < <(python -u ipes_fixed_cmd.py --print_cmd_only)

if [[ "$failed_commands" -ne 0 ]]; then
  printf '%s command(s) failed.\n' "$failed_commands" >&2
  exit 1
fi