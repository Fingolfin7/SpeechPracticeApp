#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate --noinput
if [ -n "${SPEECHPRACTICE_OWNER_USERNAME:-}" ]; then
  CLAIM_ARGS=(--username "${SPEECHPRACTICE_OWNER_USERNAME}" --apply --delete-empty-placeholder)
  if [ "${SPEECHPRACTICE_CLAIM_ALL_USERS:-0}" = "1" ]; then
    CLAIM_ARGS+=(--all-users)
  fi
  if [ "${SPEECHPRACTICE_REPLACE_SETTINGS:-0}" = "1" ]; then
    CLAIM_ARGS+=(--replace-settings)
  fi
  python manage.py claim_existing_data "${CLAIM_ARGS[@]}"
fi
