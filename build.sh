#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate --noinput
# TEMPORARY (2026-07-09): one-time rebuild of ladder gate progress from historical
# scoring jobs after the core-loop deploy. Idempotent, but remove this line once
# the first deploy with it has gone out.
python manage.py backfill_ladder_progress
