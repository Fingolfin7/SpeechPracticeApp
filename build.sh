#!/usr/bin/env bash
set -o errexit

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate --noinput
# One-time ladder-progress backfill: no-ops once progress rows exist, so it is
# safe (and nearly free) on every deploy.
python manage.py backfill_ladder_progress --if-empty
