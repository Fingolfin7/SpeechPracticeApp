# Deployment Readiness Checklist

Target: Render free web service + threaded scoring + Neon Postgres + private AWS S3 media.

Note: Render's free instance type is available for web services, but not for
background workers. This free configuration runs scoring in a background thread
inside the web process. Upgrade to a paid worker service when you want more
durable production job processing.

## Application

- [x] Production configuration is environment-driven.
- [x] OpenAI `whisper-1` is the production transcription default.
- [x] Normal recordings under 24 MB bypass local ffmpeg decoding.
- [x] Static assets are collected and served through WhiteNoise.
- [x] Audio uses Django storage and supports private S3.
- [x] Playback supports byte-range requests for mobile seeking.
- [x] Duplicate scoring submissions are protected with idempotency keys.
- [x] Production requests require Django authentication.
- [x] Free deploy uses threaded web-process scoring.
- [x] App still includes a long-running worker command for a future paid worker.
- [ ] Existing local audio has been copied to S3 and its database references updated.
- [ ] Resolve or accept the seven missing legacy audio references found by the current dry run (73 files are copyable).
- [ ] Existing SQLite application data has been imported into Neon.

## Infrastructure

- [ ] Create a Neon project and copy its pooled `DATABASE_URL`.
- [ ] Create a private S3 bucket in the chosen region.
- [ ] Create an IAM user/policy limited to that bucket.
- [ ] Create the free Render Blueprint from `render.yaml`.
- [ ] Populate every `sync: false` environment variable.
- [ ] Confirm the web service uses the generated `SECRET_KEY` environment group.
- [ ] Run `python manage.py createsuperuser` from the Render shell.

## Verification

- [x] `python manage.py makemigrations --check --dry-run`
- [x] `python manage.py migrate --noinput` against a fresh database.
- [x] `python manage.py collectstatic --no-input`
- [x] `python manage.py check --deploy` with production variables (only the intentionally deferred HSTS preload/subdomain warnings remain).
- [x] Full automated test suite passes (78 tests).
- [ ] Upload and score a browser-recorded WebM file.
- [x] Retry a submission with the same idempotency key; only one job exists.
- [x] Seek through storage-backed audio and receive `206 Partial Content`.
- [x] Verify queued-worker behavior remains covered for a future paid worker.
- [x] Verify the 390x844 mobile layout has no horizontal overflow or browser errors.
- [ ] Complete a live throttled-network upload and OpenAI scoring run after external services exist.
