# Deployment Readiness Checklist

Target: Render free web service + threaded scoring + Neon Postgres + private AWS S3 media.

Current live target:

- Render service: `speechpractice-web` on the free instance type.
- Public URL: `https://speechpractice-web.onrender.com`
- Neon project: `SpeechPractice` in AWS Europe Central 1 (Frankfurt).
- S3 bucket: `speechpractice-audio-978092319358` in `eu-north-1`.
- IAM app user: `speechpractice-render` with bucket-scoped S3 access.

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
- [x] Existing local audio has been copied to S3 and its database references updated.
  - `migrate_audio_storage` copied 73 unique local audio refs to `recordings/legacy/` in S3.
- [ ] Resolve or accept the seven missing legacy audio references found by the migration.
  - These were already missing locally before upload, so their old scoring-job references remain unresolved.
- [x] Existing SQLite application data has been imported into Neon.
  - Loaded 1,805 practice objects from the local SQLite fixture. Local encrypted `PracticeSettings` secrets were intentionally excluded.

## Infrastructure

- [x] Create a Neon project and copy its pooled `DATABASE_URL`.
- [x] Create a private S3 bucket in the chosen region.
- [x] Create an IAM user/policy limited to that bucket.
- [x] Create the free Render Blueprint from `render.yaml`.
- [x] Populate deployment-required `sync: false` environment variables for database and S3.
- [x] Add `OPENAI_API_KEY` to Render or configure the production account with a per-user OpenAI key.
  - Manually added in Render, then verified by a live OpenAI scoring job.
- [x] Confirm the web service uses the generated `SECRET_KEY` environment group.
- [x] Confirm the Render service deploys successfully and reaches the live state.
- [x] Create the production superuser against the Neon database (`kuda` / `mushunjek@gmail.com`).
  - Render Shell is not available on free instances, so this was run from the local repo with the production `DATABASE_URL`.

## Verification

- [x] `python manage.py makemigrations --check --dry-run`
- [x] `python manage.py migrate --noinput` against a fresh database.
- [x] `python manage.py collectstatic --no-input`
- [x] `python manage.py check --deploy` with production variables (only the intentionally deferred HSTS preload/subdomain warnings remain).
- [x] Full automated test suite passes (80 tests).
- [x] Render build installs dependencies, collects static files, and runs migrations against Neon.
- [x] Render health check returns `200` for `/healthz/`.
- [x] Public root URL loads the SpeechPractice sign-in page.
- [x] Run a post-migration live DB/S3 verification pass.
  - Live sessions page returned 70 sessions; imported legacy session audio returned `206 Partial Content`.
  - Repeat with `deployment/live_migration_verify.py`.
- [x] Upload and score a browser-recorded WebM file.
  - Live job `/jobs/22/status/` succeeded with OpenAI transcription and scoring.
  - Repeat with `deployment/live_smoke_test.py`.
- [x] Retry a submission with the same idempotency key; only one job exists.
- [x] Seek through storage-backed audio and receive `206 Partial Content`.
- [x] Verify queued-worker behavior remains covered for a future paid worker.
- [x] Verify the 390x844 mobile layout has no horizontal overflow or browser errors.
- [ ] Complete a live throttled-network upload and OpenAI scoring run.
  - Normal live upload/scoring is verified; explicit browser/network throttling is still pending.
