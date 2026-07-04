from __future__ import annotations

import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken
from django.core.management.base import BaseCommand

from practice.models import PracticeSettings, _fernet

# The stored secrets (API keys, Codex token bundles) are encrypted with a Fernet
# key derived from settings.SECRET_KEY. If SECRET_KEY changes (e.g. it was saved
# under the dev fallback key and later a real SECRET_KEY/DJANGO_SECRET_KEY was set),
# every secret silently fails to decrypt and looks "unset". This command decrypts
# each secret with a legacy key and re-encrypts it under the current SECRET_KEY,
# so users don't have to re-enter their credentials.
#
# It never prints secret values.

DEV_SECRET_KEY = "dev-only-speechpractice-secret-key-change-before-deploy"

SECRET_FIELDS = (
    "autumn_token",
    "openai_api_key",
    "anthropic_api_key",
    "codex_token_bundle",
)

_ENC_ATTR = {
    "autumn_token": "autumn_token_enc",
    "openai_api_key": "openai_api_key_enc",
    "anthropic_api_key": "anthropic_api_key_enc",
    "codex_token_bundle": "codex_token_bundle_enc",
}


def _fernet_for(secret_key: str) -> Fernet:
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


class Command(BaseCommand):
    help = (
        "Re-encrypt stored PracticeSettings secrets (API keys, Codex tokens) under "
        "the current SECRET_KEY, recovering values encrypted with a previous key."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--old-key",
            action="append",
            default=[],
            dest="old_keys",
            help="A legacy SECRET_KEY value to try when decrypting. May be repeated. "
            "The dev fallback key is always tried.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Report what would change without writing to the database.",
        )

    def handle(self, *args, **options):
        dry_run = bool(options["dry_run"])
        # Always include the dev fallback key; de-dup while preserving order.
        candidates: list[str] = []
        for key in [*options["old_keys"], DEV_SECRET_KEY]:
            if key and key not in candidates:
                candidates.append(key)
        legacy_fernets = [_fernet_for(key) for key in candidates]
        current = _fernet()

        recovered = 0
        for row in PracticeSettings.objects.all():
            changed_fields: list[str] = []
            for field in SECRET_FIELDS:
                attr = _ENC_ATTR[field]
                data = getattr(row, attr)
                if not data:
                    continue  # nothing stored
                data = bytes(data)
                # Already decryptable under the current key? Leave it alone.
                try:
                    current.decrypt(data)
                    continue
                except InvalidToken:
                    pass
                # Try each legacy key.
                plaintext = None
                for fernet in legacy_fernets:
                    try:
                        plaintext = fernet.decrypt(data).decode("utf-8")
                        break
                    except InvalidToken:
                        continue
                if plaintext is None:
                    self.stderr.write(
                        f"user={row.user_id} field={field}: UNRECOVERABLE "
                        f"(no candidate key decrypts it)"
                    )
                    continue
                if not dry_run:
                    setattr(row, attr, current.encrypt(plaintext.encode("utf-8")))
                changed_fields.append(field)
                recovered += 1
            if changed_fields and not dry_run:
                row.save(update_fields=[_ENC_ATTR[f] for f in changed_fields] + ["updated_at"])
            if changed_fields:
                verb = "would re-encrypt" if dry_run else "re-encrypted"
                self.stdout.write(
                    f"user={row.user_id}: {verb} {', '.join(changed_fields)}"
                )

        summary = f"{'Would recover' if dry_run else 'Recovered'} {recovered} secret field(s)."
        self.stdout.write(self.style.SUCCESS(summary))
