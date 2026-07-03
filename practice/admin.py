from __future__ import annotations

from django.contrib import admin

from .models import GeneratedPracticeScript, ImprovementCard, PracticeReview, PracticeScript, ScoringJob


@admin.register(PracticeScript)
class PracticeScriptAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "source", "author", "difficulty", "active", "updated_at")
    list_filter = ("source", "active", "difficulty", "user")
    search_fields = ("title", "body", "author", "tags", "user__username")


@admin.register(ImprovementCard)
class ImprovementCardAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "kind", "status", "mastery", "due_at", "updated_at")
    list_filter = ("kind", "status", "user")
    search_fields = ("title", "target_key", "prompt", "user__username")


@admin.register(PracticeReview)
class PracticeReviewAdmin(admin.ModelAdmin):
    list_display = ("card", "user", "score", "error_rate", "reviewed_at")
    list_filter = ("reviewed_at", "user")


@admin.register(GeneratedPracticeScript)
class GeneratedPracticeScriptAdmin(admin.ModelAdmin):
    list_display = ("card", "user", "model_provider", "created_at")
    list_filter = ("model_provider", "created_at", "user")


@admin.register(ScoringJob)
class ScoringJobAdmin(admin.ModelAdmin):
    list_display = ("script_name", "user", "card", "provider", "status", "legacy_session_id", "created_at", "finished_at")
    list_filter = ("status", "provider", "created_at", "user")
    search_fields = ("script_name", "audio_path", "error_message", "user__username")
