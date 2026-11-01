from __future__ import annotations

from django.urls import path

from . import views, native_audio_views

app_name = "practice"

urlpatterns = [
    path("experimental/audio/", native_audio_views.index, name="native_audio"),
    path("experimental/audio/create/", native_audio_views.create, name="native_audio_create"),
    path("experimental/audio/<int:pk>/", native_audio_views.detail, name="native_audio_take"),
    path("experimental/audio/<int:pk>/assess/", native_audio_views.assess, name="native_audio_assess"),
    path("experimental/audio/<int:pk>/play/", native_audio_views.audio, name="native_audio_play"),
    path("experimental/audio/<int:pk>/recover/", native_audio_views.recover, name="native_audio_recover"),
    path("experimental/audio/<int:pk>/delete/", native_audio_views.delete, name="native_audio_delete"),
    path("healthz/", views.health, name="health"),
    path("accounts/signup/", views.signup, name="signup"),
    path("", views.dashboard, name="dashboard"),
    path("progress/", views.progress, name="progress"),
    path("practice/", views.practice_run, name="practice"),
    path("jobs/<int:pk>/", views.scoring_job_detail, name="scoring_job"),
    path("jobs/<int:pk>/status/", views.scoring_job_status, name="scoring_job_status"),
    path("jobs/<int:pk>/retry/", views.retry_scoring_job, name="retry_scoring_job"),
    path("sessions/", views.session_list, name="sessions"),
    path("sessions/<int:pk>/", views.session_detail, name="session_detail"),
    path("sessions/<int:pk>/audio/", views.session_audio, name="session_audio"),
    path("sessions/<int:pk>/report.md", views.session_report, name="session_report"),
    path("sessions/<int:pk>/delete/", views.session_delete, name="session_delete"),
    path("sessions/bulk-delete/", views.session_bulk_delete, name="session_bulk_delete"),
    path("scripts/", views.script_list, name="scripts"),
    path("scripts/new/", views.script_create, name="script_create"),
    path("scripts/import/", views.script_import, name="script_import"),
    path("scripts/<int:pk>/edit/", views.script_edit, name="script_edit"),
    path("scripts/<int:pk>/delete/", views.script_delete, name="script_delete"),
    path("scripts/<int:pk>/preview/", views.script_preview, name="script_preview"),
    path("scripts/bulk-delete/", views.script_bulk_delete, name="script_bulk_delete"),
    path("cards/", views.card_list, name="cards"),
    path("cards/<int:pk>/", views.card_detail, name="card_detail"),
    path("cards/<int:pk>/delete/", views.card_delete, name="card_delete"),
    path("cards/bulk-delete/", views.card_bulk_delete, name="card_bulk_delete"),
    path("cards/<int:pk>/generate-script/", views.generate_script_for_card, name="generate_script"),
    path("practice/generate-ladder/", views.generate_practice_ladder, name="generate_ladder"),
    path("practice/ladders/<int:pk>/delete/", views.delete_practice_ladder, name="delete_ladder"),
    path("practice/ladders/bulk-delete/", views.ladder_bulk_delete, name="ladder_bulk_delete"),
    path("account/", views.account_settings, name="account"),
]
