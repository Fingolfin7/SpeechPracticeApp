from __future__ import annotations

from django.urls import path

from . import views

app_name = "practice"

urlpatterns = [
    path("", views.dashboard, name="dashboard"),
    path("progress/", views.progress, name="progress"),
    path("practice/", views.practice_run, name="practice"),
    path("jobs/<int:pk>/", views.scoring_job_detail, name="scoring_job"),
    path("jobs/<int:pk>/retry/", views.retry_scoring_job, name="retry_scoring_job"),
    path("sessions/", views.session_list, name="sessions"),
    path("sessions/<int:pk>/", views.session_detail, name="session_detail"),
    path("sessions/<int:pk>/audio/", views.session_audio, name="session_audio"),
    path("sessions/<int:pk>/report.md", views.session_report, name="session_report"),
    path("sessions/<int:pk>/delete/", views.session_delete, name="session_delete"),
    path("scripts/", views.script_list, name="scripts"),
    path("scripts/new/", views.script_create, name="script_create"),
    path("scripts/import/", views.script_import, name="script_import"),
    path("scripts/<int:pk>/edit/", views.script_edit, name="script_edit"),
    path("scripts/<int:pk>/delete/", views.script_delete, name="script_delete"),
    path("scripts/<int:pk>/preview/", views.script_preview, name="script_preview"),
    path("cards/", views.card_list, name="cards"),
    path("cards/<int:pk>/", views.card_detail, name="card_detail"),
    path("cards/<int:pk>/generate-script/", views.generate_script_for_card, name="generate_script"),
    path("account/", views.account_settings, name="account"),
]
