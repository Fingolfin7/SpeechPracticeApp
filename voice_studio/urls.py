from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="studio"),
    path("evaluate/", views.evaluate, name="studio_evaluate"),
]
