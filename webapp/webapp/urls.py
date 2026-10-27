"""URL configuration for AQI-GRU web application."""

from django.urls import include, path

urlpatterns = [
    path("", include("core.urls")),
]
