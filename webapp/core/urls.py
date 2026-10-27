from django.urls import path
from . import views

urlpatterns = [
    path("", views.forecast_view, name="forecast"),
    path("about/", views.about_view, name="about"),
]
