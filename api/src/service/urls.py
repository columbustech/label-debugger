from django.urls import path
from . import views

urlpatterns = [
    path('fetchPairs/', views.fetchSuspiciousPairs.as_view()),
    path('savetoDrive/', views.SaveToCDriveView.as_view()),
]
