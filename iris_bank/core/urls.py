# Fichier : votre_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # On laisse les chemins "propres", sans 'api/' devant
    path('register/',     views.RegisterAPIView.as_view(),    name='api_register'),
    path('login/',        views.LoginAPIView.as_view(),       name='api_login'),
    path('verify-iris/',  views.VerifyIrisAPIView.as_view(),  name='api_verify_iris'),
    path('transfer/',     views.TransferAPIView.as_view(),    name='api_transfer'),
    path('send-otp/',     views.SendOTPAPIView.as_view(),     name='api_send_otp'),
    path('verify-otp/',   views.VerifyOTPAPIView.as_view(),   name='api_verify_otp'),
    
    # La vue pour charger la page d'accueil
    path('', views.index, name='index'),
]