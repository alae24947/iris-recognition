from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core import views # Importez vos vues ici

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # CETTE LIGNE MANQUE DANS VOTRE ERREUR 404 :
    path('', views.index, name='index'), 
    
    # Routes API
    path('api/register/', views.RegisterAPIView.as_view(), name='api_register'),
    path('api/login/', views.LoginAPIView.as_view(), name='api_login'),
    path('api/verify-iris/', views.VerifyIrisAPIView.as_view(), name='api_verify_iris'),
    path('api/transfer/', views.TransferAPIView.as_view(), name='api_transfer'),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)# Remplacez 'core' par le nom de votre app


# Permet de lire les images d'iris dans le navigateur pendant le développement
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    
    # C'est cette ligne qui fait le lien magique
    # Elle dit : tout ce qui commence par /api/ va dans core.urls
    path('api/', include('core.urls')), 
    
    # Et cette ligne permet d'afficher l'index sur http://127.0.0.1:8000/
    path('', include('core.urls')),
]