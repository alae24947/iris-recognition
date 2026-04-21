import random
import base64
import os
from datetime import datetime, timedelta

from django.shortcuts import render
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.core.mail import send_mail
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

from .serializers import RegisterSerializer
from .models import UserProfile
from .ai_logic import IrisAI

# Initialisation de l'IA (S'assure que le modèle est chargé au démarrage)
iris_checker = IrisAI()

# Stockage temporaire des codes OTP en mémoire vive
_otp_store = {}

def index(request):
    """Sert la page d'accueil Algerie Bank"""
    return render(request, 'core/index.html')

@method_decorator(csrf_exempt, name='dispatch')
class SendOTPAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [] 

    def post(self, request):
        email = request.data.get('email', '').strip().lower()
        if not email:
            return Response({"status": "error", "message": "Email requis"}, status=400)

        # Génération d'un code à 6 chiffres
        otp_code = str(random.randint(100000, 999999))
        _otp_store[email] = {
            'code': otp_code, 
            'expires_at': datetime.now() + timedelta(minutes=10)
        }

        try:
            send_mail(
                "Votre code de sécurité Algerie Bank",
                f"Votre code de vérification est : {otp_code}",
                settings.DEFAULT_FROM_EMAIL,
                [email],
                fail_silently=False,
            )
            print(f"--- [DEBUG] OTP Envoyé à {email} : {otp_code} ---")
            return Response({"status": "success", "message": "Code envoyé !"})
        except Exception as e:
            print(f"--- [ERREUR SMTP] : {e} ---")
            return Response({"status": "error", "message": "Erreur d'envoi d'email"}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class VerifyOTPAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        # Debugging pour voir exactement ce que le frontend envoie
        print(f"--- [DEBUG] DATA REÇUE : {request.data} ---")
        
        email = request.data.get('email', '').strip().lower()
        # On accepte 'otp' ou 'code' pour plus de flexibilité avec le JS
        otp_received = request.data.get('otp') or request.data.get('code')
        
        if not email or not otp_received:
            return Response({"status": "error", "message": "Données incomplètes"}, status=400)

        if email in _otp_store:
            record = _otp_store[email]
            
            # 1. Vérification de l'expiration
            if datetime.now() > record['expires_at']:
                return Response({"status": "error", "message": "Code expiré"}, status=400)
            
            # 2. Comparaison stricte en format String
            if str(record['code']) == str(otp_received).strip():
                print(f"--- [DEBUG] OTP Valide pour {email} ---")
                return Response({"status": "success", "message": "Code vérifié !"})
            else:
                print(f"--- [DEBUG] OTP Incorrect : Reçu {otp_received}, attendu {record['code']} ---")
        
        return Response({"status": "error", "message": "Code invalide"}, status=400)

@method_decorator(csrf_exempt, name='dispatch')
class LoginAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email', '').strip().lower()
        password = request.data.get('password')
        
        user = authenticate(username=email, password=password)
        if user:
            profile = UserProfile.objects.get(user=user)
            return Response({
                "status": "success",
                "user": {
                    "name": user.first_name,
                    "email": user.email,
                    "balance": float(profile.balance)
                }
            })
        return Response({"status": "error", "message": "Identifiants incorrects"}, status=401)

@method_decorator(csrf_exempt, name='dispatch')
class VerifyIrisAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        email = request.data.get('email', '').strip().lower()
        captured_iris = request.data.get('iris_image')

        try:
            user = User.objects.get(username=email)
            profile = UserProfile.objects.get(user=user)
            
            # Chemin de l'image de référence enregistrée lors de l'inscription
            ref_path = profile.iris_image.path
            temp_path = os.path.join(settings.MEDIA_ROOT, f"temp_login_{email}.jpg")
            
            # Décodage Base64
            if ';base64,' in captured_iris:
                _, imgstr = captured_iris.split(';base64,')
            else:
                imgstr = captured_iris

            with open(temp_path, "wb") as f:
                f.write(base64.b64decode(imgstr))

            # Appel à la méthode .compare() de votre fichier ai_logic.py
            is_match = iris_checker.compare(temp_path, ref_path)
            
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if is_match:
                return Response({"status": "success"})
            else:
                return Response({"status": "error", "message": "Iris non reconnu"}, status=403)

        except Exception as e:
            print(f"--- [ERREUR IRIS] : {e} ---")
            return Response({"status": "error", "message": str(e)}, status=500)
@method_decorator(csrf_exempt, name='dispatch')
class RegisterAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = [] 

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            print(f"✅ Utilisateur créé : {request.data.get('email')}")
            return Response({"status": "success"}, status=status.HTTP_201_CREATED)
        
        print(f"❌ Erreur validation : {serializer.errors}")
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
@method_decorator(csrf_exempt, name='dispatch')
class TransferAPIView(APIView):
    permission_classes = [AllowAny]
    authentication_classes = []

    def post(self, request):
        sender_email = request.data.get('from_email', '').strip().lower()
        to_email = request.data.get('to_email', '').strip().lower()
        amount = float(request.data.get('amount', 0))

        try:
            sender = UserProfile.objects.get(user__username=sender_email)
            recipient = UserProfile.objects.get(user__username=to_email)
            
            if sender.balance >= amount:
                sender.balance -= amount
                recipient.balance += amount
                sender.save()
                recipient.save()
                return Response({"status": "success", "new_balance": float(sender.balance)})
            else:
                return Response({"status": "error", "message": "Solde insuffisant"}, status=400)
        except Exception:
            return Response({"status": "error", "message": "Utilisateur introuvable"}, status=404)