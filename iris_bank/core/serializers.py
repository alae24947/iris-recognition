import base64
from django.core.files.base import ContentFile
from rest_framework import serializers
from django.contrib.auth.models import User
from .models import UserProfile

class RegisterSerializer(serializers.ModelSerializer):
    # Champs pour capturer les données du frontend
    name = serializers.CharField(write_only=True)
    email = serializers.EmailField(write_only=True)
    password = serializers.CharField(write_only=True)
    phone_number = serializers.CharField(write_only=True)
    iris_image = serializers.CharField(write_only=True)

    class Meta:
        model = UserProfile
        fields = ['name', 'email', 'password', 'phone_number', 'iris_image']

    def validate_email(self, value):
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("Cet email est déjà utilisé.")
        return value

    def create(self, validated_data):
        # 1. Extraction des données
        iris_data = validated_data.pop('iris_image')
        phone = validated_data.pop('phone_number')
        password = validated_data.pop('password')
        email = validated_data.pop('email')
        full_name = validated_data.pop('name')

        # 2. Traitement de l'image Base64
        try:
            if ';base64,' in iris_data:
                header, imgstr = iris_data.split(';base64,')
                ext = header.split('/')[-1].split(';')[0]
            else:
                imgstr = iris_data
                ext = 'jpg'
            
            iris_file = ContentFile(
                base64.b64decode(imgstr), 
                name=f"iris_{email}.{ext}"
            )
        except Exception as e:
            raise serializers.ValidationError({"iris_image": "Erreur de décodage image."})

        # 3. Création de l'utilisateur (auth_user)
        user = User.objects.create_user(
            username=email,
            email=email,
            password=password,
            first_name=full_name # Stockage du Nom Complet
        )

        # 4. Création du profil lié (core_userprofile)
        profile = UserProfile.objects.create(
            user=user,
            phone_number=phone,
            iris_image=iris_file,
            balance=1000.0
        )
        return profile