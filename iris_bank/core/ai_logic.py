import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from django.conf import settings
import numpy as np

class EmbeddingNet(nn.Module):
    """Matches the architecture from complet.py"""
    def __init__(self, embed_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.backbone = backbone

    def forward(self, x):
        return F.normalize(self.backbone(x), p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self, embed_dim=128):
        super(SiameseNetwork, self).__init__()
        self.embedding_net = EmbeddingNet(embed_dim)

    def forward(self, x1, x2):
        out1 = self.embedding_net(x1)
        out2 = self.embedding_net(x2)
        return out1, out2


class IrisAI:
    def __init__(self):
        self.model = SiameseNetwork(embed_dim=128)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Match the transformations from complet.py
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # 🔴 Mode développement DÉSACTIVÉ - Calcul réel de la distance
        self.dev_mode = False  # False = mode réel avec calcul de distance
        
        # Charger le modèle pré-entraîné
        self.load_custom_model()
        
        self.model.to(self.device)
        self.model.eval()
        
        # Seuil pour la vérification (distance < threshold = match)
        # Plus le seuil est bas, plus la vérification est stricte
        self.threshold = 0.65  # Ajustez selon vos besoins

    def load_custom_model(self):
        """Charge le modèle pré-entraîné"""
        path = os.path.join(settings.BASE_DIR, 'best_siamese.pth')
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)
                print(f"✅ [IrisAI] Modèle chargé avec succès depuis {path}")
            except Exception as e:
                print(f"❌ [IrisAI] Erreur chargement modèle: {e}")
                print(f"⚠️ [IrisAI] Utilisation du modèle non entraîné (résultats aléatoires)")
        else:
            print(f"⚠️ [IrisAI] Fichier {path} introuvable.")
            print(f"⚠️ [IrisAI] Utilisation du modèle non entraîné (résultats aléatoires)")

    def load_image(self, image_path_or_file):
        """Charge une image depuis un chemin ou un fichier Django"""
        try:
            if hasattr(image_path_or_file, 'read'):
                # C'est un fichier Django ContentFile
                img = Image.open(image_path_or_file)
            elif isinstance(image_path_or_file, str):
                # C'est un chemin de fichier
                img = Image.open(image_path_or_file)
            else:
                raise ValueError("Format d'image non supporté")
            
            # Convertir en RGB si nécessaire (le modèle attend 3 canaux)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            return self.transform(img).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[IrisAI] Erreur chargement image: {e}")
            raise

    def get_distance(self, img1, img2):
        """Calcule la distance euclidienne entre deux embeddings d'iris"""
        with torch.no_grad():
            emb1, emb2 = self.model(img1, img2)
            # Distance euclidienne
            dist = F.pairwise_distance(emb1, emb2)
            return dist.item()

    def compare(self, temp_image_file, db_image_path, threshold=None):
        """Compare une image temporaire avec celle de la DB et retourne la distance"""
        if threshold is None:
            threshold = self.threshold
            
        try:
            # Charger les deux images
            print(f"[IrisAI] Chargement des images...")
            img1 = self.load_image(temp_image_file)
            img2 = self.load_image(db_image_path)
            
            # Calculer la distance réelle
            print(f"[IrisAI] Calcul de la distance...")
            distance = self.get_distance(img1, img2)
            
            # Vérifier si la distance est en dessous du seuil
            is_match = distance < threshold
            
            # Afficher les résultats détaillés
            print(f"[IrisAI] 📊 Résultat vérification:")
            print(f"[IrisAI]    - Distance calculée: {distance:.6f}")
            print(f"[IrisAI]    - Seuil: {threshold}")
            print(f"[IrisAI]    - Match: {is_match}")
            
            if is_match:
                print(f"[IrisAI] ✅ Vérification réussie - Les iris correspondent")
            else:
                print(f"[IrisAI] ❌ Vérification échouée - Les iris ne correspondent pas")
                print(f"[IrisAI]    Différence: {distance - threshold:.6f} au-dessus du seuil")
            
            return is_match
            
        except Exception as e:
            print(f"[IrisAI] ❌ Erreur lors de la comparaison: {e}")
            import traceback
            traceback.print_exc()
            return False

    def extract_features_for_training(self, image_path):
        """Extrait les features pour l'entraînement (utile pour debug)"""
        try:
            img = self.load_image(image_path)
            with torch.no_grad():
                embedding = self.model.embedding_net(img)
            return embedding.cpu().numpy()
        except Exception as e:
            print(f"[IrisAI] Erreur extraction features: {e}")
            return None
    
    def get_similarity_score(self, temp_image_file, db_image_path):
        """Retourne le score de similarité (1 - distance normalisée)"""
        try:
            img1 = self.load_image(temp_image_file)
            img2 = self.load_image(db_image_path)
            distance = self.get_distance(img1, img2)
            
            # Convertir la distance en score de similarité (0 à 1)
            similarity = max(0, 1 - distance)
            
            print(f"[IrisAI] Score de similarité: {similarity:.4f} (distance: {distance:.4f})")
            return similarity
            
        except Exception as e:
            print(f"[IrisAI] Erreur calcul similarité: {e}")
            return 0.0