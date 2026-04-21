import torch

path = "best_siamese.pth"
try:
    weights = torch.load(path, map_location='cpu')
    print(f"✅ Succès ! Le fichier contient {len(weights)} couches de neurones.")
except Exception as e:
    print(f"❌ Erreur : {e}")