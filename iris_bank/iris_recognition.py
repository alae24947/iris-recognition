import os
import re
import random
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================

CFG = {
    "data_root":    r"C:\Users\RAM Tech\Desktop\iris_images",
    "model_save_path": r"C:\Users\RAM Tech\Desktop\iris_bank\iris_bank\best_siamese.pth",
    "img_size":     224,
    "embed_dim":    128,
    "margin":       1.0,
    "batch_size":   32,
    "epochs":       20,
    "lr":           1e-4,
    "seed":         42,
    "train_ratio":  0.75,
    "val_ratio":    0.10,
    "pairs_per_id": 4,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

# Fixer les graines pour la reproductibilité
random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])

print(f"Using device: {CFG['device']}")
print(f"Data path: {CFG['data_root']}")

# ============================================================================
# CHARGEMENT DES DONNÉES
# ============================================================================

def parse_dataset(data_root: str) -> dict[str, list[str]]:
    """Parse le dataset et retourne un dictionnaire identité -> liste d'images"""
    identity_images = defaultdict(list)
    data_root = Path(data_root)

    # Parcourir toutes les images avec différentes extensions
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    for ext in extensions:
        for img_path in sorted(data_root.rglob(ext)):
            filename = img_path.stem
            # Essayer d'extraire l'identité à partir du nom de fichier
            m = re.match(r"(\d+)([LR])", filename)
            if m:
                person_id, side = m.group(1), m.group(2)
                identity = f"{person_id}{side}"
            else:
                # Utiliser le dossier parent comme identité
                parts = img_path.parts
                if len(parts) >= 2:
                    folder = parts[-2] if len(parts) >= 2 else "unknown"
                    identity = folder
                else:
                    identity = "unknown"

            identity_images[identity].append(str(img_path))

    # Garder seulement les identités avec au moins 2 images
    identity_images = {k: v for k, v in identity_images.items() if len(v) >= 2}
    print(f"Found {len(identity_images)} identities, {sum(len(v) for v in identity_images.values())} images total.")
    return identity_images


def split_identities(identity_images: dict, cfg: dict) -> tuple[dict, dict, dict]:
    """Divise les identités en train/val/test"""
    ids = list(identity_images.keys())
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])

    train_ids = ids[:n_train]
    val_ids = ids[n_train:n_train + n_val]
    test_ids = ids[n_train + n_val:]

    train_data = {k: identity_images[k] for k in train_ids}
    val_data = {k: identity_images[k] for k in val_ids}
    test_data = {k: identity_images[k] for k in test_ids}

    print(f"Split — train: {len(train_data)} ids | val: {len(val_data)} ids | test: {len(test_data)} ids")
    return train_data, val_data, test_data


def generate_pairs(identity_images: dict, pairs_per_id: int = 4) -> list[tuple]:
    """Génère des paires positives et négatives"""
    all_ids = list(identity_images.keys())
    pos_pairs = []
    neg_pairs = []

    # Paires positives (même identité)
    for identity, images in identity_images.items():
        combos = list(combinations(images, 2))
        sampled = random.sample(combos, min(pairs_per_id, len(combos)))
        for img1, img2 in sampled:
            pos_pairs.append((img1, img2, 1))

    # Paires négatives (identités différentes)
    n_neg = len(pos_pairs)
    while len(neg_pairs) < n_neg:
        id1, id2 = random.sample(all_ids, 2)
        img1 = random.choice(identity_images[id1])
        img2 = random.choice(identity_images[id2])
        neg_pairs.append((img1, img2, 0))

    pairs = pos_pairs + neg_pairs
    random.shuffle(pairs)
    print(f"  Generated {len(pos_pairs)} positive + {len(neg_pairs)} negative pairs")
    return pairs


def get_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Définit les transformations d'image"""
    base = [
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if train:
        aug = [
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
    else:
        aug = []
    post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(base + aug + post)


class IrisPairDataset(Dataset):
    """Dataset pour les paires d'images d'iris"""
    def __init__(self, pairs: list[tuple], transform: transforms.Compose):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = self.transform(Image.open(path1).convert("L"))
        img2 = self.transform(Image.open(path2).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ============================================================================
# MODÈLES
# ============================================================================

class EmbeddingNet(nn.Module):
    """Réseau d'embedding basé sur ResNet18"""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.backbone(x), p=2, dim=1)


class SiameseNet(nn.Module):
    """Réseau Siamese"""
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embedding_net = EmbeddingNet(embed_dim)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        return self.embedding_net(img1), self.embedding_net(img2)


class ContrastiveLoss(nn.Module):
    """Fonction de perte contrastive"""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = F.pairwise_distance(emb1, emb2)
        pos_loss = label * dist.pow(2)
        neg_loss = (1 - label) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()


# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================================

def compute_accuracy(emb1, emb2, labels, threshold: float = 0.5) -> float:
    """Calcule l'accuracy avec un seuil fixe"""
    dist = F.pairwise_distance(emb1, emb2).detach().cpu()
    return ((dist < threshold).float() == labels.cpu()).float().mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Entraîne le modèle pour une époque"""
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        optimizer.zero_grad()
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += compute_accuracy(emb1, emb2, labels)
    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Évalue le modèle"""
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        total_loss += loss.item()
        total_acc += compute_accuracy(emb1, emb2, labels)
    return total_loss / len(loader), total_acc / len(loader)


def train(model, train_loader, val_loader, cfg):
    """Boucle d'entraînement complète"""
    device = cfg["device"]
    criterion = ContrastiveLoss(margin=cfg["margin"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:3d}/{cfg['epochs']}  train loss {tr_loss:.4f} acc {tr_acc:.3f} | val loss {vl_loss:.4f} acc {vl_acc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), "best_siamese.pth")
            print(f"  ✓ saved best model (val_loss={vl_loss:.4f})")

    return history


# ============================================================================
# FONCTIONS D'ÉVALUATION
# ============================================================================

@torch.no_grad()
def get_distances_and_labels(model, loader, device):
    """Récupère toutes les distances et labels"""
    model.eval()
    all_dists, all_labels = [], []
    for img1, img2, labels in loader:
        img1, img2 = img1.to(device), img2.to(device)
        emb1, emb2 = model(img1, img2)
        dists = F.pairwise_distance(emb1, emb2).cpu().numpy()
        all_dists.extend(dists)
        all_labels.extend(labels.numpy())
    return np.array(all_dists), np.array(all_labels)


def find_best_threshold(distances, labels) -> float:
    """Trouve le meilleur seuil basé sur l'EER"""
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_threshold = -thresholds[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"EER = {eer:.3f} at threshold = {eer_threshold:.4f}")
    return float(eer_threshold)


def plot_distance_distributions(distances, labels, threshold: float = None):
    """Affiche la distribution des distances"""
    same_dists, diff_dists = distances[labels == 1], distances[labels == 0]
    plt.figure(figsize=(8, 4))
    plt.hist(same_dists, bins=40, alpha=0.6, color="steelblue", label="Same identity")
    plt.hist(diff_dists, bins=40, alpha=0.6, color="tomato", label="Different identity")
    if threshold is not None:
        plt.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold = {threshold:.3f}")
    plt.xlabel("Euclidean distance")
    plt.ylabel("Count")
    plt.title("Distance distribution — test set")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_training_history(history: dict):
    """Affiche l'historique d'entraînement"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Contrastive loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy (threshold=0.5)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.show()


class IrisVerifier:
    """Classe pour la vérification d'iris"""
    def __init__(self, checkpoint_path: str, threshold: float, cfg: dict = CFG):
        self.threshold = threshold
        self.device = cfg["device"]
        self.transform = get_transforms(cfg["img_size"], train=False)
        self.model = SiameseNet(embed_dim=cfg["embed_dim"]).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, image_path: str) -> torch.Tensor:
        img = self.transform(Image.open(image_path).convert("L")).unsqueeze(0).to(self.device)
        return self.model.embedding_net(img)

    @torch.no_grad()
    def verify(self, path1: str, path2: str) -> dict:
        emb1, emb2 = self.get_embedding(path1), self.get_embedding(path2)
        dist = F.pairwise_distance(emb1, emb2).item()
        match = dist < self.threshold
        return {
            "match": match,
            "distance": round(dist, 4),
            "threshold": self.threshold,
            "verdict": "MATCH ✓" if match else "NO MATCH ✗"
        }


# ============================================================================
# EXÉCUTION PRINCIPALE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IRIS RECOGNITION SYSTEM")
    print("="*60)

    # Chargement des données
    print("\n[1/6] Loading dataset...")
    identity_images = parse_dataset(CFG["data_root"])
    train_data, val_data, test_data = split_identities(identity_images, CFG)

    # Génération des paires
    print("\n[2/6] Generating pairs...")
    train_pairs = generate_pairs(train_data, CFG["pairs_per_id"])
    val_pairs = generate_pairs(val_data, CFG["pairs_per_id"])
    test_pairs = generate_pairs(test_data, CFG["pairs_per_id"])

    # Création des DataLoaders
    print("\n[3/6] Creating data loaders...")
    train_loader = DataLoader(
        IrisPairDataset(train_pairs, get_transforms(CFG["img_size"], True)),
        batch_size=CFG["batch_size"], shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        IrisPairDataset(val_pairs, get_transforms(CFG["img_size"], False)),
        batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        IrisPairDataset(test_pairs, get_transforms(CFG["img_size"], False)),
        batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True
    )

    # Initialisation du modèle
    print("\n[4/6] Initializing model...")
    model = SiameseNet(embed_dim=CFG["embed_dim"]).to(CFG["device"])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Entraînement
    print(f"\n[5/6] Training for {CFG['epochs']} epochs...\n")
    history = train(model, train_loader, val_loader, CFG)
    plot_training_history(history)

    # Évaluation
    print("\n[6/6] Evaluating on test set...")
    model.load_state_dict(torch.load("best_siamese.pth", map_location=CFG["device"]))

    val_dists, val_labels = get_distances_and_labels(model, val_loader, CFG["device"])
    threshold = find_best_threshold(val_dists, val_labels)

    test_dists, test_labels = get_distances_and_labels(model, test_loader, CFG["device"])
    accuracy = ((test_dists < threshold).astype(float) == test_labels).mean()
    print(f"Test accuracy at EER threshold: {accuracy:.3f}")

    plot_distance_distributions(test_dists, test_labels, threshold)

    # Matrice de confusion
    y_true = test_labels.astype(int)
    y_pred = (test_dists < threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    acc = (tp + tn) / total
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"\nFinal Results:")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"FAR      : {far*100:.2f}%  (impostors accepted)")
    print(f"FRR      : {frr*100:.2f}%  (genuine users rejected)")
    print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}\n")
    print(classification_report(y_true, y_pred, target_names=["NO MATCH", "MATCH"]))

    # Sauvegarde finale
    torch.save(model.state_dict(), CFG["model_save_path"])
    print(f"\n✅ Model saved to: {CFG['model_save_path']}")
    print("✅ Execution completed!")