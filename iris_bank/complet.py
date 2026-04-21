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

# =============================================================================
# CONFIGURATION - MODIFIEZ ICI LE CHEMIN VERS VOTRE DATASET
# =============================================================================

# Remplacez ce chemin par le chemin de votre dossier iris_images
DATA_ROOT = r"C:\Users\RAM Tech\Desktop\iris_images"

# Config
CFG = {
    "data_root":    DATA_ROOT,
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

print(f"Using device: {CFG['device']}")
print(f"Data root: {CFG['data_root']}")

# Vérifier si le dossier existe
if not os.path.exists(DATA_ROOT):
    print(f"ERREUR: Le dossier {DATA_ROOT} n'existe pas!")
    print("Veuillez vérifier le chemin et modifier DATA_ROOT dans le code.")
    exit(1)

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])


# =============================================================================
# Dataset Parsing Functions
# =============================================================================

def parse_dataset(data_root: str) -> dict[str, list[str]]:
    """Parse le dataset et retourne un dictionnaire identité -> liste d'images"""
    identity_images = defaultdict(list)
    data_root = Path(data_root)

    # Parcourir toutes les images PNG
    for img_path in sorted(data_root.rglob("*.png")):
        filename = img_path.stem
        m = re.match(r"(\d+)([LR])", filename)
        if m:
            person_id, side = m.group(1), m.group(2)
            identity = f"{person_id}{side}"
        else:
            # Fallback pour d'autres formats de nommage
            parts = img_path.parts
            folder   = parts[-2] if len(parts) >= 2 else "unknown"
            side_dir = parts[-1] if len(parts) >= 1 else "unknown"
            identity = f"{folder}_{side_dir[0].upper() if side_dir else 'U'}"

        identity_images[identity].append(str(img_path))

    # Garder seulement les identités avec au moins 2 images
    identity_images = {k: v for k, v in identity_images.items() if len(v) >= 2}
    print(f"Found {len(identity_images)} identities, {sum(len(v) for v in identity_images.values())} images total.")
    
    # Afficher quelques exemples
    for i, (k, v) in enumerate(list(identity_images.items())[:5]):
        print(f"  {k}: {len(v)} images")
    if len(identity_images) > 5:
        print(f"  ... and {len(identity_images) - 5} more")
    
    return identity_images


def split_identities(identity_images: dict, cfg: dict) -> tuple[dict, dict, dict]:
    """Divise les identités en train/val/test"""
    ids = list(identity_images.keys())
    random.shuffle(ids)

    n        = len(ids)
    n_train  = int(n * cfg["train_ratio"])
    n_val    = int(n * cfg["val_ratio"])

    train_ids = ids[:n_train]
    val_ids   = ids[n_train : n_train + n_val]
    test_ids  = ids[n_train + n_val:]

    train_data = {k: identity_images[k] for k in train_ids}
    val_data   = {k: identity_images[k] for k in val_ids}
    test_data  = {k: identity_images[k] for k in test_ids}

    print(f"Split — train: {len(train_data)} ids | val: {len(val_data)} ids | test: {len(test_data)} ids")
    return train_data, val_data, test_data


def generate_pairs(identity_images: dict, pairs_per_id: int = 4) -> list[tuple]:
    """Génère des paires d'images (positives et négatives)"""
    all_ids   = list(identity_images.keys())
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
    attempts = 0
    max_attempts = n_neg * 10
    
    while len(neg_pairs) < n_neg and attempts < max_attempts:
        id1, id2 = random.sample(all_ids, 2)
        if id1 != id2:  # S'assurer que ce sont des identités différentes
            img1 = random.choice(identity_images[id1])
            img2 = random.choice(identity_images[id2])
            neg_pairs.append((img1, img2, 0))
        attempts += 1

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
            transforms.ColorJitter(brightness=0.3, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        aug = []
    post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(base + aug + post)


# =============================================================================
# Dataset Class
# =============================================================================

class IrisPairDataset(Dataset):
    def __init__(self, pairs: list[tuple], transform: transforms.Compose):
        self.pairs     = pairs
        self.transform = transform

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        img1 = self.transform(Image.open(path1).convert("L"))
        img2 = self.transform(Image.open(path2).convert("L"))
        return img1, img2, torch.tensor(label, dtype=torch.float32)


# =============================================================================
# Neural Network Models
# =============================================================================

class EmbeddingNet(nn.Module):
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
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embedding_net = EmbeddingNet(embed_dim)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor):
        return self.embedding_net(img1), self.embedding_net(img2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = F.pairwise_distance(emb1, emb2)
        pos_loss = label * dist.pow(2)
        neg_loss = (1 - label) * F.relu(self.margin - dist).pow(2)
        return (pos_loss + neg_loss).mean()


def compute_accuracy(emb1, emb2, labels, threshold: float = 0.5) -> float:
    dist = F.pairwise_distance(emb1, emb2).detach().cpu()
    return ((dist < threshold).float() == labels.cpu()).float().mean().item()


# =============================================================================
# Training Functions
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
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
        total_acc  += compute_accuracy(emb1, emb2, labels)
    return total_loss / len(loader), total_acc / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for img1, img2, labels in loader:
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, labels)
        total_loss += loss.item()
        total_acc  += compute_accuracy(emb1, emb2, labels)
    return total_loss / len(loader), total_acc / len(loader)


def train(model, train_loader, val_loader, cfg):
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


# =============================================================================
# Evaluation Functions
# =============================================================================

@torch.no_grad()
def get_distances_and_labels(model, loader, device):
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
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer_threshold = -thresholds[eer_idx]
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"EER = {eer:.3f} at threshold = {eer_threshold:.4f}")
    return float(eer_threshold)


def plot_distance_distributions(distances, labels, threshold: float = None):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Val")
    ax1.set_title("Contrastive loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Val")
    ax2.set_title("Accuracy (threshold=0.5)")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Inference Class
# =============================================================================

class IrisVerifier:
    def __init__(self, checkpoint_path: str, threshold: float, cfg: dict = CFG):
        self.threshold = threshold
        self.device    = cfg["device"]
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
        return {"match": match, "distance": round(dist, 4), "threshold": self.threshold, 
                "verdict": "MATCH ✓" if match else "NO MATCH ✗"}


# =============================================================================
# Main Execution
# =============================================================================

print("\n" + "=" * 60)
print("IRIS RECOGNITION SYSTEM - SIAMESE NETWORK")
print("=" * 60)

# Charger les données
print("\n[1] Loading dataset...")
identity_images = parse_dataset(CFG["data_root"])

if len(identity_images) == 0:
    print("ERREUR: Aucune image trouvée! Vérifiez le chemin et le format des fichiers.")
    exit(1)

# Diviser les données
print("\n[2] Splitting data...")
train_data, val_data, test_data = split_identities(identity_images, CFG)

# Générer les paires
print("\n[3] Generating pairs...")
train_pairs = generate_pairs(train_data, CFG["pairs_per_id"])
val_pairs   = generate_pairs(val_data,   CFG["pairs_per_id"])
test_pairs  = generate_pairs(test_data,  CFG["pairs_per_id"])

# Créer les DataLoaders
print("\n[4] Creating DataLoaders...")
train_loader = DataLoader(IrisPairDataset(train_pairs, get_transforms(CFG["img_size"], True)), 
                          batch_size=CFG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(IrisPairDataset(val_pairs, get_transforms(CFG["img_size"], False)), 
                          batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(IrisPairDataset(test_pairs, get_transforms(CFG["img_size"], False)), 
                          batch_size=CFG["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

# Créer le modèle
print("\n[5] Creating model...")
model = SiameseNet(embed_dim=CFG["embed_dim"]).to(CFG["device"])
print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# Entraînement
print(f"\n[6] Training for {CFG['epochs']} epochs...\n")
history = train(model, train_loader, val_loader, CFG)

# Visualiser l'historique
plot_training_history(history)

# Évaluation
print("\n[7] Evaluating on test set...")
model.load_state_dict(torch.load("best_siamese.pth", map_location=CFG["device"]))

# Trouver le meilleur seuil
val_dists, val_labels = get_distances_and_labels(model, val_loader, CFG["device"])
threshold = find_best_threshold(val_dists, val_labels)

# Tester sur le jeu de test
test_dists, test_labels = get_distances_and_labels(model, test_loader, CFG["device"])
accuracy = ((test_dists < threshold).astype(float) == test_labels).mean()
print(f"Test accuracy at EER threshold: {accuracy:.3f}")

# Afficher les distributions
plot_distance_distributions(test_dists, test_labels, threshold)

# =============================================================================
# Confusion Matrix
# =============================================================================

print("\n[8] Confusion Matrix Analysis...")

y_true = test_labels.astype(int)
y_pred = (test_dists < threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
total = tp + tn + fp + fn
acc   = (tp + tn) / total
far   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
frr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Final Results — Test Set", fontsize=13, fontweight="bold")

# Graphique des courbes
epochs = range(1, len(history["train_loss"]) + 1)
best_ep = int(np.argmin(history["val_loss"])) + 1
ax = axes[0]
ax.plot(epochs, history["train_loss"], label="Train loss", color="#2196F3", linewidth=2)
ax.plot(epochs, history["val_loss"],   label="Val loss",   color="#4CAF50", linewidth=2)
ax.plot(epochs, history["train_acc"],  label="Train acc",  color="#2196F3", linewidth=2, linestyle="--")
ax.plot(epochs, history["val_acc"],    label="Val acc",    color="#4CAF50", linewidth=2, linestyle="--")
ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1.5, label=f"Best epoch {best_ep}")
ax.set_title("Loss & Accuracy curves", fontweight="bold")
ax.set_xlabel("Epoch")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Matrice de confusion
ax = axes[1]
cell_labels = [[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]]
cell_colors = [["#4CAF50", "#f44336"], ["#FF9800", "#4CAF50"]]
for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, 1-i), 1, 1,
                     color=cell_colors[i][j], alpha=0.82, ec="white", lw=3))
        name, val = cell_labels[i][j].split("\n")
        ax.text(j+0.5, 1-i+0.62, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color="white")
        ax.text(j+0.5, 1-i+0.30, val,  ha="center", va="center",
                fontsize=22, fontweight="bold", color="white")

ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.set_xticks([0.5, 1.5])
ax.set_xticklabels(["Predicted\nNO MATCH", "Predicted\nMATCH"], fontsize=9)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(["Actual\nMATCH", "Actual\nNO MATCH"], fontsize=9)
ax.set_title("Confusion Matrix", fontweight="bold")
for spine in ax.spines.values(): 
    spine.set_visible(False)
ax.tick_params(length=0)

plt.tight_layout()
plt.show()

print(f"\nAccuracy : {acc*100:.2f}%")
print(f"FAR      : {far*100:.2f}%  (impostors accepted)")
print(f"FRR      : {frr*100:.2f}%  (genuine users rejected)")
print(f"TP={tp}  TN={tn}  FP={fp}  FN={fn}\n")
print(classification_report(y_true, y_pred, target_names=["NO MATCH", "MATCH"]))

# =============================================================================
# Demo avec le modèle entraîné
# =============================================================================

print("\n[9] Demo - Testing with sample images...")
verifier = IrisVerifier("best_siamese.pth", threshold=threshold)

# Tester avec quelques paires du jeu de test
sample_indices = np.random.choice(len(test_pairs), min(5, len(test_pairs)), replace=False)
for idx in sample_indices:
    path1, path2, label = test_pairs[idx]
    expected = "MATCH" if label == 1 else "NO MATCH"
    result = verifier.verify(path1, path2)
    status = "✓" if (result["match"] == (label == 1)) else "✗"
    print(f"  [{status}] Expected: {expected:8s} | {result['verdict']:12s} | distance={result['distance']:.4f}")

print("\n" + "=" * 60)
print("Training completed successfully!")
print("=" * 60)