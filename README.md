# 👁️ Iris Biometric Authentication — Siamese ResNet18

 
Biometric iris verification system using a Siamese Neural Network with a fine-tuned ResNet18 backbone. Trained on the UPOL iris dataset, achieving **95% accuracy** and **5% EER** on unseen identities — deployable as a REST microservice.
 
---
 
##  How It Works
 
The model learns an embedding function `f(x)` such that:
- Same iris → embeddings are **close** (distance < 0.442)
- Different iris → embeddings are **far apart** (distance > 0.442)
Decision threshold is optimized via **Equal Error Rate (EER)**.
 
```
Image A ──┐
           ├──► ResNet18 Backbone ──► 128D Embedding ──► Euclidean Distance ──► MATCH / NO MATCH
Image B ──┘
```

---
 
##  Architecture
 
### EmbeddingNet (backbone)
- **Base:** ResNet18 pre-trained on ImageNet
- **Head:** Linear(512 → 128) + BatchNorm1d
- **Output:** L2-normalized 128D embedding vector
- **Frozen layers:** `conv1`, `layer1`, `layer2` (generic texture features)
- **Trainable layers:** `layer3`, `layer4`, projection head
### SiameseNet
- Two **identical branches** sharing the same weights
- Feeds two images → returns two embeddings
### ContrastiveLoss
```
L = y · d² + (1 - y) · max(margin - d, 0)²
```
- `y = 1` → same identity → minimize distance
- `y = 0` → different identity → push beyond margin (1.0)
---
 
## 📊 Dataset
 
| Property | Value |
|---|---|
| Source | UPOL Iris Database (Kaggle) |
| Total images | 384 |
| Identities | 128 (L/R eyes treated separately) |
| Images per identity | ~3 |
| Acquisition device | TOPCON TRC50IA + SONY DXC-950P 3CCD |
 
### Split Strategy
Split is done **by identity** (not by image) to prevent data leakage:
 
| Split | Identities | Ratio |
|---|---|---|
| Train | 96 | 75% |
| Val | 12 | 10% |
| Test | 20 | 15% |
 
Pairs are **balanced**: equal positive and negative pairs per split (Train: 288 / 288).
 
---
 
##  Hyperparameters
 
| Parameter | Value | Reason |
|---|---|---|
| Optimizer | Adam lr=1e-4 | Preserves pre-trained weights |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) | Handles plateaus |
| Margin | 1.0 | Clear separation boundary |
| Batch size | 32 | Memory / gradient variance tradeoff |
| Epochs | 20 | With best checkpoint saving |
| Embedding dim | 128 | Sufficient capacity without overfitting |
| Seed | 42 | Full reproducibility |
 
---
 
## 📈 Results
 
### Test Set (20 unseen identities, 120 pairs)
 
| Metric | Value |
|---|---|
| Accuracy | **95%** |
| FAR (False Accept Rate) | 5% |
| FRR (False Reject Rate) | 5% |
| Precision / Recall / F1 | 0.95 / 0.95 / 0.95 |
| EER Threshold | 0.442 |
 
### vs VGG16 Baseline (same conditions)
 
| Metric | VGG16 | ResNet18 | Gain |
|---|---|---|---|
| Accuracy | 81.67% | **95%** | +13.33 pts |
| FAR | 16.67% | **5%** | ÷ 3 |
| FRR | 20% | **5%** | ÷ 4 |
| Parameters | 138M | **11.2M** | 12× lighter |
| Time / epoch | 45s | **18s** | 2.5× faster |
 
### Validation Scenarios
 
| Scenario | Distance | Verdict |
|---|---|---|
| Same person, same iris | 0.25 |  MATCH |
| Same person, left vs right iris | 0.8717 | NO MATCH |
| Two different people | 1.37 |  NO MATCH |
 
### Real-World Test (smartphone photos)
 
| Test | Images | Distance | Verdict |
|---|---|---|---|
| Same iris | al1.jpg vs al2.jpg | 0.2125 |  MATCH |
| Different people | ak1.jpg vs al1.jpg | 0.8126 |  NO MATCH |
 
---
 
##  Quick Start
 
### Installation
```bash
pip install torch torchvision pillow scikit-learn matplotlib kagglehub
```
 
### Run Inference
```python
from iris_auth.verify import IrisVerifier
 
verifier = IrisVerifier("best_siamese.pth", threshold=0.442)
result = verifier.verify("path/to/iris1.jpg", "path/to/iris2.jpg")
 
print(result)
# {'match': True, 'distance': 0.2125, 'threshold': 0.442, 'verdict': 'MATCH ✓'}
```
 
### Training
Open and run `iriss-reco-resnet18.ipynb` on Kaggle (GPU recommended).
 
Environment: Python 3.12 · PyTorch · NVIDIA Tesla T4 · Kaggle Notebooks
 
---
