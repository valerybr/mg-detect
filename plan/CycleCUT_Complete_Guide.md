# CycleCUT: Complete Implementation Guide
### Bilateral Anomaly Detection for Mammography
#### Version 1.0 — Master Student Implementation Recipe

---

## How To Read This Document

This document describes two experimental plans for the same system:

- **Plan A** — Train on normal pairs only. Fully unsupervised. Start here.
- **Plan B** — Add synthetic corruptions to Plan A. Run only after Plan A is working.

Read the entire document before writing any code. Every section builds on the previous one.

---

## 1. The Problem

Radiologists detect breast cancer by comparing the left and right breast. If something looks different on one side compared to the same region on the other side, it is suspicious. We want to teach a computer to do the same thing — without being told where the lesions are.

The core idea:

```
1. Train on normal breast pairs (left, right) only
2. Learn what normal bilateral differences look like
3. At test time: if a difference is unusual compared to normal → flag as anomaly
```

The output is a **patch-level heatmap** over each breast:

```
heatmap[i,j] ≈ 0    → normal patch
heatmap[i,j] >> 0   → anomalous patch → possible lesion
```

---

## 2. System Overview

The system has four components trained in sequence:

```
Component 1: CycleCUT
  Input:  normal bilateral pairs (left breast, right breast)
  Output: two trained generators G_AB and G_BA
  Purpose: learn to translate one breast to the other

Component 2: Residual Extraction
  Input:  trained G_AB, G_BA + all training pairs
  Output: residual maps r_A and r_B per patient
  Purpose: compute what is different between real and predicted breast

Component 3: Patch Encoder
  Input:  residual patches from normal pairs
  Output: trained CNN encoder φ
  Purpose: compress 32×32 residual patch to 128-dimensional feature vector

Component 4: Normalizing Flow
  Input:  encoded normal residual patches
  Output: trained flow p_flow
  Purpose: learn the distribution of normal residuals
```

At test time:

```
Input:   new patient (A_test, B_test)
Output:  heatmap[i,j] = anomaly score per patch
```

---

## 3. Mathematical Notation

```
A         = left breast image,  shape 512×512×1
B         = right breast image, shape 512×512×1
G_AB      = generator that maps A → predicted B
G_BA      = generator that maps B → predicted A
r_A       = |A − G_BA(B)|   = residual on A side
r_B       = |B − G_AB(A)|   = residual on B side
φ         = patch encoder CNN
z         = φ(patch) ∈ R^128 = patch feature vector
p_flow    = normalizing flow distribution
score_A   = −log p_flow(φ(r_A patch))
score_B   = −log p_flow(φ(r_B patch))
heatmap   = score_B − score_A   (per patch)
```

---

## 4. Data

### 4.1 Datasets

Use these datasets in this order:

| Dataset | Cases | Annotations | Role |
|---|---|---|---|
| VinDr-Mammo | 5,000 | Bounding boxes + BI-RADS density | Primary train/test |
| INbreast | 410 | Segmentation masks | External validation |
| CBIS-DDSM | 2,620 | Segmentation masks | Lesion bank (Plan B only) |

Download VinDr-Mammo from: https://physionet.org/content/vindr-mammo/1.0.0/
Download INbreast from: https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset
Download CBIS-DDSM from: https://www.cancerimagingarchive.net/collection/cbis-ddsm/

### 4.2 Which Images To Use

Each patient has four images: left CC, right CC, left MLO, right MLO.

**Use CC view only for the first version.** CC view is the most symmetric — the nipple is centered and the breast is compressed symmetrically. MLO view has the pectoral muscle and is harder to align.

```
For each patient keep:
  left_CC  = A
  right_CC = B
```

### 4.3 Which Cases To Include and Exclude

**Include in training:** patients where both CC views are normal (no lesion on either side).

**Exclude from all splits:**
- Patients with implants
- Post-surgical patients (mastectomy, lumpectomy)
- Patients with bilateral cancer (lesion on both sides) — approximately 10-15% of cancer cases
- Images with severe artifacts or poor quality

**For test set only:** include patients with unilateral lesion (lesion on one side, other side is clean). These are your positive cases.

### 4.4 How To Identify Normal vs. Lesion Cases in VinDr-Mammo

VinDr-Mammo provides a CSV file `finding_annotations.csv`. A case is normal if it has no bounding box annotations and BI-RADS category 1 or 2.

```python
import pandas as pd

df = pd.read_csv('finding_annotations.csv')

# Normal cases: no findings
normal_cases = df[df['finding_categories'].isna()]['study_id'].unique()

# Lesion cases: has at least one finding
lesion_cases = df[df['finding_categories'].notna()]['study_id'].unique()

# Bilateral cancer: lesion on both left and right
bilateral = df.groupby('study_id')['laterality'].apply(
    lambda x: set(x) == {'L', 'R'}
)
bilateral_ids = bilateral[bilateral].index.tolist()

# Exclude bilateral from both sets
normal_cases = [c for c in normal_cases if c not in bilateral_ids]
lesion_cases = [c for c in lesion_cases if c not in bilateral_ids]
```

### 4.5 Data Split

```python
from sklearn.model_selection import train_test_split

# Normal cases: split 70/15/15
train_normal, temp = train_test_split(normal_cases, test_size=0.30, random_state=42)
val_normal, test_normal = train_test_split(temp, test_size=0.50, random_state=42)

# Lesion cases: all go to test set
test_lesion = lesion_cases

# Final splits:
# train:      train_normal only          → CycleCUT + Flow training
# validation: val_normal only            → early stopping (no lesion labels)
# test:       test_normal + test_lesion  → evaluation
```

**Important:** The validation set contains only normal cases. You cannot use lesion cases for early stopping without breaking the unsupervised claim.

### 4.6 Preprocessing — Step by Step

Run this pipeline on every image before training. Save preprocessed images to disk as float32 numpy arrays.

**Step 1: DICOM to PNG**
```python
import pydicom
import numpy as np
from PIL import Image

def dicom_to_array(dicom_path):
    dcm = pydicom.dcmread(dicom_path)
    img = dcm.pixel_array.astype(np.float32)

    # Apply DICOM windowing if present
    if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
        center = float(dcm.WindowCenter)
        width = float(dcm.WindowWidth)
        img = np.clip(img, center - width/2, center + width/2)

    # Normalize to [0, 1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img
```

**Step 2: Breast segmentation**

Use a pretrained U-Net to remove background and pectoral muscle. Use the open-source model from: https://github.com/GalileoHealth/open-breast-seg

```python
# Load pretrained segmentation model
seg_model = load_pretrained_segmentation()

def segment_breast(img):
    mask = seg_model.predict(img)      # binary mask, 1 = breast tissue
    img_masked = img * mask            # zero out background
    return img_masked, mask
```

If you cannot find a pretrained segmentation model, use Otsu thresholding as a baseline:

```python
from skimage.filters import threshold_otsu
def segment_breast_otsu(img):
    thresh = threshold_otsu(img)
    mask = img > thresh
    return img * mask, mask
```

**Step 3: Flip right breast**

This is mandatory. CUT defines correspondence by spatial position. If you do not flip, patch (i,j) on the left breast corresponds to the wrong location on the right.

```python
def align_breasts(left_img, right_img):
    # Flip right breast horizontally so both nipples face same direction
    right_flipped = np.fliplr(right_img)
    return left_img, right_flipped
```

**Step 4: Resize**

```python
from skimage.transform import resize

def resize_image(img, target_size=(512, 512)):
    return resize(img, target_size, anti_aliasing=True, preserve_range=True)
```

**Step 5: Normalize**

Normalize each image independently (not dataset-wide) because mammogram intensities vary by machine and patient.

```python
def normalize(img):
    mean = img.mean()
    std = img.std() + 1e-8
    return (img - mean) / std
```

**Step 6: Save**

```python
import os

def preprocess_and_save(dicom_path_left, dicom_path_right, save_dir, patient_id):
    left = dicom_to_array(dicom_path_left)
    right = dicom_to_array(dicom_path_right)

    left, left_mask = segment_breast(left)
    right, right_mask = segment_breast(right)

    left, right = align_breasts(left, right)

    left = resize_image(left)
    right = resize_image(right)

    left = normalize(left)
    right = normalize(right)

    # Add channel dimension: 512×512 → 512×512×1
    left = left[:, :, np.newaxis]
    right = right[:, :, np.newaxis]

    np.save(os.path.join(save_dir, f'{patient_id}_left.npy'), left)
    np.save(os.path.join(save_dir, f'{patient_id}_right.npy'), right)
```

**Step 7: Quality check**

After preprocessing, visually inspect 20 random pairs. Verify:
- Both breasts are roughly the same size after resizing
- Nipple positions are approximately mirrored
- No large black artifacts from failed segmentation

---

## 5. Component 1: CycleCUT

### 5.1 What CycleCUT Does

CycleCUT trains two generators simultaneously:
- `G_AB`: given left breast A, predict what right breast B should look like
- `G_BA`: given right breast B, predict what left breast A should look like

It combines two training signals:
1. **CUT (Contrastive Unpaired Translation):** patch-level contrastive loss that ensures each patch in the output comes from the corresponding spatial location in the input — not from somewhere else.
2. **Cycle consistency:** ensures that if you translate A → B_fake → A_reconstructed, you get back to A. This forces both generators to be mutually consistent.

The combination is important: CUT alone gives sharp translations but no cross-direction consistency. Cycle alone can produce blurry translations. Together they produce sharp, consistent bilateral predictions.

### 5.2 Why Cycle Consistency Matters For This Task

On normal pairs, after training with cycle consistency, both residuals converge to the same distribution:

```
r_A ~ p(normal asymmetry)
r_B ~ p(normal asymmetry)
```

This means on a normal pair, the flow will assign similar likelihoods to r_A and r_B. Their difference will be near zero. On a lesion pair, one side is disrupted and the difference deviates from zero. The cycle loss is what makes this calibration possible.

### 5.3 Architecture

#### Generator G_AB and G_BA

Both generators use the same ResNet architecture. They do NOT share weights.

```python
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)   # residual connection


class Generator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9):
        super().__init__()

        # Encoder
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),           # 512→512, 64 channels
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1),    # 512→256, 128ch
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1),  # 256→128, 256ch
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True),
        ]

        # Bottleneck: 9 ResBlocks
        resblocks = [ResBlock(ngf*4) for _ in range(n_blocks)]

        # Decoder
        decoder = [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()    # output in [-1, 1]
        ]

        self.encoder = nn.Sequential(*encoder)
        self.resblocks = nn.Sequential(*resblocks)
        self.decoder = nn.Sequential(*decoder)

        # Store intermediate features for CUT loss
        self.enc_layers = [self.encoder[1], self.encoder[4], self.encoder[7]]

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.resblocks(feat)
        return self.decoder(feat)

    def get_features(self, x):
        """Return intermediate encoder features for CUT loss"""
        feats = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if layer in self.enc_layers:
                feats.append(x)
        return feats
```

#### Discriminator D_A and D_B

PatchGAN discriminator. Outputs a 32×32 real/fake map — each output pixel covers a 70×70 receptive field of the input.

```python
class Discriminator(nn.Module):
    def __init__(self, input_nc=1, ndf=64):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, stride=2, padding=1),    # 512→256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, stride=2, padding=1),       # 256→128
            nn.InstanceNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1),     # 128→64
            nn.InstanceNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, stride=1, padding=1),     # 64→63
            nn.InstanceNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, 1, 4, stride=1, padding=1)          # 63→62 ≈ 32×32
        )

    def forward(self, x):
        return self.model(x)
```

#### Patch MLP (for CUT contrastive loss)

Takes encoder feature maps and projects them to normalized embeddings.

```python
class PatchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        # x: B×C×H×W → sample patches → B×N×C
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B*H*W, C)   # flatten spatial
        x = self.net(x)
        x = nn.functional.normalize(x, dim=-1)         # L2 normalize
        return x
```

### 5.4 Loss Functions

#### Loss 1: CUT Patch Contrastive (PatchNCE)

For each patch at position (i,j) in the generated image:
- Query: patch embedding at (i,j) in G(A)
- Positive: patch embedding at (i,j) in A (same location, input image)
- Negatives: all other patches in A

```python
def patchnce_loss(feat_q, feat_k, temperature=0.07):
    """
    feat_q: query features from G(A), shape N×D
    feat_k: key features from A,    shape N×D
    N = number of patches, D = embedding dim
    """
    # Similarity between query and all keys
    logits = torch.mm(feat_q, feat_k.T) / temperature    # N×N

    # Diagonal = positive (same position)
    labels = torch.arange(logits.shape[0]).to(logits.device)

    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss
```

Apply at multiple encoder scales (shallow + deep):

```python
def cut_loss(G, mlp_list, real_A, fake_B, temperature=0.07):
    # Get features from encoder at multiple scales
    feats_A = G.get_features(real_A)      # list of feature maps
    feats_B = G.get_features(fake_B)      # list of feature maps

    total_loss = 0
    for feat_A, feat_B, mlp in zip(feats_A, feats_B, mlp_list):
        # Project to embedding space
        emb_A = mlp(feat_A)    # N×256
        emb_B = mlp(feat_B)    # N×256

        total_loss += patchnce_loss(emb_B, emb_A, temperature)

    return total_loss / len(feats_A)
```

#### Loss 2: Adversarial (LSGAN)

Use least-squares GAN — more stable than standard GAN.

```python
def adversarial_loss_G(D, fake):
    """Generator wants discriminator to output 1 (real) for its fakes"""
    return torch.mean((D(fake) - 1) ** 2)

def adversarial_loss_D(D, real, fake):
    """Discriminator wants 1 for real, 0 for fake"""
    loss_real = torch.mean((D(real) - 1) ** 2)
    loss_fake = torch.mean((D(fake.detach())) ** 2)
    return (loss_real + loss_fake) * 0.5
```

#### Loss 3: Cycle Consistency

```python
def cycle_loss(G_AB, G_BA, real_A, real_B):
    """
    Forward cycle: A → G_AB → fake_B → G_BA → reconstructed_A ≈ A
    Backward cycle: B → G_BA → fake_A → G_AB → reconstructed_B ≈ B
    """
    fake_B = G_AB(real_A)
    reconstructed_A = G_BA(fake_B)
    loss_A = torch.mean(torch.abs(real_A - reconstructed_A))

    fake_A = G_BA(real_B)
    reconstructed_B = G_AB(fake_A)
    loss_B = torch.mean(torch.abs(real_B - reconstructed_B))

    return loss_A + loss_B
```

#### Loss 4: Identity Loss

Forces G_AB(B) ≈ B — if input is already from the target domain, do not change it. This prevents the generator from shifting tissue density gratuitously.

```python
def identity_loss(G_AB, G_BA, real_A, real_B):
    """
    G_AB(B) should equal B (B is already in domain B)
    G_BA(A) should equal A (A is already in domain A)
    """
    loss_A = torch.mean(torch.abs(G_BA(real_A) - real_A))
    loss_B = torch.mean(torch.abs(G_AB(real_B) - real_B))
    return loss_A + loss_B
```

#### Total Loss

```python
lambda_cyc = 10.0    # cycle consistency weight
lambda_idt = 5.0     # identity loss weight
# Note: lambda_idt = lambda_cyc / 2 is standard practice

def total_generator_loss(G_AB, G_BA, D_A, D_B, mlp_AB, mlp_BA,
                          real_A, real_B):
    fake_B = G_AB(real_A)
    fake_A = G_BA(real_B)

    # Adversarial
    loss_adv = adversarial_loss_G(D_B, fake_B) + adversarial_loss_G(D_A, fake_A)

    # CUT contrastive (per direction)
    loss_cut = cut_loss(G_AB, mlp_AB, real_A, fake_B)
    loss_cut += cut_loss(G_BA, mlp_BA, real_B, fake_A)

    # Cycle
    loss_cyc = lambda_cyc * cycle_loss(G_AB, G_BA, real_A, real_B)

    # Identity
    loss_idt = lambda_idt * identity_loss(G_AB, G_BA, real_A, real_B)

    return loss_adv + loss_cut + loss_cyc + loss_idt
```

### 5.5 Dataset and DataLoader

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, random

class BilateralDataset(Dataset):
    def __init__(self, patient_ids, data_dir):
        self.patient_ids = patient_ids
        self.data_dir = data_dir

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        A = np.load(os.path.join(self.data_dir, f'{pid}_left.npy'))   # 512×512×1
        B = np.load(os.path.join(self.data_dir, f'{pid}_right.npy'))  # 512×512×1

        # To torch tensor: H×W×C → C×H×W
        A = torch.FloatTensor(A).permute(2, 0, 1)
        B = torch.FloatTensor(B).permute(2, 0, 1)

        return A, B

train_dataset = BilateralDataset(train_normal, data_dir='data/preprocessed/')
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
```

**Batch size is 1.** This is standard for CUT and CycleGAN. Do not change it — PatchNCE loss is computed within each sample, not across the batch.

### 5.6 Training Loop

```python
import torch.optim as optim

# Initialize models
G_AB = Generator().cuda()
G_BA = Generator().cuda()
D_A  = Discriminator().cuda()
D_B  = Discriminator().cuda()
mlp_AB = [PatchMLP(input_dim).cuda() for input_dim in [64, 128, 256]]
mlp_BA = [PatchMLP(input_dim).cuda() for input_dim in [64, 128, 256]]

# Optimizers
opt_G = optim.Adam(
    list(G_AB.parameters()) + list(G_BA.parameters()) +
    sum([list(m.parameters()) for m in mlp_AB + mlp_BA], []),
    lr=2e-4, betas=(0.5, 0.999)
)
opt_D = optim.Adam(
    list(D_A.parameters()) + list(D_B.parameters()),
    lr=2e-4, betas=(0.5, 0.999)
)

n_epochs = 200
decay_start = 100    # start LR decay at epoch 100

def get_lr_lambda(epoch):
    if epoch < decay_start:
        return 1.0
    else:
        return 1.0 - (epoch - decay_start) / (n_epochs - decay_start)

scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, get_lr_lambda)
scheduler_D = optim.lr_scheduler.LambdaLR(opt_D, get_lr_lambda)

for epoch in range(n_epochs):
    for i, (real_A, real_B) in enumerate(train_loader):
        real_A = real_A.cuda()
        real_B = real_B.cuda()

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        # --- Update Discriminators ---
        opt_D.zero_grad()
        loss_D_A = adversarial_loss_D(D_A, real_A, fake_A)
        loss_D_B = adversarial_loss_D(D_B, real_B, fake_B)
        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        opt_D.step()

        # --- Update Generators ---
        opt_G.zero_grad()
        loss_G = total_generator_loss(
            G_AB, G_BA, D_A, D_B, mlp_AB, mlp_BA, real_A, real_B
        )
        loss_G.backward()
        opt_G.step()

    scheduler_G.step()
    scheduler_D.step()

    # Save checkpoint every 10 epochs
    if epoch % 10 == 0:
        torch.save({'G_AB': G_AB.state_dict(), 'G_BA': G_BA.state_dict()},
                   f'checkpoints/cyclecut_epoch{epoch}.pth')

        # Visual check: save sample translations
        with torch.no_grad():
            sample_A, sample_B = next(iter(train_loader))
            sample_A = sample_A.cuda()
            fake_B_sample = G_AB(sample_A)
            save_image(fake_B_sample, f'samples/fake_B_epoch{epoch}.png')
```

### 5.7 What To Monitor During Training

Check these at every 10th epoch:

1. **Generator loss** should decrease in the first 50 epochs then stabilize.
2. **Discriminator loss** should stabilize around 0.5 (random chance) — if it goes to zero the generator has collapsed.
3. **Visual output:** G_AB(A) should look like a plausible breast. It should not be: (a) identical to A (identity shortcut), (b) a blurry average, (c) a noisy artifact.
4. **Identity check:** G_AB(B) − B should be near zero everywhere.
5. **Cycle check:** G_BA(G_AB(A)) should closely resemble A.

### 5.8 Hardware and Training Time

```
GPU:      2× NVIDIA A100 (40GB) or equivalent
          Minimum: 1× RTX 3090 (24GB) with batch size 1
RAM:      32GB
Storage:  ~100GB for VinDr-Mammo preprocessed
Time:     ~48 hours for 200 epochs on VinDr-Mammo
```

---

## 6. Component 2: Residual Extraction

### 6.1 What This Step Does

After CycleCUT training is complete and frozen, compute residual maps for all training pairs. These are the training data for the normalizing flow.

### 6.2 How To Compute

```python
G_AB.eval()
G_BA.eval()

residuals = []

with torch.no_grad():
    for patient_id in train_normal:
        A = load_image(patient_id, 'left').cuda()    # 1×1×512×512
        B = load_image(patient_id, 'right').cuda()   # 1×1×512×512

        B_fake = G_AB(A)    # predicted B from A
        A_fake = G_BA(B)    # predicted A from B

        r_A = torch.abs(A - A_fake).cpu().numpy()   # 1×1×512×512
        r_B = torch.abs(B - B_fake).cpu().numpy()   # 1×1×512×512

        # Save
        np.save(f'residuals/{patient_id}_rA.npy', r_A[0, 0])   # 512×512
        np.save(f'residuals/{patient_id}_rB.npy', r_B[0, 0])   # 512×512
```

**Important:** Use absolute difference, not signed difference. The flow will learn the distribution of magnitudes, not directions.

---

## 7. Component 3: Patch Encoder

### 7.1 What This Does

A raw 32×32 residual patch has 1024 dimensions — too high for a normalizing flow to model reliably. The patch encoder compresses it to a 128-dimensional feature vector while preserving relevant texture information.

### 7.2 Architecture

```python
class PatchEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),      # 32×32×32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 16×16×64
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8×8×128
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),             # 1×1×128
            nn.Flatten(),                        # 128
            nn.Linear(128, out_dim),             # 128
            nn.ReLU(inplace=True)
        )
        self.out_dim = out_dim

    def forward(self, x):
        z = self.net(x)
        return nn.functional.normalize(z, dim=-1)    # L2 normalize → unit sphere
```

### 7.3 How To Extract Patches

```python
def extract_patches(residual_map, patch_size=32, stride=16):
    """
    residual_map: H×W numpy array
    Returns: list of patches, each patch_size×patch_size
    Also returns: list of (row, col) positions for reconstruction
    """
    H, W = residual_map.shape
    patches = []
    positions = []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = residual_map[r:r+patch_size, c:c+patch_size]
            patches.append(patch)
            positions.append((r, c))

    return np.array(patches), positions    # N×32×32, N×2
```

For a 512×512 image with patch_size=32, stride=16:
```
Number of patches per image = ((512-32)/16 + 1)² = 31² = 961 patches
```

### 7.4 Pretraining the Encoder with SimCLR

Train the encoder to produce similar embeddings for augmented views of the same patch and dissimilar embeddings for different patches.

```python
import torchvision.transforms as T

def augment_patch(patch):
    """Apply random augmentations to a residual patch"""
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(28, padding=4),
        T.Resize(32),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
    ])
    # patch: 32×32 numpy → PIL → augment → tensor
    patch_pil = Image.fromarray((patch * 255).astype(np.uint8))
    aug1 = transforms(patch_pil)
    aug2 = transforms(patch_pil)
    return aug1, aug2


def simclr_loss(z1, z2, temperature=0.5):
    """
    z1, z2: augmented views, shape B×D
    """
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)   # 2B×D

    # Cosine similarity matrix
    sim = torch.mm(z, z.T) / temperature   # 2B×2B

    # Mask out self-similarity
    mask = torch.eye(2*B).bool().to(z.device)
    sim = sim.masked_fill(mask, float('-inf'))

    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.arange(B).to(z.device)
    labels = torch.cat([labels + B, labels])   # 2B

    return nn.CrossEntropyLoss()(sim, labels)


# Training
encoder = PatchEncoder(out_dim=128).cuda()
opt_enc = optim.Adam(encoder.parameters(), lr=1e-3)

for epoch in range(50):
    for batch_patches in patch_loader:    # batch of random residual patches
        aug1, aug2 = zip(*[augment_patch(p) for p in batch_patches])
        aug1 = torch.stack(aug1).cuda()
        aug2 = torch.stack(aug2).cuda()

        z1 = encoder(aug1)
        z2 = encoder(aug2)

        loss = simclr_loss(z1, z2)
        opt_enc.zero_grad()
        loss.backward()
        opt_enc.step()
```

After SimCLR pretraining, **freeze the encoder**. Do not update it during flow training.

---

## 8. Component 4: Normalizing Flow

### 8.1 What This Does

The normalizing flow learns the distribution `p(z | normal)` — what feature vectors look like when the residual comes from a normal bilateral pair.

At test time, it assigns a likelihood score to every patch. Low likelihood = that patch residual is unusual = possible lesion.

### 8.2 Why A Normalizing Flow (Not a VAE or Classifier)

A normalizing flow gives **exact log-likelihood** — not an approximation (unlike VAE) and not a classification score that requires labels (unlike a classifier). This is essential: we never see lesion patches during training, so we cannot train a classifier.

### 8.3 Architecture: Neural Spline Flow

Use the `nflows` library. Install: `pip install nflows`

```python
from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAutoregressiveFlow
from nflows.transforms.permutations import ReversePermutation

def build_flow(input_dim=128, n_layers=8, hidden_dim=256):
    transforms = []
    for i in range(n_layers):
        transforms.append(ReversePermutation(features=input_dim))
        transforms.append(
            MaskedAutoregressiveFlow(
                features=input_dim,
                hidden_features=hidden_dim,
                num_blocks=2,
                use_residual_blocks=True,
                use_random_masks=False,
                activation=torch.relu,
                dropout_probability=0.0,
                use_batch_norm=False
            )
        )

    transform = CompositeTransform(transforms)
    base_distribution = StandardNormal(shape=[input_dim])
    flow = Flow(transform=transform, distribution=base_distribution)
    return flow

flow = build_flow(input_dim=128, n_layers=8, hidden_dim=256).cuda()
```

### 8.4 Training Dataset For The Flow

Extract ALL patches from ALL normal training residuals and save them:

```python
all_patches = []

for patient_id in train_normal:
    r_A = np.load(f'residuals/{patient_id}_rA.npy')    # 512×512
    r_B = np.load(f'residuals/{patient_id}_rB.npy')    # 512×512

    patches_A, _ = extract_patches(r_A, patch_size=32, stride=16)
    patches_B, _ = extract_patches(r_B, patch_size=32, stride=16)

    all_patches.append(patches_A)
    all_patches.append(patches_B)

all_patches = np.concatenate(all_patches, axis=0)   # M×32×32
# M ≈ 2 × N_patients × 961 patches per image
# For 2000 normal patients: M ≈ 3.8 million patches

np.save('flow_training_patches.npy', all_patches)
```

### 8.5 Training The Flow

```python
class PatchFlowDataset(Dataset):
    def __init__(self, patches, encoder):
        self.patches = patches
        self.encoder = encoder

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]   # 32×32
        patch_tensor = torch.FloatTensor(patch).unsqueeze(0)   # 1×32×32

        with torch.no_grad():
            z = self.encoder(patch_tensor.unsqueeze(0))    # 1×128
        return z.squeeze(0)   # 128


flow_dataset = PatchFlowDataset(all_patches, encoder)
flow_loader  = DataLoader(flow_dataset, batch_size=256, shuffle=True, num_workers=4)

opt_flow = optim.Adam(flow.parameters(), lr=1e-4)
scheduler_flow = optim.lr_scheduler.CosineAnnealingLR(opt_flow, T_max=100)

for epoch in range(100):
    epoch_loss = 0
    for z_batch in flow_loader:
        z_batch = z_batch.cuda()
        loss = -flow.log_prob(z_batch).mean()   # negative log likelihood
        opt_flow.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
        opt_flow.step()
        epoch_loss += loss.item()

    scheduler_flow.step()
    print(f'Epoch {epoch}: NLL = {epoch_loss / len(flow_loader):.4f}')

    # Validate on held-out normal patches every 10 epochs
    if epoch % 10 == 0:
        val_nll = compute_nll_on_val(flow, encoder, val_normal)
        print(f'  Val NLL: {val_nll:.4f}')
```

**What to watch:**
- Training NLL should decrease steadily
- Validation NLL should track training NLL (no overfitting)
- If NLL variance is near zero → flow collapsed. Reduce LR, add gradient clipping.

---

## 9. Test Time Inference

### 9.1 Full Pipeline

```python
def compute_anomaly_heatmap(A_test, B_test, G_AB, G_BA, encoder, flow,
                             patch_size=32, stride=16):
    """
    A_test, B_test: 1×1×512×512 tensors
    Returns: heatmap as 2D numpy array (upsampled to 512×512)
    """
    G_AB.eval(); G_BA.eval(); encoder.eval(); flow.eval()

    with torch.no_grad():
        # Step 1: Generate bilateral predictions
        B_fake = G_AB(A_test)
        A_fake = G_BA(B_test)

        # Step 2: Compute residuals
        r_A = torch.abs(A_test - A_fake).cpu().numpy()[0, 0]   # 512×512
        r_B = torch.abs(B_test - B_fake).cpu().numpy()[0, 0]   # 512×512

        # Step 3: Extract patches
        patches_A, positions = extract_patches(r_A, patch_size, stride)
        patches_B, _         = extract_patches(r_B, patch_size, stride)

        # Step 4: Encode patches
        patches_A_t = torch.FloatTensor(patches_A).unsqueeze(1).cuda()  # N×1×32×32
        patches_B_t = torch.FloatTensor(patches_B).unsqueeze(1).cuda()

        z_A = encoder(patches_A_t)   # N×128
        z_B = encoder(patches_B_t)   # N×128

        # Step 5: Compute flow scores (negative log likelihood)
        score_A = -flow.log_prob(z_A).cpu().numpy()   # N
        score_B = -flow.log_prob(z_B).cpu().numpy()   # N

        # Step 6: Likelihood ratio per patch
        ratio = score_B - score_A   # N
        # ratio > 0 → B side anomalous
        # ratio < 0 → A side anomalous
        # ratio ≈ 0 → both normal

        # Step 7: Reconstruct spatial heatmap
        H_map = (512 - patch_size) // stride + 1   # 31
        W_map = (512 - patch_size) // stride + 1   # 31
        heatmap_low = ratio.reshape(H_map, W_map)   # 31×31

        # Step 8: Upsample to original resolution
        heatmap = cv2.resize(heatmap_low, (512, 512),
                             interpolation=cv2.INTER_LINEAR)   # 512×512

    return heatmap, r_A, r_B


def get_image_score(heatmap, method='max'):
    """Convert heatmap to single image-level anomaly score"""
    if method == 'max':
        return heatmap.max()
    elif method == 'top_k_mean':
        k = int(0.05 * heatmap.size)   # top 5% of patches
        return np.sort(heatmap.flatten())[-k:].mean()
```

---

## 10. Evaluation

### 10.1 Metrics

```python
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

def evaluate(model_scores, labels):
    """
    model_scores: list of image-level anomaly scores
    labels:       list of binary labels (0=normal, 1=lesion)
    """
    auroc = roc_auc_score(labels, model_scores)
    auprc = average_precision_score(labels, model_scores)

    # Sensitivity at 90% specificity
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, model_scores)
    idx = np.argmin(np.abs(fpr - 0.10))   # specificity = 1 - fpr = 0.90
    sens_at_spec90 = tpr[idx]

    return {'AUROC': auroc, 'AUPRC': auprc, 'Sens@Spec90': sens_at_spec90}
```

### 10.2 Localization Metric

```python
def pointing_game(heatmap, lesion_bbox):
    """
    heatmap:     512×512 anomaly map
    lesion_bbox: (x_min, y_min, x_max, y_max)
    Returns True if heatmap maximum is inside the bounding box
    """
    max_pos = np.unravel_index(heatmap.argmax(), heatmap.shape)
    r, c = max_pos
    x_min, y_min, x_max, y_max = lesion_bbox
    return (y_min <= r <= y_max) and (x_min <= c <= x_max)
```

### 10.3 Stratified Evaluation

Always report results stratified by:
- BI-RADS density (1, 2, 3, 4)
- Lesion type (mass, calcification, asymmetry, distortion)
- Lesion size (small: <1cm, medium: 1-2cm, large: >2cm)

---

---

# PLAN A: Normal Pairs Only

## A.1 What Is Different From Base Recipe

Plan A uses the base recipe exactly as written above. No synthetic corruptions. No lesion data in any form during training.

## A.2 Training Sequence

```
Step 1: Preprocess all images (Section 4.6)
Step 2: Train CycleCUT on train_normal only (Section 5.6)
Step 3: Extract residuals from train_normal (Section 6.2)
Step 4: Pretrain patch encoder with SimCLR on normal residuals (Section 7.4)
Step 5: Train normalizing flow on normal residual patches (Section 8.5)
Step 6: Evaluate on test set (Section 9, 10)
```

## A.3 Experiments — Run In This Order

---

### Experiment A1: OOD Verification ← RUN THIS FIRST

**Scientific question:** Are residual patches from lesion-containing breasts out-of-distribution relative to the flow trained on normal pairs?

**Why first:** This is the foundational assumption of the entire method. If it fails, Plan A cannot work and Plan B is necessary.

**How to run:**

```python
# Compute flow scores on normal test patches
normal_scores = []
for patient_id in test_normal:
    r_A = compute_residual(patient_id, side='A')
    r_B = compute_residual(patient_id, side='B')
    patches_A, _ = extract_patches(r_A)
    patches_B, _ = extract_patches(r_B)
    scores = compute_flow_scores(patches_A + patches_B)   # all patches
    normal_scores.extend(scores)

# Compute flow scores on lesion patches only (inside bounding box)
lesion_scores = []
for patient_id in test_lesion:
    r_B = compute_residual(patient_id, side='B')   # assume lesion on B
    patches_in_bbox = extract_patches_in_bbox(r_B, lesion_bbox[patient_id])
    scores = compute_flow_scores(patches_in_bbox)
    lesion_scores.extend(scores)

# AUROC at patch level
from sklearn.metrics import roc_auc_score
labels = [0]*len(normal_scores) + [1]*len(lesion_scores)
scores = normal_scores + lesion_scores
patch_auroc = roc_auc_score(labels, scores)
print(f'Patch-level AUROC: {patch_auroc:.3f}')

# Plot histograms
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))
plt.hist(normal_scores, bins=100, alpha=0.5, label='Normal patches', density=True)
plt.hist(lesion_scores, bins=100, alpha=0.5, label='Lesion patches', density=True)
plt.xlabel('Flow score (−log p)')
plt.ylabel('Density')
plt.legend()
plt.title(f'Patch-level AUROC = {patch_auroc:.3f}')
plt.savefig('results/A1_ood_verification.png')
```

**What to report:**
- Patch-level AUROC (random = 0.5, target: > 0.65)
- Histogram plot of score distributions

**Decision rule:**
```
If patch AUROC > 0.65 → OOD assumption holds → continue with Plan A
If patch AUROC < 0.65 → OOD assumption weak → proceed to Plan B
```

---

### Experiment A2: Bilateral Symmetry Verification

**Scientific question:** Are r_A and r_B identically distributed on normal pairs? This must hold for the shared flow and likelihood ratio to be valid.

**How to run:**

```python
from scipy.stats import ks_2samp
import numpy as np

scores_A = []
scores_B = []
ratio_scores = []

for patient_id in test_normal:
    # Compute heatmap
    heatmap, r_A, r_B = compute_anomaly_heatmap(...)

    # Image-level scores
    score_A = -compute_mean_flow_score(r_A)
    score_B = -compute_mean_flow_score(r_B)

    scores_A.append(score_A)
    scores_B.append(score_B)
    ratio_scores.append(score_B - score_A)

# KS test
stat, p_value = ks_2samp(scores_A, scores_B)
print(f'KS test: stat={stat:.4f}, p={p_value:.4f}')
print(f'Expected: p > 0.05 (distributions are the same)')

# Ratio distribution
print(f'Mean ratio on normal pairs: {np.mean(ratio_scores):.4f}')
print(f'Std ratio on normal pairs:  {np.std(ratio_scores):.4f}')
print(f'Expected: mean ≈ 0')

# Plot
plt.figure(figsize=(10, 4))
plt.hist(ratio_scores, bins=50, density=True)
plt.axvline(0, color='r', linestyle='--', label='Zero')
plt.xlabel('score_B − score_A')
plt.title(f'Normal pair ratio distribution (mean={np.mean(ratio_scores):.3f})')
plt.savefig('results/A2_symmetry.png')
```

**What to report:**
- KS test statistic and p-value
- Mean and std of ratio on normal pairs
- Histogram of ratio distribution

**Decision rule:**
```
If p > 0.05 → symmetric → use shared flow (one flow for both sides)
If p < 0.05 → asymmetric → train separate flows per side, report both
```

---

### Experiment A3: Does Cycle Loss Help?

**Scientific question:** Does the cycle consistency loss reduce normal residual variance compared to independent CUT (without cycle loss)?

**How to run:** Train two models:

```
Model 1: CycleCUT (with cycle loss, lambda_cyc=10)
Model 2: CUT_independent (no cycle loss, lambda_cyc=0)
```

For both, compute residual standard deviation on validation normal pairs:

```python
def residual_stats(patient_ids, G_AB, G_BA):
    all_r_A = []
    all_r_B = []
    for pid in patient_ids:
        r_A, r_B = compute_residuals(pid, G_AB, G_BA)
        all_r_A.append(r_A.std())
        all_r_B.append(r_B.std())
    return np.mean(all_r_A), np.mean(all_r_B)

std_A_cyclecut, std_B_cyclecut = residual_stats(val_normal, G_AB_cyclecut, G_BA_cyclecut)
std_A_indep,    std_B_indep    = residual_stats(val_normal, G_AB_indep,    G_BA_indep)

print(f'CycleCUT residual std:    r_A={std_A_cyclecut:.4f}, r_B={std_B_cyclecut:.4f}')
print(f'Independent CUT std:      r_A={std_A_indep:.4f},    r_B={std_B_indep:.4f}')
print(f'Expected: CycleCUT < Independent')
```

---

### Experiment A4: Does Flow Add Value Over Raw Magnitude?

**Scientific question:** Is the flow score better than simply comparing residual magnitudes?

**How to run:**

```python
# Method 1: Raw magnitude difference
def raw_magnitude_score(A_test, B_test, G_AB, G_BA):
    r_A, r_B = compute_residuals(A_test, B_test, G_AB, G_BA)
    score_A = r_A.max()
    score_B = r_B.max()
    return score_B - score_A

# Method 2: Flow likelihood ratio
def flow_score(A_test, B_test, G_AB, G_BA, encoder, flow):
    heatmap, _, _ = compute_anomaly_heatmap(...)
    return heatmap.max()

# Evaluate both on test set
magnitude_scores = [raw_magnitude_score(A, B, G_AB, G_BA) for A, B in test_pairs]
flow_scores      = [flow_score(A, B, ...) for A, B in test_pairs]
labels           = [label for _, _, label in test_pairs]

print(f'Raw magnitude AUROC: {roc_auc_score(labels, magnitude_scores):.3f}')
print(f'Flow score AUROC:    {roc_auc_score(labels, flow_scores):.3f}')
```

**Decision:** If flow AUROC ≤ magnitude AUROC, the flow adds no value — report raw magnitude as the baseline and explain what the flow should theoretically add.

---

### Experiment A5: Ablation Table

Train all variants and report in one table:

| Model | Image AUROC | Patch AUROC | Pointing Game |
|---|---|---|---|
| Single-breast flow baseline | | | |
| Bilateral raw magnitude | | | |
| CUT one-direction + flow | | | |
| CycleCUT (no cycle loss) + flow | | | |
| **CycleCUT + flow (Plan A)** | | | |

For each: report mean ± std over 3 random seeds.

---

### Experiment A6: Heatmap Localization

**Scientific question:** Does the heatmap visually localize to the lesion?

**How to run:**

```python
import matplotlib.pyplot as plt

for patient_id in test_lesion[:10]:   # inspect 10 cases
    A, B = load_pair(patient_id)
    heatmap, r_A, r_B = compute_anomaly_heatmap(A, B, ...)
    bbox = lesion_bboxes[patient_id]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(A[0,0].cpu(), cmap='gray', vmin=-2, vmax=2)
    axes[0].set_title('Left breast (A)')

    axes[1].imshow(B[0,0].cpu(), cmap='gray', vmin=-2, vmax=2)
    rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
                          fill=False, color='red', linewidth=2)
    axes[1].add_patch(rect)
    axes[1].set_title('Right breast (B) + lesion bbox')

    axes[2].imshow(r_B, cmap='hot')
    axes[2].set_title('Residual r_B')

    axes[3].imshow(heatmap, cmap='hot')
    axes[3].set_title('Anomaly heatmap')

    plt.savefig(f'results/heatmap_{patient_id}.png')
    plt.close()

# Quantitative: pointing game
pointing_scores = [
    pointing_game(compute_heatmap(pid), lesion_bboxes[pid])
    for pid in test_lesion
]
print(f'Pointing game accuracy: {np.mean(pointing_scores):.3f}')
print(f'Random baseline: ~{mean_bbox_fraction:.3f}')
```

---

---

# PLAN B: Normal Pairs + Synthetic Corruptions

## B.1 What Plan B Adds To Plan A

Plan B adds one component between CycleCUT training and residual extraction:

**Healing supervision:** show the generator lesion-like inputs during training, teach it to produce a healthy output. This ensures that when the generator sees a real lesion at test time, it produces a clearly different residual — strengthening the OOD signal.

**Run Plan A first.** Run Plan B only if Experiment A1 shows patch AUROC < 0.65, or if you want to demonstrate improvement over the unsupervised baseline.

## B.2 What Changes In Training

CycleCUT training changes in one way: add a healing loss using synthetic corruptions and add real lesion images as discriminator negatives.

Everything else (encoder, flow, inference) is identical to Plan A.

## B.3 Synthetic Corruption Generation

Start with Type 1 (Gaussian blobs). Add Types 2 and 3 only if Experiment B3 shows they matter.

#### Type 1: Gaussian Blob

```python
import numpy as np

def add_gaussian_blob(image, breast_mask):
    """
    image:       512×512 numpy array (normalized mammogram)
    breast_mask: 512×512 binary mask (1 = breast tissue)
    Returns:     corrupted image
    """
    corrupted = image.copy()

    # Sample parameters
    amplitude = np.random.uniform(0.2, 0.8)
    sigma = np.random.uniform(5, 20)   # pixels

    # Sample center inside breast mask
    mask_pixels = np.argwhere(breast_mask > 0)
    center_idx = np.random.randint(len(mask_pixels))
    cy, cx = mask_pixels[center_idx]

    # Create blob
    Y, X = np.ogrid[:512, :512]
    blob = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))

    corrupted += blob
    return corrupted
```

#### Type 2: Copy-Paste from Lesion Bank

```python
from scipy.ndimage import gaussian_filter

def copy_paste_lesion(image, lesion_patch, breast_mask):
    """
    image:         512×512 target image
    lesion_patch:  H×W lesion patch from CBIS-DDSM
    breast_mask:   512×512 binary mask
    """
    corrupted = image.copy()
    H, W = lesion_patch.shape

    # Random position inside breast mask
    mask_pixels = np.argwhere(breast_mask > 0)
    valid = mask_pixels[
        (mask_pixels[:, 0] > H//2) & (mask_pixels[:, 0] < 512-H//2) &
        (mask_pixels[:, 1] > W//2) & (mask_pixels[:, 1] < 512-W//2)
    ]
    cy, cx = valid[np.random.randint(len(valid))]

    # Soft blending mask
    blend_mask = np.ones((H, W))
    border = 5
    blend_mask[:border, :] = np.linspace(0, 1, border)[:, None]
    blend_mask[-border:, :] = np.linspace(1, 0, border)[:, None]
    blend_mask[:, :border] = np.minimum(blend_mask[:, :border],
                                         np.linspace(0, 1, border)[None, :])
    blend_mask[:, -border:] = np.minimum(blend_mask[:, -border:],
                                          np.linspace(1, 0, border)[None, :])

    # Insert
    r0, r1 = cy - H//2, cy + H//2
    c0, c1 = cx - W//2, cx + W//2
    corrupted[r0:r1, c0:c1] = (
        (1 - blend_mask) * corrupted[r0:r1, c0:c1] +
        blend_mask * lesion_patch
    )
    return corrupted
```

**How to build the lesion bank from CBIS-DDSM:**

```python
def build_lesion_bank(cbis_ddsm_dir, save_dir, n_patches=500):
    """Extract lesion patches from CBIS-DDSM segmentation masks"""
    patches = []
    for case in os.listdir(cbis_ddsm_dir):
        img = load_image(os.path.join(cbis_ddsm_dir, case, 'image.dcm'))
        mask = load_mask(os.path.join(cbis_ddsm_dir, case, 'mask.dcm'))

        # Extract bounding box of lesion
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]

        patch = img[r_min:r_max, c_min:c_max]
        patch = resize_image(patch, (64, 64))   # standardize size
        patches.append(patch)

        if len(patches) >= n_patches:
            break

    np.save(os.path.join(save_dir, 'lesion_bank.npy'), np.array(patches))
    return patches
```

## B.4 Modified CycleCUT Training

Add healing loss to the total generator loss. The rest is unchanged.

```python
def healing_loss(G_AB, G_BA, real_A, real_B, corrupt_B, breast_mask_A, breast_mask_B):
    """
    G_AB(real_A) should predict clean B (not corrupted B)
    G_BA(corrupt_B) should predict clean A
    """
    # Predict from healthy A → compare to clean B
    predicted_B = G_AB(real_A)
    loss_1 = torch.mean(torch.abs(predicted_B - real_B))

    # Predict from corrupted B → should recover clean A
    predicted_A = G_BA(corrupt_B)
    loss_2 = torch.mean(torch.abs(predicted_A - real_A))

    return loss_1 + loss_2


lambda_heal = 5.0

# In the training loop, for each batch:
# Randomly corrupt B with 50% probability per batch
if np.random.random() < 0.5:
    corrupt_B = add_gaussian_blob(real_B, breast_mask_B)
    corrupt_B = torch.FloatTensor(corrupt_B).cuda()
    loss_heal = lambda_heal * healing_loss(G_AB, G_BA, real_A, real_B, corrupt_B, ...)
else:
    corrupt_B = None
    loss_heal = 0.0

loss_G = loss_adv + loss_cut + loss_cyc + loss_idt + loss_heal
```

## B.5 Discriminator Negatives From Real Lesion Images

Add unpaired lesion images to the discriminator's fake pool during training.

```python
# Create a separate loader for lesion images (unpaired)
lesion_dataset = SingleBreastDataset(lesion_cases_B, data_dir)
lesion_loader  = DataLoader(lesion_dataset, batch_size=1, shuffle=True)
lesion_iter    = iter(lesion_loader)

# In the discriminator update step:
def get_lesion_image():
    global lesion_iter
    try:
        return next(lesion_iter).cuda()
    except StopIteration:
        lesion_iter = iter(lesion_loader)
        return next(lesion_iter).cuda()

# Modified discriminator loss for D_B:
def adversarial_loss_D_with_negatives(D_B, real_B, fake_B, lesion_B):
    loss_real    = torch.mean((D_B(real_B) - 1) ** 2)
    loss_fake_G  = torch.mean(D_B(fake_B.detach()) ** 2)
    loss_fake_L  = torch.mean(D_B(lesion_B) ** 2)   # lesion = also fake
    return (loss_real + 0.5*loss_fake_G + 0.5*loss_fake_L) * 0.5

# Use this instead of adversarial_loss_D for D_B only
lesion_B_sample = get_lesion_image()
loss_D_B = adversarial_loss_D_with_negatives(D_B, real_B, fake_B, lesion_B_sample)
```

## B.6 Training Sequence for Plan B

```
Step 1: Preprocess all images (same as Plan A)
Step 2: Build lesion bank from CBIS-DDSM (Section B.3)
Step 3: Train CycleCUT with healing loss + discriminator negatives (Sections B.4, B.5)
Step 4: Extract residuals from train_normal only (same as Plan A)
Step 5: Pretrain patch encoder with SimCLR (same as Plan A)
Step 6: Train normalizing flow (same as Plan A)
Step 7: Evaluate (same as Plan A)
```

## B.7 Experiments — Plan B

---

### Experiment B1: Does Healing Work?

**Scientific question:** Does G_BA produce a different (healthier) output when given a lesion breast as input, compared to Plan A?

```python
# For 10 test lesion cases:
for patient_id in test_lesion[:10]:
    B_lesion = load_image(patient_id, 'right')   # has lesion
    A_healthy = load_image(patient_id, 'left')   # no lesion

    with torch.no_grad():
        # Plan A generator
        A_fake_planA = G_BA_planA(B_lesion)
        r_A_planA = torch.abs(A_healthy - A_fake_planA)

        # Plan B generator
        A_fake_planB = G_BA_planB(B_lesion)
        r_A_planB = torch.abs(A_healthy - A_fake_planB)

    # Coverage: what fraction of lesion bbox is in top-10% of residual?
    coverage_A = compute_coverage(r_A_planA, lesion_bbox[patient_id], top_pct=0.10)
    coverage_B = compute_coverage(r_A_planB, lesion_bbox[patient_id], top_pct=0.10)
    print(f'{patient_id}: Plan A coverage={coverage_A:.3f}, Plan B coverage={coverage_B:.3f}')

    # Visualize G_BA outputs
    fig, axes = plt.subplots(1, 4, figsize=(16,4))
    axes[0].imshow(B_lesion[0,0].cpu(), cmap='gray')
    axes[0].set_title('Lesion breast input')
    axes[1].imshow(A_fake_planA[0,0].cpu(), cmap='gray')
    axes[1].set_title('G_BA output (Plan A)')
    axes[2].imshow(A_fake_planB[0,0].cpu(), cmap='gray')
    axes[2].set_title('G_BA output (Plan B)')
    axes[3].imshow(lesion_bbox_overlay(B_lesion, lesion_bbox[patient_id]), cmap='gray')
    axes[3].set_title('Lesion location reference')
    plt.savefig(f'results/B1_healing_{patient_id}.png')
```

---

### Experiment B2: Does Plan B Improve AUROC Over Plan A?

```python
# Both models evaluated on same test set
plan_A_scores = evaluate_all(test_pairs, G_AB_A, G_BA_A, encoder_A, flow_A)
plan_B_scores = evaluate_all(test_pairs, G_AB_B, G_BA_B, encoder_B, flow_B)

auroc_A = roc_auc_score(labels, plan_A_scores)
auroc_B = roc_auc_score(labels, plan_B_scores)

# Bootstrap confidence intervals
from sklearn.utils import resample
n_bootstrap = 1000
deltas = []
for _ in range(n_bootstrap):
    idx = resample(range(len(labels)))
    a = roc_auc_score([labels[i] for i in idx], [plan_A_scores[i] for i in idx])
    b = roc_auc_score([labels[i] for i in idx], [plan_B_scores[i] for i in idx])
    deltas.append(b - a)

ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
print(f'Plan A AUROC: {auroc_A:.3f}')
print(f'Plan B AUROC: {auroc_B:.3f}')
print(f'Delta: {auroc_B - auroc_A:.3f} (95% CI: [{ci_low:.3f}, {ci_high:.3f}])')
```

If the CI includes zero → improvement is not statistically significant → Plan A is sufficient.

---

### Experiment B3: Corruption Type Ablation

```
Train four Plan B variants:
  B_blob:   Gaussian blobs only
  B_paste:  Copy-paste only (requires CBIS-DDSM lesion bank)
  B_all:    Both combined

Report AUROC for each, stratified by lesion type (masses vs. calcifications)
```

**Key question:** Do more realistic corruptions help? If blobs suffice, use blobs.

---

### Experiment B4: Corruption Gap Test

**Scientific question:** Does training with mass-like corruptions generalize to calcification detection?

```python
# Train on corruptions that mimic masses (large blobs, copy-paste masses)
# Test on:
#   (a) Mass cases only
#   (b) Calcification cases only

mass_cases      = [p for p in test_lesion if lesion_type[p] == 'mass']
calcif_cases    = [p for p in test_lesion if lesion_type[p] == 'calcification']

auroc_mass  = evaluate(mass_cases, plan_B_model)
auroc_calcif = evaluate(calcif_cases, plan_B_model)
print(f'Mass AUROC:           {auroc_mass:.3f}')
print(f'Calcification AUROC: {auroc_calcif:.3f}')
print(f'Gap: {auroc_mass - auroc_calcif:.3f}')
```

Large gap (> 0.1) → corruption type matters. Use diverse corruptions.
Small gap (< 0.05) → model generalizes → OOD signal is robust.

---

### Experiment B5: Label Efficiency

**Scientific question:** How many normal training pairs does the method need?

```python
fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
aurocs = []

for frac in fractions:
    n = int(frac * len(train_normal))
    subset = train_normal[:n]

    # Train CycleCUT + flow on subset
    model = train_full_pipeline(subset)
    auroc = evaluate(test_pairs, model)
    aurocs.append(auroc)
    print(f'{int(frac*100)}% normal pairs ({n} cases): AUROC = {auroc:.3f}')

# Plot learning curve
plt.plot([int(f*100) for f in fractions], aurocs, marker='o')
plt.xlabel('% of normal training pairs used')
plt.ylabel('AUROC')
plt.title('Label efficiency curve')
plt.savefig('results/B5_label_efficiency.png')
```

---

## 11. Final Comparison Table (Both Plans)

Report this table in the paper:

| Model | AUROC | Patch AUROC | Pointing Game | Notes |
|---|---|---|---|---|
| Single-breast baseline | | | | Lower bound |
| Raw magnitude bilateral | | | | No flow |
| Plan A: CycleCUT + flow | | | | Fully unsupervised |
| Plan B: + Gaussian blobs | | | | Weakly supervised |
| Plan B: + Copy-paste | | | | |
| Plan B: + Discriminator neg | | | | |
| Plan B: Full | | | | |

---

## 12. Decision Guide

```
Start here:
│
├── Run Experiment A1 (OOD verification)
│     │
│     ├── Patch AUROC > 0.65?
│     │     YES → Plan A is working
│     │           Run A2, A3, A4, A5, A6
│     │           Plan B = ablation only
│     │
│     └── Patch AUROC < 0.65?
│           NO  → Plan A is not sufficient
│                 Move to Plan B immediately
│                 Run B1 first (healing verification)
│
├── After Plan A results:
│     Always run Plan B as well
│     Compare: does healing improve AUROC significantly?
│     If yes → Plan B is the main method, Plan A is the baseline
│     If no  → Plan A is the main method, Plan B is an ablation
│
└── Final paper structure:
      Plan A AUROC < Plan B AUROC significantly:
        → Lead with Plan B, Plan A as unsupervised variant
      Plan A AUROC ≈ Plan B AUROC:
        → Lead with Plan A (simpler, stronger claim), Plan B as extension
```

---

## 13. Checklist Before Submitting Results

- [ ] Right breast flipped before training
- [ ] Bilateral cancer cases excluded from training
- [ ] Validation set contains only normal cases
- [ ] Flow never sees lesion patches during training
- [ ] AUROC reported with 3 random seed mean ± std
- [ ] Stratified by BI-RADS density
- [ ] Stratified by lesion type (mass vs. calcification)
- [ ] Pointing game baseline reported (random = bbox_area / image_area)
- [ ] Raw magnitude baseline included in ablation
- [ ] Single-breast baseline included in ablation
