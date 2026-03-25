# CycleCUT: Required Reading List
### Organized by Component — Read In This Order

---

## How To Use This List

Each reference includes:
- **What it is:** one sentence on the core contribution
- **Why you need it:** exactly how it connects to CycleCUT
- **What to read:** which sections to prioritize

Read the papers in the order listed within each section. Do not skip the foundational papers — the implementation details you need are in them.

---

## Section 1: Foundational Methods (Must Read All)

---

### 1.1 CUT — The Core Generator

**Park, T., Efros, A.A., Zhang, R., Zhu, J.Y. (2020)**
*Contrastive Learning for Unpaired Image-to-Image Translation*
ECCV 2020
https://arxiv.org/abs/2007.15651

**What it is:** Replaces CycleGAN's cycle consistency with a patch-wise contrastive loss (PatchNCE). Each patch in the output is pulled toward the same spatial location in the input and pushed away from all other locations.

**Why you need it:** G_AB and G_BA in CycleCUT are CUT generators. The PatchNCE loss implementation, the patch MLP architecture, and the multi-scale feature extraction all come directly from this paper.

**What to read:** Sections 3 (method), 4 (implementation details). Pay close attention to Figure 2 (PatchNCE diagram) and Table 1 (hyperparameters).

---

### 1.2 CycleGAN — The Cycle Consistency Component

**Zhu, J.Y., Park, T., Isola, P., Efros, A.A. (2017)**
*Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*
ICCV 2017
https://arxiv.org/abs/1703.10593

**What it is:** Trains two generators (G_AB, G_BA) with a cycle consistency loss — translating A→B→A should recover A. Allows unpaired image translation.

**Why you need it:** CycleCUT combines CUT with cycle consistency from this paper. The cycle loss formula, identity loss, and the bidirectional training setup are all taken from CycleGAN. Also understand why identity loss matters — Section 3.2.

**What to read:** Sections 3 (formulation), 4 (implementation). Read the identity loss paragraph carefully — it is directly relevant to preventing intensity shift in mammograms.

---

### 1.3 Normalizing Flows — The Anomaly Scorer

**Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., Lakshminarayanan, B. (2021)**
*Normalizing Flows for Probabilistic Modeling and Inference*
JMLR 2021
https://arxiv.org/abs/1912.02762

**What it is:** Comprehensive review of normalizing flows — invertible neural networks that transform a simple distribution (Gaussian) to a complex data distribution, enabling exact likelihood computation.

**Why you need it:** The normalizing flow in CycleCUT is the entire anomaly scoring mechanism. You need to understand: (a) what exact likelihood means and why it matters, (b) how coupling layers work, (c) what can go wrong during training (mode collapse, numerical instability).

**What to read:** Sections 2 (background), 3 (coupling flows), 6.1 (anomaly detection application). You can skip the variational inference sections.

---

### 1.4 Neural Spline Flows — The Specific Flow Architecture

**Durkan, C., Bekasov, A., Murray, I., Papamakarios, G. (2019)**
*Neural Spline Flows*
NeurIPS 2019
https://arxiv.org/abs/1906.04032

**What it is:** Replaces affine coupling layers with rational quadratic splines — more expressive transformations that better model complex distributions while remaining exactly invertible.

**Why you need it:** This is the specific flow architecture recommended for CycleCUT. The implementation uses masked autoregressive flows with spline transformations. Understand the spline parameterization and why it is more expressive than RealNVP.

**What to read:** Sections 3 (rational quadratic splines), 4 (experiments on density estimation). Read the appendix on numerical stability.

---

### 1.5 SimCLR — Patch Encoder Pretraining

**Chen, T., Kornblith, S., Norouzi, M., Hinton, G. (2020)**
*A Simple Framework for Contrastive Learning of Visual Representations*
ICML 2020
https://arxiv.org/abs/2002.05709

**What it is:** Trains image encoders without labels by pulling augmented views of the same image together and pushing different images apart in embedding space. Uses NT-Xent (InfoNCE) loss.

**Why you need it:** The patch encoder φ is pretrained with SimCLR on normal residual patches. The augmentation strategy, temperature parameter, projection head design, and batch size considerations all come from this paper.

**What to read:** Sections 2 (method), 3 (data augmentation — critical), Appendix B (implementation details). Pay attention to the effect of batch size and temperature on contrastive learning quality.

---

## Section 2: Anomaly Detection (Must Read)

---

### 2.1 The OOD Framework

**Ruff, L., et al. (2021)**
*A Unifying Review of Deep and Shallow Anomaly Detection*
Proceedings of the IEEE 2021
https://arxiv.org/abs/2009.11732

**What it is:** Comprehensive taxonomy of anomaly detection methods — one-class classification, density estimation, reconstruction-based, and self-supervised approaches.

**Why you need it:** CycleCUT is a density estimation anomaly detector. This paper gives you the vocabulary and framework to position the method correctly in related work, understand the theoretical guarantees (and limitations), and anticipate reviewer questions about baselines.

**What to read:** Sections 2 (problem formulation), 3.3 (density-based methods), 4 (deep methods). Table 1 is a useful reference for positioning against baselines.

---

### 2.2 Normalizing Flows for Anomaly Detection

**Rudolph, M., Wandt, B., Rosenhahn, B. (2021)**
*Same Same But DifferNet: Semi-Hard Negative Mining Contrastive Learning for Industrial Anomaly Detection*
WACV 2021
https://arxiv.org/abs/2008.12577

**What it is:** Uses normalizing flows on CNN feature vectors for industrial anomaly detection — training on normal images only, flagging OOD at test time via likelihood score.

**Why you need it:** This is the closest existing work to your flow-on-residual-features approach. The architecture (CNN encoder → normalizing flow on features), the patch-level scoring, and the OOD evaluation protocol are directly transferable to CycleCUT. Study their implementation carefully — your approach extends it to bilateral residuals.

**What to read:** Entire paper — it is short (8 pages). Pay close attention to their feature extraction strategy and the multi-scale approach.

---

### 2.3 Patch-Level Anomaly Detection

**Roth, K., et al. (2022)**
*Towards Total Recall in Industrial Anomaly Detection*
CVPR 2022
https://arxiv.org/abs/2106.08265

**What it is:** PatchCore — stores a memory bank of normal patch features, flags anomalies via nearest-neighbor distance to this bank. State-of-the-art on MVTec anomaly detection benchmark.

**Why you need it:** PatchCore is your strongest single-breast baseline. It operates on exactly the same patch feature representation as CycleCUT's flow, but without the bilateral signal. You must implement it as a baseline and understand why bilateral information should help beyond it.

**What to read:** Sections 3 (method), 4 (experiments). Understand the coreset subsampling — important for scaling to mammography dataset size.

---

### 2.4 OOD Detection Theory

**Nalisnick, E., Matsukawa, A., Teh, Y.W., Gorur, D., Lakshminarayanan, B. (2019)**
*Do Deep Generative Models Know What They Don't Know?*
ICLR 2019
https://arxiv.org/abs/1810.09136

**What it is:** Shows that deep generative models (VAEs, flows) can assign higher likelihood to OOD data than training data — a fundamental failure mode of likelihood-based anomaly detection.

**Why you need it:** This paper describes exactly the failure mode that could affect CycleCUT's flow scorer. A reviewer will raise this. You need to understand when it occurs (typically when OOD data has simpler structure than training data) and argue why your residual representation avoids it — or test for it explicitly in Experiment A1.

**What to read:** Entire paper (6 pages). Read the discussion section carefully — it explains when likelihood is and is not a reliable anomaly score.

---

## Section 3: Mammography and Medical Imaging (Must Read)

---

### 3.1 Bilateral Asymmetry in Mammography — Clinical Basis

**Kopans, D.B. (2006)**
*Breast Imaging (3rd Edition)*
Lippincott Williams & Wilkins

**What it is:** Standard radiology textbook. Chapter on bilateral comparison describes exactly how radiologists use contralateral comparison — the clinical motivation for the entire project.

**Why you need it:** You need to understand what radiologists actually look for when comparing bilateral views before claiming to mimic their process. The clinical definition of asymmetry, its subtypes (global, focal, developing), and its significance are all here. Reviewers with clinical background will check whether your clinical motivation is accurate.

**What to read:** Chapter 5 (mammographic interpretation), specifically the section on asymmetric densities and bilateral comparison. If you cannot access the book, read the ACR BI-RADS Atlas (free online) Chapter 5 instead.

---

### 3.2 Deep Learning for Mammography — Survey

**Sechopoulos, I., Mann, R.M. (2020)**
*Stand-alone artificial intelligence for breast cancer detection in mammography: comparison with 11 radiologists*
JNCI 2020
https://doi.org/10.1093/jnci/djaa123

**What it is:** Comparison of AI mammography systems against radiologists on a large screening dataset. Establishes what AUROC performance levels mean clinically.

**Why you need it:** You need to know what AUROC values are clinically meaningful. This paper gives you calibration — an AUROC of 0.85 in this setting is roughly competitive with a single radiologist. Without this context, you cannot interpret your results.

**What to read:** Results section and Table 2. Also read the discussion on operating points (sensitivity/specificity tradeoffs in screening).

---

### 3.3 VinDr-Mammo Dataset Paper

**Nguyen, H.T., et al. (2023)**
*VinDr-Mammo: A large-scale benchmark dataset for computer-aided diagnosis in full-field digital mammography*
Scientific Data 2023
https://www.nature.com/articles/s41597-023-02100-7

**What it is:** Description of the VinDr-Mammo dataset — 5,000 mammograms with bounding box annotations and BI-RADS ratings.

**Why you need it:** This is your primary dataset. Read the data collection protocol, annotation procedure, class distribution, and known biases before designing your experiments. The label quality section is especially important — it tells you which annotations to trust.

**What to read:** Entire paper. Pay close attention to the inter-reader agreement statistics and the distribution of BI-RADS categories.

---

### 3.4 Anomaly Detection in Medical Imaging — Survey

**Baur, C., Wiestler, B., Albarqouni, S., Navab, N. (2021)**
*Autoencoders for Unsupervised Anomaly Segmentation in Brain MR Images: A Comparative Study*
Medical Image Analysis 2021
https://arxiv.org/abs/1904.07734

**What it is:** Compares reconstruction-based anomaly detection approaches (VAE, AE, GAN-based) for detecting brain lesions — training on healthy scans, testing on pathological scans.

**Why you need it:** This is the most direct precedent for your approach in medical imaging. Their finding that reconstruction residuals localize lesions but are sensitive to reconstruction quality directly predicts the challenges you will face. Their evaluation protocol (pixel-level AUROC, Dice at threshold) should inform yours.

**What to read:** Sections 2 (methods), 4 (results), 5 (discussion — especially the failure modes).

---

### 3.5 Breast Bilateral Symmetry — Quantitative Analysis

**Scutt, D., Lancaster, G.A., Manning, J.T. (2006)**
*Breast asymmetry and predisposition to breast cancer*
Breast Cancer Research 2006
https://breast-cancer-research.biomedcentral.com/articles/10.1186/bcr1371

**What it is:** Quantifies the degree of normal bilateral asymmetry in breasts — size, shape, and density differences between left and right in healthy women.

**Why you need it:** Your entire method assumes that normal asymmetry is learnable and distinguishable from pathological asymmetry. This paper gives you the empirical basis for that assumption — and tells you how large normal asymmetry actually is. This is the evidence you cite when justifying the bilateral approach.

**What to read:** Results section (asymmetry measurements) and the discussion on asymmetry as a cancer risk factor.

---

## Section 4: Related Methods You Must Know As Baselines

---

### 4.1 pix2pix — Supervised Image Translation

**Isola, P., Zhu, J.Y., Zhou, T., Efros, A.A. (2017)**
*Image-to-Image Translation with Conditional Adversarial Networks*
CVPR 2017
https://arxiv.org/abs/1611.07004

**What it is:** Conditional GAN for paired image translation — learns a direct mapping from input to output when paired training data exists.

**Why you need it:** pix2pix is the supervised upper bound for bilateral translation — if you had paired (lesion breast, healthy breast) data, you would use this. Understanding why you cannot use pix2pix (no healthy ground truth for lesion cases) clarifies why unpaired translation (CUT/CycleGAN) is necessary. Also: the PatchGAN discriminator architecture used in CycleCUT comes from this paper.

**What to read:** Sections 3 (PatchGAN discriminator), 4 (training details).

---

### 4.2 GAN Anomaly Detection

**Schlegl, T., Seeböck, P., Waldstein, S.M., Schmidt-Erfurth, U., Langs, G. (2017)**
*Unsupervised Anomaly Detection with Generative Adversarial Networks*
IPMI 2017
https://arxiv.org/abs/1703.05921

**What it is:** AnoGAN — trains a GAN on normal images, then at test time finds the latent code that best reconstructs an input. Anomalies produce large reconstruction residuals.

**Why you need it:** AnoGAN is the historical precedent for GAN-based anomaly detection in medical imaging. CycleCUT's residual-based anomaly map is conceptually similar but avoids the expensive test-time optimization. You must cite and compare against this or its faster successors (f-AnoGAN).

**What to read:** Sections 2 (method), 3 (experiments). Note why test-time optimization is a practical limitation.

---

### 4.3 DRAEM — Reconstruction-Based Anomaly With Synthetic Corruptions

**Zavrtanik, V., Kruse, M., Skočaj, D. (2021)**
*DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection*
ICCV 2021
https://arxiv.org/abs/2108.07610

**What it is:** Trains a reconstruction network on normal images augmented with synthetic corruptions, then trains a discriminator to detect the difference between original and reconstructed images. The synthetic corruptions give the discriminator real anomaly examples.

**Why you need it:** Plan B's synthetic corruption strategy is conceptually related to DRAEM. Understanding DRAEM's design — specifically how corruption type affects generalization and the reconstruction+discrimination two-stage approach — will help you design and justify Plan B's corruption strategy.

**What to read:** Sections 3 (method), 4 (ablation on corruption types). The ablation on corruption realism is directly relevant to Experiment B3.

---

### 4.4 Bilateral Mammography — Deep Learning

**Liu, Y., et al. (2021)**
*From Unilateral to Bilateral: Aggregating Multi-View Mammography for Cancer Detection*
MICCAI 2021

**What it is:** Uses attention-based bilateral aggregation — one breast attends to the other to improve cancer classification. Supervised (uses cancer labels).

**Why you need it:** This is the most directly related deep learning paper using bilateral mammography. You must understand what they do, how it differs from CycleCUT (supervised vs. unsupervised, classification vs. anomaly detection), and why your approach is complementary or superior in the annotation-free setting.

**What to read:** Sections 3 (bilateral attention), 4 (results). Note their AUROC numbers as a supervised upper bound for your comparison.

---

### 4.5 RealNVP — Alternative Flow Architecture

**Dinh, L., Sohl-Dickstein, J., Bengio, S. (2017)**
*Density Estimation Using Real-Valued Non-Volume Preserving Transformations*
ICLR 2017
https://arxiv.org/abs/1605.08803

**What it is:** Introduces coupling layers for normalizing flows — an efficient architecture for exact density estimation with cheap Jacobian computation.

**Why you need it:** RealNVP is the simpler alternative to Neural Spline Flows. If NSF is unstable during training, fall back to RealNVP. Understanding both lets you make an informed architectural choice and justify it.

**What to read:** Sections 3 (coupling layers), 4 (experiments). The Jacobian computation explanation (Section 3.3) is essential for understanding why flows enable exact likelihood.

---

## Section 5: Evaluation Methods (Read When You Reach Evaluation)

---

### 5.1 FROC Analysis

**Bunch, P.C., Hamilton, J.F., Sanderson, G.K., Simmons, A.H. (1977)**
*A Free-Response Approach to the Measurement and Characterization of Radiographic-Observer Performance*
Journal of Applied Photographic Engineering

**What it is:** Free-Response ROC (FROC) — evaluation metric for detection tasks where the number of findings per image varies. Reports sensitivity as a function of false positives per image.

**Why you need it:** If you have bounding box annotations, FROC is the standard evaluation for lesion detection in radiology — more informative than standard AUROC for detection tasks. Reviewers with medical imaging background will expect it.

**What to read:** The concept — a one-paragraph explanation suffices. Then read the FROC section in any recent mammography detection paper (e.g., the VinDr paper above) to see how it is computed in practice.

---

### 5.2 Calibration of Anomaly Scores

**Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017)**
*On Calibration of Modern Neural Networks*
ICML 2017
https://arxiv.org/abs/1706.04599

**What it is:** Shows that modern neural networks are poorly calibrated — high confidence does not mean high accuracy. Introduces temperature scaling for post-hoc calibration.

**Why you need it:** CycleCUT's flow score is not a probability — it is an unbounded log-likelihood. To convert it to a clinically meaningful score (e.g., probability of malignancy), you need calibration. This paper gives you the tools (Platt scaling, temperature scaling) and the evaluation metric (Expected Calibration Error).

**What to read:** Sections 2 (calibration definition), 4 (temperature scaling). Short and practical.

---

## Section 6: Background Reading (Read If Unfamiliar)

Read these only if you are unfamiliar with the prerequisite concepts. They are not cited in the paper but will save you significant debugging time.

---

### 6.1 GAN Training

**Goodfellow, I., et al. (2014)**
*Generative Adversarial Nets*
NeurIPS 2014
https://arxiv.org/abs/1406.2661

Read if: you have never implemented a GAN.
Focus: training dynamics, why mode collapse happens, the minimax game formulation.

---

### 6.2 Instance Normalization

**Ulyanov, D., Vedaldi, A., Lempitsky, V. (2017)**
*Instance Normalization: The Missing Ingredient for Fast Stylization*
https://arxiv.org/abs/1607.08022

Read if: you are unsure why generators use InstanceNorm instead of BatchNorm.
Focus: one page — the answer is that BatchNorm destroys style information across the batch, which is catastrophic for image translation.

---

### 6.3 Contrastive Learning Background

**Oord, A., Li, Y., Vinyals, O. (2018)**
*Representation Learning with Contrastive Predictive Coding*
https://arxiv.org/abs/1807.03748

Read if: you are unfamiliar with InfoNCE loss.
Focus: Section 2 (InfoNCE derivation) — the loss used in both CUT and SimCLR comes from this paper.

---

## Reading Order Recommendation

```
Week 1 (before writing any code):
  1.1 CUT
  1.2 CycleGAN
  3.3 VinDr-Mammo dataset
  3.1 Kopans (skim clinical chapter)

Week 2 (before implementing the flow):
  1.3 Normalizing Flows review
  1.4 Neural Spline Flows
  2.2 DifferNet (closest existing work)
  2.4 Nalisnick et al. (OOD failure modes)

Week 3 (before running experiments):
  2.1 Anomaly detection review
  2.3 PatchCore (implement as baseline)
  3.4 Baur et al. (medical anomaly detection)
  4.3 DRAEM (for Plan B motivation)

Week 4 (before writing the paper):
  3.2 Sechopoulos (clinical context)
  4.4 Liu et al. (bilateral deep learning baseline)
  5.1 FROC
  5.2 Calibration
```
