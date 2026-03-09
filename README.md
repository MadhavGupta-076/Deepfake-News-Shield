# Deepfake-News Shield: Detecting False Articles and Manipulated Photos

Deepfake-News Shield is a complete deep learning system designed to detect misleading, fabricated and manipulated news content.
It focuses on cleaning noisy text and images, training multiple neural models across both modalities, and combining them using a feature-level fusion strategy to produce reliable fake/real predictions.

---

## 📌 Problem Statement

Fake news spreads rapidly across the internet and easily influences public opinion.
When false articles are supported with edited or manipulated photos, they appear even more convincing.
Manually verifying every piece of information becomes impossible when the volume is huge.

Because of this, there is a strong need for an **automated deep learning system capable of detecting both fake text and manipulated media with high accuracy**.
This project builds a complete multimodal pipeline — text-based fake news detection (SEM 7) extended with a full image branch (SEM 8), with the Final Multi-Fusion Model as the next step.

---

## 🎯 Core Objectives

**1.** To build a true multimodal preprocessing pipeline by downloading images and processing both text and image data in parallel.

**2.** To merge text and image data using entry ID as primary key and remove exact duplicates using pHashing based True Multimodal Duplicate Elimination.

**3.** To design and train text classification models using RoBERTa and Hybrid Bi-LSTM+Attention+GRU and combine them into a Fused Text Model.

**4.** To design and train image classification models using ViT-base-384 and EfficientNetB3 and combine them into a Fused Image Model.

**5.** To design a Final Multi-Fusion Model by combining the Fused Text Model and Fused Image Model for accurate fake news detection (in progress).

---

## ✨ Key Features

### Multimodal Preprocessing Pipeline

**Text Branch:**
* Repairs Unicode issues and corrupted characters
* Removes HTML tags, boilerplate text and embedded metadata
* Applies duplicate and near-duplicate detection
* Normalizes special symbols, punctuation and numerical patterns
* Enforces a token length suitable for transformers

**Image Branch:**
* Downloads images from URLs present in the IFND dataset
* Fixes orientation issues and handles corrupted files
* Converts all images to RGB format
* Normalizes resolution to 256x256
* Merges text and image data using entry ID as primary key
* Removes exact duplicates using pHashing technique (Text + Image + Label checking)

### Multiple Model Architectures

**Text Models:**
* **RoBERTa** for deep semantic understanding (768-dim CLS embedding)
* **Hybrid Bi-LSTM + Attention + GRU** for sequential and contextual patterns (128-dim GRU hidden vector)
* **Fused Text Model** combining both via feature-level fusion (896 → 512 → 2)

**Image Models:**
* **EfficientNetB3** for capturing local and hierarchical visual features (1536-dim)
* **ViT-base-384** for capturing global visual context via vit_base_patch16_384 from timm (768-dim)
* **Fused Image Model** combining both via feature-level fusion (2304 → 1024 → 2)

---

## 📊 Dataset Details

### Cleaned Multimodal IFND Dataset

| Property | Details |
|---|---|
| Raw IFND Entries | 56,714 |
| Total Cleaned Entries | 29,533 |
| Real Samples (True) | 20,063 |
| Fake Samples (False) | 9,470 |
| Train Split (80%) | 23,626 |
| Validation Split (10%) | 2,953 |
| Test Split (10%) | 2,954 |
| Class Weight FALSE | 1.5593 |
| Class Weight TRUE | 0.7360 |

---

## 🧼 Preprocessing Summary

**Text Preprocessing (16 steps across 4 phases):**
* Phase 1: Data Ingestion and Encoding Correction (UTF-8, Unicode, Mojibake, Symbol, Arabic noise)
* Phase 2: Structural Cleaning and Deduplication (Null handling, exact duplicate removal, near-dupe removal)
* Phase 3: Content Standardization and Formatting (HTML removal, URL/email/hashtag fix, whitespace normalization)
* Phase 4: Final Filtering (512 token limit, min-length enforcement, source boilerplate mitigation)

**Image Preprocessing (2 phases):**
* Phase 1: Image Downloading and Pre-Processing Adjustments (orientation fix, corruption handling)
* Phase 2: Image Standardization (RGB conversion, resolution normalization to 256x256)

---

## 🤖 Model Details

### RoBERTa
* Pretrained roberta-base loaded via HuggingFace transformers
* Fine-tuned with warmup scheduler, AMP, gradient clipping and early stopping
* Feature extraction: 768-dim CLS token

### Hybrid Bi-LSTM + Attention + GRU
* Custom architecture: Embedding → Bi-LSTM → Attention → GRU → Classifier
* Trained with class-weighted CrossEntropyLoss, gradient clipping and early stopping
* Feature extraction: 128-dim GRU hidden vector

### Fused Text Model
* RoBERTa (768) + Hybrid (128) = 896-dim fused vector
* Fusion head: 896 → 512 → 2 with ReLU, Dropout and Softmax
* Both backbones frozen — only fusion head trained

### EfficientNetB3
* Pretrained on ImageNet via torchvision (EfficientNet_B3_Weights.IMAGENET1K_V1)
* Custom classifier head: Linear → BatchNorm → SiLU → Dropout → Linear
* Progressive unfreezing: top 3 blocks at epoch 3, top 6 blocks at epoch 6
* Feature extraction: 1536-dim via AdaptiveAvgPool + flatten

### ViT-base-384
* Pretrained vit_base_patch16_384 loaded via timm 1.0.24
* Custom classifier head: LayerNorm → Dropout → Linear → GELU → Dropout → Linear
* Progressive unfreezing: top 3 ViT blocks at epoch 3, top 6 blocks at epoch 6
* Feature extraction: 768-dim CLS token

### Fused Image Model
* Both backbones loaded from best saved checkpoints and fully frozen
* Feature concatenation: EfficientNetB3 (1536) + ViT (768) = 2304-dim
* Fusion head: LayerNorm → Dropout → Linear(2304→1024) → GELU → Dropout → Linear(1024→2)
* Only fusion head trained (trainable params: ~2.1M)

---

## 📈 Results

### Text Model Performance

| Model | Test Accuracy | Macro F1 | Test AUC |
|---|---|---|---|
| RoBERTa | 0.9719 | 0.9676 | 0.9822 |
| Hybrid (Bi-LSTM+Attn+GRU) | 0.9411 | 0.9319 | 0.9713 |
| Fused Text Model | 0.9722 | 0.9678 | 0.9862 |

### Image Model Performance

| Model | Test Accuracy | Macro F1 | Test AUC | Training Time |
|---|---|---|---|---|
| EfficientNetB3 | 0.7508 | 0.6930 | 0.7247 | 2hr 3min |
| ViT-base-384 | 0.7519 | 0.7039 | 0.7323 | 5hr 7min |
| Fused Image Model | 0.7617 | 0.7119 | 0.7358 | 4hr 11min |

### Confusion Matrix — Fused Text Model
```
[[1554   77]
 [  30 3615]]
```

### Confusion Matrix — Fused Image Model
```
[[511  436]
 [268 1739]]
```

---

## 🚀 Installation

### Requirements

* Python 3.10+
* CUDA-supported GPU (recommended: P100 16GB or higher)
* PyTorch 2.9.0+cu126
* timm 1.0.24

### Key Dependencies

```
torch==2.9.0+cu126
torchvision==0.24.0+cu126
timm==1.0.24
numpy==2.0.2
pandas==2.3.3
Pillow==11.3.0
scikit-learn==1.6.1
matplotlib==3.10.0
CUDA==12.6
```

---

## 💻 Usage

### Preprocess the Dataset

```
python src/preprocessing_text.py --input data/IFND.csv --output data/IFND_clean_text.csv
python src/preprocessing_image.py --input data/IFND.csv --output data/Cleaned_Multimodal_IFND.csv
```

### Train Text Models

```
python experiments/roberta_train.py
python experiments/hybrid_train.py
python experiments/fusion_text_train.py
```

### Train Image Models

```
python experiments/efficientnet_b3_train.py
python experiments/vit_base_384_train.py
python experiments/fusion_image_train.py
```

### Example Inference

```python
from src.infer import FusionImageInference

detector = FusionImageInference(
    effnet_path="models/efficientnet_b3-best.pt",
    vit_path="models/vit_base_384-best.pt",
    fusion_path="models/fusion_image-best.pt"
)

result = detector.predict(image_path="sample.jpg")
print(result["label"], result["probability"])
```

---

## 📚 Documentation

This repository includes:

* Major project report (methodology + results)
* Presentation slides (SEM 7 + SEM 8)
* Notebooks for training, preprocessing and evaluation
* Scripts for text and image model training and inference

---

## 🔮 Future Work

* Build Final Multi-Fusion Model combining Fused Text Model and Fused Image Model
* Explore advanced fusion techniques: Joint balancing (weighted feature-level fusion) and Cross-attention based fusion
* Deploy the complete system as a Flask web application for real-time fake news detection

---

## 📄 Publication

M. Gupta, S. Khan, H. Thakur, D. Saini, "Deepfake-News Shield: Detecting False Articles and Manipulated Photos," Major Project I (18B19CI791), JUIT, **Status: Under Review**

---

## 👥 Contributors

Developed as a B.Tech major project at **Jaypee University of Information Technology**
* SEM 7: July – December 2025
* SEM 8: January – May 2026

**Team Members:**
- **Madhav Gupta** (221030283)
- **Soha Khan** (221031049)
- **Harshit Thakur** (221031013)
- **Divyam Saini** (221030070)

**Supervisor:**
Dr. Deepak Gupta, Assistant Professor (SG)
Department of CSE & IT, JUIT, Waknaghat

---

## 🙏 Acknowledgments

We thank:
- **Dr. Deepak Gupta** for invaluable guidance
- **JUIT** for computational resources
- The creators of the **IFND dataset**
- The open-source communities behind PyTorch, timm and torchvision

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---
