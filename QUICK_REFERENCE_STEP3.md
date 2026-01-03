# QUICK REFERENCE: STEP 3 FEATURE EXTRACTION

## 1-Minute Overview

**What:** Extract 2074-dimensional feature vectors from Kolam images  
**How:** Combine 26 handcrafted (classical CV) + 2048 CNN (ResNet-50) features  
**Why:** Enable both explainable rule-based validation + accurate deep learning  
**Result:** Feature files (.npy, .csv) ready for classification  

---

## Installation (1 command)

```bash
pip install torch torchvision scipy scikit-image scikit-learn
```

---

## Run Feature Extraction (1 command)

```bash
python scripts/06_feature_extraction.py --validate
```

**Time:** 3-5 min (GPU) or 15-20 min (CPU)  
**Output:** Features in `kolam_dataset/04_feature_extraction/`

---

## 26 Handcrafted Features Summary

| # | Feature | Range | Meaning |
|---|---------|-------|---------|
| 1 | `dot_count` | 0-500 | Number of dots |
| 2 | `dot_spacing_mean` | 0-100 px | Avg distance between dots |
| 3 | `dot_spacing_std` | 0-50 px | Spacing variation |
| 4 | `grid_regularity` | 0-1 | Grid uniformity |
| 5 | `dot_size_mean` | 0-20 px | Dot radius |
| 6 | `dot_density` | 0-100% | Dots area % |
| 7 | `rotational_symmetry_90` | 0-1 | 90° rotation match |
| 8 | `rotational_symmetry_180` | 0-1 | 180° rotation match |
| 9 | `horizontal_symmetry` | 0-1 | L-R reflection |
| 10 | `vertical_symmetry` | 0-1 | T-B reflection |
| 11 | `symmetry_type` | 0-3 | Dominant type |
| 12 | `loop_count` | 0-100 | Closed loops |
| 13 | `intersection_count` | 0-500 | Curve intersections |
| 14 | `branch_count` | 0-500 | Junction points |
| 15 | `connectivity_ratio` | 0-1 | Connected fraction |
| 16 | `dominant_curve_length` | 0-5000 px | Longest curve |
| 17 | `stroke_thickness_mean` | 0-10 px | Line width |
| 18 | `stroke_thickness_std` | 0-5 px | Thickness variation |
| 19 | `curvature_mean` | 0-5 | Avg curve sharpness |
| 20 | `curvature_max` | 0-10 | Max sharpness |
| 21 | `compactness` | 0-1 | Shape solidity |
| 22 | `fractal_dimension` | 1-2 | Pattern complexity |
| 23 | `edge_continuity` | 0-100% | Connected edges |
| 24 | `pattern_fill` | 0-100% | Pattern coverage |
| 25 | `local_variance` | 0-255 | Intensity variation |
| 26 | `smoothness_metric` | 0-1 | Edge smoothness |

---

## Kolam Type Indicators

| Feature | Pulli | Chukku | Line | Freehand |
|---------|-------|--------|------|----------|
| `dot_count` | **HIGH** | Medium | Low | Very Low |
| `loop_count` | Low | **HIGH** | Medium | Medium |
| `rotational_symmetry_90` | Medium | Low | **HIGH** | Low |
| `fractal_dimension` | Low | Medium | Low | **HIGH** |
| `smoothness_metric` | High | High | **HIGH** | Low |

---

## Output Files (per split)

```
train_features.npy              # (700, 2074) combined features
train_features_handcrafted.npy  # (700, 26) handcrafted only
train_features_cnn.npy          # (700, 2048) CNN only
train_features.csv              # CSV with headers
train_metadata.json             # Per-sample metadata
train_validation.json           # Validation report
train_feature_distributions.png # Histogram plot
train_pca_projection.png        # 2D projection
train_correlation_matrix.png    # Correlation heatmap
```

---

## Load Features in Python (copy-paste ready)

```python
import numpy as np
import json

# Load features
features = np.load('kolam_dataset/04_feature_extraction/train_features.npy')  # (700, 2074)
h_features = np.load('kolam_dataset/04_feature_extraction/train_features_handcrafted.npy')  # (700, 26)
c_features = np.load('kolam_dataset/04_feature_extraction/train_features_cnn.npy')  # (700, 2048)

# Load metadata
with open('kolam_dataset/04_feature_extraction/train_metadata.json') as f:
    meta = json.load(f)

# Get labels
labels = np.array([s['label'] for s in meta['samples']])

# Get feature names
with open('kolam_dataset/04_feature_extraction/feature_names.json') as f:
    names = json.load(f)  # names['combined'] = list of 2074 names

# Example: Train a classifier
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(features, labels)
accuracy = clf.score(features, labels)
```

---

## Command Options

```bash
# Basic extraction
python scripts/06_feature_extraction.py

# With visualization plots
python scripts/06_feature_extraction.py --validate

# With detailed reports
python scripts/06_feature_extraction.py --report

# Force CPU (no GPU)
python scripts/06_feature_extraction.py --device cpu

# Custom paths
python scripts/06_feature_extraction.py --input kolam_dataset/02_split_data --output kolam_dataset/04_feature_extraction

# All options combined
python scripts/06_feature_extraction.py --validate --report --device cpu --input <path> --output <path>
```

---

## Performance (Approximate)

| Task | Time |
|------|------|
| Handcrafted features (1 image) | 0.5-1.0s |
| CNN features (1 image, GPU) | 0.1-0.2s |
| CNN features (1 image, CPU) | 1-2s |
| Entire pipeline (1000 images, GPU) | 3-5 min |
| Entire pipeline (1000 images, CPU) | 15-20 min |
| With visualization plots | +2 min |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "torch not found" | `pip install torch torchvision` |
| GPU out of memory | `python 06_feature_extraction.py --device cpu` |
| Input directory not found | Complete Step 1 first |
| Very slow extraction | Check if GPU available with `python -c "import torch; print(torch.cuda.is_available())"` |
| NaN values in output | Check input images, rerun Step 2 preprocessing |

---

## Feature Fusion Formula

$$\mathbf{F}_{combined} = [0.3 \cdot \frac{\mathbf{H}_{raw} - H_{min}}{H_{max} - H_{min}}; 0.7 \cdot \frac{\mathbf{C}_{raw}}{||\mathbf{C}_{raw}||_2}]$$

- **H**: Handcrafted features (normalized 0-1)
- **C**: CNN features (L2 normalized)
- **Result**: 2074-dimensional vectors

---

## Feature Engineering Workflow

```
Step 1: Load Image
    ↓
Step 2: Extract Handcrafted (26-dim)
    ├─ Dot detection
    ├─ Symmetry analysis
    ├─ Topology analysis
    ├─ Geometric features
    └─ Continuity metrics
    ↓
Step 3: Extract CNN (2048-dim)
    └─ ResNet-50 layer4[2]
    ↓
Step 4: Normalize
    ├─ Handcrafted: min-max [0,1]
    └─ CNN: L2 unit norm
    ↓
Step 5: Fuse & Weight
    └─ [0.3 × H; 0.7 × C]
    ↓
Step 6: Output (2074-dim)
    └─ Ready for classification
```

---

## Key Files

| File | Purpose |
|------|---------|
| `STEP3_README.md` | How to run & interpret |
| `STEP3_FEATURE_EXTRACTION_DESIGN.md` | Technical details |
| `scripts/06_feature_extraction.py` | Main pipeline |
| `scripts/feature_extraction/*.py` | Feature modules |
| `kolam_dataset/04_feature_extraction/*` | Output features |

---

## Next Step: Classification (Step 4)

```python
# With extracted features, train classifier:
from sklearn.ensemble import RandomForestClassifier

X_train = np.load('kolam_dataset/04_feature_extraction/train_features.npy')
y_train = labels_train

clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

---

## Common Kolam Classifications

| Type | Characteristics | Key Features |
|------|-----------------|--------------|
| **Pulli Kolam** | Dot-based grid | High `dot_count`, `dot_density` |
| **Chukku Kolam** | Loops around dots | High `loop_count`, `connectivity_ratio` |
| **Line Kolam** | Geometric patterns | High symmetry (rot/refl) |
| **Freehand Kolam** | Artistic designs | High `fractal_dimension` |

---

## Validation Checklist

- [x] Features extracted (2074-dim)
- [x] No NaN/Inf values
- [x] Correct shape (N × 2074)
- [x] Normalization statistics saved
- [x] Feature names mapped
- [x] Metadata generated
- [x] Validation plots created
- [x] Ready for Step 4

---

**Status:** ✅ STEP 3 COMPLETE  
**Next:** Step 4 - Classification  
**Questions?** See `STEP3_README.md` or `STEP3_FEATURE_EXTRACTION_DESIGN.md`
