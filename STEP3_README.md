# STEP 3: FEATURE EXTRACTION — EXECUTION GUIDE

## Overview

This guide explains Step 3 of the Kolam Pattern Classification system: **Feature Extraction**. This step extracts meaningful, explainable, and discriminative features from preprocessed Kolam images.

**Key Objective:** Combine handcrafted classical vision features with deep CNN features to create a hybrid feature representation (2074-dim) that supports both classification and rule-based validation.

---

## Why Feature Extraction?

Even though we'll use CNNs for classification, explicit feature extraction provides:

1. **Explainability:** Handcrafted features directly relate to Kolam properties (dots, symmetry, loops)
2. **Hybrid Robustness:** CNN features capture visual patterns; handcrafted features validate geometric constraints
3. **Rule-Based Validation:** Enable cultural/geometric correctness checking before classification
4. **Generalization:** Reduce overfitting on small Kolam dataset
5. **Interpretability for Review:** Academic and cultural experts can understand extracted properties

---

## Quick Start (3 Commands)

### 1. Full Feature Extraction

```bash
cd "c:\Users\princ\Desktop\MACHINE TRAINING"
python scripts/06_feature_extraction.py
```

**Time:** 10-20 minutes (CPU) or 2-5 minutes (GPU)

**Output:** Feature files, metadata, and normalization statistics

### 2. With Validation Plots

```bash
python scripts/06_feature_extraction.py --validate
```

**Generates:** Distribution plots, PCA projection, correlation heatmap

### 3. Force CPU Usage

```bash
python scripts/06_feature_extraction.py --device cpu
```

---

## Detailed Execution

### Prerequisites

Ensure you have completed:
- ✅ Step 1: Dataset Preparation
- ✅ Step 2: Image Preprocessing

Check that preprocessed images exist:
```
kolam_dataset/02_split_data/train/[kolam_type]/image.png
kolam_dataset/02_split_data/val/[kolam_type]/image.png
kolam_dataset/02_split_data/test/[kolam_type]/image.png
```

### Install Dependencies

```bash
# Install required packages
pip install opencv-python numpy torch torchvision scipy scikit-image scikit-learn matplotlib
```

Key packages:
- **torch, torchvision:** Deep learning (CNN feature extraction)
- **scikit-image:** Image processing (topology analysis)
- **scipy:** Scientific computing (distance transforms, label analysis)
- **opencv-python:** Image processing (edge detection, morphology)
- **scikit-learn:** ML utilities (PCA for visualization)

### Run Full Pipeline

```bash
python scripts/06_feature_extraction.py
```

**Expected Output:**
```
======================================================================
KOLAM PATTERN FEATURE EXTRACTION
======================================================================
Timestamp: 2025-12-28 14:30:45
Input: kolam_dataset/02_split_data
Output: kolam_dataset/04_feature_extraction

[1/5] Loading dataset information...
  ✓ Found 700 images in train split
  ✓ Found 150 images in val split
  ✓ Found 150 images in test split

[2/5] Initializing extractors...
  Loading pre-trained resnet50 on cuda...
  ✓ Model loaded: resnet50
  ✓ Device: cuda
  ✓ Feature dimension: 2048
✓ Extractors initialized

[3/5] Extracting features...

  [Processing TRAIN]
  Total images: 700
  Extracting handcrafted features...
    ✓ Extracted 700 × 26 features
  Extracting CNN features...
    ✓ Extracted 700 × 2048 features
  Fusing features...
    ✓ Combined features: 700 × 2074
  Validating features...
    ✓ Validation complete
  Creating metadata...
    ✓ Metadata created

  ... (similar for val and test)

[4/5] Saving features...
  ✓ Saved train_features.npy (700, 2074)
  ✓ Saved train_features.csv
  ✓ Saved train_metadata.json
  ✓ Saved train_validation.json

[5/5] Validation & Visualization...

✅ FEATURE EXTRACTION COMPLETE
======================================================================
```

---

## Feature Extraction Details

### 1. Handcrafted Features (26-dim)

Classical computer vision features extracted directly from image geometry.

#### 1.1 Dot Grid Features (6)
- **dot_count:** Number of detected dots (circular regions)
- **dot_spacing_mean:** Average distance between nearest dots
- **dot_spacing_std:** Variation in dot spacing
- **grid_regularity:** How uniform is dot spacing (0-1)
- **dot_size_mean:** Average dot radius (pixels)
- **dot_density:** Percentage of image area covered by dots (0-100)

*Kolam Type Indicator:* Pulli Kolam has high dot_count and dot_density

#### 1.2 Symmetry Features (5)
- **rotational_symmetry_90:** Match with 90° rotation (0-1)
- **rotational_symmetry_180:** Match with 180° rotation (0-1)
- **horizontal_symmetry:** Left-right reflection match (0-1)
- **vertical_symmetry:** Top-bottom reflection match (0-1)
- **symmetry_type:** Dominant symmetry (0=none, 1=rot90, 2=rot180, 3=reflectional)

*Kolam Type Indicator:* Line Kolam often has rotational/reflectional symmetry

#### 1.3 Topology Features (5)
- **loop_count:** Number of closed curves/loops
- **intersection_count:** Number of curve intersections
- **branch_count:** Number of branching points (T-junctions, etc.)
- **connectivity_ratio:** Ratio of connected component size to total pattern pixels (0-1)
- **dominant_curve_length:** Length of longest continuous curve (pixels)

*Kolam Type Indicator:* Chukku Kolam has high loop_count and continuity

#### 1.4 Geometric Features (6)
- **stroke_thickness_mean:** Average line/stroke thickness (pixels)
- **stroke_thickness_std:** Variation in stroke thickness
- **curvature_mean:** Average curvature of curves (1/radius)
- **curvature_max:** Maximum curvature (sharpest curves)
- **compactness:** Shape compactness metric (0-1), high = filled/solid
- **fractal_dimension:** Complexity estimate (1.0-2.0), higher = more intricate

*Kolam Type Indicator:* Freehand Kolam has higher curvature_mean and fractal_dimension

#### 1.5 Continuity & Texture (4)
- **edge_continuity:** Percentage of edges that are connected (0-100)
- **pattern_fill:** Percentage of image covered by pattern (0-100)
- **local_variance:** Variance in local intensity patches
- **smoothness_metric:** Inverse of edge roughness (0-1), high = smooth

*Kolam Type Indicator:* Freehand Kolam has higher pattern_fill and lower smoothness

---

### 2. CNN Features (2048-dim)

Deep features extracted from pre-trained ResNet-50.

#### Architecture Selection
- **Model:** ResNet-50 (pre-trained on ImageNet, 15M images)
- **Extraction Layer:** `layer4[2]` (final residual block before pooling)
- **Dimensionality:** 2048-dim vectors
- **Why ResNet-50?**
  - Deep enough for semantic pattern understanding
  - Skip connections prevent gradient problems
  - Fast inference (GPU: 0.1-0.2s/image; CPU: 1-2s/image)
  - Proven transfer learning performance

#### Feature Extraction Process
1. Load image and resize to 224×224
2. Convert BGR to RGB
3. Apply ImageNet normalization (mean, std)
4. Pass through ResNet-50
5. Extract activations from `layer4[2]`
6. Global average pooling → 2048-dim vector

#### Why These Features Matter
- Capture high-level visual patterns (e.g., dot arrangements, curve patterns)
- Learn semantic representations optimized by ImageNet training
- Reusable for similarity matching and retrieval
- Complement handcrafted features with learned representations

---

### 3. Feature Fusion (2074-dim)

Combine handcrafted and CNN features into unified representation.

#### Normalization Strategy

**Handcrafted Features:**
- Min-Max scaling: $\tilde{x}_i = \frac{x_i - \min_i}{\max_i - \min_i}$
- Results in [0, 1] range
- Fitted on training set, applied to val/test

**CNN Features:**
- L2 normalization: $\tilde{\mathbf{f}} = \frac{\mathbf{f}}{||\mathbf{f}||_2}$
- Converts to unit vectors (∥f∥ = 1)
- Invariant to scale

#### Fusion Process

```
Handcrafted (26-dim) → Min-Max Normalize → Weight: 0.3
                                         ↓
                                    Concatenate
                                         ↓
CNN (2048-dim) ────────→ L2 Normalize → Weight: 0.7
```

**Combined Feature Vector:**
$$\mathbf{F}_{combined} = [0.3 \cdot \mathbf{H}_{norm}; 0.7 \cdot \mathbf{C}_{norm}]$$

**Result:** 2074-dim feature vector

#### Why This Weighting?
- **0.7 CNN (70%):** More discriminative power for classification
- **0.3 Handcrafted (30%):** Interpretability and rule-based validation
- Tunable via `--handcrafted-weight` and `--cnn-weight` arguments

---

## Output Files

After running Step 3, you'll have:

```
kolam_dataset/04_feature_extraction/
├── train_features.npy               # (700, 2074) - Combined features
├── train_features_handcrafted.npy   # (700, 26) - Handcrafted only
├── train_features_cnn.npy           # (700, 2048) - CNN only
├── train_features.csv               # CSV version
├── train_metadata.json              # Per-sample metadata
├── train_validation.json            # Validation report
│
├── val_features.npy                 # (150, 2074)
├── val_features_handcrafted.npy
├── val_features_cnn.npy
├── val_features.csv
├── val_metadata.json
├── val_validation.json
│
├── test_features.npy                # (150, 2074)
├── test_features_handcrafted.npy
├── test_features_cnn.npy
├── test_features.csv
├── test_metadata.json
├── test_validation.json
│
├── feature_names.json               # Feature name mapping
├── normalization_stats.json         # Min/max stats per feature
│
└── (if --validate flag used)
    ├── train_feature_distributions.png    # Histogram per class
    ├── train_pca_projection.png           # 2D PCA visualization
    ├── train_correlation_matrix.png       # Correlation heatmap
    ├── val_feature_distributions.png
    ├── val_pca_projection.png
    ├── val_correlation_matrix.png
    ├── test_feature_distributions.png
    ├── test_pca_projection.png
    └── test_correlation_matrix.png
```

### File Descriptions

**train_features.npy**
- NumPy array shape: (700, 2074)
- Format: float32
- Content: Combined handcrafted + CNN features for all training images
- Usage: `features = np.load('train_features.npy')`

**train_features.csv**
- CSV format with headers
- 700 rows, 2074 columns
- Header row: feature names
- Values: space-delimited float

**train_metadata.json**
- Per-sample metadata:
  - Image filename
  - Label and label name
  - Handcrafted feature statistics (mean, std, min, max)
  - CNN feature statistics (mean, std, norm)
- Class distribution summary

**normalization_stats.json**
- Min/max values for each handcrafted feature
- Fusion weights used
- Dimensions of each feature group
- Use for reproducing normalization on new data

**feature_names.json**
- List of 26 handcrafted feature names
- List of 2048 CNN feature names
- Combined list of 2074 names

---

## Feature Validation

### Validation Checks

The script performs automatic sanity checks:

1. **Dimension Check**
   - Verify correct number of features (2074)
   - Ensure feature array shape is (N, 2074)

2. **NaN/Inf Detection**
   - Count NaN values (should be 0)
   - Count Inf values (should be 0)
   - Indicates numerical stability issues

3. **Range Validation**
   - Check feature values in reasonable range
   - Identify outliers or unexpected scales

4. **Sanity Checks**
   - Detect all-zero samples (should be < 10% of data)
   - Detect all-one samples (should be rare)
   - Ensure diversity in feature values

5. **Feature Statistics**
   - Per-feature mean, std, min, max
   - Identify skewed or constant features

### Sample Validation Output

```
============================================================
VALIDATION SUMMARY - TRAIN
============================================================

Dataset:
  Samples: 700
  Features: 2074

Validation Checks:
  ✓ PASS - dimension: Expected D > 0, got D=2074
  ✓ PASS - nan_inf: NaNs: 0, Infs: 0
  ✓ PASS - range: Range: [0.0000, 1.0000]
  ✓ PASS - sanity: All-zero samples: 3, All-one: 0

Feature Statistics (first 10):
  dot_count:
    Mean: 45.3, Std: 22.1
    Range: [5.0, 150.2]
  ...
```

---

## Visualization Outputs

### 1. Feature Distributions (train_feature_distributions.png)

Shows histogram of first 12 handcrafted features, colored by class.

**What to Look For:**
- Clear separation between classes → good discriminative power
- Overlapping distributions → similar Kolam types
- Multi-modal distributions → within-class variation

### 2. PCA Projection (train_pca_projection.png)

2D visualization of 2074-dim feature space using PCA.

**Interpretation:**
- Well-separated clusters → classes are distinct
- PC1 and PC2 explained variance % (goal: >50% combined)
- Outliers in feature space

### 3. Correlation Matrix (train_correlation_matrix.png)

Heatmap showing correlation between handcrafted features.

**What to Look For:**
- Red (positive correlation): Features move together
- Blue (negative correlation): Inverse relationship
- White (no correlation): Independent features

---

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
pip install torch torchvision
pip install opencv-python numpy scipy scikit-image scikit-learn matplotlib
```

### Issue: Out of Memory (OOM) errors

**Cause:** GPU memory exhausted by ResNet-50
**Solution:**
```bash
# Reduce batch size
python 06_feature_extraction.py --device cpu

# Or modify batch_size in script (line ~250):
# batch_size = 16  # instead of 32
```

### Issue: "Input directory not found"

**Cause:** Step 1 (dataset splitting) not completed
**Solution:**
```bash
# Run Step 1 first
python scripts/01_create_structure.py
python scripts/02_clean_dataset.py
python scripts/03_split_dataset.py
```

### Issue: Slow Feature Extraction

**Cause:** Using CPU instead of GPU
**Solution:**
```bash
# Verify GPU availability in Python:
python -c "import torch; print(torch.cuda.is_available())"

# If true, GPU will be used automatically
python 06_feature_extraction.py
```

### Issue: Features contain NaN/Inf values

**Cause:** Edge case in image processing
**Solution:**
- Check for malformed images in dataset
- Inspect `train_validation.json` for failed samples
- Reprocess failing images in Step 2

---

## Performance Expectations

| Metric | Value |
|--------|-------|
| Handcrafted feature extraction time | 0.5-1.0s per image (CPU) |
| CNN feature extraction time (GPU) | 0.1-0.2s per image |
| CNN feature extraction time (CPU) | 1-2s per image |
| **Total time for 1000 images** | 3-5 min (GPU) / 15-30 min (CPU) |
| Feature vector dimensionality | 2074 |
| Memory per image (features) | 8.3 KB |
| Reproducibility | 100% deterministic |

---

## Using Extracted Features

### Load Features in Python

```python
import numpy as np

# Load combined features
features = np.load('kolam_dataset/04_feature_extraction/train_features.npy')
print(features.shape)  # (700, 2074)

# Load only handcrafted features
h_features = np.load('kolam_dataset/04_feature_extraction/train_features_handcrafted.npy')
print(h_features.shape)  # (700, 26)

# Load only CNN features
c_features = np.load('kolam_dataset/04_feature_extraction/train_features_cnn.npy')
print(c_features.shape)  # (700, 2048)

# Load feature names
import json
with open('kolam_dataset/04_feature_extraction/feature_names.json') as f:
    names = json.load(f)
print(len(names['combined']))  # 2074
```

### Load Labels and Metadata

```python
import json
import pandas as pd

# Load metadata
with open('kolam_dataset/04_feature_extraction/train_metadata.json') as f:
    metadata = json.load(f)

# Get labels
labels = [s['label'] for s in metadata['samples']]
print(set(labels))  # {0, 1, 2, 3}

# Get image filenames
filenames = [s['filename'] for s in metadata['samples']]
```

### Next Steps (Step 4)

Use these features for:
1. **Classification:** Train classifier on combined features
2. **Rule-Based Validation:** Use handcrafted features to verify Kolam properties
3. **Similarity Matching:** Use CNN features for image retrieval
4. **Explainability:** Analyze which handcrafted features discriminate classes

---

## Cultural and Academic Considerations

### Kolam Authenticity

This feature extraction preserves the integrity of Kolam patterns:

- **Symmetry features** respect traditional geometric properties
- **Topology features** analyze the continuous nature of authentic patterns
- **Dot grid features** capture pulli (dot) structure
- **Handcrafted features** are interpretable by cultural experts

### Explainability

Unlike pure CNN features, handcrafted features can be explained to:
- **Artists:** "Your pattern has high loop_count and high symmetry"
- **Academics:** Quantified measures of pattern complexity
- **Reviewers:** Transparent feature computation pipeline

---

## References

- ResNet-50: He et al., "Deep Residual Learning for Image Recognition" (CVPR 2015)
- Feature Fusion: Kittler et al., "On Combining Classifiers" (IEEE TPAMI 1998)
- Image Topology: Edelsbrunner & Harer, "Persistent Homology" (2010)
- Symmetry Detection: Atallah & Magdon-Ismail, "Shape Matching" (2010)

---

## Support & Documentation

- **Design Document:** See `STEP3_FEATURE_EXTRACTION_DESIGN.md` for detailed technical explanations
- **Code Documentation:** Inline comments in Python modules explain each feature
- **Source Code:** Review `scripts/feature_extraction/` for implementation details

---

## Next: Step 4 - Classification

Once features are extracted, proceed to **Step 4: CNN-Based Classification and Rule-Based Validation**

```bash
# Coming next: Classification module
python scripts/07_train_classifier.py
```

---

**Date Created:** December 28, 2025  
**Author:** Senior Computer Vision & Machine Learning Engineer  
**Status:** ✅ Step 3 Complete
