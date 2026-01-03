# STEP 3: FEATURE EXTRACTION - COMPLETE DESIGN DOCUMENT

## Executive Summary

**Objective:** Extract meaningful, explainable, and discriminative features from Kolam images that capture geometry, symmetry, dot structure, loops, and topological properties.

**Key Principle:** Feature extraction is essential even with CNNs because:
1. **Explainability:** Handcrafted features provide interpretable, domain-specific insights
2. **Hybrid Robustness:** Combining CNN features with geometric features improves classification
3. **Rule-based Validation:** Handcrafted features enable validation against cultural/geometric constraints
4. **Small Dataset Optimization:** With limited Kolam images, handcrafted features reduce overfitting

---

## Architecture Overview

```
Input Kolam Images (224×224)
          ↓
    ┌─────┴─────┐
    ↓           ↓
Handcrafted   CNN Features
Features      (ResNet-50)
    ↓           ↓
    └─────┬─────┘
          ↓
   Feature Fusion
   (Normalize & Concatenate)
          ↓
Feature Vectors (1024-dim)
          ↓
Save as .npy & .csv
```

---

## Part 1: HANDCRAFTED FEATURE EXTRACTION

### 1.1 Dot Grid Detection (6 features)

**Purpose:** Identify and quantify the discrete dot pattern structure

**Features:**
- `dot_count`: Total number of detected dots
- `dot_spacing_mean`: Average spacing between dots (pixels)
- `dot_spacing_std`: Standard deviation of dot spacing
- `grid_regularity`: Ratio of expected to actual dot spacing (0-1)
- `dot_size_mean`: Average detected dot radius (pixels)
- `dot_density`: Percentage of image area covered by dots (0-100)

**Algorithm:**
1. Apply morphological opening to isolate dots
2. Find contours and filter by circularity
3. Compute spacing via Delaunay triangulation
4. Calculate regularity metric

**Code Reference:** `KolamHandcraftedFeatures.extract_dot_grid_features()`

---

### 1.2 Symmetry Analysis (5 features)

**Purpose:** Quantify rotational and reflectional symmetry

**Features:**
- `rotational_symmetry_90`: Match score for 90° rotation (0-1)
- `rotational_symmetry_180`: Match score for 180° rotation (0-1)
- `horizontal_symmetry`: Reflectional symmetry (left-right) (0-1)
- `vertical_symmetry`: Reflectional symmetry (top-bottom) (0-1)
- `symmetry_type`: Dominant symmetry (category: 0=none, 1=rot90, 2=rot180, 3=reflectional)

**Algorithm:**
1. Compute image transforms (rotations, flips)
2. Calculate SSIM or MSE between original and transformed
3. Identify dominant symmetry

**Code Reference:** `KolamHandcraftedFeatures.extract_symmetry_features()`

---

### 1.3 Loop and Curve Topology (5 features)

**Purpose:** Analyze closed curves, intersections, and connectivity

**Features:**
- `loop_count`: Number of closed loops detected
- `intersection_count`: Number of curve intersections
- `branch_count`: Number of branching points
- `connectivity_ratio`: Connected pixels / total pixels (0-1)
- `dominant_curve_length`: Length of longest continuous curve (pixels)

**Algorithm:**
1. Apply thinning (skeleton extraction)
2. Detect endpoints and junctions
3. Trace curves and count loops (via Euler characteristic)
4. Compute connectivity metrics

**Code Reference:** `KolamHandcraftedFeatures.extract_topology_features()`

---

### 1.4 Geometric Features (6 features)

**Purpose:** Capture shape properties and stroke characteristics

**Features:**
- `stroke_thickness_mean`: Average line thickness (pixels)
- `stroke_thickness_std`: Standard deviation of thickness
- `curvature_mean`: Average curvature of curves (1/radius)
- `curvature_max`: Maximum curvature point
- `compactness`: Shape compactness metric (0-1)
- `fractal_dimension`: Fractal complexity estimate (1.0-2.0)

**Algorithm:**
1. Compute medial axis (skeleton)
2. Measure thickness from skeleton to boundaries
3. Fit splines to curves and compute curvature
4. Calculate fractal dimension via box-counting

**Code Reference:** `KolamHandcraftedFeatures.extract_geometric_features()`

---

### 1.5 Texture and Continuity (4 features)

**Purpose:** Analyze pattern continuity and local structure

**Features:**
- `edge_continuity`: Percentage of connected edges (0-100)
- `pattern_fill`: Percentage of non-background pixels (0-100)
- `local_variance`: Variance of local intensity patches
- `smoothness_metric`: Inverse of edge roughness (0-1)

**Algorithm:**
1. Edge detection and connection analysis
2. Connected component analysis
3. Local patch statistics (variance)
4. Edge smoothness via Laplacian

**Code Reference:** `KolamHandcraftedFeatures.extract_continuity_features()`

---

## Part 2: CNN-BASED FEATURE EXTRACTION

### 2.1 Architecture Selection: ResNet-50

**Why ResNet-50?**
- Pre-trained on ImageNet (15M images)
- Deep enough for complex pattern recognition (50 layers)
- Skip connections prevent vanishing gradients
- Fast inference for batch processing
- Well-established in CV literature

### 2.2 Layer Selection & Feature Dimensionality

**Selected Layer:** `layer4[2]` (final residual block before average pooling)

**Why This Layer?**
- **High-level Features:** Captures semantic patterns (loops, symmetry, topology)
- **Spatial Information:** Maintains spatial relationships (not flattened)
- **Dimensionality:** 2048 channels × 7×7 spatial = rich feature representation
- **Transfer Learning:** Optimal balance between pre-training and dataset-specific learning

**Feature Extraction Process:**
1. Load pre-trained ResNet-50 (ImageNet weights)
2. Remove classification head
3. Pass image through network
4. Extract features from `layer4[2]`
5. Global average pooling → 2048-dim vector

**Code Reference:** `CNNFeatureExtractor.extract_cnn_features()`

---

## Part 3: FEATURE NORMALIZATION & FUSION

### 3.1 Normalization Strategy

**Handcrafted Features Normalization:**
- Min-Max scaling (0-1) based on dataset statistics
- Per-feature: $\tilde{x} = \frac{x - x_{min}}{x_{max} - x_{min}}$

**CNN Features Normalization:**
- L2 normalization (unit norm)
- Per-sample: $\tilde{\mathbf{f}} = \frac{\mathbf{f}}{||\mathbf{f}||_2}$

### 3.2 Feature Fusion

**Strategy:** Concatenation with weighted importance

**Combined Feature Vector:**
$$\mathbf{F}_{combined} = [\alpha \cdot \mathbf{H}_{norm}; \beta \cdot \mathbf{C}_{norm}]$$

Where:
- $\mathbf{H}_{norm}$: Normalized handcrafted features (26-dim)
- $\mathbf{C}_{norm}$: Normalized CNN features (2048-dim)
- $\alpha = 0.3$: Weight for handcrafted features
- $\beta = 0.7$: Weight for CNN features
- Result: 2074-dim feature vector (or configurable)

**Justification:**
- CNN features dominate (0.7) → better classification power
- Handcrafted features (0.3) → explainability and rule-based validation
- Weighted sum allows fine-tuning importance

**Code Reference:** `FeatureFusion.fuse_features()`

---

## Part 4: PYTHON IMPLEMENTATION

### 4.1 Module Organization

```
scripts/
├── 06_feature_extraction.py         # Main extraction script
├── feature_extraction/
│   ├── __init__.py
│   ├── handcrafted_features.py      # Handcrafted feature class
│   ├── cnn_features.py              # CNN feature extraction
│   ├── feature_fusion.py            # Fusion and normalization
│   └── feature_validation.py        # Sanity checks and visualization
└── feature_configs.py               # Configuration constants
```

### 4.2 Key Classes

**KolamHandcraftedFeatures**
- Methods for each feature category
- Batch processing support
- Error handling and fallbacks

**CNNFeatureExtractor**
- ResNet-50 initialization
- GPU/CPU detection
- Batch feature extraction

**FeatureFusion**
- Normalization pipelines
- Feature concatenation
- Statistical metadata

**FeatureValidator**
- Consistency checks
- Visualization generation
- Failure detection

---

## Part 5: FEATURE VALIDATION

### 5.1 Sanity Checks

1. **Dimension Validation:** Check output shapes
2. **NaN/Inf Detection:** Identify numerical instabilities
3. **Range Validation:** Ensure features within expected bounds
4. **Class Distribution:** Analyze feature distributions per class
5. **Correlation Analysis:** Detect highly correlated features

### 5.2 Visualization

1. **Feature Distribution Plots:** Per-class histograms
2. **PCA Visualization:** 2D projection of 2074-dim space
3. **Correlation Heatmap:** Feature correlation matrix
4. **Per-Sample Plots:** Show intermediate processing steps

**Code Reference:** `FeatureValidator.validate_and_visualize()`

---

## Part 6: OUTPUT ARTIFACTS

### 6.1 Feature Files

```
kolam_dataset/
└── 04_feature_extraction/
    ├── train_features.npy           # (700, 2074) - train features
    ├── train_features.csv           # CSV version with headers
    ├── val_features.npy             # (150, 2074)
    ├── val_features.csv
    ├── test_features.npy            # (150, 2074)
    ├── test_features.csv
    ├── feature_names.json           # Feature name mapping
    ├── feature_statistics.json      # Min/max/mean per feature
    ├── image_filenames.json         # Mapping: index → image path
    └── labels.csv                   # Image labels and metadata
```

### 6.2 Configuration & Metadata

**feature_names.json**
```json
{
  "handcrafted": [
    "dot_count",
    "dot_spacing_mean",
    ...  // 26 features total
  ],
  "cnn": ["cnn_feature_0", ..., "cnn_feature_2047"],
  "total_dim": 2074
}
```

**feature_statistics.json**
```json
{
  "dot_count": {"min": 5, "max": 150, "mean": 45.2, "std": 20.1},
  ...
}
```

### 6.3 Scripts

- **06_feature_extraction.py:** Main execution script
- **feature_extraction_modules.py:** Core feature classes
- **STEP3_README.md:** Usage guide

---

## Part 7: EXECUTION WORKFLOW

### 7.1 Quick Start

```bash
# 1. Extract all features (train/val/test)
python scripts/06_feature_extraction.py

# 2. Validate features and generate plots
python scripts/06_feature_extraction.py --validate

# 3. Generate detailed reports
python scripts/06_feature_extraction.py --report
```

### 7.2 Expected Output

```
KOLAM FEATURE EXTRACTION
========================
[1/3] Extracting handcrafted features...
  Train: 700/700 ✓
  Val:   150/150 ✓
  Test:  150/150 ✓

[2/3] Extracting CNN features...
  Train: 700/700 ✓
  Val:   150/150 ✓
  Test:  150/150 ✓

[3/3] Fusing and normalizing features...
  Saving to 04_feature_extraction/

✅ Feature extraction complete!
   - Total features: 2074-dim
   - Handcrafted: 26-dim
   - CNN: 2048-dim
   - Files saved: train_features.npy, val_features.npy, test_features.npy
```

---

## Performance Expectations

| Metric | Expected Value |
|--------|---|
| Handcrafted feature extraction time | ~0.5-1s per image |
| CNN feature extraction time | ~0.1-0.2s per image (GPU), ~1-2s (CPU) |
| Total extraction time (all splits) | 5-15 minutes (1000 images on CPU) |
| Feature vector dimensionality | 2074 |
| Memory per image | ~8.3 KB (float32) |
| Reproducibility | 100% (deterministic) |

---

## Next Steps (Step 4)

- Use these 2074-dim features for classification
- Combine with CNN classifier for end-to-end learning
- Implement rule-based validation using handcrafted features
- Feature importance analysis via SHAP or permutation

---

## References

- ResNet-50: He et al., "Deep Residual Learning for Image Recognition" (2015)
- Symmetry Detection: Tate & Cubitt, "Symmetry in General Relativity" (2000)
- Topology in Images: Edelsbrunner & Harer, "Persistent Homology" (2010)
- Feature Fusion: Kittler et al., "On Combining Classifiers" (1998)

