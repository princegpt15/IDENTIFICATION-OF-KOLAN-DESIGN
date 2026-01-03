# STEP 2: IMAGE PREPROCESSING PIPELINE
## Kolam Pattern Classification System

---

## Overview

This document defines the complete image preprocessing pipeline that transforms raw Kolam images into standardized, CNN-ready inputs while preserving essential pattern features.

---

## 1. PREPROCESSING PIPELINE DESIGN

### 1.1 Pipeline Architecture

```
RAW IMAGE (variable size, color, noisy)
    ↓
[1] LOAD & VALIDATE
    ↓
[2] ASPECT RATIO PRESERVATION & RESIZE
    ↓
[3] GRAYSCALE CONVERSION
    ↓
[4] NOISE REDUCTION
    ↓
[5] ADAPTIVE THRESHOLDING
    ↓
[6] MORPHOLOGICAL REFINEMENT (optional)
    ↓
[7] EDGE PRESERVATION CHECK
    ↓
PREPROCESSED IMAGE (224×224, clean, binary)
```

### 1.2 Pipeline Stages Explained

#### **Stage 1: Load & Validate**
- **Purpose:** Ensure image is readable and meets basic requirements
- **Checks:** File integrity, resolution > 0, color channels
- **Action:** Skip corrupted files, log errors

#### **Stage 2: Aspect Ratio Preservation & Resize**
- **Purpose:** Standardize to CNN input size without distortion
- **Method:** Pad to square aspect ratio, then resize to 224×224
- **Why:** CNNs require fixed input size; padding preserves pattern geometry
- **Alternative:** Center crop (loses border patterns) - NOT USED

#### **Stage 3: Grayscale Conversion**
- **Purpose:** Reduce dimensionality, focus on pattern structure
- **Method:** Weighted conversion (0.299R + 0.587G + 0.114B)
- **Why:** Kolam patterns are monochromatic; color doesn't add information
- **Benefit:** 3x smaller memory footprint

#### **Stage 4: Noise Reduction**
- **Purpose:** Remove dust, sensor noise, compression artifacts
- **Method:** Bilateral filter (edge-preserving smoothing)
- **Parameters:** 
  - d=5 (diameter)
  - sigmaColor=75 (filter strength)
  - sigmaSpace=75 (spatial influence)
- **Why Bilateral?**
  - Preserves sharp edges (critical for Kolam curves)
  - Smooths uniform regions
  - Better than Gaussian for pattern preservation
  
**Alternatives Considered:**
- ❌ Gaussian Blur: Blurs edges too much
- ❌ Median Filter: Good for salt-pepper noise but slower
- ✅ Bilateral Filter: Best edge preservation

#### **Stage 5: Adaptive Thresholding**
- **Purpose:** Convert to binary (black/white) for clear pattern extraction
- **Method:** Adaptive Gaussian thresholding
- **Parameters:**
  - Block size: 11 (local neighborhood)
  - C: 2 (constant subtracted from mean)
- **Why Adaptive?**
  - Handles varying lighting conditions
  - Better than global Otsu for uneven illumination
  - Preserves thin lines in shadows
  
**Comparison:**
| Method | Pros | Cons | Use Case |
|--------|------|------|----------|
| **Fixed Threshold** | Fast, simple | Fails with lighting variation | Controlled lighting |
| **Otsu** | Automatic, global | Sensitive to illumination | Uniform backgrounds |
| **Adaptive** | Robust, local | Slightly slower | Real-world images ✅ |

#### **Stage 6: Morphological Refinement (Conditional)**
- **Purpose:** Clean up small noise artifacts, connect broken strokes
- **Operations:**
  1. **Opening** (Erosion → Dilation): Remove small white noise
  2. **Closing** (Dilation → Erosion): Fill small gaps in lines
- **Kernel:** 3×3 rectangular
- **When to Apply:**
  - Only if noise percentage > 5%
  - Only if stroke connectivity is broken
- **Risk:** Over-processing can merge separate pattern elements
- **Mitigation:** Use minimal kernel size, validate output

#### **Stage 7: Edge Preservation Check**
- **Purpose:** Validate that preprocessing didn't destroy pattern structure
- **Checks:**
  1. Edge density (Canny edge count)
  2. Connectivity (connected components)
  3. Stroke width consistency
- **Action:** Flag images with <50% edge retention

---

## 2. DATA AUGMENTATION STRATEGY

### 2.1 Symmetry-Preserving Augmentations

Kolam patterns have inherent symmetry. Augmentation must respect this property.

#### **Allowed Augmentations:**

1. **Rotation (90°, 180°, 270°)**
   - ✅ Preserves radial symmetry
   - ✅ Common viewing angles
   - ❌ DO NOT use arbitrary angles (breaks pattern alignment)

2. **Horizontal/Vertical Flip**
   - ✅ Preserves bilateral symmetry
   - ✅ Mirrors are valid Kolam interpretations

3. **Brightness Adjustment (±10%)**
   - ✅ Simulates lighting variations
   - ❌ DO NOT exceed ±15% (affects thresholding)

4. **Contrast Enhancement (±10%)**
   - ✅ Helps with faded patterns
   - Limited range to avoid distortion

5. **Gaussian Noise (σ=0.01)**
   - ✅ Simulates sensor noise
   - Very low intensity to avoid pattern loss

#### **Forbidden Augmentations:**
- ❌ Arbitrary rotation (e.g., 45°) - breaks grid alignment
- ❌ Shearing - distorts geometry
- ❌ Heavy perspective transforms - changes pattern structure
- ❌ Elastic deformations - breaks symmetry rules
- ❌ Color jittering - Kolams are monochromatic

### 2.2 Augmentation Application Strategy

```
Training Set:
  - Original: 100%
  - Rotated (90°/180°/270°): 3× augmentation
  - Flipped (H/V): 2× augmentation
  - Brightness varied: Applied to 50%
  
Validation/Test Sets:
  - NO AUGMENTATION (evaluate on clean data)
```

**Augmentation Factor:** 2-3× training data (conservative)

---

## 3. TECHNICAL SPECIFICATIONS

### 3.1 Image Specifications

**Input (Raw):**
- Format: JPG, PNG
- Size: Variable (≥224×224)
- Color: RGB (3 channels)
- Quality: Variable (noisy, lighting variations)

**Output (Preprocessed):**
- Format: PNG (lossless)
- Size: 224×224 pixels (fixed)
- Color: Grayscale (1 channel) or Binary (for rule-based)
- Quality: Clean, noise-free, normalized

### 3.2 Processing Parameters

```python
PREPROCESSING_CONFIG = {
    'target_size': (224, 224),
    'pad_color': 255,  # White padding
    'bilateral_filter': {
        'd': 5,
        'sigmaColor': 75,
        'sigmaSpace': 75
    },
    'adaptive_threshold': {
        'block_size': 11,
        'C': 2,
        'method': cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    },
    'morphology': {
        'kernel_size': (3, 3),
        'iterations': 1,
        'apply_if_noise_ratio': 0.05
    }
}

AUGMENTATION_CONFIG = {
    'rotation_angles': [0, 90, 180, 270],
    'flip_horizontal': True,
    'flip_vertical': True,
    'brightness_range': (-0.1, 0.1),
    'contrast_range': (0.9, 1.1),
    'gaussian_noise_sigma': 0.01
}
```

---

## 4. FOLDER STRUCTURE

```
kolam_dataset/
├── 02_split_data/              # Input (from Step 1)
│   ├── train/
│   ├── val/
│   └── test/
│
├── 03_preprocessed_data/       # Output (Step 2)
│   ├── train/
│   │   ├── pulli_kolam/
│   │   ├── chukku_kolam/
│   │   ├── line_kolam/
│   │   └── freehand_kolam/
│   ├── val/
│   │   └── (same structure)
│   └── test/
│       └── (same structure)
│
├── 04_augmented_data/          # Augmented training data
│   └── train/
│       ├── pulli_kolam/
│       ├── chukku_kolam/
│       ├── line_kolam/
│       └── freehand_kolam/
│
└── preprocessing_reports/      # Validation & statistics
    ├── preprocessing_stats.json
    ├── edge_preservation_report.txt
    ├── sample_comparisons/     # Before/after visualizations
    │   ├── comparison_001.png
    │   ├── comparison_002.png
    │   └── ...
    └── failed_images.txt       # Images that failed validation
```

---

## 5. QUALITY ASSURANCE

### 5.1 Validation Checks

**Per-Image Checks:**
1. Edge density preserved (>50% of original Canny edges)
2. Connected components reasonable (<20 separate regions)
3. Stroke width consistent (std deviation < 30%)
4. No complete white/black images (dynamic range check)

**Batch Checks:**
1. Mean image statistics (brightness, contrast)
2. Class balance maintained
3. Augmentation distribution correct
4. File count verification

### 5.2 Visual Inspection

For each category, generate side-by-side comparisons:
- Original image
- After noise reduction
- After thresholding
- Final preprocessed output

Sample 10 random images per category for manual review.

---

## 6. PERFORMANCE CONSIDERATIONS

### 6.1 Processing Speed

**Expected Performance:**
- Single image: ~50-100ms
- Batch of 100: ~5-10 seconds
- Full dataset (2000 images): ~2-3 minutes

**Optimization:**
- Use `cv2.bilateralFilter` (C++ optimized)
- Batch I/O operations
- Parallel processing for augmentation (optional)

### 6.2 Memory Management

**Per Image Memory:**
- Raw (RGB, 1024×1024): ~3 MB
- Preprocessed (Gray, 224×224): ~50 KB
- Total for 2000 images: ~100 MB (manageable)

**Strategy:**
- Process images in sequence (no need for batch loading)
- Save immediately after processing
- Clear memory between batches

---

## 7. FAILURE MODES & MITIGATION

### 7.1 Common Issues

**Issue 1: Over-thresholding (pattern lost)**
- **Symptom:** Large white areas, broken lines
- **Cause:** Adaptive threshold too aggressive
- **Fix:** Increase block size or reduce C value

**Issue 2: Under-thresholding (background noise)**
- **Symptom:** Grainy background, unclear pattern
- **Cause:** Weak thresholding
- **Fix:** Decrease block size or increase C value

**Issue 3: Broken strokes**
- **Symptom:** Disconnected lines in continuous patterns
- **Cause:** Aggressive noise reduction
- **Fix:** Apply morphological closing

**Issue 4: Merged patterns**
- **Symptom:** Separate elements connected
- **Cause:** Over-aggressive morphological operations
- **Fix:** Reduce kernel size or skip morphology

### 7.2 Automated Detection

```python
def validate_preprocessing(original, processed):
    """
    Returns True if preprocessing is acceptable
    """
    # Check 1: Edge preservation
    edges_orig = cv2.Canny(original, 50, 150)
    edges_proc = cv2.Canny(processed, 50, 150)
    edge_retention = np.sum(edges_proc) / np.sum(edges_orig)
    if edge_retention < 0.5:
        return False, "Edge loss too high"
    
    # Check 2: Dynamic range
    if np.std(processed) < 30:
        return False, "Image too uniform"
    
    # Check 3: Not completely black/white
    if np.mean(processed) < 10 or np.mean(processed) > 245:
        return False, "Extreme brightness"
    
    return True, "Valid"
```

---

## 8. INTEGRATION WITH STEP 3

**Preprocessed Output Format:**
- Grayscale PNG (224×224) for CNN training
- Binary PNG for rule-based validation (optional)
- Both versions saved in separate subfolders

**Annotation Preservation:**
- CSV annotation files updated with preprocessed paths
- Metadata preserved (category, split, etc.)

---

## 9. EXECUTION SUMMARY

**Input:** `kolam_dataset/02_split_data/` (from Step 1)  
**Output:** `kolam_dataset/03_preprocessed_data/` (for Step 3)  
**Scripts:**
- `preprocess_pipeline.py` - Core preprocessing functions
- `batch_preprocess.py` - Apply to train/val/test
- `augment_data.py` - Generate augmented training data
- `validate_preprocessing.py` - Visual comparison & QA

**Estimated Time:** 5-10 minutes for full dataset

---

## 10. BEST PRACTICES

1. **Always keep raw data** - Never overwrite originals
2. **Validate samples manually** - Automated checks can miss subtle issues
3. **Document parameter choices** - Preprocessing is dataset-specific
4. **Version control preprocessed data** - Track changes to pipeline
5. **Monitor edge cases** - Some Kolam styles may need special handling

---

**Status:** ✅ DESIGN COMPLETE  
**Next:** Implement preprocessing scripts  
**Author:** Senior CV & ML Engineer  
**Date:** December 27, 2025
