# STEP 2: IMAGE PREPROCESSING - EXECUTION GUIDE

## Overview
This guide explains how to execute Step 2 (Image Preprocessing) for the Kolam Pattern Classification system using VS Code.

---

## Prerequisites

### Completed Steps
- ✅ Step 1: Dataset preparation complete
- ✅ Images organized in `kolam_dataset/02_split_data/`
- ✅ Train/val/test splits exist

### System Requirements
- Python 3.8+ installed
- OpenCV, NumPy, Matplotlib

### Install Dependencies
```bash
# If not already installed
pip install opencv-python numpy matplotlib tqdm
```

---

## Quick Start (3 Commands)

```bash
# 1. Preprocess all images (train/val/test)
python scripts/batch_preprocess.py

# 2. Generate augmented training data (optional but recommended)
python scripts/augment_data.py

# 3. Validate preprocessing quality
python scripts/validate_preprocessing.py
```

---

## Detailed Execution Steps

### Step 1: Batch Preprocessing (Required)

**Purpose:** Convert all raw images to standardized, CNN-ready format

**Script:** `batch_preprocess.py`

**Execution:**
```bash
cd "c:\Users\princ\Desktop\MACHINE TRAINING"
python scripts\batch_preprocess.py
```

**What it does:**
1. Loads images from `kolam_dataset/02_split_data/`
2. Applies complete preprocessing pipeline:
   - Resize to 224×224 with padding
   - Convert to grayscale
   - Bilateral noise reduction
   - Adaptive thresholding
   - Morphological refinement (if needed)
   - Edge preservation validation
3. Saves preprocessed images to `kolam_dataset/03_preprocessed_data/`
4. Generates comprehensive reports

**Expected Output:**
```
kolam_dataset/
├── 03_preprocessed_data/
│   ├── train/ (preprocessed training images)
│   ├── val/ (preprocessed validation images)
│   └── test/ (preprocessed test images)
│
└── preprocessing_reports/
    ├── preprocessing_metadata.json
    ├── preprocessing_stats.json
    └── failed_images.txt (if any)
```

**Duration:** ~5-10 minutes for full dataset

**Success Criteria:**
- Success rate > 95%
- Mean edge preservation > 0.6
- No critical errors

---

### Step 2: Data Augmentation (Recommended)

**Purpose:** Generate augmented training data to improve model generalization

**Script:** `augment_data.py`

**Execution:**
```bash
python scripts\augment_data.py
```

**What it does:**
1. Loads preprocessed training images
2. Applies symmetry-preserving augmentations:
   - 90°, 180°, 270° rotations
   - Horizontal and vertical flips
   - Brightness adjustments (±10%)
3. Saves augmented data to `kolam_dataset/04_augmented_data/`

**Expected Output:**
```
kolam_dataset/
└── 04_augmented_data/
    └── train/
        ├── pulli_kolam/ (~3x original count)
        ├── chukku_kolam/ (~3x original count)
        ├── line_kolam/ (~3x original count)
        └── freehand_kolam/ (~3x original count)
```

**Augmentation Factor:** 3x (conservative, proven effective)

**Note:** Validation and test sets are NOT augmented (evaluate on clean data)

---

### Step 3: Validate Preprocessing (Recommended)

**Purpose:** Visually inspect preprocessing quality and verify correctness

**Script:** `validate_preprocessing.py`

**Execution:**
```bash
python scripts\validate_preprocessing.py
```

**What it does:**
1. Creates side-by-side comparisons (original vs preprocessed)
2. Shows all preprocessing stages for sample images
3. Calculates preprocessing statistics
4. Saves visual reports

**Expected Output:**
```
kolam_dataset/
└── preprocessing_reports/
    ├── sample_comparisons/
    │   ├── comparison_pulli_kolam.png
    │   ├── comparison_chukku_kolam.png
    │   ├── comparison_line_kolam.png
    │   ├── comparison_freehand_kolam.png
    │   ├── stages_pulli_kolam_*.png (detailed pipeline view)
    │   └── ... (more visualizations)
    │
    └── preprocessing_statistics.json
```

**Manual Review:**
- Open comparison images in `sample_comparisons/`
- Verify pattern structure is preserved
- Check that edges are clear and connected
- Ensure no over-thresholding or noise artifacts

---

## Preprocessing Pipeline Details

### Pipeline Stages

```
Raw Image (RGB, variable size)
    ↓
[1] Load & Validate
    ↓
[2] Resize to 224×224 (with padding to preserve aspect ratio)
    ↓
[3] Convert to Grayscale
    ↓
[4] Bilateral Filter (noise reduction, edge-preserving)
    ↓
[5] Adaptive Gaussian Thresholding (binarization)
    ↓
[6] Morphological Refinement (conditional, if noise > 5%)
    ↓
[7] Edge Preservation Validation
    ↓
Preprocessed Image (Grayscale, 224×224, clean)
```

### Configuration Parameters

Default parameters (optimized for Kolam patterns):

```python
{
    'target_size': (224, 224),
    'bilateral_filter': {
        'd': 5,                # Diameter
        'sigmaColor': 75,      # Filter strength
        'sigmaSpace': 75       # Spatial influence
    },
    'adaptive_threshold': {
        'block_size': 11,      # Local neighborhood size
        'C': 2                 # Constant subtracted from mean
    },
    'morphology': {
        'kernel_size': (3, 3),
        'iterations': 1,
        'apply_if_noise_ratio': 0.05
    }
}
```

### Why These Parameters?

| Parameter | Value | Justification |
|-----------|-------|---------------|
| **Bilateral d=5** | Small diameter | Preserves fine Kolam curves |
| **sigma=75** | Medium strength | Balances noise reduction vs edge preservation |
| **Block size=11** | Medium | Adapts to local lighting, not too sensitive |
| **C=2** | Small constant | Conservative thresholding (avoid over-processing) |
| **Morphology 3×3** | Minimal | Clean noise without merging pattern elements |

---

## Customization

### Adjust Preprocessing Parameters

Edit `scripts/batch_preprocess.py`:

```python
# Initialize preprocessor with custom config
custom_config = {
    'bilateral_filter': {
        'd': 7,              # Increase for more smoothing
        'sigmaColor': 100,   # Increase for stronger filtering
        'sigmaSpace': 100
    },
    'adaptive_threshold': {
        'block_size': 15,    # Increase for larger local neighborhoods
        'C': 3               # Increase for lighter thresholding
    }
}

preprocessor = KolamPreprocessor(
    target_size=(224, 224),
    config=custom_config
)
```

### Change Augmentation Factor

Edit `scripts/augment_data.py`:

```python
augment_training_data(
    augmentation_factor=5  # Increase to 5x for more aggressive augmentation
)
```

### Enable Binary Output

If you need binary (black/white) images for rule-based validation:

Edit `scripts/batch_preprocess.py`:

```python
batch_preprocess(
    save_binary=True  # Set to True
)
```

---

## Troubleshooting

### Issue: Images look over-processed (too much noise removed)
**Solution:**
- Reduce bilateral filter strength: `sigmaColor=50, sigmaSpace=50`
- Increase adaptive threshold `C` value: `C=3`

### Issue: Background noise remains
**Solution:**
- Increase bilateral filter: `d=7, sigmaColor=100`
- Decrease threshold `C`: `C=1`
- Enable morphological refinement

### Issue: Broken lines/strokes
**Solution:**
- Apply morphological closing
- Reduce threshold block size: `block_size=9`

### Issue: Over-thresholding (pattern lost)
**Solution:**
- Increase `C` value: `C=4`
- Use larger block size: `block_size=15`
- Check original image quality

### Issue: Import errors
**Solution:**
```bash
pip install --upgrade opencv-python numpy matplotlib tqdm
```

### Issue: Slow processing
**Normal:** ~50-100ms per image
**If slower:** Check system resources, close other applications

---

## Validation Checklist

Before proceeding to Step 3 (Model Training), verify:

- [ ] All images preprocessed successfully (>95% success rate)
- [ ] Edge preservation score > 0.6 (mean)
- [ ] Visual inspection shows clear patterns
- [ ] No excessive noise or artifacts
- [ ] Thresholding is appropriate (not too light/dark)
- [ ] Augmented data generated (if using)
- [ ] Validation reports reviewed
- [ ] Sample comparisons look correct

---

## Understanding the Output

### Preprocessed Images
- **Format:** PNG (lossless)
- **Size:** 224×224 pixels (fixed)
- **Color:** Grayscale (1 channel)
- **Range:** 0-255 (uint8)

### Quality Metrics

**Edge Preservation Score:**
- > 0.8: Excellent (edges well preserved)
- 0.6-0.8: Good (acceptable for most cases)
- < 0.6: Poor (may need parameter adjustment)

**Success Rate:**
- > 98%: Excellent
- 95-98%: Good
- < 95%: Review failed images and adjust parameters

---

## File Structure After Step 2

```
kolam_dataset/
│
├── 02_split_data/              # Input (from Step 1)
│   ├── train/
│   ├── val/
│   └── test/
│
├── 03_preprocessed_data/       # Output (for Step 3)
│   ├── train/
│   ├── val/
│   ├── test/
│   └── preprocessing_config.json
│
├── 04_augmented_data/          # Augmented training data
│   ├── train/ (~3x original)
│   └── augmentation_config.json
│
└── preprocessing_reports/      # Quality assurance
    ├── preprocessing_metadata.json
    ├── preprocessing_stats.json
    ├── preprocessing_statistics.json
    ├── augmentation_report.json
    ├── failed_images.txt
    └── sample_comparisons/
        ├── comparison_*.png
        └── stages_*.png
```

---

## Performance Expectations

### Processing Speed
- Single image: 50-100ms
- Full dataset (2000 images): ~5-10 minutes
- Augmentation: ~2-3 minutes

### Memory Usage
- Per image: ~50 KB (preprocessed)
- Full dataset in memory: ~100 MB
- No special GPU required

---

## Integration with Step 3

**For CNN Training:**
- Use: `kolam_dataset/04_augmented_data/train/` (if augmented)
- Or: `kolam_dataset/03_preprocessed_data/train/` (if no augmentation)
- Val/Test: Always use `03_preprocessed_data/val/` and `test/`

**Image Loading in Training Script:**
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'kolam_dataset/04_augmented_data/train',
    target_size=(224, 224),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)
```

---

## Advanced Usage

### Process Single Image (for testing)

```python
from scripts.preprocess_pipeline import KolamPreprocessor

preprocessor = KolamPreprocessor(target_size=(224, 224))
output, metadata = preprocessor.preprocess(
    'path/to/input.jpg',
    'path/to/output.png'
)

print(f"Success: {metadata['success']}")
print(f"Edge score: {metadata['edge_preservation_score']:.3f}")
```

### Custom Augmentation

```python
from scripts.augment_data import KolamAugmentor

augmentor = KolamAugmentor(config={
    'rotation_angles': [90, 180, 270],
    'flip_horizontal': True,
    'brightness_range': (-0.15, 0.15)  # More aggressive
})

augmentor.augment_dataset(
    input_dir='custom/input',
    output_dir='custom/output',
    augmentation_factor=5
)
```

---

## Next Steps

Once Step 2 is complete:

1. ✅ Verify all preprocessing completed successfully
2. ✅ Review visual comparisons
3. ✅ Check preprocessing statistics
4. ⏳ **Proceed to Step 3:** CNN architecture design and training
5. ⏳ Use preprocessed data for model training
6. ⏳ Compare model performance with/without augmentation

---

## Support

**Documentation:**
- Design: `STEP2_PREPROCESSING_DESIGN.md`
- This guide: `STEP2_README.md`

**Scripts:**
- Core pipeline: `scripts/preprocess_pipeline.py`
- Batch processing: `scripts/batch_preprocess.py`
- Augmentation: `scripts/augment_data.py`
- Validation: `scripts/validate_preprocessing.py`

**Reports:**
- Location: `kolam_dataset/preprocessing_reports/`
- Review for quality assurance

---

**Status:** Ready for execution  
**Estimated Time:** 10-15 minutes total  
**Last Updated:** December 27, 2025

---

## Best Practices

1. **Always validate:** Run `validate_preprocessing.py` after batch processing
2. **Keep originals:** Never overwrite `02_split_data/`
3. **Review samples:** Manually inspect at least 10 images per category
4. **Monitor metrics:** Check edge preservation scores
5. **Document changes:** If you adjust parameters, note why
6. **Version control:** Save preprocessing configs
7. **Test incrementally:** Process one category first if unsure

---

**Ready to preprocess your Kolam dataset! 🎨✨**
