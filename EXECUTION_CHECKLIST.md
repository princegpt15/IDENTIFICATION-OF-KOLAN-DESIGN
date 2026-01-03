# ✅ STEP 1 EXECUTION CHECKLIST

Use this checklist to track your progress through Step 1.

---

## 📋 PRE-EXECUTION SETUP

- [ ] Read [STEP1_README.md](STEP1_README.md) for complete instructions
- [ ] Review [STEP1_DATASET_DESIGN.md](STEP1_DATASET_DESIGN.md) for specifications
- [ ] Install Python 3.8+ (verify: `python --version`)
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Ensure 2GB+ free disk space

---

## 🚀 PHASE 1: STRUCTURE CREATION

- [ ] Run: `python scripts/01_create_structure.py`
- [ ] Verify `kolam_dataset/` folder created
- [ ] Check that all category folders exist
- [ ] Confirm `STRUCTURE_SUMMARY.txt` generated

**Expected Output:**
- 60+ directories created
- Placeholder README files in key folders
- Structure summary document

---

## 📸 PHASE 2: DATA COLLECTION (Manual)

### Pulli Kolam (Target: 200-500 images)
- [ ] Collect images from various sources
- [ ] Save to: `kolam_dataset/00_raw_data/pulli_kolam/`
- [ ] Verify images are ≥224×224 resolution
- [ ] Use descriptive filenames (e.g., `pulli_001.jpg`)

### Chukku Kolam (Target: 200-500 images)
- [ ] Collect images from various sources
- [ ] Save to: `kolam_dataset/00_raw_data/chukku_kolam/`
- [ ] Verify images are ≥224×224 resolution
- [ ] Use descriptive filenames (e.g., `chukku_001.jpg`)

### Line Kolam (Target: 200-500 images)
- [ ] Collect images from various sources
- [ ] Save to: `kolam_dataset/00_raw_data/line_kolam/`
- [ ] Verify images are ≥224×224 resolution
- [ ] Use descriptive filenames (e.g., `line_001.jpg`)

### Freehand Kolam (Target: 200-500 images)
- [ ] Collect images from various sources
- [ ] Save to: `kolam_dataset/00_raw_data/freehand_kolam/`
- [ ] Verify images are ≥224×224 resolution
- [ ] Use descriptive filenames (e.g., `freehand_001.jpg`)

**Collection Tips:**
- Top-down viewpoint (±15° tolerance)
- Clean, contrasting background
- Good lighting (avoid harsh shadows)
- Sharp focus (no blur)
- Full pattern visible (not cropped)

---

## 🧹 PHASE 3: DATA CLEANING

- [ ] Run: `python scripts/02_clean_dataset.py`
- [ ] Review terminal output for statistics
- [ ] Check rejection rate (<20% is good)
- [ ] Open `kolam_dataset/reports/cleaning_report.txt`
- [ ] Verify cleaned images in `01_cleaned_data/` folders
- [ ] If rejection rate is high, review rejected images manually

**Quality Checks:**
- [ ] Resolution validation passed
- [ ] Blur detection working
- [ ] Brightness filtering correct
- [ ] No corrupted files reported

---

## 🏷️ PHASE 4: ANNOTATION GENERATION

- [ ] Run: `python scripts/04_generate_annotations.py`
- [ ] Check `kolam_dataset/annotations/` folder created
- [ ] Verify `cleaned_annotations.csv` exists
- [ ] Open CSV file to inspect annotations
- [ ] Confirm all images have category labels
- [ ] Review `sample_annotations.csv` for schema

**Optional Enhancement:**
- [ ] Add metadata: dot_count, symmetry_type
- [ ] Fill in grid_type, complexity fields
- [ ] Rate quality_score (1-10)
- [ ] Document source information

---

## ✂️ PHASE 5: DATASET SPLITTING

- [ ] Run: `python scripts/03_split_dataset.py`
- [ ] Review split statistics in terminal
- [ ] Verify train/val/test folders created
- [ ] Check `kolam_dataset/reports/split_statistics.txt`
- [ ] Confirm split ratios ≈ 70/15/15
- [ ] Verify equal samples per class in each split

**Balance Verification:**
- [ ] Train set: ~350 images per class
- [ ] Val set: ~75 images per class
- [ ] Test set: ~75 images per class
- [ ] All classes have equal counts (±5 tolerance)

---

## ✅ PHASE 6: VALIDATION

- [ ] Run: `python scripts/05_validate_dataset.py`
- [ ] Review all validation checks in terminal
- [ ] Check `kolam_dataset/reports/validation_report.txt`
- [ ] Open `sample_visualization.png` to inspect images
- [ ] Verify no data leakage detected
- [ ] Confirm all annotations complete

**Validation Checklist:**
- [ ] All folders created correctly
- [ ] Equal samples per class (±5 tolerance)
- [ ] No corrupted images
- [ ] All images have annotations
- [ ] No missing labels in CSV
- [ ] Train/val/test splits sum to 100%
- [ ] No data leakage detected
- [ ] Sample visualization looks correct

---

## 📊 FINAL REVIEW

- [ ] Check total images collected: _____ (target: 2,000)
- [ ] Images per class: _____ (target: 500)
- [ ] Rejection rate: _____% (target: <20%)
- [ ] Class balance tolerance: _____ (target: ±5)
- [ ] All reports generated successfully
- [ ] Sample images manually inspected
- [ ] Annotations quality verified

---

## 📁 VERIFY FILE STRUCTURE

Check that these folders/files exist:

```
kolam_dataset/
├── 00_raw_data/
│   ├── pulli_kolam/        [ ] _____ images
│   ├── chukku_kolam/       [ ] _____ images
│   ├── line_kolam/         [ ] _____ images
│   └── freehand_kolam/     [ ] _____ images
│
├── 01_cleaned_data/
│   ├── pulli_kolam/        [ ] _____ images
│   ├── chukku_kolam/       [ ] _____ images
│   ├── line_kolam/         [ ] _____ images
│   └── freehand_kolam/     [ ] _____ images
│
├── 02_split_data/
│   ├── train/              [ ] _____ images
│   ├── val/                [ ] _____ images
│   └── test/               [ ] _____ images
│
├── annotations/
│   ├── cleaned_annotations.csv       [ ]
│   ├── train_annotations.csv         [ ]
│   ├── val_annotations.csv           [ ]
│   └── test_annotations.csv          [ ]
│
└── reports/
    ├── cleaning_report.json          [ ]
    ├── split_statistics.json         [ ]
    ├── validation_report.txt         [ ]
    └── sample_visualization.png      [ ]
```

---

## 🎯 READY FOR STEP 2?

Before proceeding to Step 2 (Model Training), ensure:

- [ ] All phases above completed successfully
- [ ] Dataset contains ≥800 images (200 per class minimum)
- [ ] Class balance within ±5 tolerance
- [ ] All validation checks passed
- [ ] Sample images manually reviewed
- [ ] Annotations quality verified
- [ ] Reports generated and reviewed
- [ ] No outstanding errors or warnings

**If all checkboxes above are checked, you're ready for Step 2! ✅**

---

## 🆘 TROUBLESHOOTING

### Issue: Script fails with import errors
**Solution:**
```bash
pip install --upgrade opencv-python numpy pandas matplotlib tqdm
```

### Issue: No images found after cleaning
**Solution:**
- Verify images are in `00_raw_data/<category>/` folders
- Check image file extensions (.jpg, .png)
- Lower quality thresholds in `02_clean_dataset.py` if needed

### Issue: Imbalanced classes
**Solution:**
- Collect more images for underrepresented classes
- Ensure each category has similar image counts before cleaning

### Issue: High rejection rate (>20%)
**Solution:**
- Review rejected images manually
- Adjust thresholds in `02_clean_dataset.py`:
  - `min_resolution` (default: 224)
  - `blur_threshold` (default: 50)
  - `brightness_min` (default: 15)
  - `brightness_max` (default: 240)

---

## 📝 NOTES & OBSERVATIONS

Use this space to track any issues, decisions, or observations:

```
Date: ___________
Notes:
-
-
-
```

---

**Status:** Step 1 Complete ✅  
**Date Completed:** ___________  
**Next Step:** Proceed to Step 2 - CNN Model Architecture & Training

---

**Project:** Kolam Pattern Classification System  
**Version:** 1.0  
**Last Updated:** December 27, 2025
