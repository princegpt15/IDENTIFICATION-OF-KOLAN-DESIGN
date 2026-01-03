# STEP 1: DATASET PREPARATION - EXECUTION GUIDE

## Overview
This guide explains how to execute Step 1 (Data Collection & Dataset Preparation) for the Kolam Pattern Classification system.

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- 2GB+ free disk space (for dataset)
- Windows/Linux/macOS

### Python Dependencies
Install required packages:

```bash
pip install opencv-python numpy pandas matplotlib tqdm
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

---

## Execution Workflow

### Phase 1: Setup Dataset Structure (5 minutes)

**Script:** `01_create_structure.py`

**Purpose:** Creates the complete folder hierarchy for organizing kolam images.

**Execution:**
```bash
cd "c:\Users\princ\Desktop\MACHINE TRAINING"
python scripts/01_create_structure.py
```

**Output:**
- Creates `kolam_dataset/` directory with all subdirectories
- Generates placeholder README files
- Creates `STRUCTURE_SUMMARY.txt`

**Verification:**
- Check that `kolam_dataset/` folder exists
- Verify all category folders are created
- Confirm train/val/test split folders exist

---

### Phase 2: Data Collection (1-2 weeks)

**Manual Task:** Collect Kolam images from various sources

**Instructions:**

1. **Gather Images**
   - Take photographs of real kolams (top-down view)
   - Download from cultural archives (with proper attribution)
   - Scan kolam design books
   - Use kolam drawing apps

2. **Organize by Category**
   - Place images in appropriate folders:
     - `kolam_dataset/00_raw_data/pulli_kolam/`
     - `kolam_dataset/00_raw_data/chukku_kolam/`
     - `kolam_dataset/00_raw_data/line_kolam/`
     - `kolam_dataset/00_raw_data/freehand_kolam/`

3. **Naming Convention**
   - Use descriptive names: `pulli_001.jpg`, `chukku_temple_01.png`
   - Avoid special characters
   - Use consistent extensions (.jpg, .png)

**Target:** 200-500 images per category

---

### Phase 3: Data Cleaning (2-3 days)

**Script:** `02_clean_dataset.py`

**Purpose:** Validates and filters images based on quality criteria.

**Execution:**
```bash
python scripts/02_clean_dataset.py
```

**What it does:**
- Checks image resolution (minimum 224x224)
- Detects blurry images
- Filters over/underexposed images
- Identifies corrupted files
- Copies valid images to `01_cleaned_data/`

**Output:**
- Cleaned images in `kolam_dataset/01_cleaned_data/<category>/`
- `reports/cleaning_report.json` - detailed statistics
- `reports/cleaning_report.txt` - human-readable summary

**Review:**
- Check rejection rate (should be <20%)
- Manually inspect rejected images if rate is too high
- Adjust thresholds in script if needed

---

### Phase 4: Annotation Generation (3-5 days)

**Script:** `04_generate_annotations.py`

**Purpose:** Creates CSV and JSON annotation files with metadata.

**Execution:**
```bash
python scripts/04_generate_annotations.py
```

**What it does:**
- Scans all cleaned images
- Extracts basic metadata (resolution, file size)
- Creates annotation entries
- Generates CSV and JSON files

**Output:**
- `annotations/cleaned_annotations.csv`
- `annotations/sample_annotations.csv` (example schema)
- JSON versions of all annotation files

**Manual Enhancement (Optional):**
Edit annotation CSV files to add metadata:
- `dot_count`: Number of dots (for pulli/chukku kolams)
- `symmetry_type`: radial/bilateral/asymmetric/none
- `grid_type`: square/diamond/triangular/none
- `complexity`: low/medium/high
- `quality_score`: 1-10 rating
- `source`: field_photography/web/scanned/synthetic

---

### Phase 5: Dataset Splitting (1 hour)

**Script:** `03_split_dataset.py`

**Purpose:** Splits cleaned data into train/val/test sets (70/15/15).

**Execution:**
```bash
python scripts/03_split_dataset.py
```

**What it does:**
- Performs stratified random split
- Ensures equal class representation
- Uses fixed random seed (reproducible)
- Copies images to split directories

**Output:**
- `kolam_dataset/02_split_data/train/<category>/`
- `kolam_dataset/02_split_data/val/<category>/`
- `kolam_dataset/02_split_data/test/<category>/`
- `reports/split_statistics.json`
- `reports/split_statistics.txt`

**Verification:**
- Check that train:val:test ≈ 70:15:15
- Verify each category has equal samples per split
- Confirm no data leakage

---

### Phase 6: Dataset Validation (30 minutes)

**Script:** `05_validate_dataset.py`

**Purpose:** Comprehensive validation and quality checks.

**Execution:**
```bash
python scripts/05_validate_dataset.py
```

**What it does:**
- Validates directory structure
- Checks class balance
- Validates annotation files
- Detects data leakage
- Generates sample visualizations

**Output:**
- `reports/validation_report.txt`
- `reports/sample_visualization.png` (grid of sample images)
- Terminal output with validation results

**Review Checklist:**
- [ ] All folders created correctly
- [ ] Equal samples per class (±5 tolerance)
- [ ] No corrupted images
- [ ] All images have annotations
- [ ] No missing labels in CSV
- [ ] Train/val/test splits sum to 100%
- [ ] No data leakage detected
- [ ] Sample visualization looks correct

---

## Complete Execution Sequence

For a full end-to-end run (after collecting raw images):

```bash
# Step 1: Create structure
python scripts/01_create_structure.py

# Step 2: Collect images manually (place in 00_raw_data/)

# Step 3: Clean dataset
python scripts/02_clean_dataset.py

# Step 4: Generate annotations
python scripts/04_generate_annotations.py

# Step 5: Split dataset
python scripts/03_split_dataset.py

# Step 6: Validate everything
python scripts/05_validate_dataset.py
```

---

## Directory Structure After Completion

```
kolam_dataset/
├── 00_raw_data/                    # Original collected images
│   ├── pulli_kolam/
│   ├── chukku_kolam/
│   ├── line_kolam/
│   └── freehand_kolam/
│
├── 01_cleaned_data/                # Validated, quality-checked images
│   ├── pulli_kolam/
│   ├── chukku_kolam/
│   ├── line_kolam/
│   └── freehand_kolam/
│
├── 02_split_data/                  # Ready for training
│   ├── train/
│   │   ├── pulli_kolam/
│   │   ├── chukku_kolam/
│   │   ├── line_kolam/
│   │   └── freehand_kolam/
│   ├── val/
│   │   └── ... (same structure)
│   └── test/
│       └── ... (same structure)
│
├── annotations/                    # Label files
│   ├── cleaned_annotations.csv
│   ├── train_annotations.csv
│   ├── val_annotations.csv
│   ├── test_annotations.csv
│   └── sample_annotations.csv
│
├── reports/                        # Statistics and validation
│   ├── cleaning_report.json
│   ├── split_statistics.json
│   ├── validation_report.txt
│   └── sample_visualization.png
│
└── scripts/                        # Automation scripts
    ├── 01_create_structure.py
    ├── 02_clean_dataset.py
    ├── 03_split_dataset.py
    ├── 04_generate_annotations.py
    └── 05_validate_dataset.py
```

---

## Troubleshooting

### Issue: No images found after cleaning
**Solution:** 
- Check if images are in raw_data folders
- Verify image formats (.jpg, .png)
- Lower quality thresholds in `02_clean_dataset.py`

### Issue: Imbalanced classes after splitting
**Solution:**
- Collect more images for underrepresented classes
- Use data augmentation (will be covered in later steps)

### Issue: Import errors (cv2, matplotlib)
**Solution:**
```bash
pip install --upgrade opencv-python matplotlib numpy
```

### Issue: Images too small (<224x224)
**Solution:**
- Re-collect images at higher resolution
- Or modify `min_resolution` parameter in cleaning script

---

## Quality Targets

Before proceeding to Step 2 (Model Training), ensure:

✅ **Minimum Requirements:**
- 800+ total images (200 per class)
- <20% rejection rate during cleaning
- ±5 class balance tolerance
- No data leakage
- All annotations complete

✅ **Recommended Targets:**
- 2000+ total images (500 per class)
- <10% rejection rate
- Perfect class balance
- Metadata fields populated
- High-quality sample diversity

---

## Next Steps

Once Step 1 is complete and validated:

1. **Backup Dataset:** Save a copy of `kolam_dataset/` folder
2. **Document Collection:** Note sources, dates, and any issues
3. **Proceed to Step 2:** Model architecture design and training pipeline
4. **Data Augmentation:** Will be implemented during training (Step 2)

---

## Support & Resources

**Documentation:**
- See `STEP1_DATASET_DESIGN.md` for detailed specifications
- Check individual script docstrings for parameter details

**Dataset Statistics:**
- View `reports/` folder for all generated statistics
- Use `validation_report.txt` as final checklist

**Customization:**
- All scripts have configurable parameters
- Edit threshold values in validation scripts
- Modify split ratios if needed (default 70/15/15)

---

**Status:** Ready for execution  
**Estimated Time:** 2-3 weeks (depends on data collection)  
**Last Updated:** 2025-12-27
