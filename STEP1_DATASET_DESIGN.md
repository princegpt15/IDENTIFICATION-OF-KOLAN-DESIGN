# STEP 1: DATA COLLECTION & DATASET PREPARATION
## Kolam Pattern Classification System

---

## 1. DATASET DESIGN

### 1.1 Kolam Categories (4 Classes)

| Category | Description | Key Features |
|----------|-------------|--------------|
| **Pulli Kolam** | Dot-based kolam with dots forming grid | Regular dot patterns, grid structure, symmetrical |
| **Chukku Kolam** | Continuous loop patterns around dots | Flowing lines, loops around dots, no breaks |
| **Line Kolam** | Straight line geometric patterns | Linear elements, angular designs, geometric shapes |
| **Freehand Kolam** | Free-flowing artistic designs | No grid, creative, organic patterns |

### 1.2 Dataset Size Targets

**Target Distribution (Balanced):**
- Pulli Kolam: 500 images
- Chukku Kolam: 500 images
- Line Kolam: 500 images
- Freehand Kolam: 500 images
- **Total: 2000 images**

**Minimum viable dataset:** 200 images per class (800 total)

### 1.3 Image Constraints

#### Resolution Requirements:
- **Minimum:** 224x224 pixels (standard CNN input)
- **Recommended:** 512x512 to 1024x1024 pixels
- **Maximum:** 2048x2048 pixels (for memory efficiency)

#### Viewpoint Requirements:
- **Top-down view** (directly above the kolam)
- Angle tolerance: ±15 degrees from vertical
- No extreme perspective distortion

#### Background Requirements:
- **Preferred:** Clean, contrasting background (white floor, plain surface)
- **Acceptable:** Light-colored floors, rangoli boards
- **Reject:** Cluttered backgrounds, multiple patterns in frame

#### Format Requirements:
- **Accepted formats:** JPG, PNG, JPEG
- **Color mode:** RGB (3 channels)
- **Bit depth:** 8-bit per channel

---

## 2. DATA COLLECTION PLAN

### 2.1 Collection Sources

#### Primary Sources:
1. **Real-world Photography**
   - Household kolams (morning rituals)
   - Temple entrances
   - Cultural events and competitions
   - Kolam workshops

2. **Digital Archives**
   - Kolam enthusiast websites
   - Cultural heritage databases
   - Pinterest/Instagram (with proper attribution)
   - YouTube video frames (competition recordings)

3. **Scanned Images**
   - Kolam design books
   - Paper-based rangoli designs
   - Traditional pattern archives

4. **Synthetic Data (if needed)**
   - Digital kolam drawing apps
   - Procedurally generated patterns
   - Augmented variations of real images

### 2.2 Image Capture Guidelines

#### Lighting:
- ✅ Natural daylight or bright indoor lighting
- ✅ Even illumination across the pattern
- ❌ Harsh shadows or extreme highlights
- ❌ Night photography with poor lighting

#### Orientation:
- ✅ Camera parallel to floor/surface
- ✅ Pattern centered in frame
- ❌ Tilted or angled shots
- ❌ Partial patterns cut off

#### Framing:
- ✅ Pattern fills 70-90% of frame
- ✅ Include full pattern with small margin
- ❌ Too much empty space around pattern
- ❌ Pattern edges cut off

### 2.3 Acceptable vs Rejectable Conditions

| Aspect | ✅ Accept | ❌ Reject |
|--------|----------|-----------|
| **Clarity** | Sharp, in-focus images | Blurry, out-of-focus |
| **Completeness** | Full pattern visible | Partial/incomplete patterns |
| **Interference** | Clean pattern | Footprints, smudges, or damage |
| **Background** | Plain, contrasting surface | Busy/patterned backgrounds |
| **Overlap** | Single kolam design | Multiple overlapping designs |
| **Resolution** | ≥224x224 pixels | <224x224 pixels |

---

## 3. DATASET STRUCTURE

### 3.1 Folder Hierarchy

```
kolam_dataset/
├── 00_raw_data/
│   ├── pulli_kolam/
│   ├── chukku_kolam/
│   ├── line_kolam/
│   └── freehand_kolam/
│
├── 01_cleaned_data/
│   ├── pulli_kolam/
│   ├── chukku_kolam/
│   ├── line_kolam/
│   └── freehand_kolam/
│
├── 02_split_data/
│   ├── train/
│   │   ├── pulli_kolam/
│   │   ├── chukku_kolam/
│   │   ├── line_kolam/
│   │   └── freehand_kolam/
│   ├── val/
│   │   ├── pulli_kolam/
│   │   ├── chukku_kolam/
│   │   ├── line_kolam/
│   │   └── freehand_kolam/
│   └── test/
│       ├── pulli_kolam/
│       ├── chukku_kolam/
│       ├── line_kolam/
│       └── freehand_kolam/
│
├── annotations/
│   ├── raw_annotations.csv
│   ├── cleaned_annotations.csv
│   ├── train_annotations.csv
│   ├── val_annotations.csv
│   └── test_annotations.csv
│
├── reports/
│   ├── dataset_statistics.json
│   ├── class_distribution.png
│   └── validation_report.txt
│
└── scripts/
    ├── 01_create_structure.py
    ├── 02_clean_dataset.py
    ├── 03_split_dataset.py
    ├── 04_generate_annotations.py
    └── 05_validate_dataset.py
```

---

## 4. ANNOTATION STRATEGY

### 4.1 Labeling Methodology

**Manual Labeling Process:**
1. Visual inspection by domain expert
2. Verify pattern type matches category definition
3. Record metadata (dots, symmetry, complexity)
4. Double-check ambiguous cases

**Validation Rules:**
- Each image must have exactly one primary label
- Ambiguous patterns reviewed by 2+ annotators
- Majority vote for conflict resolution

### 4.2 Annotation Schema

**CSV Format:**
```csv
filename,category,category_id,dot_count,symmetry_type,grid_type,complexity,quality_score,source,date_collected
```

**JSON Format:**
```json
{
  "filename": "pulli_001.jpg",
  "category": "pulli_kolam",
  "category_id": 0,
  "metadata": {
    "dot_count": 25,
    "symmetry_type": "radial",
    "grid_type": "square",
    "complexity": "medium",
    "quality_score": 8.5
  },
  "collection_info": {
    "source": "field_photography",
    "date_collected": "2025-12-27",
    "photographer": "anonymous"
  }
}
```

**Field Definitions:**
- `filename`: Image filename with extension
- `category`: Class name (string)
- `category_id`: Integer label (0=pulli, 1=chukku, 2=line, 3=freehand)
- `dot_count`: Number of dots (for pulli/chukku), null otherwise
- `symmetry_type`: radial/bilateral/asymmetric/none
- `grid_type`: square/diamond/triangular/none
- `complexity`: low/medium/high (subjective)
- `quality_score`: 1-10 rating (image quality)
- `source`: field_photography/web/scanned/synthetic
- `date_collected`: YYYY-MM-DD format

---

## 5. DATA CLEANING RULES

### 5.1 Discard Criteria (Auto-Reject)

**Technical Issues:**
- Resolution < 224x224 pixels
- Corrupted or unreadable files
- Non-RGB images (grayscale without proper handling)
- File size < 10KB (likely corrupted)

**Content Issues:**
- Multiple distinct patterns in one image
- Pattern <50% complete
- >30% of pattern obscured
- Wrong subject (not a kolam)

**Quality Issues:**
- Motion blur (edge sharpness test)
- Extreme overexposure (>90% white pixels)
- Extreme underexposure (>90% dark pixels)

### 5.2 Manual Review Required

- Ambiguous category assignment
- Hybrid patterns (e.g., line + freehand mix)
- Unusual angles but good quality
- Artistic variations on traditional patterns

### 5.3 Programmatic Filtering

```python
def should_discard(image_path):
    """
    Returns True if image should be discarded
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        return True  # Corrupted
    
    # Check resolution
    h, w = img.shape[:2]
    if min(h, w) < 224:
        return True
    
    # Check if too dark/bright
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if np.mean(gray) > 240 or np.mean(gray) < 15:
        return True
    
    # Check blur (variance of Laplacian)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:  # Threshold for blur
        return True
    
    return False
```

---

## 6. DATA BALANCING

### 6.1 Class Balance Strategy

**Target:** Exactly equal samples per class in training set

**Balancing Approaches:**

1. **Undersampling (if class has excess)**
   - Randomly remove samples from majority class
   - Keep high-quality samples preferentially

2. **Oversampling (if class has deficit)**
   - Duplicate high-quality images
   - Apply augmentation to duplicates to create variation
   - Limit: Max 2x duplication per original image

3. **Hybrid Approach**
   - Balance to mean count across classes
   - Use augmentation for final balancing

### 6.2 Augmentation for Balancing

**Acceptable augmentations:**
- Rotation: ±15 degrees
- Horizontal/vertical flip
- Brightness adjustment: ±15%
- Slight zoom: 0.9x to 1.1x
- Gaussian noise: σ=0.01

**Forbidden augmentations:**
- Extreme rotations (>30°)
- Color shifts (changes pattern perception)
- Heavy blur or distortion

---

## 7. DATASET SPLITTING

### 7.1 Split Ratio

- **Training:** 70% (1400 images, 350 per class)
- **Validation:** 15% (300 images, 75 per class)
- **Test:** 15% (300 images, 75 per class)

### 7.2 Splitting Strategy

**Stratified Random Split:**
- Ensure equal class representation in each split
- Use fixed random seed for reproducibility
- No data leakage between splits

**Implementation:**
```python
from sklearn.model_selection import train_test_split

# First split: train vs (val+test)
train, temp = train_test_split(
    data, 
    test_size=0.3, 
    stratify=labels,
    random_state=42
)

# Second split: val vs test
val, test = train_test_split(
    temp,
    test_size=0.5,
    stratify=temp_labels,
    random_seed=42
)
```

### 7.3 Data Leakage Prevention

**Rules:**
1. No augmented versions split across sets
2. If duplicates exist, keep in same split
3. Images from same kolam (different angles) stay together
4. No test set access during training/validation

---

## 8. EXECUTION TIMELINE

### Phase 1: Setup (1 day)
- Create folder structure
- Setup annotation templates

### Phase 2: Collection (1-2 weeks)
- Gather images from all sources
- Organize into raw_data folders

### Phase 3: Cleaning (2-3 days)
- Manual inspection
- Programmatic filtering
- Move to cleaned_data

### Phase 4: Annotation (3-5 days)
- Label all cleaned images
- Generate CSV/JSON annotations
- Verify metadata

### Phase 5: Splitting & Validation (1 day)
- Run split script
- Generate statistics
- Validate balance

---

## 9. QUALITY ASSURANCE

### 9.1 Validation Checklist

- [ ] All folders created correctly
- [ ] Equal samples per class (±5 tolerance)
- [ ] No corrupted images
- [ ] All images have annotations
- [ ] No missing labels in CSV
- [ ] Train/val/test splits sum to 100%
- [ ] No data leakage detected
- [ ] Sample visualization looks correct

### 9.2 Statistics to Track

- Total images per category
- Images per split
- Mean/median image resolution
- Mean file size
- Quality score distribution
- Source distribution

---

## 10. DELIVERABLES CHECKLIST

### Files & Scripts:
- ✅ `01_create_structure.py` - Folder creation
- ✅ `02_clean_dataset.py` - Image cleaning
- ✅ `03_split_dataset.py` - Dataset splitting
- ✅ `04_generate_annotations.py` - Annotation generation
- ✅ `05_validate_dataset.py` - Validation checks

### Documentation:
- ✅ `STEP1_DATASET_DESIGN.md` - This document
- ✅ `STEP1_README.md` - Execution guide
- ✅ Sample annotation files

### Outputs:
- ✅ Complete folder structure
- ✅ Annotation templates (CSV/JSON)
- ✅ Validation reports
- ✅ Dataset statistics

---

**Status:** ✅ STEP 1 DESIGN COMPLETE
**Next Step:** Execute scripts to prepare actual dataset
**Estimated Time:** 2-3 weeks for full data collection
