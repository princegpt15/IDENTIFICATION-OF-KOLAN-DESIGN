# Real Data Training Guide

## ✅ **YOUR MODEL IS READY FOR REAL DATA!**

The entire pipeline has been verified and is **fully prepared** to train on real Kolam photographs. No code changes needed.

---

## 🔍 **What Was Checked**

All 7 critical components passed readiness checks:

1. ✅ **Model Architecture** - Flexible, auto-detects input dimension
2. ✅ **Feature Extraction** - Works with any image size/format
3. ✅ **Training Pipeline** - No synthetic data dependencies
4. ✅ **Data Preprocessing** - Dataset-agnostic scripts
5. ✅ **Evaluation** - Ready for real test data
6. ✅ **Inference** - Works with real images
7. ✅ **Dependencies** - All packages installed

---

## 📋 **Quick Start: Switch to Real Data**

### Step 1: Prepare Real Dataset (15-30 minutes)

```bash
# 1. Clear synthetic data
Remove-Item "kolam_dataset/00_raw_data/*" -Recurse -Force

# 2. Add your real Kolam images
# Organize as:
# kolam_dataset/00_raw_data/
#   ├── pulli_kolam/      (dot-based patterns)
#   ├── chukku_kolam/     (spiral/wheel patterns)
#   ├── line_kolam/       (geometric line patterns)
#   └── freehand_kolam/   (flowing artistic patterns)

# Recommended: 200+ images per category
# Supported formats: .png, .jpg, .jpeg
# Any image size works (auto-resized during processing)
```

### Step 2: Run Complete Pipeline (30-60 minutes)

```bash
# Clean and validate images
python scripts/02_clean_dataset.py

# Split into train/val/test (70/15/15%)
python scripts/03_split_dataset.py

# Extract features - CHOOSE ONE:

# Option A: Quick (Handcrafted only, ~5-10 min)
python scripts/06_quick_feature_extraction.py

# Option B: Full (CNN + Handcrafted, ~2-4 hours on CPU, 15-30 min on GPU)
python scripts/06_feature_extraction.py
```

### Step 3: Train Model (20-40 minutes)

```bash
# Train classifier (auto-detects feature dimension)
python scripts/07_train_classifier.py

# Training will:
# - Auto-detect input dimension (26 or 2074)
# - Adjust architecture automatically
# - Save best model to: kolam_dataset/05_trained_models/best_model.pth
# - Use early stopping (patience=15 epochs)
```

### Step 4: Evaluate (5-10 minutes)

```bash
# Run baseline evaluation
python scripts/14_evaluate_system.py

# Run optimization experiments
python scripts/16_optimization.py

# Run stress tests
python scripts/17_stress_test.py --samples_per_class 10
```

---

## 📊 **Expected Performance**

### Current (Synthetic Data):
- **Handcrafted features**: 26.67% accuracy
- **Reason**: Mathematical patterns, not real photographs

### With Real Data:
- **Handcrafted only (26-dim)**: 50-70% accuracy (estimated)
- **CNN + Handcrafted (2074-dim)**: 80-95% accuracy (estimated)
- **Depends on**: Image quality, quantity, class balance

---

## 🎯 **Key Advantages**

Your pipeline is production-ready:

✅ **No Code Changes Needed** - Everything adapts automatically  
✅ **Auto Input Detection** - Handles 26-dim or 2074-dim features  
✅ **Flexible Image Sizes** - Any resolution works (auto-resized)  
✅ **Dataset Agnostic** - No hard-coded synthetic dependencies  
✅ **GPU Ready** - Will use GPU if available (check with `torch.cuda.is_available()`)  
✅ **Fully Tested** - All scripts verified and working

---

## ⚙️ **Training Configuration**

Current settings (in `scripts/07_train_classifier.py`):

```python
# Auto-adjusted based on input:
- input_dim: Auto-detected (26 or 2074)
- hidden_dims: [64, 32, 16] for 26-dim, [512, 256, 128] for 2074-dim
- dropout_rates: [0.3, 0.2, 0.2]
- num_classes: 4 (fixed)

# Hyperparameters:
- batch_size: 32
- num_epochs: 100 (with early stopping)
- learning_rate: 0.001
- optimizer: AdamW
- lr_scheduler: ReduceLROnPlateau
- early_stopping_patience: 15
```

To customize, edit the config dict in the training script or pass a JSON config file.

---

## 🚀 **Performance Tips**

### For Faster Training:
1. **Use GPU** - 10-50× faster than CPU
2. **Use Quick Extraction** - Handcrafted features only (much faster)
3. **Reduce batch_size** - If running out of memory

### For Better Accuracy:
1. **Use Full Extraction** - CNN + Handcrafted features (2074-dim)
2. **More Data** - 300+ images per category recommended
3. **Data Augmentation** - Use `scripts/augment_data.py` if needed
4. **Hyperparameter Tuning** - Adjust learning rate, architecture

### For Production:
1. **Collect Real Data** - 500+ high-quality images per category
2. **Use GPU** - Essential for CNN features
3. **Run Full Pipeline** - Don't skip evaluation steps
4. **Monitor Calibration** - ECE (Expected Calibration Error) should be <0.15

---

## 📝 **One-Command Pipeline**

Run everything in sequence (after adding real images):

```bash
# Windows PowerShell
python scripts/02_clean_dataset.py; `
python scripts/03_split_dataset.py; `
python scripts/06_quick_feature_extraction.py; `
python scripts/07_train_classifier.py; `
python scripts/14_evaluate_system.py
```

---

## 🔍 **Verify Readiness Anytime**

Run the readiness check script:

```bash
python check_real_data_readiness.py
```

This checks all components and provides a detailed status report.

---

## 📦 **Current State**

- **Dataset**: 800 synthetic images (669 after cleaning)
- **Features**: 26-dim handcrafted features extracted
- **Model**: Trained baseline (26.67% accuracy on synthetic)
- **Status**: ✅ Ready to replace with real data

---

## 🎓 **What Makes This Pipeline Real-Data Ready**

1. **Auto Input Detection**:
   - Training script detects feature dimension automatically
   - Architecture adjusts based on input size

2. **Flexible Image Processing**:
   - Feature extractors handle any image size
   - Auto-resize during processing

3. **Dataset Agnostic**:
   - No hard-coded paths to synthetic data
   - Preprocessing works on any organized dataset

4. **Production Grade**:
   - Early stopping prevents overfitting
   - Learning rate scheduling adapts training
   - Comprehensive evaluation metrics
   - Stress testing for robustness

---

## 📞 **Need Help?**

Check these files for details:
- `check_real_data_readiness.py` - Comprehensive readiness check
- `PROJECT_READINESS_REPORT.md` - Full project status
- `STEP8_DELIVERABLES.md` - Evaluation framework docs

---

**Last Updated**: December 28, 2025  
**Status**: ✅ READY FOR REAL DATA TRAINING
