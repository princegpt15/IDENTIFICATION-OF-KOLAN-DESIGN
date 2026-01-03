# Real Kolam Dataset - Quick Setup Guide

## 🚀 **FASTEST METHOD: Browser Extension + Bulk Download**

Since automated scripts are hitting rate limits, here's the most reliable approach:

### **Step 1: Install a Bulk Image Downloader**

Choose one of these FREE browser extensions:

**For Chrome/Edge:**
- **Download All Images** - https://chrome.google.com/webstore (search "Download All Images")
- **Image Downloader** - Simple and effective
- **Bulk Image Downloader** - Feature-rich

**For Firefox:**
- **DownThemAll!** - Popular and reliable
- **Save All Images** - Easy to use

### **Step 2: Open These Google Image Searches**

**Pulli Kolam (100 images):**
```
https://www.google.com/search?q=pulli+kolam&tbm=isch
https://www.google.com/search?q=kambi+kolam+dots&tbm=isch
```

**Chukku Kolam (100 images):**
```
https://www.google.com/search?q=chukku+kolam&tbm=isch
https://www.google.com/search?q=sikku+kolam&tbm=isch
```

**Line Kolam (100 images):**
```
https://www.google.com/search?q=line+kolam&tbm=isch
https://www.google.com/search?q=geometric+kolam&tbm=isch
```

**Freehand Kolam (100 images):**
```
https://www.google.com/search?q=freehand+kolam&tbm=isch
https://www.google.com/search?q=freestyle+kolam+art&tbm=isch
```

### **Step 3: Download Images**

1. Scroll down to load ~100-150 images
2. Click the extension icon
3. Select "Download all images on page"
4. Save to the correct folder:
   - Pulli → `kolam_dataset/00_raw_data/pulli_kolam/`
   - Chukku → `kolam_dataset/00_raw_data/chukku_kolam/`
   - Line → `kolam_dataset/00_raw_data/line_kolam/`
   - Freehand → `kolam_dataset/00_raw_data/freehand_kolam/`

**Time: 15-20 minutes total**

---

## 📱 **ALTERNATIVE: Use Pinterest**

Pinterest often has better quality Kolam images:

```
https://www.pinterest.com/search/pins/?q=pulli%20kolam
https://www.pinterest.com/search/pins/?q=chukku%20kolam
https://www.pinterest.com/search/pins/?q=line%20kolam
https://www.pinterest.com/search/pins/?q=freehand%20kolam
```

Right-click images → "Save image as..."

---

## 🎯 **QUICKEST TEST: Use a Small Sample First**

Download just 20-30 images per category to test the pipeline:

**Current folder structure (ready):**
```
kolam_dataset/00_raw_data/
├── pulli_kolam/       (add images here)
├── chukku_kolam/      (add images here)
├── line_kolam/        (add images here)
└── freehand_kolam/    (add images here)
```

---

## ✅ **After Adding Images**

Check what you have:
```powershell
Get-ChildItem "kolam_dataset/00_raw_data" -Recurse -File | Measure-Object
```

Then run the pipeline:
```bash
# 1. Clean and validate
python scripts/02_clean_dataset.py

# 2. Split into train/val/test
python scripts/03_split_dataset.py

# 3. Extract features (fast method)
python scripts/06_quick_feature_extraction.py

# 4. Train model
python scripts/07_train_classifier.py

# 5. Evaluate
python scripts/14_evaluate_system.py
```

---

## 💡 **Pro Tips**

- **Start small**: 20-30 images per category is enough to test
- **Quality > Quantity**: Clear, well-lit images work better
- **Remove watermarks**: Avoid heavily watermarked images
- **Check categories**: Make sure you put images in correct folders
- **More = Better**: But 50+ per category is a good start

---

## 🔥 **Ready to Go!**

The folders are set up. Just add images and run the pipeline!
