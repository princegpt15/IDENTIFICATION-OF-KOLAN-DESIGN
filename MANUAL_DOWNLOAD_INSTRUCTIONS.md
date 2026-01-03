C:\Users\princ\Desktop\MACHINE TRAINING\kolam_dataset\00_raw_data\pulli_kolam\# MANUAL DOWNLOAD GUIDE - Kolam Dataset
## Step-by-Step Instructions

### ✅ FOLDERS ARE READY!
Your folders are created and waiting for images:
```
kolam_dataset/00_raw_data/
├── pulli_kolam/       (0 images - add here)
├── chukku_kolam/      (0 images - add here)
├── line_kolam/        (0 images - add here)
└── freehand_kolam/    (0 images - add here)
```

---

## 📥 DOWNLOAD PROCESS

### Step 1: Open Search Links

**Click these links or run: `open_download_links.bat`**

**Pulli Kolam** (Dot-based patterns):
- https://www.google.com/search?q=pulli+kolam+pattern&tbm=isch
- https://www.google.com/search?q=kambi+kolam+dots&tbm=isch
- Look for: Patterns made with dots connected by lines

**Chukku Kolam** (Spiral/Wheel patterns):
- https://www.google.com/search?q=chukku+kolam+spiral&tbm=isch
- https://www.google.com/search?q=sikku+kolam+wheel&tbm=isch
- Look for: Circular, spiral, radiating patterns

**Line Kolam** (Geometric patterns):
- https://www.google.com/search?q=line+kolam+geometric&tbm=isch
- https://www.google.com/search?q=ner+pulli+kolam&tbm=isch
- Look for: Straight lines, symmetrical geometric designs

**Freehand Kolam** (Artistic patterns):
- https://www.google.com/search?q=freehand+kolam+art&tbm=isch
- https://www.google.com/search?q=freestyle+kolam+modern&tbm=isch
- Look for: Flowing, curved, artistic patterns

---

### Step 2: Download Images (Per Category)

1. **Scroll down** to load 100-150 images
2. **Right-click** on an image → "Save image as..."
3. **Save location**: Choose the correct folder
   - Pulli → `kolam_dataset\00_raw_data\pulli_kolam\`
   - Chukku → `kolam_dataset\00_raw_data\chukku_kolam\`
   - Line → `kolam_dataset\00_raw_data\line_kolam\`
   - Freehand → `kolam_dataset\00_raw_data\freehand_kolam\`

4. **Filename**: Use simple names (image1.jpg, kolam1.png, etc.)
5. **Repeat** for 50-100 images per category

**⏱️ Time estimate**: 10-15 minutes per category (40-60 min total)

---

### Step 3: Quick Selection Tips

**✅ GOOD IMAGES:**
- Clear, high resolution
- Well-lit with good contrast
- Complete pattern (not cropped)
- Minimal or no watermarks
- Single kolam per image

**❌ AVOID:**
- Blurry or low quality
- Heavy watermarks/text
- Multiple kolams in one image
- Too dark or washed out
- Extreme angles

---

## 📊 RECOMMENDED QUANTITIES

**Minimum (for testing):**
- 20 images per category = 80 total
- Training will work but accuracy may be lower

**Good (for decent results):**
- 50 images per category = 200 total  
- Balanced training with reasonable accuracy

**Excellent (for best results):**
- 100+ images per category = 400+ total
- Best accuracy and generalization

**Start with minimum (20 each) to test the pipeline, then add more!**

---

## ✅ CHECK YOUR PROGRESS

Run this command anytime to see your progress:
```powershell
Get-ChildItem "kolam_dataset\00_raw_data" -Recurse -File | Group-Object Directory | ForEach-Object { Write-Host "$($_.Name.Split('\')[-1]): $($_.Count) images" }
```

Or use the quick checker:
```bash
python -c "from pathlib import Path; [print(f'{d.name}: {len(list(d.glob(\"*.*\")))} images') for d in Path('kolam_dataset/00_raw_data').iterdir() if d.is_dir()]"
```

---

## 🚀 AFTER DOWNLOADING - Run the Pipeline

Once you have at least 20 images per category:

```bash
# 1. Clean and validate images
python scripts/02_clean_dataset.py

# 2. Split into train/val/test (70/15/15%)
python scripts/03_split_dataset.py

# 3. Extract features (handcrafted - fast)
python scripts/06_quick_feature_extraction.py

# 4. Train the model
python scripts/07_train_classifier.py

# 5. Evaluate results
python scripts/14_evaluate_system.py
```

**Total time**: ~30-40 minutes for complete pipeline

---

## 💡 PRO TIPS

1. **Download in batches**: Do 20 images per category first, test pipeline, then add more
2. **Name consistently**: Simple filenames work best (image1.jpg, image2.jpg...)
3. **Mix formats OK**: .jpg, .png, .jpeg all work fine
4. **Review quality**: After downloading, quickly review and delete bad images
5. **More is better**: But 50-100 per category gives good results

---

## 🎯 EXPECTED RESULTS

**With 20-50 images per category:**
- Handcrafted features: 40-60% accuracy
- CNN features: 60-75% accuracy

**With 100+ images per category:**
- Handcrafted features: 50-70% accuracy
- CNN features: 75-90% accuracy

**Compare to current synthetic baseline: 26.67%**
Real data will be MUCH better!

---

## 🔧 TROUBLESHOOTING

**"Can't save to folder"**
- Make sure folders exist (run `open_download_links.bat` first)
- Check folder path is correct

**"Too many images"**
- No problem! More is better
- Cleaning script will validate all images

**"Not sure which category"**
- Use your best judgment
- Cleaning script will help identify issues

**"Found duplicate images"**
- That's OK, cleaning script will handle it
- Or manually delete obvious duplicates

---

## 📞 READY TO START?

Run the batch file to open all links:
```bash
open_download_links.bat
```

Or open links manually from above and start downloading!

**Current status: FOLDERS READY, 0 IMAGES - Waiting for your downloads!**
