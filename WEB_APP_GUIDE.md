# KOLAM WEB APP - QUICK START GUIDE
====================================

## 🚀 LAUNCH THE WEB INTERFACE

### Method 1: Double-click (Windows)
```
Double-click: launch_web_app.bat
```

### Method 2: Command Line
```bash
streamlit run kolam_web_app.py
```

### Method 3: PowerShell
```powershell
cd "c:\Users\princ\Desktop\MACHINE TRAINING"
streamlit run kolam_web_app.py
```

---

## 🌐 ACCESS THE APP

The app will automatically open in your browser at:
**http://localhost:8501**

If it doesn't open automatically:
1. Look for the URL in the terminal
2. Copy and paste it into your browser

---

## ✨ FEATURES

### 📷 Single Image Tab
- Upload any Kolam image (JPG/PNG)
- Get instant classification
- View confidence gauge (0-100%)
- See probability distribution for all classes
- Beautiful visualizations

### 📁 Batch Upload Tab
- Upload multiple images at once
- Process them all simultaneously
- View results in a table
- Download results as CSV

### 📊 Statistics Tab
- Model performance metrics
- Per-class F1-scores
- Architecture diagram
- Training details

---

## 🎯 HOW TO USE

1. **Launch** the app (see methods above)
2. **Upload** a Kolam image in the "Single Image" tab
3. **Click** "Classify Image" button
4. **View** results with confidence scores
5. **Try** batch processing for multiple images

---

## 📸 TEST IMAGES

You can test with images from:
```
kolam_dataset/02_split_data/test/
├── pulli_kolam/
├── chukku_kolam/
├── line_kolam/
└── freehand_kolam/
```

---

## 🛠️ TROUBLESHOOTING

### App won't start?
```bash
pip install streamlit plotly
```

### Model not found?
Make sure this file exists:
```
kolam_dataset/05_trained_models/balanced_training/best_model_balanced.pth
```

### Port already in use?
```bash
streamlit run kolam_web_app.py --server.port 8502
```

---

## 🎨 FEATURES OVERVIEW

✅ Real-time image classification
✅ Interactive confidence gauges
✅ Probability bar charts
✅ Batch processing
✅ CSV export
✅ Beautiful UI with custom styling
✅ Model statistics dashboard
✅ Responsive design

---

## 📊 PERFORMANCE

- Model Accuracy: 91% F1-Score
- Inference Time: <1 second per image
- Supports: JPG, JPEG, PNG
- All 4 classes balanced

---

## 🎯 NEXT STEPS

After using the web app:

1. **Share it:** Run on your network with `--server.address 0.0.0.0`
2. **Deploy it:** Deploy to Streamlit Cloud (free!)
3. **Customize it:** Edit kolam_web_app.py to add features
4. **Integrate it:** Use as part of larger application

---

**Enjoy classifying Kolam patterns! 🎨**
