# 🎨 Kolam Pattern Classifier

> **Traditional Indian Art Meets Modern AI** - A deep learning system to classify Indian Kolam patterns with 91% accuracy

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52+-ff4b4b.svg)](https://streamlit.io/)

## ✨ Features

- 🎯 **91% Macro F1-Score** - Highly accurate classification
- 🚀 **Real-time Classification** - Instant results with confidence scores
- 🎨 **Beautiful Web Interface** - Streamlit UI with Kolam-inspired design
- 📊 **Interactive Visualizations** - Plotly charts and gauges
- 🔄 **Batch Processing** - Classify multiple images at once
- ⚖️ **Balanced Training** - Focal Loss for class imbalance

## 🌟 Overview

Kolam is a traditional Indian art form where patterns are drawn using rice flour. This project uses machine learning to classify Kolam patterns into 4 distinct categories.

### Kolam Types

1. **🔴 Chukki Kolam** - Dot-based geometric patterns with connected lines
2. **📏 Line Kolam** - Continuous line drawings without lifting the hand
3. **🎨 Freehand Kolam** - Creative freestyle designs with artistic freedom
4. **⚫ Pulli Kolam** - Grid-based patterns with dots as foundation points

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch web app
streamlit run kolam_web_app.py
```

Open your browser to `http://localhost:8501`

## 📊 Performance

| Metric | Score |
|--------|-------|
| **Macro F1-Score** | 91.0% |
| **Test Accuracy** | 90.67% |
| **Training Samples** | 17,280 |

### Per-Class Performance

| Class | F1-Score | Precision | Recall |
|-------|----------|-----------|--------|
| Pulli Kolam | 89.8% | 90.2% | 89.4% |
| Chukki Kolam | 87.1% | 86.5% | 87.7% |
| Line Kolam | 94.7% | 94.1% | 95.3% |
| Freehand Kolam | 92.4% | 93.0% | 91.8% |

## 🏗️ Model Architecture

- **Input:** 26 handcrafted features
- **Architecture:** 128 → 64 → 32 neurons
- **Loss Function:** Focal Loss (α=0.25, γ=2.0)
- **Training:** WeightedRandomSampler for balanced batches

## 💻 Usage

### Web Interface (Recommended)

```bash
streamlit run kolam_web_app.py
```

### Command Line

```bash
# Single image
python classify_kolam_image.py image.jpg

# Batch processing
python batch_classify_kolams.py folder/
```

## 📁 Project Structure

```
├── kolam_web_app.py              # Main Streamlit interface
├── classify_kolam_image.py       # CLI classifier
├── balance_and_retrain.py        # Training script
├── requirements.txt              # Dependencies
└── kolam_dataset/                # Dataset & models
```

## 📧 Contact

**Prince Kumar**
- Email: princekr89360@gmail.com
- GitHub: [@princekr89360](https://github.com/princekr89360)

---

<div align="center">
  <p>✦ ❋ ✿ ❀ ❈ ✿ ❋ ✦</p>
  <p><b>Preserving Traditional Indian Art through AI</b></p>
  <p>Made with ❤️ & 🤖</p>
</div>
