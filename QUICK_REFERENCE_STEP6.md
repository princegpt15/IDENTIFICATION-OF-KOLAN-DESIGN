# STEP 6: CONFIDENCE SCORING - QUICK REFERENCE

## 🚀 ONE-MINUTE START
```bash
cd "c:\Users\princ\Desktop\MACHINE TRAINING"
python scripts/11_demo_confidence.py
```

## 📐 THE FORMULA
```
C_final = (0.65 × P_cnn + 0.35 × S_rule) × (1 - 0.20 × H_norm)
         └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
          CNN weight        Rule weight      Entropy penalty
```

## 📊 CONFIDENCE LEVELS
| Level | Range | Action | Symbol |
|-------|-------|--------|--------|
| Very High | 90-100% | Auto-accept | ✓✓ |
| High | 75-90% | Auto-accept (log) | ✓ |
| Medium | 60-75% | Flag for review | ⚠️ |
| Low | 40-60% | Human verification | 🚨 |
| Very Low | 0-40% | Reject/manual | ❌ |

## 💻 BASIC USAGE
```python
from scripts.confidence_scoring import AdvancedConfidenceCalculator

calc = AdvancedConfidenceCalculator()
result = calc.compute_confidence(
    cnn_probabilities=[0.85, 0.10, 0.03, 0.02],
    rule_score=0.78
)

print(f"Confidence: {result['confidence']:.1f}%")
print(f"Level: {result['confidence_level']}")
```

## 🎯 WHAT IT DOES
- ✅ Combines CNN + Rules + Entropy
- ✅ Detects overconfidence
- ✅ Explains decisions
- ✅ Monitors calibration
- ✅ Manages thresholds
- ✅ No retraining needed

## 📁 KEY FILES
```
scripts/confidence_scoring/     # Core package
├── confidence_calculator.py    # Main engine
├── entropy_analyzer.py         # Consistency metrics
├── explainer.py               # Explanations
├── validator.py               # Overconfidence checks
├── calibration_monitor.py     # Calibration tracking
└── threshold_manager.py       # Threshold control

scripts/
├── 11_demo_confidence.py      # Interactive demo
├── 12_analyze_calibration.py  # Calibration tool
└── 13_inference_with_confidence.py  # Enhanced inference

Documentation:
├── STEP6_CONFIDENCE_DESIGN.md    # Full design (14 sections)
├── STEP6_README.md               # Quick start guide
├── STEP6_DELIVERABLES.md         # Complete deliverables
└── STEP6_EXECUTION_SUMMARY.md    # Project summary
```

## 🛠️ COMMON TASKS

### Run Demo
```bash
python scripts/11_demo_confidence.py
```

### Analyze Calibration
```bash
python scripts/12_analyze_calibration.py --split val --save-report
```

### Enhanced Inference
```bash
# Single image
python scripts/13_inference_with_confidence.py --image path/to/image.jpg --verbose

# Batch
python scripts/13_inference_with_confidence.py --image-dir path/to/images/ --min-confidence 75
```

## 🎚️ PARAMETER TUNING

### Default (Balanced)
```python
α = 0.65  # CNN weight
β = 0.35  # Rule weight
γ = 0.20  # Entropy penalty
```

### Conservative
```python
α = 0.60, β = 0.40, γ = 0.30
```

### Aggressive
```python
α = 0.70, β = 0.30, γ = 0.15
```

## 🔍 OVERCONFIDENCE DETECTION

Automatically detects:
- CNN-Rule disagreement (>30% diff) → -10%
- Out-of-distribution (CNN high, rules low) → -15%
- Entropy conflicts → -8%
- Extreme probabilities (>98%) → Warning

## 📈 CALIBRATION METRICS
- **ECE** < 0.05 → Well-calibrated ✓
- **ECE** > 0.10 → Needs adjustment ⚠️
- **ECE** > 0.15 → Urgent recalibration 🚨

## 🎯 EXAMPLE RESULTS

**Perfect Pattern:**
```
CNN: 95% | Rules: 92% | Entropy: 0.15
→ Confidence: 91.2% [VERY HIGH] ✓✓
→ Action: AUTO-ACCEPT
```

**Ambiguous Pattern:**
```
CNN: 58% | Rules: 62% | Entropy: 0.82
→ Confidence: 49.7% [LOW] 🚨
→ Action: HUMAN VERIFICATION REQUIRED
```

**CNN-Rule Conflict:**
```
CNN: 88% | Rules: 25% | Entropy: 0.35
→ Confidence: 61.4% [MEDIUM] ⚠️
→ Action: FLAG FOR REVIEW
→ Warning: Possible out-of-distribution sample
```

## 📚 DOCUMENTATION
- **Design:** [STEP6_CONFIDENCE_DESIGN.md](STEP6_CONFIDENCE_DESIGN.md)
- **Usage:** [STEP6_README.md](STEP6_README.md)
- **Summary:** [STEP6_EXECUTION_SUMMARY.md](STEP6_EXECUTION_SUMMARY.md)

## ✅ STATUS
**STEP 6: COMPLETE** ✓
- 3,100+ lines of code
- 2,000+ lines of documentation
- 6 core classes
- 3 demonstration scripts
- Production ready

---

**Quick Start:** `python scripts/11_demo_confidence.py`  
**Full Guide:** `STEP6_README.md`  
**Design Details:** `STEP6_CONFIDENCE_DESIGN.md`
