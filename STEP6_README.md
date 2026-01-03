# STEP 6: CONFIDENCE SCORE GENERATION
## Quick Start Guide

**Version:** 1.0  
**Date:** December 28, 2025  
**Status:** ✅ Production Ready

---

## 🎯 OVERVIEW

This module implements advanced confidence scoring for Kolam pattern classification, combining CNN predictions, rule-based validation, and entropy analysis into a single, interpretable confidence score.

**Key Formula:**
```
C_final = (α × P_cnn + β × S_rule) × (1 - γ × H_norm)
```

Where:
- **α = 0.65** (CNN weight)
- **β = 0.35** (Rule weight)  
- **γ = 0.20** (Entropy penalty)

---

## 🚀 QUICK START

### 1. Run Demo (5 minutes)
```bash
cd "c:\Users\princ\Desktop\MACHINE TRAINING"
python scripts/11_demo_confidence.py
```

**What you'll see:**
- Perfect pattern → 91% confidence → Auto-accept
- Ambiguous pattern → 50% confidence → Human review
- Poor quality → 32% confidence → Reject
- CNN-Rule conflict → Warnings and adjustments

### 2. Analyze Calibration (if validation data available)
```bash
python scripts/12_analyze_calibration.py --split val --save-report
```

**Output:**
- ECE (Expected Calibration Error)
- Reliability diagram
- Per-bin statistics
- Saved to `kolam_dataset/06_confidence_analysis/`

### 3. Use in Inference
```bash
# Single image with full explanation
python scripts/13_inference_with_confidence.py \
    --image "path/to/kolam.jpg" \
    --verbose

# Batch processing with confidence filtering
python scripts/13_inference_with_confidence.py \
    --image-dir "path/to/images/" \
    --min-confidence 75 \
    --save-results results.json
```

---

## 💻 PROGRAMMATIC USAGE

### Basic Confidence Calculation

```python
from scripts.confidence_scoring import AdvancedConfidenceCalculator
import numpy as np

# Initialize calculator
calc = AdvancedConfidenceCalculator(
    alpha=0.65,  # CNN weight
    beta=0.35,   # Rule weight
    gamma=0.20   # Entropy penalty
)

# Your CNN predictions (softmax probabilities)
cnn_probs = np.array([0.85, 0.10, 0.03, 0.02])

# Your rule validation score (0-1)
rule_score = 0.78

# Compute confidence
result = calc.compute_confidence(
    cnn_probs, 
    rule_score, 
    return_components=True
)

print(f"Confidence: {result['confidence']:.1f}%")
print(f"Level: {result['confidence_level']}")
```

### With Conflict Detection

```python
# Detect CNN-Rule disagreements and other issues
result = calc.compute_with_adjustment(
    cnn_probs,
    rule_score,
    detect_conflicts=True
)

if result['warnings']:
    print("⚠️ Warnings:")
    for warning in result['warnings']:
        print(f"  • {warning}")
```

### Generate Explanation

```python
from scripts.confidence_scoring import ConfidenceExplainer

explainer = ConfidenceExplainer()
explanation = explainer.explain(result)

# Console output
print(explanation['console_output'])

# Reasoning steps
for step in explanation['reasoning_steps']:
    print(f"  - {step}")
```

### Validate for Overconfidence

```python
from scripts.confidence_scoring import OverconfidenceValidator

validator = OverconfidenceValidator()
validation = validator.validate(result)

if validation['is_overconfident']:
    print(f"Status: {validation['status']}")
    for flag in validation['flags']:
        print(f"  {flag['type']}: {flag['description']}")
```

### Manage Thresholds

```python
from scripts.confidence_scoring import ThresholdManager

# Initialize with profile
threshold_mgr = ThresholdManager(profile='standard')
threshold_mgr.initialize_context_presets()

# Get decision for confidence score
decision = threshold_mgr.get_decision(
    confidence=78.5,
    context='critical'  # or 'batch', 'screening', None
)

print(f"Action: {decision['action']}")
print(f"Rationale: {decision['rationale']}")
```

### Calibration Monitoring

```python
from scripts.confidence_scoring import CalibrationMonitor

monitor = CalibrationMonitor(num_bins=10)

# Add predictions as you make them
for pred, conf, true_label in zip(predictions, confidences, labels):
    monitor.add_prediction(pred, conf, true_label)

# Compute calibration
cal_data = monitor.compute_calibration()

print(f"ECE: {cal_data['ece']:.4f}")
print(f"Status: {cal_data['calibration_status']}")

# Generate report
print(monitor.generate_calibration_report())

# Save reliability diagram
monitor.plot_reliability_diagram(save_path='reliability.png')
```

---

## 📊 CONFIDENCE LEVELS & ACTIONS

| Level | Range | Action | Use Case |
|-------|-------|--------|----------|
| **VERY_HIGH** | 90-100% | ✓✓ Auto-accept | Production deployment, batch processing |
| **HIGH** | 75-90% | ✓ Auto-accept (log) | Normal operations with audit trail |
| **MEDIUM** | 60-75% | ⚠️ Flag for review | Border cases, quality check |
| **LOW** | 40-60% | 🚨 Human verification | Ambiguous patterns |
| **VERY_LOW** | 0-40% | ❌ Reject/manual | Poor quality, corrupted images |

---

## 🎛️ PARAMETER TUNING

### Default Parameters (Balanced)
```python
α = 0.65  # CNN gets more weight (learned patterns)
β = 0.35  # Rules validate (domain knowledge)
γ = 0.20  # Moderate entropy penalty
```

### Conservative Mode (Minimize False Positives)
```python
α = 0.60  # Reduce CNN dominance
β = 0.40  # Increase rule importance
γ = 0.30  # Stronger entropy penalty
```

### Aggressive Mode (Maximize Automation)
```python
α = 0.70  # Trust CNN more
β = 0.30  # Less strict rules
γ = 0.15  # Lighter entropy penalty
```

### How to Change
```python
calc = AdvancedConfidenceCalculator(alpha=0.60, beta=0.40, gamma=0.30)
```

---

## 🔍 UNDERSTANDING OUTPUTS

### Component Breakdown Example
```
Confidence Breakdown:
  CNN Probability:     82.0% → 53.3% (weight: 0.65)
  Rule Validation:     74.0% → 25.9% (weight: 0.35)
  ─────────────────────────────────────────
  Base Confidence:     79.2%
  Entropy Penalty:    -0.7%
  ─────────────────────────────────────────
  Final Confidence:    78.5%
```

**Interpretation:**
- CNN contributes 53.3% to final score
- Rules contribute 25.9% to final score
- Entropy reduces confidence by 0.7%
- Final score: 78.5% (HIGH level)

### Reasoning Steps Example
```
1. ✓ CNN confidently predicts Pulli Kolam (82.0%)
2. ✓ Rules validate prediction (74.0% compliance)
3. ✓ Prediction is decisive (consistency: 91.3%)
4. ✓ CNN and rules are in strong agreement
5. ✓ Overall: HIGH confidence - prediction is reliable
```

---

## ⚙️ THRESHOLD PROFILES

### Available Profiles

**Standard (Default)**
```python
threshold_mgr = ThresholdManager(profile='standard')
# Very High: 90%, High: 75%, Medium: 60%, Low: 40%
```

**Conservative (Critical Applications)**
```python
threshold_mgr = ThresholdManager(profile='conservative')
# Very High: 90%, High: 80%, Medium: 70%, Low: 50%
```

**Aggressive (Maximize Automation)**
```python
threshold_mgr = ThresholdManager(profile='aggressive')
# Very High: 80%, High: 65%, Medium: 50%, Low: 30%
```

### Context Presets

```python
threshold_mgr.initialize_context_presets()

# Critical: Museum cataloging, research
decision = threshold_mgr.get_decision(confidence, context='critical')

# Batch: Large-scale processing
decision = threshold_mgr.get_decision(confidence, context='batch')

# Screening: Initial sorting
decision = threshold_mgr.get_decision(confidence, context='screening')
```

---

## 🛡️ OVERCONFIDENCE DETECTION

The system automatically detects:

1. **CNN-Rule Disagreement**
   - Trigger: |CNN_prob - Rule_score| > 30%
   - Action: -10% confidence penalty

2. **Out-of-Distribution (OOD)**
   - Trigger: CNN > 85% AND Rules < 50%
   - Action: -15% penalty + warning

3. **Entropy Conflict**
   - Trigger: CNN > 80% AND Entropy > 70%
   - Action: -8% penalty

4. **Extreme Probabilities**
   - Trigger: CNN > 98%
   - Action: Warning (possible overfitting)

---

## 📈 CALIBRATION BEST PRACTICES

### 1. Initial Calibration
```bash
# Run on validation set
python scripts/12_analyze_calibration.py --split val --save-report

# Check ECE
# Target: ECE < 0.05 (well-calibrated)
# If ECE > 0.10: Consider parameter adjustment
```

### 2. Adjust Based on Results

**If Overconfident (Predictions > Accuracy):**
```python
# Increase entropy penalty
calc = AdvancedConfidenceCalculator(gamma=0.30)

# Or use conservative thresholds
threshold_mgr = ThresholdManager(profile='conservative')
```

**If Underconfident (Predictions < Accuracy):**
```python
# Decrease entropy penalty
calc = AdvancedConfidenceCalculator(gamma=0.10)

# Or use aggressive thresholds
threshold_mgr = ThresholdManager(profile='aggressive')
```

### 3. Monitor in Production
```python
# Continuously track
monitor = CalibrationMonitor()

# Periodically recompute
cal_data = monitor.compute_calibration()
if cal_data['ece'] > 0.10:
    print("⚠️ Recalibration needed")
```

---

## 🐛 TROUBLESHOOTING

### Issue: All confidences too high
**Cause:** Model overconfident  
**Solution:**
```python
# Increase entropy penalty
calc = AdvancedConfidenceCalculator(gamma=0.30)
# Or use conservative thresholds
```

### Issue: All confidences too low
**Cause:** Parameters too strict  
**Solution:**
```python
# Decrease entropy penalty
calc = AdvancedConfidenceCalculator(gamma=0.10)
# Or reduce rule weight
calc = AdvancedConfidenceCalculator(beta=0.25)
```

### Issue: Many overconfidence warnings
**Cause:** CNN-Rule disagreement  
**Solution:**
1. Check rule definitions
2. Retrain CNN or recalibrate rules
3. Increase disagreement penalty

### Issue: Low automation rate
**Cause:** Thresholds too strict  
**Solution:**
```python
threshold_mgr = ThresholdManager(profile='aggressive')
# Or adjust specific thresholds
threshold_mgr.set_thresholds(high=70, medium=55)
```

---

## 📝 INTEGRATION CHECKLIST

- [ ] Run demo script to understand behavior
- [ ] Analyze calibration on validation set
- [ ] Choose appropriate threshold profile
- [ ] Integrate into inference pipeline
- [ ] Set up monitoring for production
- [ ] Define human review workflow for flagged cases
- [ ] Establish feedback loop for continuous improvement

---

## 📚 ADDITIONAL RESOURCES

### Documentation
- [STEP6_CONFIDENCE_DESIGN.md](STEP6_CONFIDENCE_DESIGN.md) - Full design document
- [STEP6_DELIVERABLES.md](STEP6_DELIVERABLES.md) - Complete deliverables list
- [STEP6_EXECUTION_SUMMARY.md](STEP6_EXECUTION_SUMMARY.md) - Project summary

### Code Files
- `scripts/confidence_scoring/` - Core modules
- `scripts/11_demo_confidence.py` - Interactive demo
- `scripts/12_analyze_calibration.py` - Calibration tool
- `scripts/13_inference_with_confidence.py` - Enhanced inference

### References
1. Guo et al. (2017) - "On Calibration of Modern Neural Networks"
2. Platt (1999) - "Probabilistic Outputs for SVMs"
3. Niculescu-Mizil & Caruana (2005) - "Predicting Good Probabilities"

---

## 💡 TIPS & BEST PRACTICES

1. **Always start with demo** - Understand behavior before deployment
2. **Calibrate regularly** - Check ECE monthly or after model updates
3. **Context matters** - Use appropriate thresholds for each use case
4. **Monitor overconfidence** - Track warnings and adjust
5. **Trust the warnings** - System flags are there for good reason
6. **Tune gradually** - Small parameter changes can have big effects
7. **Document decisions** - Log why you chose specific thresholds

---

## 🎓 FOR RESEARCHERS

### Reproducibility
All formulas, parameters, and thresholds are documented. Results are reproducible with:
```python
np.random.seed(42)
torch.manual_seed(42)
```

### Citation
If you use this confidence scoring system in research:
```
Kolam Pattern Classification with Hybrid Confidence Scoring
Advanced Confidence Calculator combining CNN predictions, 
rule-based validation, and entropy-based consistency metrics.
December 2025.
```

---

## ✅ VALIDATION CHECKLIST

Before deployment, verify:
- [ ] ECE < 0.05 on validation set
- [ ] Overconfidence rate < 20%
- [ ] Automation rate meets requirements
- [ ] All demos run successfully
- [ ] Thresholds appropriate for use case
- [ ] Monitoring system in place

---

**Ready to deploy? Run the demo first!**
```bash
python scripts/11_demo_confidence.py
```

**Questions? Check the design document:**
[STEP6_CONFIDENCE_DESIGN.md](STEP6_CONFIDENCE_DESIGN.md)
