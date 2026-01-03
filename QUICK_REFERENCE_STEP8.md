# STEP 8: QUICK REFERENCE

## 🚀 One-Line Commands

```bash
# Baseline Evaluation
python scripts/14_evaluate_system.py --test_dir kolam_dataset/02_split_data/test

# Error Analysis
python scripts/15_error_analysis.py --test_dir kolam_dataset/02_split_data/test

# Optimization
python scripts/16_optimization.py --baseline evaluation_results/baseline/metrics_*.json --test_dir kolam_dataset/02_split_data/test

# Stress Test
python scripts/17_stress_test.py --test_dir kolam_dataset/02_split_data/test --samples_per_class 5

# Compare
python scripts/18_compare_performance.py --baseline evaluation_results/baseline/metrics_*.json --optimized evaluation_results/optimized/metrics_*.json
```

## 📊 Key Metrics

| Metric | Formula | Target | Interpretation |
|--------|---------|--------|----------------|
| **Accuracy** | Correct / Total | ≥85% | Overall correctness |
| **Precision** | TP / (TP + FP) | ≥80% | Correct positives |
| **Recall** | TP / (TP + FN) | ≥80% | Coverage of positives |
| **F1-Score** | 2×(P×R)/(P+R) | ≥82% | Balanced measure |
| **ECE** | Σ(accuracy-conf)×weight | <0.10 | Calibration quality |
| **MCE** | max(accuracy-conf) | <0.15 | Worst calibration |

## 🔧 Python API Quick Reference

```python
# Metrics
from scripts.evaluation import MetricsCalculator
calc = MetricsCalculator(class_names=['c1','c2','c3','c4'])
metrics = calc.calculate_all_metrics(y_true, y_pred, confidences=conf)
print(calc.format_metrics_summary(metrics))

# Errors
from scripts.evaluation import ErrorAnalyzer
analyzer = ErrorAnalyzer(class_names=['c1','c2','c3','c4'])
analysis = analyzer.analyze_errors(y_true, y_pred, conf, paths)
print(analyzer.generate_error_report(analysis))

# Confidence
from scripts.evaluation import ConfidenceEvaluator
evaluator = ConfidenceEvaluator(n_bins=10)
evaluation = evaluator.evaluate_confidence(y_true, y_pred, conf)
print(evaluator.generate_confidence_report(evaluation))

# Optimization
from scripts.evaluation import OptimizationEngine
engine = OptimizationEngine()
optimized_img = engine.optimize_preprocessing(img, 'adaptive_threshold')
thresholds = engine.optimize_thresholds(y_true, conf, 'per_class')
weights = engine.optimize_weights(y_true, cnn_scores, rule_scores)

# Stress Test
from scripts.evaluation import StressTester
tester = StressTester(seed=42)
tests = tester.generate_stress_test_suite(image, output_dir='stress/')
results = tester.evaluate_stress_tests(tests, preds, truth)
print(tester.generate_stress_report(results))
```

## 📁 Output Files

| File | Description | Size |
|------|-------------|------|
| `metrics_*.json` | Complete metrics | ~5KB |
| `error_analysis_*.json` | Error breakdown | ~10KB |
| `confidence_*.json` | Calibration data | ~5KB |
| `evaluation_report_*.txt` | Combined report | ~5KB |
| `confusion_matrix.png` | Confusion viz | ~100KB |
| `stress_test_results_*.json` | Stress results | ~20KB |
| `comparison_*.json` | Before/after | ~5KB |

## 🎯 Optimization Strategies

| ID | Strategy | Expected Gain | Parameters |
|----|----------|---------------|------------|
| E1 | Adaptive Threshold | +2% | block_size=11, c=2 |
| E2 | Bilateral Filter | +1% | d=9, sigma=75 |
| E3 | CLAHE | +1.5% | clip_limit=2.0 |
| E4 | Robust Scaling | +1% | - |
| E5 | Feature Selection | +0.5% | threshold=0.01 |
| E6 | Per-Class Thresholds | +1% F1 | per class |
| E7 | Weight Balance | +2% | α, β search |
| E8 | Confidence Weight | +1.5% | γ search |

## 🧪 Stress Test Types

| Type | Variations | Count |
|------|------------|-------|
| Noise | Low, Medium, High | 3 |
| Blur | Gaussian/Motion × 3 levels | 6 |
| Rotation | -30°, -15°, +15°, +30° | 4 |
| Crop | 50%, 70%, 90% | 3 |
| Brightness | -50, -25, +25, +50 | 4 |
| Contrast | 0.5x, 0.75x, 1.5x, 2.0x | 4 |
| Occlusion | 10%, 20%, 30% | 3 |
| **Total** | | **27** per image |

## ⚠️ Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Low accuracy (<80%) | Poor features/data | Review data, retrain |
| High ECE (>0.15) | Bad calibration | Apply temperature scaling |
| Class imbalance errors | Uneven data | Rebalance or weighted loss |
| Poor stress test (<50%) | Not robust | Add augmentation |
| Import errors | Missing deps | pip install -r requirements.txt |

## 📋 Checklist

### Before Running
- [ ] Test data available in `kolam_dataset/02_split_data/test/`
- [ ] Trained model exists
- [ ] Feature extractor working
- [ ] Dependencies installed

### Baseline Evaluation
- [ ] Run script 14 (evaluate_system.py)
- [ ] Check accuracy ≥85%
- [ ] Check Macro-F1 ≥82%
- [ ] Check ECE <0.10
- [ ] Review error patterns

### Optimization
- [ ] Run script 16 (optimization.py)
- [ ] Test E1-E3 (preprocessing)
- [ ] Test E6 (thresholds)
- [ ] Test E7 (weights)
- [ ] Measure improvements

### Stress Testing
- [ ] Run script 17 (stress_test.py)
- [ ] Check overall ≥60%
- [ ] Identify weak stress types
- [ ] Review failure examples

### Comparison
- [ ] Run script 18 (compare_performance.py)
- [ ] Verify improvements
- [ ] Generate charts
- [ ] Document findings

## 🔗 Related Commands

```bash
# View results
cat evaluation_results/baseline/evaluation_report_*.txt
cat evaluation_results/baseline/metrics_*.json | python -m json.tool

# Find latest results
ls -lt evaluation_results/baseline/ | head
ls -lt evaluation_results/optimized/ | head

# Open visualizations
start evaluation_results/errors/confusion_matrix.png
start evaluation_results/comparison/comparison_chart.png

# Quick stats
python -c "import json; print(json.load(open('evaluation_results/baseline/metrics_*.json'))['accuracy'])"
```

## 📞 Support

- Documentation: [STEP8_README.md](STEP8_README.md)
- Design: [STEP8_EVALUATION_DESIGN.md](STEP8_EVALUATION_DESIGN.md)
- Deliverables: [STEP8_DELIVERABLES.md](STEP8_DELIVERABLES.md)
- Summary: [STEP8_EXECUTION_SUMMARY.md](STEP8_EXECUTION_SUMMARY.md)

---

**Quick Tip**: Start with script 14 for baseline, then script 15 for errors, then script 16 for optimization!
