# STEP 4: CLASSIFICATION MODEL DESIGN & TRAINING - COMPLETE DESIGN

## Executive Summary

**Objective:** Design and train a hybrid classification system combining CNN-based feature learning with rule-based domain validation to accurately classify Kolam patterns while ensuring structural and cultural correctness.

**Key Innovation:** Unlike pure CNN approaches, this hybrid system uses explicit rule validation to catch geometrically invalid patterns, providing both high accuracy and interpretability.

---

## Part 1: HYBRID APPROACH JUSTIFICATION

### 1.1 Why Hybrid (CNN + Rule-Based)?

**Pure CNN Limitations:**
- Can misclassify edge cases that violate geometric constraints
- Lacks cultural/structural validation
- "Black box" with limited interpretability
- May predict valid class but with incorrect geometry

**Rule-Based Limitations:**
- Requires extensive manual feature engineering
- Brittle to variations in drawing style
- Cannot learn semantic patterns from data
- Struggles with freehand/artistic variations

**Hybrid Advantages:**
- ✅ **Accuracy:** CNN learns discriminative features
- ✅ **Validation:** Rules enforce geometric correctness
- ✅ **Interpretability:** Rules explain "why" a prediction is made/rejected
- ✅ **Robustness:** CNN handles variations, rules prevent invalid predictions
- ✅ **Confidence:** Combined score reflects both learned and structural evidence

### 1.2 System Architecture

```
Input: Kolam Image
     ↓
[Step 3: Feature Extraction]
     ├─ Handcrafted Features (26-dim)
     └─ CNN Features (2048-dim)
     ↓
[Step 4A: CNN Classifier]
     ↓
Initial Prediction + Softmax Probability
     ↓
[Step 4B: Rule-Based Validator]
     ├─ Check Dot Grid Rules
     ├─ Check Symmetry Rules
     ├─ Check Topology Rules
     └─ Check Geometric Rules
     ↓
Rule Consistency Score (0-1)
     ↓
[Step 4C: Confidence Fusion]
Final Prediction + Combined Confidence (0-100%)
     ↓
Output: {class, confidence, rule_violations, explanation}
```

### 1.3 System Inputs & Outputs

**Inputs:**
- Combined features (2074-dim) from Step 3
- Handcrafted features (26-dim) for rule validation
- Class labels (0: Pulli, 1: Chukku, 2: Line, 3: Freehand)

**Outputs:**
- **Primary:** Class prediction (0-3)
- **Confidence:** Combined CNN + rule score (0-100%)
- **Rule Violations:** List of failed constraints (if any)
- **Explanation:** Why this class was predicted

---

## Part 2: CNN CLASSIFIER DESIGN

### 2.1 Architecture Selection

**Approach:** Feature-based MLP (Multilayer Perceptron) classifier

**Justification:**
- We already have rich 2074-dim features from Step 3
- No need for end-to-end CNN training (features are pre-extracted)
- MLP is sufficient for high-dimensional feature classification
- Faster training, fewer parameters, less overfitting risk

**Alternative (Not Used):** End-to-end CNN on raw images
- Would ignore Step 3 feature extraction
- Requires more training data and compute
- Less interpretable
- Redundant given high-quality features

### 2.2 Network Architecture

**KolamFeatureClassifier:**

```
Input Layer: 2074-dim feature vector
     ↓
FC1: 2074 → 512 (ReLU + Dropout 0.4)
     ↓
FC2: 512 → 256 (ReLU + Dropout 0.3)
     ↓
FC3: 256 → 128 (ReLU + Dropout 0.2)
     ↓
FC4: 128 → 4 (Softmax)
     ↓
Output: 4-class probability distribution
```

**Layer Justification:**
- **FC1 (2074→512):** Dimensionality reduction, learns feature combinations
- **FC2 (512→256):** Hierarchical feature abstraction
- **FC3 (256→128):** Compact representation before classification
- **FC4 (128→4):** Final class logits

**Activation Functions:**
- **ReLU:** Non-linearity, computationally efficient, prevents vanishing gradients
- **Softmax:** Converts logits to probability distribution

**Regularization:**
- **Dropout (0.4, 0.3, 0.2):** Prevents overfitting, forces redundancy
- **Weight Decay (L2):** Penalizes large weights during training
- **Batch Normalization (optional):** Could be added after each FC layer

### 2.3 Loss Function & Optimizer

**Loss Function:** Cross-Entropy Loss
$$\mathcal{L} = -\sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

Where:
- $y_{i,c}$: True one-hot encoded label
- $\hat{y}_{i,c}$: Predicted probability for class $c$

**Why Cross-Entropy?**
- Standard for multi-class classification
- Penalizes confident wrong predictions more heavily
- Works well with softmax output

**Optimizer:** Adam
- **Learning Rate:** 0.001 (with decay)
- **Betas:** (0.9, 0.999)
- **Weight Decay:** 0.0001

**Why Adam?**
- Adaptive learning rates per parameter
- Combines momentum and RMSprop benefits
- Fast convergence
- Well-suited for high-dimensional features

### 2.4 Training Configuration

**Hyperparameters:**
```python
batch_size = 32
num_epochs = 100
learning_rate = 0.001
lr_decay_factor = 0.5
lr_decay_patience = 10
early_stopping_patience = 15
```

**Class Weighting (if imbalanced):**
```python
# Inverse frequency weighting
class_weights = total_samples / (num_classes * class_counts)
```

---

## Part 3: FEATURE UTILIZATION STRATEGY

### 3.1 Classification Input Choice

**Selected Approach:** Fused handcrafted + CNN features (2074-dim)

**Justification:**
1. **Maximum Information:** Uses both interpretable and learned features
2. **Proven in Step 3:** Already validated and normalized
3. **Efficiency:** No need to reprocess images
4. **Consistency:** Same features used in training and inference

### 3.2 Feature Split for Rule Validation

**Classification:** Full 2074-dim features
**Rule Validation:** 26-dim handcrafted features only

**Why?**
- CNN features (2048-dim) are abstract, not directly interpretable
- Handcrafted features (26-dim) map to physical Kolam properties
- Rules need explicit geometric measurements (dots, symmetry, loops)

---

## Part 4: RULE-BASED VALIDATION LAYER

### 4.1 Rule Categories

Four rule categories based on Kolam type characteristics:

#### Rule Category 1: Pulli Kolam Validation
```
IF predicted_class == "Pulli Kolam":
    RULE 1: dot_count >= 20
    RULE 2: grid_regularity >= 0.4
    RULE 3: dot_density >= 5.0%
    RULE 4: dot_spacing_std < 30 pixels
    
    Consistency Score = (rules_passed / total_rules)
```

#### Rule Category 2: Chukku Kolam Validation
```
IF predicted_class == "Chukku Kolam":
    RULE 1: loop_count >= 3
    RULE 2: connectivity_ratio >= 0.6
    RULE 3: dominant_curve_length >= 500 pixels
    RULE 4: edge_continuity >= 50%
    
    Consistency Score = (rules_passed / total_rules)
```

#### Rule Category 3: Line Kolam Validation
```
IF predicted_class == "Line Kolam":
    RULE 1: (rotational_symmetry_90 >= 0.5) OR (rotational_symmetry_180 >= 0.5)
    RULE 2: (horizontal_symmetry >= 0.5) OR (vertical_symmetry >= 0.5)
    RULE 3: smoothness_metric >= 0.6
    RULE 4: compactness >= 0.3
    
    Consistency Score = (rules_passed / total_rules)
```

#### Rule Category 4: Freehand Kolam Validation
```
IF predicted_class == "Freehand Kolam":
    RULE 1: fractal_dimension >= 1.5
    RULE 2: pattern_fill >= 40%
    RULE 3: curvature_mean >= 1.5
    RULE 4: dot_count < 30 (should have fewer dots)
    
    Consistency Score = (rules_passed / total_rules)
```

### 4.2 Rule Conflict Resolution

**Scenario 1: CNN predicts Pulli, but rules fail (e.g., dot_count = 5)**
```python
if rule_consistency < threshold (e.g., 0.5):
    # Flag as low confidence
    final_confidence *= rule_consistency
    add_warning("Predicted Pulli Kolam but only 5 dots detected")
    
    # Optionally suggest alternative class
    if chukku_rules.score > 0.7:
        suggest_alternative("Consider Chukku Kolam")
```

**Scenario 2: CNN is uncertain (max_prob < 0.6), rules are decisive**
```python
if max_cnn_prob < 0.6:
    # Check which class has highest rule consistency
    best_rule_class = argmax(rule_scores)
    
    if rule_scores[best_rule_class] >= 0.75:
        # Override CNN with rule-based decision
        final_class = best_rule_class
        final_confidence = 0.5 * max_cnn_prob + 0.5 * rule_scores[best_rule_class]
```

**Scenario 3: CNN confident, rules agree**
```python
if max_cnn_prob >= 0.8 and rule_consistency >= 0.75:
    # High confidence prediction
    final_confidence = 0.7 * max_cnn_prob + 0.3 * rule_consistency
    status = "High Confidence"
```

### 4.3 Rule Engine Implementation

**Class:** `RuleBasedValidator`

**Methods:**
- `validate_pulli_kolam(features) -> (score, violations)`
- `validate_chukku_kolam(features) -> (score, violations)`
- `validate_line_kolam(features) -> (score, violations)`
- `validate_freehand_kolam(features) -> (score, violations)`
- `validate_prediction(predicted_class, features) -> (score, violations, suggestions)`

**Output:**
```python
{
    'rule_score': 0.75,  # 3 out of 4 rules passed
    'violations': ['dot_count too low: 15 (expected >= 20)'],
    'passed_rules': ['grid_regularity OK', 'dot_density OK', 'dot_spacing OK'],
    'suggestions': []
}
```

---

## Part 5: TRAINING PIPELINE

### 5.1 Data Loading

```python
class KolamFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, features_path, labels_path):
        self.features = np.load(features_path)  # (N, 2074)
        self.labels = load_labels(labels_path)  # (N,)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32),
               torch.tensor(self.labels[idx], dtype=torch.long)
```

### 5.2 Training Loop

```python
for epoch in range(num_epochs):
    # Training phase
    model.train()
    for batch_features, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
    
    # Validation phase
    model.eval()
    val_loss, val_accuracy = evaluate(model, val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model, 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break
```

### 5.3 Class Imbalance Handling

**Techniques:**
1. **Weighted Loss:** Use `torch.nn.CrossEntropyLoss(weight=class_weights)`
2. **Oversampling (Optional):** `WeightedRandomSampler` for minority classes
3. **Data Augmentation (Already done in Step 2)**

### 5.4 Regularization

- **Dropout:** 0.4 → 0.3 → 0.2 (decreasing through layers)
- **Weight Decay:** 0.0001 in optimizer
- **Early Stopping:** Patience of 15 epochs
- **Learning Rate Decay:** 0.5× every 10 epochs without improvement

---

## Part 6: EVALUATION METRICS

### 6.1 Standard Classification Metrics

**Accuracy:**
$$\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

**Per-Class Metrics:**
```
For each class c:
    Precision_c = TP_c / (TP_c + FP_c)
    Recall_c = TP_c / (TP_c + FN_c)
    F1_c = 2 * (Precision_c * Recall_c) / (Precision_c + Recall_c)
```

**Macro-Averaged F1:**
$$F1_{macro} = \frac{1}{C} \sum_{c=1}^{C} F1_c$$

### 6.2 Confusion Matrix

```
              Predicted
           Pulli  Chukku  Line  Freehand
Actual
Pulli       TP     FP     FP      FP
Chukku      FN     TP     FP      FP
Line        FN     FN     TP      FP
Freehand    FN     FN     FN      TP
```

### 6.3 Hybrid Evaluation

**Rule-Augmented Accuracy:**
```
Correct with rule agreement / Total predictions
```

**Average Rule Consistency:**
```
Mean(rule_scores for all correct predictions)
```

### 6.4 Confidence Calibration

**Expected Calibration Error (ECE):**
- Measures how well predicted probabilities match actual accuracy
- Lower is better (well-calibrated)

---

## Part 7: CONFIDENCE SCORING

### 7.1 Confidence Fusion Formula

$$\text{Confidence} = \alpha \cdot P_{CNN} + \beta \cdot S_{rule}$$

Where:
- $P_{CNN}$: Softmax probability for predicted class (0-1)
- $S_{rule}$: Rule consistency score (0-1)
- $\alpha = 0.7$: Weight for CNN (learned evidence)
- $\beta = 0.3$: Weight for rules (structural evidence)

**Interpretation:**
- **0.90-1.00:** Very High Confidence (CNN agrees with rules)
- **0.75-0.90:** High Confidence
- **0.60-0.75:** Medium Confidence
- **0.40-0.60:** Low Confidence (requires review)
- **0.00-0.40:** Very Low Confidence (likely incorrect)

### 7.2 Confidence Adjustment Rules

```python
# Base confidence
base_confidence = 0.7 * cnn_prob + 0.3 * rule_score

# Adjust based on agreement
if cnn_prob > 0.8 and rule_score > 0.75:
    # Both confident and agree
    final_confidence = min(base_confidence * 1.1, 1.0)
    
elif cnn_prob < 0.6 and rule_score < 0.5:
    # Both uncertain
    final_confidence = base_confidence * 0.8
    
elif abs(cnn_prob - rule_score) > 0.3:
    # Disagreement penalty
    final_confidence = base_confidence * 0.9
```

---

## Part 8: IMPLEMENTATION MODULES

### 8.1 File Structure

```
scripts/
├── 07_train_classifier.py              # Main training script
├── 08_evaluate_model.py                # Evaluation script
├── 09_inference.py                     # Inference script
│
└── classification/
    ├── __init__.py
    ├── classifier_model.py             # CNN classifier architecture
    ├── rule_validator.py               # Rule-based validation engine
    ├── confidence_fusion.py            # Confidence scoring
    ├── training_utils.py               # Training helpers
    └── evaluation_metrics.py           # Metrics computation
```

### 8.2 Key Classes

**KolamFeatureClassifier:**
- PyTorch `nn.Module`
- 4-layer MLP: 2074 → 512 → 256 → 128 → 4
- Dropout and ReLU activations

**RuleBasedValidator:**
- Validates predictions using 26 handcrafted features
- Returns rule consistency score and violations
- Per-class rule sets

**HybridPredictor:**
- Combines CNN predictions with rule validation
- Confidence fusion
- Final prediction with explanation

**TrainingManager:**
- Orchestrates training loop
- Handles checkpointing and logging
- Learning rate scheduling and early stopping

---

## Part 9: MODEL SAVING & LOADING

### 9.1 Checkpoint Format

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_accuracy': val_accuracy,
    'class_names': ['Pulli', 'Chukku', 'Line', 'Freehand'],
    'feature_dim': 2074,
    'num_classes': 4,
    'hyperparameters': {...}
}

torch.save(checkpoint, 'checkpoints/model_epoch_{}.pth')
```

### 9.2 Inference Loading

```python
def load_trained_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model = KolamFeatureClassifier(
        input_dim=checkpoint['feature_dim'],
        num_classes=checkpoint['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names']
```

---

## Part 10: OUTPUT ARTIFACTS

### 10.1 Model Files

```
kolam_dataset/05_trained_models/
├── best_model.pth                      # Best validation accuracy
├── final_model.pth                     # Last epoch
├── checkpoint_epoch_*.pth              # Periodic checkpoints
└── model_config.json                   # Hyperparameters
```

### 10.2 Training Logs

```
kolam_dataset/05_trained_models/logs/
├── training_log.txt                    # Epoch-wise metrics
├── loss_curve.png                      # Train/val loss plot
├── accuracy_curve.png                  # Train/val accuracy plot
└── tensorboard/                        # TensorBoard logs (optional)
```

### 10.3 Evaluation Reports

```
kolam_dataset/05_trained_models/evaluation/
├── classification_report.txt           # Precision, recall, F1
├── confusion_matrix.png                # Visual confusion matrix
├── confusion_matrix.csv                # Raw confusion matrix
├── per_class_metrics.json              # Detailed metrics
├── rule_consistency_report.txt         # Rule validation stats
└── misclassified_samples.json          # Analysis of errors
```

---

## Part 11: EXPECTED PERFORMANCE

### 11.1 Target Metrics

| Metric | Target | Reasoning |
|--------|--------|-----------|
| Overall Accuracy | > 85% | 4-class classification, moderate difficulty |
| Macro F1-Score | > 0.83 | Balanced performance across classes |
| Per-Class Accuracy | > 80% | All classes should perform well |
| Rule Consistency | > 75% | Most predictions should align with rules |
| Training Time | < 5 min | On GPU with 700 training samples |

### 11.2 Common Misclassifications

**Expected Confusion:**
- **Pulli ↔ Chukku:** Both have dots, differ in loop structure
- **Line ↔ Freehand:** Geometric vs artistic, boundary can blur
- **Chukku ↔ Freehand:** Complex loops vs free curves

**Rule Engine Helps:**
- Pulli misclassified as Chukku → Rules check dot_count vs loop_count
- Line misclassified as Freehand → Rules check symmetry

---

## Part 12: INTERPRETABILITY & EXPLAINABILITY

### 12.1 Prediction Explanation Format

```python
{
    'predicted_class': 'Pulli Kolam',
    'confidence': 87.5,
    'cnn_probability': 0.92,
    'rule_score': 0.75,
    'rule_validation': {
        'passed': ['dot_count OK (35 dots)', 'grid_regularity OK (0.68)'],
        'failed': ['dot_density low (4.2%, expected >= 5.0%)'],
        'warnings': []
    },
    'top_3_predictions': [
        ('Pulli Kolam', 0.92),
        ('Chukku Kolam', 0.05),
        ('Line Kolam', 0.02)
    ],
    'feature_contribution': {
        'dot_count': 'high',
        'grid_regularity': 'high',
        'loop_count': 'low'
    }
}
```

### 12.2 Visualization

- **Confidence Distribution:** Histogram of prediction confidences
- **Feature Importance:** Which handcrafted features influence each class
- **Attention Maps (Optional):** If using end-to-end CNN

---

## Part 13: WORKFLOW SUMMARY

```
1. Load Features
   └─ train_features.npy (700, 2074)
   └─ val_features.npy (150, 2074)
   └─ test_features.npy (150, 2074)

2. Train CNN Classifier
   └─ KolamFeatureClassifier (MLP)
   └─ 100 epochs with early stopping
   └─ Save checkpoints

3. Evaluate on Validation Set
   └─ Compute accuracy, F1, confusion matrix
   └─ Generate plots

4. Rule Validation
   └─ For each prediction, check rules
   └─ Compute rule consistency scores

5. Hybrid Confidence Scoring
   └─ Fuse CNN + rule scores
   └─ Generate final predictions

6. Test Set Evaluation
   └─ Final metrics on unseen data
   └─ Generate evaluation report

7. Save Model & Artifacts
   └─ best_model.pth
   └─ evaluation reports
   └─ training logs
```

---

## Part 14: NEXT STEPS (Beyond Step 4)

- **Step 5:** Web UI for real-time classification
- **Step 6:** Mobile app deployment
- **Step 7:** Active learning for continuous improvement
- **Step 8:** Explainable AI dashboard

---

## References

- Multi-class Classification: Bishop, "Pattern Recognition and Machine Learning" (2006)
- Rule-Based Systems: Russell & Norvig, "Artificial Intelligence: A Modern Approach" (2020)
- Confidence Calibration: Guo et al., "On Calibration of Modern Neural Networks" (2017)
- Hybrid AI: Marcus, "The Next Decade in AI: Four Steps Towards Robust AI" (2020)

---

**Status:** ✅ Design Complete  
**Next:** Implementation of classification modules  
**Date:** December 28, 2025
