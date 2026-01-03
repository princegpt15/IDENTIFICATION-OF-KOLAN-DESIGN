# Step 5: Category Mapping - Quick Reference

Fast access guide for common tasks and APIs.

## Quick Start (3 lines)

```python
from scripts.category_mapping import CategoryMapper
mapper = CategoryMapper("kolam_knowledge_base")
result = mapper.map_category(cnn_output, features, rule_scores)
```

## Common Tasks

### 1. Basic Classification

```python
result = mapper.map_category(
    cnn_output={'class_id': 0, 'probabilities': np.array([0.85, 0.10, 0.03, 0.02])},
    features=features_2074dim,
    rule_scores={'pulli_kolam': 0.89, 'chukku_kolam': 0.22, ...}
)
print(f"{result['category']} ({result['confidence']:.0%})")
```

### 2. With Detailed Explanation

```python
result = mapper.map_category(..., explanation_level='detailed')
mapper.print_mapping_summary(result, detailed=True)
```

### 3. With Design Matching

```python
result = mapper.map_category(..., include_similarity=True)
print(f"Design: {result['design']} ({result['similarity_score']:.0%})")
```

### 4. Batch Processing

```python
results = mapper.map_batch(all_cnn_outputs, all_features, all_rule_scores)
avg_confidence = np.mean([r['confidence'] for r in results])
```

### 5. Handle Uncertainty

```python
if result['conflict_resolution']['status'] == 'ambiguous':
    alternatives = result['conflict_resolution']['alternatives']
    for alt in alternatives:
        print(f"  {alt['category']}: {alt['score']:.0%}")
```

## Key APIs

### CategoryMapper

```python
CategoryMapper(
    kb_path="kolam_knowledge_base",
    use_similarity=True,
    verbose=False
)

# Main method
map_category(
    cnn_output: dict,
    features: np.ndarray,
    rule_scores: dict,
    include_similarity: bool = False,
    explanation_level: str = 'summary'  # 'summary'/'basic'/'detailed'
) -> dict

# Batch processing
map_batch(
    predictions: list,
    features: np.ndarray,
    rules: list
) -> list
```

### KnowledgeBase

```python
kb = KnowledgeBase("kolam_knowledge_base")

kb.get_category_info("pulli_kolam")  # Category definition
kb.get_prototypes("pulli_kolam")     # Prototypes list
kb.get_constraints("pulli_kolam")    # Validation rules
kb.add_prototype(...)                 # Add new prototype
```

### SimilarityScorer

```python
scorer = SimilarityScorer(kb)

similarity, breakdown = scorer.compute_similarity(
    query_features,
    prototype_features,
    category='pulli_kolam',
    method='combined'  # 'cosine'/'euclidean'/'weighted'/'combined'
)

matches = scorer.find_best_matches(features, 'pulli_kolam', k=3)
```

### ConflictResolver

```python
resolver = ConflictResolver()

result = resolver.resolve(cnn_output, rule_scores, features)
# Returns: decision, confidence, conflict_type, reasoning, status
```

### CategoryExplainer

```python
explainer = CategoryExplainer(kb)

explanation = explainer.explain_mapping(
    mapping_result,
    cnn_output,
    rule_scores,
    similarity_results,
    level='detailed'  # 'summary'/'basic'/'detailed'
)

formatted = explainer.format_for_display(explanation, level='detailed')
```

## Configuration

### Thresholds (category_mapper.py)

```python
CNN_THRESHOLD = 0.60        # Minimum CNN confidence
RULE_THRESHOLD = 0.50       # Minimum rule confidence
SIMILARITY_THRESHOLD = 0.70 # Minimum similarity
```

### Conflict Resolver (conflict_resolver.py)

```python
HIGH_CNN_CONFIDENCE = 0.75   # CNN highly confident
MEDIUM_CNN_CONFIDENCE = 0.60 # CNN moderately confident
HIGH_RULE_CONFIDENCE = 0.70  # Rules highly confident
MEDIUM_RULE_CONFIDENCE = 0.50 # Rules moderately confident
```

### Similarity Weights (similarity_scorer.py)

```python
STRUCTURAL_WEIGHT = 0.40     # Euclidean
VISUAL_WEIGHT = 0.40         # Cosine
CATEGORY_WEIGHT = 0.20       # Weighted
```

## Input/Output Format

### Input

```python
cnn_output = {
    'class_id': int,              # 0-3
    'probabilities': np.ndarray   # Shape: (4,)
}

features = np.ndarray             # Shape: (2074,)

rule_scores = {
    'pulli_kolam': float,         # 0-1
    'chukku_kolam': float,
    'line_kolam': float,
    'freehand_kolam': float
}
```

### Output

```python
result = {
    'category': str,              # Final category
    'confidence': float,          # 0-1
    'design': str,                # Matched design (if similarity)
    'similarity_score': float,    # 0-1 (if similarity)
    'cnn_prediction': {...},
    'rule_validation': {...},
    'conflict_resolution': {...},
    'explanation': {...},
    'metadata': {...}
}
```

## Troubleshooting

### Knowledge Base Not Found

```
Error: Knowledge base directory not found
Fix: Ensure 'kolam_knowledge_base/' in working directory
```

### Feature Dimension Mismatch

```
Error: Expected 2074 features, got X
Fix: Use features from Step 3 (26 handcrafted + 2048 CNN)
```

### Low Confidence

```
Warning: Ambiguous classification
Fix: Use detailed explanation to diagnose. Check alternatives.
```

### Enable Debug Mode

```python
mapper = CategoryMapper("kolam_knowledge_base", verbose=True)
```

## File Locations

```
scripts/category_mapping/
├── __init__.py
├── knowledge_base.py
├── similarity_scorer.py
├── conflict_resolver.py
├── explainer.py
└── category_mapper.py

kolam_knowledge_base/
├── categories.json
├── constraints.json
├── metadata.json
└── prototypes/
    ├── pulli_kolam/
    │   └── grid_5x5.json
    ├── chukku_kolam/
    │   └── serpent_pattern.json
    ├── line_kolam/
    │   └── eightfold_mandala.json
    └── freehand_kolam/
        └── peacock_design.json

scripts/
└── 10_test_category_mapping.py

Documentation:
├── STEP5_README.md
├── STEP5_DELIVERABLES.md
├── QUICK_REFERENCE_STEP5.md
└── STEP5_EXECUTION_SUMMARY.md
```

## Testing

```bash
# Run tests
python scripts/10_test_category_mapping.py

# Verbose output
python scripts/10_test_category_mapping.py --verbose

# All tests
python scripts/10_test_category_mapping.py --test-all
```

## Categories

- **Pulli Kolam** (புள்ளி கோலம்) - Dot-grid based
- **Chukku Kolam** (சுக்கு கோலம்) - Continuous loop
- **Line Kolam** (கோடு கோலம்) - Symmetry-based
- **Freehand Kolam** (வரைதல் கோலம்) - Artistic

## Performance

- Single sample: ~5-10 ms
- With similarity: ~15-25 ms
- Batch (100): ~1-2 seconds
- Batch (1000): ~10-20 seconds

---

**See also**: [STEP5_README.md](STEP5_README.md) for detailed documentation