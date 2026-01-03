# Step 5: Category Mapping System

Complete intelligent category mapping for Kolam pattern classification with cultural context and explainability.

## Overview

The Category Mapping System bridges the semantic gap between raw model outputs (class indices, probabilities) and meaningful Kolam categories. It provides:

- **Semantic Understanding**: Maps numeric predictions to culturally correct Kolam categories
- **Similarity Matching**: Finds specific designs that match input patterns
- **Conflict Resolution**: Intelligently resolves disagreements between CNN and rules
- **Explainability**: Generates human-readable explanations at multiple levels
- **Cultural Correctness**: Ensures taxonomic accuracy using domain knowledge

## Architecture

### Three-Stage Pipeline

```
Stage 1: Primary Mapping
├── Input: CNN predictions (class_id, probabilities)
├── Process: Map to base category
└── Output: Initial category assignment

Stage 2: Rule Validation & Conflict Resolution
├── Input: CNN prediction + Rule scores
├── Process: Analyze agreement/disagreement
├── Strategies: 
│   ├── Agreement → Boost confidence
│   ├── CNN confident, Rules reject → Analyze alternatives
│   ├── CNN uncertain, Rules clear → Trust rules
│   └── Both uncertain → Return top-3, flag ambiguous
└── Output: Final category + confidence + conflict type

Stage 3: Similarity Matching (Optional)
├── Input: Features (2074-dim) + Category
├── Process: Compare against prototype designs
├── Metrics: Cosine + Euclidean + Weighted
└── Output: Specific design match + similarity score
```

### Component Architecture

```
CategoryMapper (Main Orchestrator)
├── KnowledgeBase (JSON loader)
│   ├── categories.json (4 Kolam categories)
│   ├── constraints.json (validation rules)
│   ├── metadata.json (cultural context)
│   └── prototypes/*.json (design templates)
├── SimilarityScorer (Multi-metric matching)
│   ├── Cosine similarity (CNN features)
│   ├── Euclidean similarity (handcrafted)
│   └── Weighted similarity (category-specific)
├── ConflictResolver (Decision logic)
│   ├── Agreement confirmation
│   ├── CNN vs Rules resolution
│   └── Uncertainty handling
└── CategoryExplainer (Human-readable output)
    ├── Summary (1 sentence)
    ├── Basic (paragraph)
    └── Detailed (comprehensive)
```

## Components

### 1. KnowledgeBase

Manages JSON-based knowledge about Kolam categories.

**Files:**
- `kolam_knowledge_base/categories.json` - Category definitions
- `kolam_knowledge_base/constraints.json` - Validation rules
- `kolam_knowledge_base/metadata.json` - Cultural/mathematical info
- `kolam_knowledge_base/prototypes/{category}/*.json` - Design templates

**Key Features:**
- Load all KB files on initialization
- Map class indices (0-3) to category names
- Access category information, prototypes, constraints
- Add new prototypes dynamically without code changes
- Export KB statistics and summaries

**Usage:**
```python
from scripts.category_mapping import KnowledgeBase

kb = KnowledgeBase("kolam_knowledge_base")

# Get category info
info = kb.get_category_info("pulli_kolam")
print(info['tamil_name'])  # "புள்ளி கோலம்"

# Get prototypes
prototypes = kb.get_prototypes("pulli_kolam")

# Get constraints
rules = kb.get_constraints("pulli_kolam")

# Statistics
stats = kb.get_category_statistics()
print(f"Total categories: {stats['total_categories']}")
```

### 2. SimilarityScorer

Multi-metric similarity computation for pattern matching.

**Similarity Methods:**

1. **Cosine Similarity** - For CNN features (2048-dim)
   - Direction-based comparison
   - Insensitive to magnitude
   - Best for deep features
   - Formula: `(a·b) / (||a|| × ||b||)`

2. **Euclidean Similarity** - For handcrafted features (26-dim)
   - Distance-based comparison
   - Sensitive to absolute values
   - Good for geometric features
   - Formula: `1 / (1 + normalized_distance)`

3. **Weighted Similarity** - Category-specific
   - Uses importance weights per category
   - Focuses on discriminative features
   - Customizable per category

4. **Combined Similarity** - Best of all worlds
   - `0.40 × structural + 0.40 × visual + 0.20 × category_specific`
   - Balanced approach

**Usage:**
```python
from scripts.category_mapping import SimilarityScorer

scorer = SimilarityScorer(kb)

# Compute similarity
similarity, breakdown = scorer.compute_similarity(
    query_features,
    prototype_features,
    category='pulli_kolam',
    method='combined'
)

# Find best matches
matches = scorer.find_best_matches(
    features,
    category='pulli_kolam',
    k=3
)

for match in matches:
    print(f"{match['prototype']}: {match['similarity']:.3f}")
```

### 3. ConflictResolver

Resolves disagreements between CNN and rule validation.

**Conflict Types:**

1. **Agreement** - Both CNN and rules agree
   - Boost confidence: `0.7×CNN + 0.3×Rules`
   - High confidence output

2. **CNN Confident, Rules Reject**
   - CNN says A with high confidence (≥0.75)
   - Rules strongly favor B (≥0.70)
   - Strategy: Analyze rule alternative, use similarity for tiebreak

3. **CNN Uncertain, Rules Clear**
   - CNN confidence < 0.60
   - Rules have clear favorite (≥0.70)
   - Strategy: Trust rules

4. **Both Uncertain**
   - CNN confidence < 0.60
   - All rule scores < 0.60
   - Strategy: Return top-3 candidates, flag as ambiguous

**Usage:**
```python
from scripts.category_mapping import ConflictResolver

resolver = ConflictResolver(similarity_scorer=scorer)

result = resolver.resolve(cnn_output, rule_scores, features)

print(f"Decision: {result['decision']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Conflict Type: {result['conflict_type']}")
print(f"Reasoning: {result['reasoning']}")
```

### 4. CategoryExplainer

Generates human-readable explanations for mapping decisions.

**Explanation Levels:**

1. **Summary** - One sentence
   ```
   "Classified as Pulli Kolam with 87% confidence based on strong agreement between CNN prediction and structural rules."
   ```

2. **Basic** - Short paragraph
   ```
   Summary + key reasoning points + confidence assessment
   ```

3. **Detailed** - Comprehensive breakdown
   - CNN prediction analysis (top-3)
   - Rule validation scores
   - Similarity matching results
   - Key discriminative features
   - Decision reasoning
   - Alternative categories considered
   - Confidence interpretation

**Usage:**
```python
from scripts.category_mapping import CategoryExplainer

explainer = CategoryExplainer(kb)

explanation = explainer.explain_mapping(
    mapping_result,
    cnn_output,
    rule_scores,
    similarity_results,
    level='detailed'
)

# Console formatted output
formatted = explainer.format_for_display(explanation, level='detailed')
print(formatted)
```

### 5. CategoryMapper

Main orchestrator for the complete 3-stage pipeline.

**Key Methods:**

- `map_category()` - Map single sample
- `map_batch()` - Map multiple samples
- `print_mapping_summary()` - Human-readable output

**Usage:**
```python
from scripts.category_mapping import CategoryMapper

# Initialize
mapper = CategoryMapper(
    kb_path="kolam_knowledge_base",
    use_similarity=True,
    verbose=True
)

# Map category
result = mapper.map_category(
    cnn_output={
        'class_id': 0,
        'probabilities': np.array([0.85, 0.10, 0.03, 0.02])
    },
    features=features_2074dim,
    rule_scores={
        'pulli_kolam': 0.89,
        'chukku_kolam': 0.22,
        'line_kolam': 0.28,
        'freehand_kolam': 0.15
    },
    include_similarity=True,
    explanation_level='detailed'
)

# Print summary
mapper.print_mapping_summary(result, detailed=True)

# Access results
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Design: {result['design']}")
print(f"Explanation: {result['explanation']['summary']}")
```

## Knowledge Base

### Categories

**4 Kolam Categories:**

1. **Pulli Kolam** (புள்ளி கோலம்) - Dot-Grid Based
   - Characteristics: Regular grid of dots, lines connect dots
   - Difficulty: Beginner to Intermediate
   - Cultural: Daily household kolams, mathematical beauty

2. **Chukku Kolam** (சுக்கு கோலம்) - Continuous Loop
   - Characteristics: Single continuous path, high connectivity
   - Difficulty: Intermediate to Advanced
   - Cultural: Infinity symbolism, spiritual significance

3. **Line Kolam** (கோடு கோலம்) - Symmetry-Based
   - Characteristics: High symmetry (rotational/reflective), no dots
   - Difficulty: Intermediate to Advanced
   - Cultural: Mandala patterns, cosmic harmony

4. **Freehand Kolam** (வரைதல் கோலம்) - Artistic
   - Characteristics: Representational figures, high complexity
   - Difficulty: Advanced to Expert
   - Cultural: Wedding/festival occasions, artistic expression

### Structural Constraints

Each category has validation rules:

**Pulli Kolam Rules:**
- `dot_count ≥ 20` (weight: 0.8)
- `grid_regularity ≥ 0.40` (weight: 1.0)
- `dot_density ≥ 5%` (weight: 0.6)
- `dot_spacing_std < 30` (weight: 0.7)

**Chukku Kolam Rules:**
- `loop_count ≥ 3` (weight: 0.8)
- `connectivity ≥ 0.60` (weight: 1.0)
- `curve_length ≥ 500` (weight: 0.7)
- `edge_continuity ≥ 0.50` (weight: 0.8)

**Line Kolam Rules:**
- `symmetry (reflective OR rotational) ≥ 0.50` (weight: 0.9 each)
- `smoothness ≥ 0.60` (weight: 0.7)
- `compactness ≥ 0.30` (weight: 0.6)

**Freehand Kolam Rules:**
- `fractal_dimension ≥ 1.5` (weight: 0.8)
- `pattern_fill ≥ 40%` (weight: 0.7)
- `curvature ≥ 1.5` (weight: 0.6)
- `dot_count < 30` (weight: 0.5)

### Prototypes

**Current Prototypes (1 per category):**

1. **Pulli**: 5×5 Grid Pattern (25 dots, beginner-intermediate)
2. **Chukku**: Serpent Pattern (12 loops, continuous, advanced)
3. **Line**: Eight-fold Mandala (8-fold symmetry, intermediate-advanced)
4. **Freehand**: Peacock Design (artistic figure, expert)

**Adding New Prototypes:**

```python
# Extract features for new design
new_features = extract_features(image)  # From Step 3

# Add to knowledge base
kb.add_prototype(
    category='pulli_kolam',
    design_name='7x7_Grid_Complex',
    features=new_features,
    metadata={
        'description': '7×7 grid with diagonal connections',
        'difficulty': 'intermediate',
        'drawing_time': '8-12 minutes',
        'cultural_notes': 'Festival kolam pattern'
    },
    similarity_threshold=0.75
)
```

## Integration with Previous Steps

### Input Requirements

**From Step 4 (Classification):**
```python
cnn_output = {
    'class_id': int,           # 0-3
    'probabilities': np.array  # [p0, p1, p2, p3]
}

rule_scores = {
    'pulli_kolam': float,      # 0-1
    'chukku_kolam': float,
    'line_kolam': float,
    'freehand_kolam': float
}
```

**From Step 3 (Feature Extraction):**
```python
features = np.array(2074)  # [26 handcrafted + 2048 CNN features]
```

### Output Format

```python
result = {
    'category': str,                    # Final category name
    'confidence': float,                # 0-1
    'design': str,                      # Matched design (if similarity used)
    'similarity_score': float,          # 0-1 (if similarity used)
    
    'cnn_prediction': {
        'class_id': int,
        'category': str,
        'confidence': float,
        'top_k': list
    },
    
    'rule_validation': {
        'scores': dict,
        'best_category': str,
        'confidence': float
    },
    
    'conflict_resolution': {
        'conflict_type': str,
        'decision': str,
        'confidence': float,
        'reasoning': str,
        'status': str
    },
    
    'explanation': {
        'summary': str,
        'reasoning': str,
        'cnn_analysis': str,
        'rule_analysis': str,
        'similarity_analysis': str,
        'key_features': list,
        'alternatives': list,
        'confidence_interpretation': str
    },
    
    'metadata': {
        'timestamp': str,
        'processing_time': float,
        'explanation_level': str
    }
}
```

## Usage Examples

### Example 1: Basic Classification

```python
from scripts.category_mapping import CategoryMapper
import numpy as np

# Initialize mapper
mapper = CategoryMapper("kolam_knowledge_base")

# Prepare inputs (from Steps 3-4)
cnn_output = {
    'class_id': 0,
    'probabilities': np.array([0.85, 0.10, 0.03, 0.02])
}

features = np.load('features.npy')  # From Step 3

rule_scores = {
    'pulli_kolam': 0.89,
    'chukku_kolam': 0.22,
    'line_kolam': 0.28,
    'freehand_kolam': 0.15
}

# Map category
result = mapper.map_category(
    cnn_output,
    features,
    rule_scores,
    explanation_level='summary'
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Explanation: {result['explanation']['summary']}")
```

### Example 2: Batch Processing

```python
# Load test data
test_predictions = np.load('test_predictions.npy')
test_features = np.load('test_features.npy')
test_rule_scores = np.load('test_rule_scores.npy')

# Process batch
results = mapper.map_batch(
    test_predictions,
    test_features,
    test_rule_scores
)

# Analyze results
categories = [r['category'] for r in results]
confidences = [r['confidence'] for r in results]

print(f"Average confidence: {np.mean(confidences):.2%}")
print(f"Category distribution: {Counter(categories)}")
```

### Example 3: With Similarity Matching

```python
result = mapper.map_category(
    cnn_output,
    features,
    rule_scores,
    include_similarity=True,  # Enable design matching
    explanation_level='detailed'
)

print(f"Category: {result['category']}")
print(f"Design: {result['design']}")
print(f"Similarity: {result['similarity_score']:.2%}")

# Print full explanation
mapper.print_mapping_summary(result, detailed=True)
```

### Example 4: Handling Conflicts

```python
# CNN says Pulli, but rules favor Chukku
cnn_output = {
    'class_id': 0,
    'probabilities': np.array([0.78, 0.15, 0.05, 0.02])
}

rule_scores = {
    'pulli_kolam': 0.35,
    'chukku_kolam': 0.82,
    'line_kolam': 0.25,
    'freehand_kolam': 0.15
}

result = mapper.map_category(cnn_output, features, rule_scores)

conflict = result['conflict_resolution']
print(f"Conflict Type: {conflict['conflict_type']}")
print(f"CNN said: {conflict.get('cnn_prediction')}")
print(f"Rules favor: {conflict.get('rule_alternative')}")
print(f"Final Decision: {conflict['decision']}")
print(f"Reasoning: {conflict['reasoning']}")
```

### Example 5: Uncertain Classification

```python
# Both CNN and rules uncertain
cnn_output = {
    'class_id': 0,
    'probabilities': np.array([0.35, 0.30, 0.20, 0.15])
}

rule_scores = {
    'pulli_kolam': 0.45,
    'chukku_kolam': 0.50,
    'line_kolam': 0.40,
    'freehand_kolam': 0.35
}

result = mapper.map_category(cnn_output, features, rule_scores)

if result['conflict_resolution']['status'] == 'ambiguous':
    print("⚠ Ambiguous classification")
    print("Top candidates:")
    for alt in result['explanation']['alternatives']:
        print(f"  - {alt['category']}: {alt['score']:.2%}")
```

## Testing

### Run Test Suite

```bash
# Basic tests
python scripts/10_test_category_mapping.py

# Verbose output
python scripts/10_test_category_mapping.py --verbose

# All comprehensive tests
python scripts/10_test_category_mapping.py --test-all
```

### Test Coverage

The test suite validates:

1. **Knowledge Base Loading** - JSON files load correctly
2. **Similarity Scoring** - All metrics compute properly
3. **Conflict Resolution** - All scenarios handled
4. **Category Mapping** - Complete pipeline works
5. **Explanation Generation** - All levels generate correctly
6. **Full Pipeline** - End-to-end integration

### Expected Output

```
======================================================================
KOLAM CATEGORY MAPPING SYSTEM - TEST SUITE
======================================================================

TEST 1: Knowledge Base Loading
✓ Knowledge base loaded successfully
Total categories: 4
Total prototypes: 4

TEST 2: Similarity Scoring
✓ Similarity scorer test passed

TEST 3: Conflict Resolution
✓ Conflict resolver test passed

TEST 4: Complete Category Mapping
✓ Category mapper test passed

TEST 5: Explanation Generation
✓ Explainer test passed

TEST 6: Full Pipeline with Detailed Explanation
✓ Full pipeline test passed

======================================================================
TEST SUMMARY
======================================================================
  knowledge_base................................. ✓ PASS
  similarity_scorer.............................. ✓ PASS
  conflict_resolver.............................. ✓ PASS
  category_mapper................................ ✓ PASS
  explainer...................................... ✓ PASS
  full_pipeline.................................. ✓ PASS

Total: 6/6 tests passed

🎉 ALL TESTS PASSED! Category mapping system is working correctly.
```

## Extending the System

### Adding New Categories

1. Update `categories.json`:
```json
{
  "categories": {
    "new_category": {
      "id": 4,
      "name": "new_category",
      "tamil_name": "புதிய வகை",
      "description": "...",
      "characteristics": [...],
      "cultural_significance": "...",
      "difficulty": "intermediate",
      "structural_requirements": [...]
    }
  }
}
```

2. Add constraints in `constraints.json`
3. Create prototype JSON file
4. Update class mapping in `knowledge_base.py` if needed

### Adding Prototypes

```python
# Programmatically
kb.add_prototype(
    category='pulli_kolam',
    design_name='New_Design',
    features=feature_vector,
    metadata={...}
)

# Or manually create JSON file:
# kolam_knowledge_base/prototypes/pulli_kolam/new_design.json
```

### Customizing Similarity Weights

Edit category-specific weights in `similarity_scorer.py`:

```python
category_weights = {
    'pulli_kolam': {
        'dot_count': 0.8,
        'grid_regularity': 1.0,
        'symmetry': 0.6,
        ...
    }
}
```

### Adjusting Confidence Thresholds

Modify thresholds in `CategoryMapper`:

```python
self.CNN_THRESHOLD = 0.60        # Minimum CNN confidence
self.RULE_THRESHOLD = 0.50       # Minimum rule confidence
self.SIMILARITY_THRESHOLD = 0.70 # Minimum similarity
```

Or in `ConflictResolver`:

```python
self.HIGH_CNN_CONFIDENCE = 0.75
self.HIGH_RULE_CONFIDENCE = 0.70
```

## Performance Characteristics

### Computational Complexity

- **Knowledge Base Loading**: O(1) - One-time initialization
- **Primary Mapping**: O(1) - Direct lookup
- **Rule Validation**: O(K) where K = number of categories (4)
- **Similarity Scoring**: O(P×D) where P = prototypes, D = dimensions
  - Cosine: O(512) for CNN features
  - Euclidean: O(26) for handcrafted
  - Combined: O(538)
- **Conflict Resolution**: O(K) - Compare category scores
- **Explanation Generation**: O(1) - String formatting
- **Overall**: O(P×D + K) ≈ O(P×D) dominated by similarity

### Memory Requirements

- Knowledge Base: ~100 KB (4 categories + 4 prototypes)
- Per Sample:
  - Input features: 2074 × 4 bytes = 8.3 KB
  - Working memory: ~20 KB
  - Output result: ~5 KB
- Total per sample: ~35 KB

### Speed Benchmarks

(Approximate, on standard CPU)

- Map single category: ~5-10 ms
- With similarity matching: ~15-25 ms
- Batch (100 samples): ~1-2 seconds
- Batch (1000 samples): ~10-20 seconds

## Troubleshooting

### Common Issues

**1. Knowledge Base Not Found**
```
Error: Knowledge base directory not found
Solution: Ensure 'kolam_knowledge_base/' exists in working directory
```

**2. Missing JSON Files**
```
Error: categories.json not found
Solution: Verify all KB files present (categories, constraints, metadata, prototypes)
```

**3. Feature Dimension Mismatch**
```
Error: Expected 2074 features, got X
Solution: Ensure features from Step 3 (26 handcrafted + 2048 CNN)
```

**4. Rule Scores Format**
```
Error: Invalid rule_scores format
Solution: Ensure dict with all 4 categories: {'pulli_kolam': float, ...}
```

**5. Low Confidence Results**
```
Warning: Ambiguous classification
Solution: Check if input pattern is unusual/hybrid. Use detailed explanation to diagnose.
```

### Debug Mode

Enable verbose output:

```python
mapper = CategoryMapper("kolam_knowledge_base", verbose=True)
```

This prints:
- Stage 1: CNN prediction
- Stage 2: Conflict resolution details
- Stage 3: Similarity matching progress

### Logging

Add logging for production use:

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('category_mapping')

# Mapper will log to this logger
mapper = CategoryMapper("kolam_knowledge_base", verbose=True)
```

## Future Enhancements

### Planned Features

1. **Subcategories**: Finer-grained classification within categories
2. **Regional Variations**: Tamil Nadu vs Karnataka vs Kerala styles
3. **Temporal Evolution**: Ancient vs Modern styles
4. **Uncertainty Quantification**: Bayesian confidence intervals
5. **Active Learning**: Request human labels for ambiguous cases
6. **Multi-label Classification**: Patterns with multiple influences
7. **Similarity Clustering**: Discover new design families

### Research Directions

1. **Transfer Learning**: Fine-tune for regional variants
2. **Few-Shot Learning**: Classify with minimal examples
3. **Generative Models**: Generate new Kolam designs
4. **Cultural Embedding**: Learn cultural significance vectors
5. **Explainable AI**: Deeper interpretability methods

## References

### Academic

- Transforming Tradition: A Method for Maintaining System
- Kolam Designs and Their Mathematical Foundations
- Cultural Heritage Preservation through AI

### Cultural

- Traditional Kolam Practitioners (Chennai, Tamil Nadu)
- Dakshinachitra Cultural Centre
- Tamil Heritage Foundation

### Technical

- scikit-learn Documentation: Similarity Metrics
- NumPy User Guide: Array Operations
- JSON Schema Specification

## License

Part of Kolam Pattern Classification System
Academic Project - 2025

## Contact

For questions or contributions:
- Project Repository: [link]
- Documentation: [link]
- Issues: [link]

---

**Next**: See [STEP5_DELIVERABLES.md](STEP5_DELIVERABLES.md) for complete implementation inventory.
