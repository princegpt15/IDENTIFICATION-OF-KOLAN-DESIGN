# STEP 5: CATEGORY MAPPING SYSTEM - TECHNICAL DESIGN

**Project:** Kolam Pattern Classification  
**Step:** 5 - Category Mapping  
**Date:** December 28, 2025  
**Status:** Design Complete

---

## TABLE OF CONTENTS

1. [Overview](#1-overview)
2. [Category Mapping Strategy](#2-category-mapping-strategy)
3. [Feature-Based Similarity Matching](#3-feature-based-similarity-matching)
4. [Rule Confirmation & Conflict Resolution](#4-rule-confirmation--conflict-resolution)
5. [Knowledge Base Design](#5-knowledge-base-design)
6. [Implementation Architecture](#6-implementation-architecture)
7. [Explainability System](#7-explainability-system)
8. [Validation & Testing](#8-validation--testing)
9. [Integration with Steps 1-4](#9-integration-with-steps-1-4)
10. [Output Artifacts](#10-output-artifacts)
11. [Performance Considerations](#11-performance-considerations)
12. [Future Extensions](#12-future-extensions)

---

## 1. OVERVIEW

### 1.1 Purpose

Step 5 creates an intelligent **Category Mapping System** that transforms raw CNN predictions into culturally correct, interpretable Kolam classifications. It bridges the gap between:

- **Machine Output:** Class indices (0,1,2,3), probability vectors
- **Human Understanding:** "Pulli Kolam with 5×5 grid", "Chukku Kolam - Serpent design"

### 1.2 Key Challenges

1. **Semantic Gap:** Raw predictions lack cultural context
2. **Ambiguity:** Visually similar patterns may belong to different categories
3. **Conflict Resolution:** CNN vs. rule-based validation disagreements
4. **Explainability:** Users need to understand WHY a mapping was made
5. **Extensibility:** New Kolam designs should be easily added

### 1.3 Solution Approach

**Three-Stage Mapping Pipeline:**

```
Stage 1: Primary Mapping
CNN Output (class_id, probabilities) → Base Category (Pulli/Chukku/Line/Freehand)

Stage 2: Similarity Refinement
Features (2074-dim) + Knowledge Base → Specific Design Match (if applicable)

Stage 3: Rule Validation & Explanation
Rules + Conflicts → Final Category + Confidence + Reasoning
```

### 1.4 Design Principles

- ✅ **Culturally Correct:** Respects traditional Kolam taxonomy
- ✅ **Transparent:** Every decision is explainable
- ✅ **Modular:** Easy to extend with new categories
- ✅ **Integrated:** Works seamlessly with Steps 1-4
- ✅ **Lightweight:** No external APIs, runs locally

---

## 2. CATEGORY MAPPING STRATEGY

### 2.1 Taxonomy Hierarchy

```
Level 0: Raw Model Output
├─ Class 0 (probability: 0.85)
├─ Class 1 (probability: 0.10)
├─ Class 2 (probability: 0.03)
└─ Class 3 (probability: 0.02)

        ↓ [Primary Mapping]

Level 1: Base Category
├─ Pulli Kolam (Dot-based grid patterns)
├─ Chukku Kolam (Loop-based continuous patterns)
├─ Line Kolam (Symmetric line patterns)
└─ Freehand Kolam (Complex freehand designs)

        ↓ [Similarity Matching]

Level 2: Specific Design (Optional)
├─ Pulli Kolam
│   ├─ 3×3 Grid Pattern
│   ├─ 5×5 Grid Pattern
│   ├─ Diamond Grid Pattern
│   └─ Circular Grid Pattern
│
├─ Chukku Kolam
│   ├─ Serpent Pattern
│   ├─ Spiral Pattern
│   ├─ Meander Pattern
│   └─ Complex Loop Pattern
│
├─ Line Kolam
│   ├─ Four-fold Symmetry
│   ├─ Eight-fold Symmetry
│   ├─ Mandala Pattern
│   └─ Star Pattern
│
└─ Freehand Kolam
    ├─ Floral Design
    ├─ Peacock Design
    ├─ Abstract Design
    └─ Goddess Figure
```

### 2.2 Primary Mapping Logic

**Input:** CNN prediction (class_id, probabilities), rule validation scores

**Mapping Table:**

| Class ID | Base Category    | Probability Threshold | Rule Minimum Score |
|----------|------------------|----------------------|-------------------|
| 0        | Pulli Kolam      | 0.60                 | 0.50              |
| 1        | Chukku Kolam     | 0.60                 | 0.50              |
| 2        | Line Kolam       | 0.60                 | 0.50              |
| 3        | Freehand Kolam   | 0.60                 | 0.50              |

**Decision Logic:**

```python
def primary_mapping(class_id, probabilities, rule_scores):
    """
    Map CNN prediction to base category.
    
    Returns:
        category: str - Base category name
        confidence: float - Mapping confidence [0-1]
        status: str - 'confident', 'uncertain', 'conflict'
    """
    
    # 1. Check probability threshold
    if probabilities[class_id] < 0.60:
        status = 'uncertain'
    
    # 2. Check rule validation
    if rule_scores[class_id] < 0.50:
        status = 'conflict'
    
    # 3. Map to category
    category = CATEGORY_MAP[class_id]
    
    # 4. Compute final confidence
    confidence = 0.7 * probabilities[class_id] + 0.3 * rule_scores[class_id]
    
    return category, confidence, status
```

### 2.3 Edge Case Handling

**Case 1: Low Confidence Prediction**
- Probability < 0.60
- **Action:** Return top-3 candidates with confidence scores
- **Explanation:** "Multiple categories possible, showing top matches"

**Case 2: Rule Conflict**
- CNN says "Pulli", rules say "Not Pulli"
- **Action:** Analyze feature similarity to prototypes
- **Explanation:** "CNN prediction conflicts with structural rules, using similarity matching"

**Case 3: Near-Tie Predictions**
- Top-2 probabilities within 0.10
- **Action:** Use rule scores as tie-breaker
- **Explanation:** "Close match between X and Y, rules favor X"

**Case 4: All Low Probabilities**
- Max probability < 0.40
- **Action:** Flag as "Unknown/Ambiguous"
- **Explanation:** "Pattern does not match known categories"

---

## 3. FEATURE-BASED SIMILARITY MATCHING

### 3.1 Similarity Objectives

After primary mapping, use features to:
1. **Confirm category:** Find similar patterns in same category
2. **Identify design:** Match to specific named designs (if knowledge base has them)
3. **Detect outliers:** Flag patterns that don't match any prototype

### 3.2 Feature Selection Strategy

**Available Features (from Step 3):**

```
Total: 2074 dimensions
├─ Handcrafted: 26 dimensions
│   ├─ Geometric: aspect_ratio, extent, solidity, compactness (4)
│   ├─ Shape: perimeter, area, convex_hull_area, circularity (4)
│   ├─ Structural: dot_count, loop_count, edge_count, curve_length (4)
│   ├─ Spatial: symmetry metrics, grid_regularity, dot_spacing (5)
│   ├─ Texture: smoothness, fractal_dimension, pattern_fill (3)
│   └─ Advanced: connectivity, curvature, edge_continuity (6)
│
└─ CNN Features: 2048 dimensions
    └─ ResNet50 pre-logits layer (deep semantic features)
```

**Feature Selection for Similarity:**

| Purpose                     | Features Used                                  | Dimensions | Weight |
|-----------------------------|------------------------------------------------|------------|--------|
| **Structural Similarity**   | Handcrafted (all 26)                           | 26         | 0.40   |
| **Visual Similarity**       | CNN features (first 512 dims, high variance)   | 512        | 0.40   |
| **Category-Specific**       | Selected features per category                 | Varies     | 0.20   |

**Category-Specific Features:**

```python
CATEGORY_FEATURES = {
    'pulli_kolam': {
        'primary': ['dot_count', 'grid_regularity', 'dot_spacing_std', 'symmetry_rotational'],
        'weight': 0.30
    },
    'chukku_kolam': {
        'primary': ['loop_count', 'connectivity_ratio', 'curve_length', 'edge_continuity'],
        'weight': 0.30
    },
    'line_kolam': {
        'primary': ['symmetry_reflective', 'symmetry_rotational', 'smoothness', 'compactness'],
        'weight': 0.30
    },
    'freehand_kolam': {
        'primary': ['fractal_dimension', 'pattern_fill_ratio', 'curvature_mean', 'complexity'],
        'weight': 0.30
    }
}
```

### 3.3 Similarity Measures

**Multi-Metric Approach:**

1. **Cosine Similarity** (for CNN features)
   ```
   sim_cosine = (A · B) / (||A|| × ||B||)
   ```
   - Range: [-1, 1], normalize to [0, 1]
   - Best for: High-dimensional semantic features
   - Insensitive to magnitude, focuses on direction

2. **Euclidean Distance** (for handcrafted features)
   ```
   dist_euclidean = sqrt(Σ(A_i - B_i)²)
   sim_euclidean = 1 / (1 + dist_normalized)
   ```
   - Range after normalization: [0, 1]
   - Best for: Geometric features with real units
   - Sensitive to actual feature values

3. **Weighted Feature Distance** (for category-specific)
   ```
   dist_weighted = Σ w_i × |A_i - B_i|
   ```
   - Weights based on feature importance
   - Focuses on discriminative features per category

**Combined Similarity Score:**

```python
def compute_similarity(query_features, prototype_features, category):
    """
    Compute multi-metric similarity score.
    
    Returns:
        similarity: float [0-1]
        breakdown: dict with individual scores
    """
    
    # 1. Extract feature subsets
    handcrafted_q = query_features[:26]
    cnn_q = query_features[26:538]  # First 512 of 2048
    
    handcrafted_p = prototype_features[:26]
    cnn_p = prototype_features[26:538]
    
    # 2. Compute individual similarities
    sim_structural = euclidean_similarity(handcrafted_q, handcrafted_p)
    sim_visual = cosine_similarity(cnn_q, cnn_p)
    sim_category = weighted_similarity(handcrafted_q, handcrafted_p, category)
    
    # 3. Combine with weights
    similarity = (
        0.40 * sim_structural +
        0.40 * sim_visual +
        0.20 * sim_category
    )
    
    return similarity, {
        'structural': sim_structural,
        'visual': sim_visual,
        'category_specific': sim_category
    }
```

### 3.4 Prototype Matching Algorithm

```python
def find_best_match(query_features, category, knowledge_base):
    """
    Find most similar prototype in knowledge base.
    
    Args:
        query_features: 2074-dim feature vector
        category: str - Base category from primary mapping
        knowledge_base: dict - Prototypes per category
    
    Returns:
        best_match: dict with design name, similarity, confidence
    """
    
    # 1. Get prototypes for this category
    prototypes = knowledge_base[category]['designs']
    
    if not prototypes:
        return None  # No specific designs available
    
    # 2. Compute similarity to each prototype
    similarities = []
    for design_name, prototype_data in prototypes.items():
        prototype_features = prototype_data['features']
        
        sim, breakdown = compute_similarity(
            query_features, 
            prototype_features, 
            category
        )
        
        similarities.append({
            'design': design_name,
            'similarity': sim,
            'breakdown': breakdown
        })
    
    # 3. Sort by similarity
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # 4. Check threshold
    best_match = similarities[0]
    
    if best_match['similarity'] < 0.70:
        # Not similar enough to any known design
        return {
            'design': f"{category} - Unknown Variant",
            'similarity': best_match['similarity'],
            'confidence': 'low',
            'note': 'No close match to known designs'
        }
    elif best_match['similarity'] >= 0.85:
        confidence = 'high'
    else:
        confidence = 'medium'
    
    return {
        'design': best_match['design'],
        'similarity': best_match['similarity'],
        'confidence': confidence,
        'breakdown': best_match['breakdown'],
        'top_3': similarities[:3]
    }
```

### 3.5 Handling Visual Similarity vs. Structural Difference

**Problem:** Two Kolams may look similar visually (CNN says same) but have different structures (rules say different).

**Example:**
- Pattern A: Chukku Kolam (continuous loops)
- Pattern B: Line Kolam with curved lines (looks similar but no loops)

**Solution:**

```python
def resolve_similarity_conflict(visual_sim, structural_sim, rules):
    """
    Prioritize structural features over visual for Kolam classification.
    
    Kolam categories are defined by structure, not just appearance.
    """
    
    # 1. Check if structural features strongly disagree
    if abs(visual_sim - structural_sim) > 0.30:
        # Significant conflict
        
        if structural_sim > 0.70:
            # Trust structural similarity
            decision = 'structural'
            confidence = structural_sim
            explanation = "Structural features match despite visual differences"
        
        elif visual_sim > 0.70:
            # Visual is strong but structural disagrees
            decision = 'uncertain'
            confidence = (visual_sim + structural_sim) / 2
            explanation = "Visual similarity high, but structural features differ"
        
        else:
            # Both uncertain
            decision = 'ambiguous'
            confidence = max(visual_sim, structural_sim)
            explanation = "Neither visual nor structural features provide clear match"
    
    else:
        # Agreement between visual and structural
        decision = 'agreed'
        confidence = (visual_sim + structural_sim) / 2
        explanation = "Visual and structural features agree"
    
    return {
        'decision': decision,
        'confidence': confidence,
        'explanation': explanation,
        'visual_sim': visual_sim,
        'structural_sim': structural_sim
    }
```

---

## 4. RULE CONFIRMATION & CONFLICT RESOLUTION

### 4.1 Rule Validation Integration

Step 5 leverages the rule-based validator from Step 4 to:
1. **Confirm** primary category mapping
2. **Reject** incorrect mappings
3. **Resolve** conflicts between CNN and rules

**Rule Validation Flow:**

```
1. CNN predicts: "Pulli Kolam" (prob=0.85)
2. Extract features: dot_count=45, grid_regularity=0.82
3. Apply Pulli rules:
   ✓ dot_count ≥ 20 (PASS)
   ✓ grid_regularity ≥ 0.40 (PASS)
   ✓ dot_density ≥ 5% (PASS)
   ✓ dot_spacing_std < 30 (PASS)
4. Rule score: 0.95 (strong validation)
5. Final: "Pulli Kolam" CONFIRMED
```

### 4.2 Conflict Types

**Type 1: CNN Confident, Rules Reject**

```
CNN: "Pulli Kolam" (prob=0.90)
Rules: Pulli score=0.30 (FAIL)
```

**Resolution Strategy:**

```python
def resolve_cnn_vs_rules(cnn_pred, cnn_prob, rule_scores):
    """
    CNN is confident, but rules reject the prediction.
    """
    
    # 1. Check if another category has better rule score
    alternative = max(rule_scores, key=rule_scores.get)
    alternative_score = rule_scores[alternative]
    
    if alternative_score > 0.70:
        # Rules strongly support alternative
        decision = alternative
        confidence = 0.5 * cnn_prob + 0.5 * alternative_score
        explanation = (
            f"CNN predicted {cnn_pred} (prob={cnn_prob:.2f}), "
            f"but structural rules strongly support {alternative} "
            f"(score={alternative_score:.2f}). "
            f"Final decision: {alternative}"
        )
        conflict_type = 'resolved_by_rules'
    
    elif alternative_score > rule_scores[cnn_pred]:
        # Alternative has better (but not strong) rule support
        decision = alternative
        confidence = 0.6 * cnn_prob + 0.4 * alternative_score
        explanation = (
            f"CNN predicted {cnn_pred}, but {alternative} has better "
            f"structural support. Moderate confidence."
        )
        conflict_type = 'resolved_with_uncertainty'
    
    else:
        # No alternative has good rule support
        decision = cnn_pred
        confidence = 0.8 * cnn_prob  # Reduce confidence
        explanation = (
            f"CNN predicted {cnn_pred}, but structural rules do not "
            f"strongly support any category. Using CNN with reduced confidence."
        )
        conflict_type = 'unresolved'
    
    return {
        'decision': decision,
        'confidence': confidence,
        'explanation': explanation,
        'conflict_type': conflict_type,
        'cnn_prediction': cnn_pred,
        'rule_alternative': alternative
    }
```

**Type 2: CNN Uncertain, Rules Clear**

```
CNN: "Line Kolam" (prob=0.55), "Chukku Kolam" (prob=0.40)
Rules: Chukku score=0.85
```

**Resolution:**

```python
def resolve_uncertain_cnn(cnn_probs, rule_scores, threshold=0.70):
    """
    CNN is uncertain (top predictions close), but rules are clear.
    """
    
    # 1. Find best rule score
    best_rule_category = max(rule_scores, key=rule_scores.get)
    best_rule_score = rule_scores[best_rule_category]
    
    if best_rule_score > threshold:
        # Rules provide clear guidance
        decision = best_rule_category
        confidence = 0.4 * cnn_probs[best_rule_category] + 0.6 * best_rule_score
        explanation = (
            f"CNN predictions were uncertain (top scores: {cnn_probs}), "
            f"but structural rules clearly support {best_rule_category} "
            f"(score={best_rule_score:.2f})"
        )
        conflict_type = 'resolved_by_rules'
    
    else:
        # Both CNN and rules uncertain
        decision = max(cnn_probs, key=cnn_probs.get)
        confidence = max(cnn_probs.values()) * 0.7
        explanation = (
            f"Both CNN and rules are uncertain. "
            f"Using top CNN prediction with low confidence."
        )
        conflict_type = 'unresolved_uncertainty'
    
    return {
        'decision': decision,
        'confidence': confidence,
        'explanation': explanation,
        'conflict_type': conflict_type
    }
```

**Type 3: Both Agree**

```
CNN: "Chukku Kolam" (prob=0.88)
Rules: Chukku score=0.82
```

**Resolution:**

```python
def confirm_agreement(cnn_pred, cnn_prob, rule_scores):
    """
    CNN and rules agree - strongest case.
    """
    
    rule_score = rule_scores[cnn_pred]
    
    # Boost confidence when both agree
    confidence = 0.7 * cnn_prob + 0.3 * rule_score
    
    if confidence > 0.85:
        confidence_level = 'very_high'
    elif confidence > 0.75:
        confidence_level = 'high'
    else:
        confidence_level = 'medium'
    
    explanation = (
        f"CNN prediction ({cnn_pred}, prob={cnn_prob:.2f}) "
        f"confirmed by structural rules (score={rule_score:.2f}). "
        f"High confidence in classification."
    )
    
    return {
        'decision': cnn_pred,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'explanation': explanation,
        'conflict_type': 'agreement'
    }
```

### 4.3 Decision Matrix

| CNN Confidence | Rule Score | Action                        | Final Confidence Weight |
|----------------|-----------|-------------------------------|------------------------|
| High (>0.75)   | High (>0.70) | **Confirm** (agreement)       | 0.70 CNN + 0.30 Rules  |
| High (>0.75)   | Low (<0.50)  | **Investigate** (conflict)    | Use similarity matching |
| Low (<0.60)    | High (>0.70) | **Trust Rules**               | 0.40 CNN + 0.60 Rules  |
| Low (<0.60)    | Low (<0.50)  | **Flag Uncertain**            | Return top-3 candidates |
| Medium (0.60-0.75) | Medium (0.50-0.70) | **Moderate Confidence** | 0.60 CNN + 0.40 Rules |

### 4.4 Conflict Logging

All conflicts are logged for analysis:

```json
{
  "sample_id": "test_0125",
  "timestamp": "2025-12-28T14:30:00",
  "conflict_type": "cnn_confident_rules_reject",
  "cnn_prediction": {
    "category": "pulli_kolam",
    "probability": 0.87,
    "top_3": [
      {"category": "pulli_kolam", "prob": 0.87},
      {"category": "line_kolam", "prob": 0.09},
      {"category": "chukku_kolam", "prob": 0.03}
    ]
  },
  "rule_scores": {
    "pulli_kolam": 0.35,
    "chukku_kolam": 0.78,
    "line_kolam": 0.42,
    "freehand_kolam": 0.15
  },
  "resolution": {
    "final_decision": "chukku_kolam",
    "confidence": 0.71,
    "reasoning": "CNN predicted pulli_kolam, but structural analysis (loop_count=8, connectivity=0.85) strongly supports chukku_kolam",
    "similarity_analysis": {
      "chukku_prototypes": [
        {"design": "spiral_pattern", "similarity": 0.82}
      ]
    }
  }
}
```

---

## 5. KNOWLEDGE BASE DESIGN

### 5.1 Structure

The knowledge base contains:
1. **Category Definitions:** Official taxonomy and descriptions
2. **Feature Prototypes:** Representative feature vectors for each design
3. **Structural Constraints:** Rules and thresholds per category
4. **Design Metadata:** Cultural information, variations, examples

**File Structure:**

```
kolam_knowledge_base/
├── categories.json          # Category definitions
├── prototypes/
│   ├── pulli_kolam/
│   │   ├── grid_3x3.json
│   │   ├── grid_5x5.json
│   │   └── diamond_grid.json
│   ├── chukku_kolam/
│   │   ├── serpent_pattern.json
│   │   └── spiral_pattern.json
│   ├── line_kolam/
│   │   ├── fourfold_symmetry.json
│   │   └── mandala_pattern.json
│   └── freehand_kolam/
│       ├── floral_design.json
│       └── peacock_design.json
├── constraints.json         # Structural rules per category
└── metadata.json            # Cultural context, descriptions
```

### 5.2 Category Definitions (categories.json)

```json
{
  "version": "1.0",
  "last_updated": "2025-12-28",
  "categories": {
    "pulli_kolam": {
      "id": 0,
      "name": "Pulli Kolam",
      "tamil_name": "புள்ளி கோலம்",
      "description": "Dot-based grid patterns where dots (pulli) are placed in a regular grid and lines are drawn connecting them following specific rules",
      "characteristics": [
        "Regular grid of dots",
        "Lines connect dots symmetrically",
        "Grid sizes: 3×3, 5×5, 7×7, etc.",
        "Mathematical and geometric",
        "Closed loops around dots"
      ],
      "cultural_significance": "Represents order and mathematical beauty in Tamil culture",
      "difficulty": "beginner_to_intermediate",
      "typical_dot_range": [9, 81],
      "examples": ["kolam_00_pulli_01.jpg", "kolam_00_pulli_02.jpg"]
    },
    "chukku_kolam": {
      "id": 1,
      "name": "Chukku Kolam",
      "tamil_name": "சுழி கோலம்",
      "description": "Continuous loop patterns drawn without lifting the hand, featuring spirals, meanders, and interlocking loops",
      "characteristics": [
        "Continuous unbroken lines",
        "Spiral and loop motifs",
        "High connectivity",
        "Flowing and organic",
        "No isolated segments"
      ],
      "cultural_significance": "Represents continuity and the cyclical nature of life",
      "difficulty": "intermediate_to_advanced",
      "typical_loop_range": [3, 20],
      "examples": ["kolam_01_chukku_01.jpg", "kolam_01_chukku_02.jpg"]
    },
    "line_kolam": {
      "id": 2,
      "name": "Line Kolam",
      "tamil_name": "கோடு கோலம்",
      "description": "Symmetric line-based patterns emphasizing geometric symmetry without dots, often featuring radial or reflective symmetry",
      "characteristics": [
        "Strong symmetry (4-fold, 8-fold)",
        "No dots or minimal dots",
        "Straight and curved lines",
        "Mandala-like structures",
        "Radial patterns"
      ],
      "cultural_significance": "Represents balance and harmony in the universe",
      "difficulty": "beginner_to_advanced",
      "symmetry_types": ["rotational", "reflective", "bilateral"],
      "examples": ["kolam_02_line_01.jpg", "kolam_02_line_02.jpg"]
    },
    "freehand_kolam": {
      "id": 3,
      "name": "Freehand Kolam",
      "tamil_name": "வரைபட கோலம்",
      "description": "Complex freehand designs depicting natural objects, deities, or abstract patterns with high artistic freedom",
      "characteristics": [
        "Representational imagery",
        "High complexity",
        "Artistic interpretation",
        "Natural motifs (flowers, birds)",
        "Less constrained by rules"
      ],
      "cultural_significance": "Represents creativity and devotion in artistic expression",
      "difficulty": "advanced",
      "common_themes": ["floral", "peacock", "deity", "abstract"],
      "examples": ["kolam_03_freehand_01.jpg", "kolam_03_freehand_02.jpg"]
    }
  }
}
```

### 5.3 Prototype Structure

Each prototype contains:
- **Design metadata:** Name, category, description
- **Feature vector:** 2074-dim feature representation
- **Key characteristics:** Distinctive feature values
- **Visual examples:** Reference image paths

**Example: prototypes/pulli_kolam/grid_5x5.json**

```json
{
  "design_name": "5×5 Grid Pulli Kolam",
  "category": "pulli_kolam",
  "description": "Classic 25-dot grid pattern with symmetric loops",
  "created_from": "kolam_dataset/00_raw_data/pulli_kolam/sample_025.jpg",
  "features": {
    "vector": [/* 2074 float values */],
    "key_characteristics": {
      "dot_count": 25,
      "grid_regularity": 0.92,
      "symmetry_rotational": 0.88,
      "dot_spacing_std": 12.5,
      "aspect_ratio": 1.02,
      "pattern_fill_ratio": 0.65
    }
  },
  "structural_properties": {
    "grid_size": "5×5",
    "symmetry_order": 4,
    "loop_closure": "complete",
    "complexity": "medium"
  },
  "variations": [
    "Can be drawn with diagonal connections",
    "May include central motif",
    "Border patterns vary"
  ],
  "similarity_threshold": 0.75,
  "examples": [
    "pulli_kolam/kolam_00_025.jpg",
    "pulli_kolam/kolam_00_047.jpg",
    "pulli_kolam/kolam_00_089.jpg"
  ]
}
```

### 5.4 Adding New Categories

**Process:**

1. **Define Category:**
   - Add entry to `categories.json`
   - Assign unique ID and name
   - Describe characteristics

2. **Create Prototypes:**
   - Extract features from representative samples
   - Create prototype JSON files
   - Define similarity thresholds

3. **Define Rules:**
   - Add validation rules to `constraints.json`
   - Set feature thresholds

4. **Update Mapper:**
   - Add category to mapping table
   - Update rule validator integration

**No code changes required for new prototypes, only JSON files!**

### 5.5 Constraints File (constraints.json)

```json
{
  "pulli_kolam": {
    "required_features": ["dot_count", "grid_regularity"],
    "rules": [
      {
        "name": "minimum_dots",
        "feature": "dot_count",
        "operator": ">=",
        "threshold": 20,
        "weight": 0.8,
        "violation_message": "Dot count too low for Pulli Kolam"
      },
      {
        "name": "grid_structure",
        "feature": "grid_regularity",
        "operator": ">=",
        "threshold": 0.40,
        "weight": 1.0,
        "violation_message": "Grid structure not regular enough"
      },
      {
        "name": "dot_density",
        "feature": "dot_density",
        "operator": ">=",
        "threshold": 0.05,
        "weight": 0.6,
        "violation_message": "Dot density too low"
      }
    ],
    "incompatible_features": {
      "loop_count": {
        "operator": "<",
        "threshold": 5,
        "reason": "High loop count indicates Chukku Kolam"
      }
    }
  },
  "chukku_kolam": {
    "required_features": ["loop_count", "connectivity_ratio"],
    "rules": [
      {
        "name": "minimum_loops",
        "feature": "loop_count",
        "operator": ">=",
        "threshold": 3,
        "weight": 0.8,
        "violation_message": "Insufficient loops for Chukku Kolam"
      },
      {
        "name": "high_connectivity",
        "feature": "connectivity_ratio",
        "operator": ">=",
        "threshold": 0.60,
        "weight": 1.0,
        "violation_message": "Lines not sufficiently connected"
      }
    ]
  }
}
```

---

## 6. IMPLEMENTATION ARCHITECTURE

### 6.1 Module Overview

```
scripts/category_mapping/
├── __init__.py                 # Package initialization
├── category_mapper.py          # Core mapping logic
├── similarity_scorer.py        # Feature similarity computation
├── knowledge_base.py           # Knowledge base loader and manager
├── explainer.py                # Explanation generation
└── conflict_resolver.py        # Conflict resolution strategies
```

### 6.2 CategoryMapper Class

**Responsibilities:**
- Orchestrate the 3-stage mapping pipeline
- Integrate CNN, rules, and similarity
- Generate final category assignment

**Interface:**

```python
class CategoryMapper:
    """
    Maps raw CNN predictions to semantic Kolam categories.
    """
    
    def __init__(self, knowledge_base, rule_validator, similarity_scorer):
        """
        Initialize mapper with required components.
        
        Args:
            knowledge_base: KnowledgeBase instance
            rule_validator: RuleBasedValidator from Step 4
            similarity_scorer: SimilarityScorer instance
        """
        pass
    
    def map_category(self, 
                     cnn_output, 
                     features, 
                     rule_scores=None,
                     include_similarity=True):
        """
        Map prediction to final category.
        
        Args:
            cnn_output: dict with 'class_id' and 'probabilities'
            features: 2074-dim feature vector
            rule_scores: dict of rule scores per category (optional)
            include_similarity: bool - whether to do similarity matching
        
        Returns:
            mapping_result: dict with:
                - category: str - Final category name
                - confidence: float [0-1]
                - design: str - Specific design name (if found)
                - similarity: float [0-1] (if applicable)
                - explanation: str - Human-readable reasoning
                - metadata: dict - Additional information
        """
        pass
    
    def map_batch(self, predictions, features_list, rule_scores_list=None):
        """
        Map multiple predictions in batch.
        """
        pass
```

### 6.3 SimilarityScorer Class

```python
class SimilarityScorer:
    """
    Compute feature-based similarity to prototypes.
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def compute_similarity(self, query_features, prototype_features, 
                          category, method='combined'):
        """
        Compute similarity score.
        
        Args:
            query_features: 2074-dim vector
            prototype_features: 2074-dim vector
            category: str - Category for weighted features
            method: 'combined', 'cosine', 'euclidean', 'weighted'
        
        Returns:
            similarity: float [0-1]
            breakdown: dict with component scores
        """
        pass
    
    def find_best_matches(self, query_features, category, top_k=3):
        """
        Find top-k most similar prototypes.
        
        Returns:
            matches: list of dicts with design, similarity, confidence
        """
        pass
    
    def cosine_similarity(self, a, b):
        """Cosine similarity for CNN features."""
        pass
    
    def euclidean_similarity(self, a, b):
        """Normalized Euclidean similarity for handcrafted features."""
        pass
    
    def weighted_similarity(self, a, b, category):
        """Category-specific weighted similarity."""
        pass
```

### 6.4 KnowledgeBase Class

```python
class KnowledgeBase:
    """
    Manage Kolam knowledge base (categories, prototypes, constraints).
    """
    
    def __init__(self, kb_path):
        """
        Load knowledge base from JSON files.
        
        Args:
            kb_path: Path to kolam_knowledge_base/ directory
        """
        self.kb_path = kb_path
        self.categories = None
        self.prototypes = {}
        self.constraints = None
        self.metadata = None
        
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """Load all JSON files."""
        pass
    
    def get_category_info(self, category_name):
        """Get category definition."""
        pass
    
    def get_prototypes(self, category_name):
        """Get all prototypes for a category."""
        pass
    
    def get_constraints(self, category_name):
        """Get structural constraints for a category."""
        pass
    
    def add_prototype(self, category, design_name, features, metadata):
        """
        Add new prototype to knowledge base.
        
        Allows extending the system without code changes.
        """
        pass
    
    def validate_category(self, category_name):
        """Check if category exists in knowledge base."""
        pass
```

### 6.5 Explainer Class

```python
class CategoryExplainer:
    """
    Generate human-readable explanations for category mappings.
    """
    
    def __init__(self, knowledge_base):
        self.kb = knowledge_base
    
    def explain_mapping(self, mapping_result, cnn_output, rule_scores, 
                       similarity_result=None):
        """
        Generate detailed explanation for mapping decision.
        
        Returns:
            explanation: dict with:
                - summary: str - One-sentence summary
                - reasoning: str - Detailed multi-line explanation
                - key_features: list - Features that influenced decision
                - confidence_breakdown: dict - Component confidences
                - alternative_categories: list - Other possibilities
        """
        pass
    
    def _explain_agreement(self, category, cnn_prob, rule_score):
        """Explain when CNN and rules agree."""
        pass
    
    def _explain_conflict(self, cnn_pred, rule_best, resolution):
        """Explain conflict resolution."""
        pass
    
    def _explain_similarity(self, similarity_result):
        """Explain similarity matching results."""
        pass
    
    def format_for_display(self, explanation):
        """Format explanation for console/log output."""
        pass
```

### 6.6 ConflictResolver Class

```python
class ConflictResolver:
    """
    Resolve conflicts between CNN predictions and rule validation.
    """
    
    def __init__(self, similarity_scorer):
        self.similarity_scorer = similarity_scorer
    
    def resolve(self, cnn_output, rule_scores, features):
        """
        Analyze conflict and determine resolution.
        
        Returns:
            resolution: dict with:
                - decision: str - Final category
                - confidence: float
                - conflict_type: str
                - reasoning: str
        """
        pass
    
    def resolve_cnn_confident_rules_reject(self, cnn_pred, cnn_prob, 
                                          rule_scores, features):
        """Handle Type 1 conflict."""
        pass
    
    def resolve_cnn_uncertain_rules_clear(self, cnn_probs, rule_scores):
        """Handle Type 2 conflict."""
        pass
    
    def confirm_agreement(self, cnn_pred, cnn_prob, rule_scores):
        """Handle agreement case."""
        pass
```

---

## 7. EXPLAINABILITY SYSTEM

### 7.1 Explanation Levels

**Level 1: Summary** (One sentence)
```
"Classified as Pulli Kolam with high confidence (0.87)"
```

**Level 2: Basic Reasoning** (Short paragraph)
```
Classified as Pulli Kolam with high confidence (0.87).
CNN prediction: Pulli Kolam (prob=0.85)
Rule validation: Strong support (score=0.89)
Both CNN and structural analysis agree on this classification.
```

**Level 3: Detailed Analysis** (Full breakdown)
```
=== CATEGORY MAPPING RESULT ===

Final Category: Pulli Kolam
Confidence: 0.87 (High)
Design: 5×5 Grid Pattern (similarity: 0.82)

--- CNN PREDICTION ---
Predicted: Pulli Kolam
Probability: 0.85
Top-3 Predictions:
  1. Pulli Kolam: 0.85
  2. Line Kolam: 0.10
  3. Chukku Kolam: 0.03

--- RULE VALIDATION ---
Pulli Kolam Rules: 0.89 (PASS)
  ✓ dot_count = 28 (threshold: ≥20)
  ✓ grid_regularity = 0.86 (threshold: ≥0.40)
  ✓ dot_density = 0.08 (threshold: ≥0.05)
  ✓ dot_spacing_std = 15.2 (threshold: <30)

--- SIMILARITY ANALYSIS ---
Best Match: 5×5 Grid Pattern (similarity: 0.82)
Similarity Breakdown:
  - Structural: 0.85 (handcrafted features)
  - Visual: 0.79 (CNN features)
  - Category-specific: 0.83 (Pulli features)

--- KEY FEATURES ---
Discriminative features for this classification:
  1. dot_count = 28 (Expected: 20-81)
  2. grid_regularity = 0.86 (Expected: >0.70)
  3. symmetry_rotational = 0.81 (Expected: >0.60)

--- REASONING ---
The Kolam shows clear characteristics of a Pulli Kolam:
- Regular grid of 28 dots arranged in ~5×5 pattern
- High grid regularity (0.86) indicates well-structured dot placement
- Strong rotational symmetry (0.81) typical of grid-based designs
- CNN and structural rules both agree on Pulli classification

Alternative possibilities considered:
  - Line Kolam: Low probability (0.10) and weak rule support (0.35)
  
Confidence Assessment: HIGH
Both machine learning and rule-based validation strongly support
this classification. The pattern closely matches known 5×5 grid
Pulli Kolam prototypes.
```

### 7.2 Feature Importance Explanation

```python
def explain_feature_importance(features, category, top_k=5):
    """
    Identify which features most influenced the decision.
    
    Uses category-specific feature importance weights.
    """
    
    category_important_features = {
        'pulli_kolam': [
            ('dot_count', 'Dot Count', 'Number of dots in the pattern'),
            ('grid_regularity', 'Grid Regularity', 'How regular the dot grid is'),
            ('dot_spacing_std', 'Dot Spacing Std', 'Consistency of spacing'),
            ('symmetry_rotational', 'Rotational Symmetry', 'Rotational order')
        ],
        'chukku_kolam': [
            ('loop_count', 'Loop Count', 'Number of closed loops'),
            ('connectivity_ratio', 'Connectivity', 'How connected the lines are'),
            ('curve_length', 'Curve Length', 'Total length of curves'),
            ('edge_continuity', 'Edge Continuity', 'Continuity of edges')
        ],
        # ... other categories
    }
    
    important = category_important_features[category]
    
    explanation = []
    for feature_name, display_name, description in important[:top_k]:
        value = features[feature_name]
        explanation.append({
            'feature': display_name,
            'value': value,
            'description': description,
            'influence': 'high' if value > threshold else 'medium'
        })
    
    return explanation
```

### 7.3 Conflict Explanation

```python
def explain_conflict(conflict_result):
    """
    Explain why there was a conflict and how it was resolved.
    """
    
    explanation = f"""
=== CONFLICT DETECTED ===

Conflict Type: {conflict_result['conflict_type']}

CNN Prediction:
  Category: {conflict_result['cnn_prediction']}
  Probability: {conflict_result['cnn_probability']:.2f}

Rule Analysis:
  {conflict_result['cnn_prediction']} Score: {conflict_result['cnn_rule_score']:.2f}
  Alternative ({conflict_result['rule_alternative']}): {conflict_result['alt_rule_score']:.2f}

Resolution Strategy:
  {conflict_result['resolution_strategy']}

Final Decision: {conflict_result['decision']}
Confidence: {conflict_result['confidence']:.2f} ({conflict_result['confidence_level']})

Reasoning:
{conflict_result['reasoning']}

Additional Analysis:
{conflict_result['additional_notes']}
"""
    
    return explanation
```

### 7.4 Visualization Support

While we don't implement UI in Step 5, we provide data structures for visualization:

```python
def get_visualization_data(mapping_result):
    """
    Prepare data for potential visualization.
    
    Returns structured data that can be used by Step 7 (UI)
    to create visual explanations.
    """
    
    return {
        'classification': {
            'category': mapping_result['category'],
            'confidence': mapping_result['confidence'],
            'design': mapping_result.get('design', 'Unknown')
        },
        'confidence_breakdown': {
            'cnn': mapping_result['cnn_prob'],
            'rules': mapping_result['rule_score'],
            'similarity': mapping_result.get('similarity', None)
        },
        'feature_values': {
            # Key feature values for radar chart, etc.
        },
        'decision_path': [
            # Steps taken to reach decision
        ],
        'alternatives': [
            # Other categories considered
        ]
    }
```

---

## 8. VALIDATION & TESTING

### 8.1 Test Categories

**Test Set 1: Clear Cases** (80% of test set)
- Strong agreement between CNN and rules
- High confidence predictions
- Expected Result: Correct mapping with high confidence

**Test Set 2: Edge Cases** (15% of test set)
- Ambiguous patterns (e.g., Line Kolam with some dots)
- Low confidence predictions
- Expected Result: Multiple candidates or uncertainty flagged

**Test Set 3: Conflict Cases** (5% of test set)
- Deliberate mismatches between appearance and structure
- CNN vs. rules disagreement
- Expected Result: Intelligent conflict resolution

### 8.2 Test Metrics

```python
def evaluate_category_mapping(mapper, test_data, ground_truth):
    """
    Evaluate mapping system performance.
    
    Metrics:
    1. Accuracy: Correct category assignments
    2. Confidence Calibration: Are high-confidence predictions more accurate?
    3. Conflict Resolution: How well are conflicts resolved?
    4. Similarity Matching: Do similar patterns match correctly?
    5. Explainability: Are explanations consistent with decisions?
    """
    
    results = {
        'total_samples': len(test_data),
        'correct_mappings': 0,
        'high_confidence_correct': 0,
        'low_confidence_correct': 0,
        'conflicts_resolved_correctly': 0,
        'similarity_matches_correct': 0,
        'top3_accuracy': 0
    }
    
    for sample in test_data:
        mapping = mapper.map_category(
            sample['cnn_output'],
            sample['features'],
            sample['rule_scores']
        )
        
        true_category = ground_truth[sample['id']]
        
        # 1. Primary accuracy
        if mapping['category'] == true_category:
            results['correct_mappings'] += 1
            
            if mapping['confidence'] > 0.80:
                results['high_confidence_correct'] += 1
            else:
                results['low_confidence_correct'] += 1
        
        # 2. Top-3 accuracy
        if 'alternatives' in mapping:
            top3 = [mapping['category']] + [a['category'] for a in mapping['alternatives'][:2]]
            if true_category in top3:
                results['top3_accuracy'] += 1
        
        # 3. Conflict resolution
        if mapping.get('conflict_type') is not None:
            if mapping['category'] == true_category:
                results['conflicts_resolved_correctly'] += 1
    
    # Calculate percentages
    results['accuracy'] = results['correct_mappings'] / results['total_samples']
    results['top3_accuracy_rate'] = results['top3_accuracy'] / results['total_samples']
    
    return results
```

### 8.3 Validation Criteria

| Criterion | Target | Importance |
|-----------|--------|-----------|
| **Overall Accuracy** | ≥85% | Critical |
| **High-Confidence Accuracy** | ≥90% | Critical |
| **Top-3 Accuracy** | ≥95% | High |
| **Conflict Resolution Rate** | ≥75% | High |
| **Similarity Match Accuracy** | ≥80% | Medium |
| **Explanation Consistency** | 100% | Critical |

### 8.4 Test Cases

```python
# Test Case 1: Clear Pulli Kolam
test_pulli = {
    'cnn_output': {'class_id': 0, 'probabilities': [0.89, 0.05, 0.04, 0.02]},
    'features': [...],  # 2074-dim with dot_count=25, grid_regularity=0.91
    'rule_scores': {'pulli_kolam': 0.92, 'chukku_kolam': 0.15, ...},
    'expected': {
        'category': 'pulli_kolam',
        'confidence_level': 'high',
        'conflict_type': 'agreement'
    }
}

# Test Case 2: Conflict - CNN says Pulli, Rules say Chukku
test_conflict = {
    'cnn_output': {'class_id': 0, 'probabilities': [0.78, 0.15, 0.05, 0.02]},
    'features': [...],  # Features suggest loops
    'rule_scores': {'pulli_kolam': 0.35, 'chukku_kolam': 0.82, ...},
    'expected': {
        'category': 'chukku_kolam',  # Should resolve to chukku
        'confidence_level': 'medium',
        'conflict_type': 'cnn_confident_rules_reject'
    }
}

# Test Case 3: Uncertain - Multiple candidates
test_uncertain = {
    'cnn_output': {'class_id': 2, 'probabilities': [0.35, 0.30, 0.25, 0.10]},
    'features': [...],
    'rule_scores': {'pulli_kolam': 0.45, 'chukku_kolam': 0.50, ...},
    'expected': {
        'category': 'chukku_kolam',  # Best rule support
        'confidence_level': 'low',
        'alternatives': ['pulli_kolam', 'line_kolam']
    }
}
```

---

## 9. INTEGRATION WITH STEPS 1-4

### 9.1 Input Requirements

**From Step 3 (Feature Extraction):**
```python
# Required: Feature vectors
features = np.load('kolam_dataset/04_feature_extraction/test/sample_001.npy')
# Shape: (2074,) containing 26 handcrafted + 2048 CNN features

# Optional: Feature metadata
metadata = json.load('kolam_dataset/04_feature_extraction/test/metadata.json')
```

**From Step 4 (Classification):**
```python
# Required: CNN predictions
cnn_output = {
    'class_id': 0,
    'probabilities': np.array([0.85, 0.10, 0.03, 0.02]),
    'logits': np.array([2.3, -0.5, -1.2, -1.8])
}

# Required: Rule validation scores
rule_scores = {
    'pulli_kolam': 0.89,
    'chukku_kolam': 0.25,
    'line_kolam': 0.42,
    'freehand_kolam': 0.18
}
```

### 9.2 Output Format

**For Step 6 (Model Interpretation):**
```python
mapping_result = {
    'category': 'pulli_kolam',
    'confidence': 0.87,
    'confidence_level': 'high',
    'design': '5×5 Grid Pattern',
    'similarity': 0.82,
    
    'cnn_prediction': {
        'class_id': 0,
        'probability': 0.85,
        'top_3': [...]
    },
    
    'rule_validation': {
        'score': 0.89,
        'passed': True,
        'violations': []
    },
    
    'similarity_analysis': {
        'best_match': '5×5 Grid Pattern',
        'similarity': 0.82,
        'breakdown': {
            'structural': 0.85,
            'visual': 0.79,
            'category_specific': 0.83
        }
    },
    
    'explanation': {
        'summary': '...',
        'reasoning': '...',
        'key_features': [...],
        'alternatives': [...]
    },
    
    'metadata': {
        'timestamp': '2025-12-28T14:30:00',
        'processing_time_ms': 45,
        'conflict_type': 'agreement'
    }
}
```

### 9.3 Integration Script

```python
def run_full_pipeline(image_path):
    """
    Run complete pipeline from image to category mapping.
    
    Steps:
    1. Load and preprocess image (Step 2)
    2. Extract features (Step 3)
    3. CNN prediction + rule validation (Step 4)
    4. Category mapping (Step 5)
    """
    
    # Step 2: Preprocess
    from scripts.preprocessing import preprocess_image
    preprocessed = preprocess_image(image_path)
    
    # Step 3: Extract features
    from scripts.feature_extraction import extract_features
    features = extract_features(preprocessed)
    
    # Step 4: Classify
    from scripts.classification import HybridPredictor
    predictor = HybridPredictor.load('models/best_model.pth')
    prediction = predictor.predict(features)
    
    # Step 5: Map category
    from scripts.category_mapping import CategoryMapper
    mapper = CategoryMapper.load('knowledge_base/')
    mapping = mapper.map_category(
        cnn_output=prediction['cnn'],
        features=features,
        rule_scores=prediction['rules']
    )
    
    return mapping
```

---

## 10. OUTPUT ARTIFACTS

### 10.1 Code Modules

```
scripts/category_mapping/
├── __init__.py                    (~30 lines)
├── category_mapper.py             (~400 lines)
├── similarity_scorer.py           (~350 lines)
├── knowledge_base.py              (~300 lines)
├── explainer.py                   (~400 lines)
└── conflict_resolver.py           (~300 lines)

Total: ~1,780 lines of code
```

### 10.2 Knowledge Base Files

```
kolam_knowledge_base/
├── categories.json                (~200 lines)
├── constraints.json               (~150 lines)
├── metadata.json                  (~100 lines)
└── prototypes/
    ├── pulli_kolam/
    │   ├── grid_3x3.json
    │   ├── grid_5x5.json
    │   └── diamond_grid.json
    ├── chukku_kolam/
    │   ├── serpent_pattern.json
    │   └── spiral_pattern.json
    ├── line_kolam/
    │   ├── fourfold_symmetry.json
    │   └── mandala_pattern.json
    └── freehand_kolam/
        ├── floral_design.json
        └── peacock_design.json

Total: ~12 JSON files
```

### 10.3 Execution Scripts

```
scripts/
├── 10_test_category_mapping.py    (~400 lines)
│   Test mapping system on samples
│
└── 11_full_pipeline.py             (~300 lines)
    Run complete pipeline with mapping

Total: ~700 lines
```

### 10.4 Documentation

```
STEP5_CATEGORY_MAPPING_DESIGN.md    (This document, ~1,000 lines)
STEP5_README.md                     (~800 lines)
STEP5_DELIVERABLES.md               (~600 lines)
QUICK_REFERENCE_STEP5.md            (~200 lines)

Total: ~2,600 lines
```

### 10.5 Test Results & Logs

```
kolam_dataset/06_category_mapping/
├── test_results/
│   ├── mapping_results.json        # All test mappings
│   ├── conflict_log.json           # Conflict cases
│   ├── accuracy_report.txt         # Performance metrics
│   └── edge_cases.json             # Edge case analysis
│
├── explanations/
│   ├── sample_001_explanation.txt
│   ├── sample_002_explanation.txt
│   └── ...
│
└── validation/
    ├── accuracy_by_category.json
    ├── confidence_calibration.json
    └── similarity_analysis.json
```

---

## 11. PERFORMANCE CONSIDERATIONS

### 11.1 Computational Complexity

**Per-Sample Processing:**

| Operation | Complexity | Time (CPU) | Time (GPU) |
|-----------|-----------|-----------|-----------|
| Primary Mapping | O(1) | <1 ms | <1 ms |
| Rule Validation | O(n) features | ~5 ms | ~2 ms |
| Similarity (all prototypes) | O(k×d) | ~20 ms | ~5 ms |
| Explanation Generation | O(1) | ~2 ms | ~2 ms |
| **Total** | | **~28 ms** | **~10 ms** |

Where:
- k = number of prototypes (~12)
- d = feature dimensions (2074)
- n = number of features (~26 for rules)

**Batch Processing (100 samples):**
- CPU: ~2.8 seconds
- GPU: ~1.0 seconds

### 11.2 Memory Requirements

- **Knowledge Base:** ~5 MB (loaded once)
- **Per-Sample Processing:** ~1 MB temporary
- **Batch (100 samples):** ~50 MB
- **Total System:** ~100 MB RAM

### 11.3 Optimization Strategies

1. **Cache Prototypes:** Load once, reuse
2. **Vectorize Similarity:** Compute all similarities in parallel
3. **Early Termination:** Skip similarity if high-confidence agreement
4. **Lazy Explanation:** Generate detailed explanations only when requested

```python
# Optimization: Skip similarity for high-confidence agreements
if cnn_prob > 0.85 and rule_score > 0.80:
    # Skip similarity matching
    return quick_mapping(cnn_pred, cnn_prob, rule_score)
```

---

## 12. FUTURE EXTENSIONS

### 12.1 Enhanced Similarity

- **Learnable Similarity Metrics:** Train metric learning model
- **Hierarchical Matching:** Match at multiple levels (category → subcategory → design)
- **Context-Aware Similarity:** Consider regional variations

### 12.2 Dynamic Knowledge Base

- **Active Learning:** Add user-validated samples as prototypes
- **Incremental Updates:** Update prototypes based on new data
- **Versioning:** Track knowledge base changes over time

### 12.3 Multi-Modal Mapping

- **Textual Descriptions:** Accept text queries ("Find spiral Kolams")
- **Sketch-Based:** Map from rough sketches
- **Cultural Context:** Incorporate festival, region, occasion metadata

### 12.4 Confidence Refinement

- **Uncertainty Quantification:** Bayesian confidence intervals
- **Ensemble Methods:** Combine multiple mapping strategies
- **Human-in-the-Loop:** Request human verification for low-confidence cases

---

## SUMMARY

**Step 5 Design Complete!**

This design document provides:

✅ **3-Stage Mapping Pipeline:** CNN → Rules → Similarity  
✅ **Multi-Metric Similarity:** Cosine + Euclidean + Weighted  
✅ **Intelligent Conflict Resolution:** 3 conflict types handled  
✅ **Extensible Knowledge Base:** JSON-based, no code changes needed  
✅ **Comprehensive Explainability:** 3 levels of explanation  
✅ **Integrated Testing:** Test cases for clear, edge, and conflict scenarios  
✅ **Modular Implementation:** 5 core classes, ~1,780 lines  
✅ **Cultural Correctness:** Respects traditional Kolam taxonomy  

**Next:** Implement the 5 modules and create knowledge base files.

---

**Document Version:** 1.0  
**Last Updated:** December 28, 2025  
**Status:** Design Complete - Ready for Implementation
