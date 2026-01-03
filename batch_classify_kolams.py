"""
BATCH KOLAM CLASSIFIER
======================
Classify multiple Kolam images at once and generate a report

Usage:
    python batch_classify_kolams.py folder_path/
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from classify_kolam_image import load_model, classify_image, CLASS_NAMES
from collections import Counter

def batch_classify(folder_path, model):
    """Classify all images in a folder"""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(folder.glob(ext))
    
    if not image_files:
        print(f"No images found in {folder_path}")
        return []
    
    print(f"\n📁 Found {len(image_files)} images")
    print("🔄 Processing...")
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        try:
            predicted_class, confidence, probabilities = classify_image(img_path, model)
            
            results.append({
                'filename': img_path.name,
                'predicted_class': CLASS_NAMES[predicted_class],
                'predicted_class_id': int(predicted_class),
                'confidence': float(confidence),
                'probabilities': {
                    name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)
                }
            })
            
            if i % 10 == 0:
                print(f"   Processed {i}/{len(image_files)} images...")
                
        except Exception as e:
            print(f"   ⚠️  Error with {img_path.name}: {e}")
            continue
    
    return results

def generate_report(results, output_file='batch_classification_report.json'):
    """Generate and save classification report"""
    
    # Statistics
    class_counts = Counter(r['predicted_class'] for r in results)
    avg_confidence = sum(r['confidence'] for r in results) / len(results) if results else 0
    
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_images': len(results),
        'class_distribution': dict(class_counts),
        'average_confidence': avg_confidence,
        'results': results
    }
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def print_summary(report):
    """Print summary of batch classification"""
    print("\n" + "=" * 70)
    print("BATCH CLASSIFICATION SUMMARY")
    print("=" * 70)
    
    print(f"\n📊 Total Images: {report['total_images']}")
    print(f"   Average Confidence: {report['average_confidence']*100:.2f}%")
    
    print(f"\n🎯 Class Distribution:")
    for class_name, count in sorted(report['class_distribution'].items()):
        percentage = (count / report['total_images']) * 100
        bar = "█" * int(percentage / 2)
        print(f"   {class_name:15}: {count:4} ({percentage:5.1f}%) {bar}")
    
    print("\n📄 Results saved to: batch_classification_report.json")
    print("=" * 70)

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_classify_kolams.py folder_path/")
        print("\nExample: python batch_classify_kolams.py kolam_dataset/02_split_data/test/pulli_kolam/")
        return
    
    folder_path = sys.argv[1]
    
    print("=" * 70)
    print("BATCH KOLAM CLASSIFIER")
    print("=" * 70)
    
    # Load model
    print("\n📥 Loading model...")
    try:
        model, checkpoint = load_model()
        print(f"   ✅ Model ready (F1: {checkpoint['val_f1']:.4f})")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Classify images
    results = batch_classify(folder_path, model)
    
    if results:
        # Generate report
        report = generate_report(results)
        print_summary(report)
        
        # Show some examples
        print("\n📋 Sample Results (first 5):")
        for i, r in enumerate(results[:5], 1):
            print(f"   {i}. {r['filename'][:30]:30} → {r['predicted_class']:15} ({r['confidence']*100:.1f}%)")
    else:
        print("\n❌ No images were successfully classified")

if __name__ == '__main__':
    main()
