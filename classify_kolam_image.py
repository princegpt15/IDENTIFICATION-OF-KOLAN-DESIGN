"""
KOLAM IMAGE CLASSIFIER - Using Balanced Model
==============================================
Classify any Kolam image using the newly trained balanced model (91% F1-Score)

Usage:
    python classify_kolam_image.py path/to/image.jpg
    python classify_kolam_image.py  # Uses test images
"""

import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import json

# Import feature extraction functions
sys.path.append('kolam_dataset/scripts')
try:
    from feature_extraction import extract_handcrafted_features
except:
    print("Note: Full feature extraction unavailable, using simplified version")

# Class names
CLASS_NAMES = ['Pulli Kolam', 'Chukku Kolam', 'Line Kolam', 'Freehand Kolam']

class ImprovedKolamClassifier(torch.nn.Module):
    """Improved classifier architecture"""
    def __init__(self, input_dim=26, num_classes=4, hidden_dims=[128, 64, 32], dropout_rates=[0.4, 0.3, 0.2]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.BatchNorm1d(hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(torch.nn.Linear(prev_dim, num_classes))
        self.model = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def extract_simple_features(image_path):
    """Extract 26 handcrafted features from image"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    features = []
    
    # 1-2: Basic stats
    features.append(np.mean(gray) / 255.0)
    features.append(np.std(gray) / 255.0)
    
    # 3-4: Edge detection
    edges = cv2.Canny(gray, 50, 150)
    features.append(np.sum(edges > 0) / edges.size)
    features.append(np.mean(edges) / 255.0)
    
    # 5-7: Color distribution
    for i in range(3):
        features.append(np.mean(img[:,:,i]) / 255.0)
    
    # 8-10: Texture (gradient)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    features.append(np.mean(np.abs(gx)) / 255.0)
    features.append(np.mean(np.abs(gy)) / 255.0)
    features.append(np.std(gx) / 255.0)
    
    # 11-13: Pattern detection (circles/dots)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=5, maxRadius=30)
    if circles is not None:
        features.append(len(circles[0]) / 100.0)  # Normalized count
        features.append(np.mean(circles[0][:, 2]) / gray.shape[0])  # Avg radius
    else:
        features.append(0.0)
        features.append(0.0)
    features.append(1.0 if circles is not None else 0.0)
    
    # 14-16: Contour analysis
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    features.append(len(contours) / 1000.0)
    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        features.append(np.mean(areas) / (gray.shape[0] * gray.shape[1]))
        features.append(np.std(areas) / (gray.shape[0] * gray.shape[1]))
    else:
        features.append(0.0)
        features.append(0.0)
    
    # 17-19: Symmetry (horizontal, vertical, rotational)
    h, w = gray.shape
    left = gray[:, :w//2]
    right = cv2.flip(gray[:, w//2:], 1)
    features.append(np.mean(np.abs(left - right[:, :left.shape[1]])) / 255.0)
    
    top = gray[:h//2, :]
    bottom = cv2.flip(gray[h//2:, :], 0)
    features.append(np.mean(np.abs(top - bottom[:top.shape[0], :])) / 255.0)
    
    features.append(np.mean(gray[:h//2, :]) / np.mean(gray[h//2:, :]) if np.mean(gray[h//2:, :]) > 0 else 1.0)
    
    # 20-22: Line detection
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
    if lines is not None:
        features.append(len(lines) / 100.0)
        lengths = [np.sqrt((x2-x1)**2 + (y2-y1)**2) for x1,y1,x2,y2 in lines[:, 0]]
        features.append(np.mean(lengths) / gray.shape[0])
    else:
        features.append(0.0)
        features.append(0.0)
    features.append(1.0 if lines is not None else 0.0)
    
    # 23-26: Additional spatial features
    features.append(h / w if w > 0 else 1.0)  # Aspect ratio
    features.append(np.sum(gray < 50) / gray.size)  # Dark pixel ratio
    features.append(np.sum(gray > 200) / gray.size)  # Bright pixel ratio
    features.append(len(np.unique(gray)) / 256.0)  # Color diversity
    
    return np.array(features[:26], dtype=np.float32)

def load_model():
    """Load the trained balanced model"""
    model_path = Path('kolam_dataset/05_trained_models/balanced_training/best_model_balanced.pth')
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Create model
    model = ImprovedKolamClassifier(input_dim=26, num_classes=4)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

def classify_image(image_path, model):
    """Classify a single image"""
    # Extract features
    features = extract_simple_features(image_path)
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()
    
    return predicted_class, confidence, probabilities.numpy()

def display_results(image_path, predicted_class, confidence, probabilities):
    """Display classification results"""
    print("\n" + "=" * 70)
    print("KOLAM CLASSIFICATION RESULT")
    print("=" * 70)
    print(f"\nImage: {Path(image_path).name}")
    print(f"\n🎯 Prediction: {CLASS_NAMES[predicted_class]}")
    print(f"   Confidence: {confidence*100:.2f}%")
    
    print(f"\n📊 All Class Probabilities:")
    for i, (name, prob) in enumerate(zip(CLASS_NAMES, probabilities)):
        bar = "█" * int(prob * 50)
        indicator = " ← PREDICTED" if i == predicted_class else ""
        print(f"   {name:15} {prob*100:5.2f}% {bar}{indicator}")
    
    print("\n" + "=" * 70)

def test_on_sample_images():
    """Test on a few sample images from test set"""
    test_dir = Path('kolam_dataset/02_split_data/test')
    
    if not test_dir.exists():
        print("Test directory not found")
        return
    
    print("\n🔍 Testing on sample images from test set...\n")
    
    # Get one image from each class
    for class_folder in test_dir.iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob('*.jpg'))[:1]
            if images:
                yield images[0]

def main():
    print("=" * 70)
    print("KOLAM IMAGE CLASSIFIER")
    print("Using Balanced Model (91% F1-Score)")
    print("=" * 70)
    
    # Load model
    print("\n📥 Loading trained model...")
    try:
        model, checkpoint = load_model()
        print(f"   ✅ Model loaded successfully")
        print(f"   Best validation F1: {checkpoint['val_f1']:.4f}")
        print(f"   Training epoch: {checkpoint['epoch']}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return
    
    # Check if image path provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        if not Path(image_path).exists():
            print(f"\n❌ Error: Image not found: {image_path}")
            return
        
        print(f"\n🖼️  Classifying image: {image_path}")
        
        try:
            predicted_class, confidence, probabilities = classify_image(image_path, model)
            display_results(image_path, predicted_class, confidence, probabilities)
        except Exception as e:
            print(f"\n❌ Error classifying image: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        # Test on sample images
        print("\n💡 No image specified, testing on sample images...")
        print("   Usage: python classify_kolam_image.py path/to/image.jpg")
        
        tested = 0
        for image_path in test_on_sample_images():
            try:
                predicted_class, confidence, probabilities = classify_image(image_path, model)
                display_results(image_path, predicted_class, confidence, probabilities)
                tested += 1
                
                if tested >= 3:  # Show 3 examples
                    break
            except Exception as e:
                print(f"Error with {image_path}: {e}")
                continue
        
        if tested == 0:
            print("\n⚠️  No test images found. Please provide an image path.")
            print("   Usage: python classify_kolam_image.py path/to/image.jpg")

if __name__ == '__main__':
    main()
