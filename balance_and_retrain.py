"""
DATASET BALANCER AND RETRAINER
===============================
Balances the dataset and retrains with improved configuration.
Fixes the class imbalance issue causing poor minority class performance.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
from collections import Counter
import shutil
from datetime import datetime

# ==================== CONFIGURATION ====================

CONFIG = {
    'data_dir': 'kolam_dataset/04_feature_extraction',
    'model_dir': 'kolam_dataset/05_trained_models',
    'output_dir': 'kolam_dataset/05_trained_models/balanced_training',
    
    # Training parameters
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'early_stopping_patience': 20,
    
    # Model architecture
    'hidden_dims': [128, 64, 32],
    'dropout_rates': [0.4, 0.3, 0.2],
    
    # Class balancing
    'use_weighted_sampler': True,
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
}

CLASS_NAMES = ['pulli_kolam', 'chukku_kolam', 'line_kolam', 'freehand_kolam']

# ==================== DATASET ====================

class KolamDataset(Dataset):
    """Kolam features dataset"""
    
    def __init__(self, features_file, metadata_file):
        # Load features (shape: [n_samples, n_features])
        self.features = torch.FloatTensor(np.load(features_file, allow_pickle=True))
        
        # Load metadata to get labels
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        self.labels = torch.LongTensor([item['label'] for item in metadata])
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# ==================== FOCAL LOSS ====================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=4):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==================== MODEL ====================

class ImprovedKolamClassifier(nn.Module):
    """Improved classifier with better architecture"""
    
    def __init__(self, input_dim, num_classes, hidden_dims, dropout_rates):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim, dropout in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# ==================== TRAINING ====================

def get_class_weights(labels):
    """Calculate class weights for balanced training"""
    class_counts = Counter(labels.numpy())
    total = len(labels)
    weights = {cls: total / (len(class_counts) * count) 
               for cls, count in class_counts.items()}
    return torch.FloatTensor([weights[i] for i in range(len(class_counts))])

def get_weighted_sampler(labels):
    """Create weighted sampler for balanced batches"""
    class_counts = Counter(labels.numpy())
    weights = [1.0 / class_counts[label] for label in labels.numpy()]
    return WeightedRandomSampler(weights, len(weights), replacement=True)

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate macro F1
    macro_f1 = calculate_macro_f1(all_labels, all_preds)
    
    return total_loss / len(loader), 100. * correct / total, macro_f1

def validate(model, loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate macro F1
    macro_f1 = calculate_macro_f1(all_labels, all_preds)
    
    return total_loss / len(loader), 100. * correct / total, macro_f1

def calculate_macro_f1(labels, preds):
    """Calculate macro F1 score"""
    from sklearn.metrics import f1_score
    return f1_score(labels, preds, average='macro', zero_division=0)

def calculate_per_class_metrics(labels, preds):
    """Calculate per-class precision, recall, F1"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )
    
    metrics = {}
    for i, class_name in enumerate(CLASS_NAMES):
        metrics[class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }
    
    return metrics

# ==================== MAIN TRAINING ====================

def main():
    print("=" * 70)
    print("BALANCED DATASET TRAINING - OPTION A")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Load datasets
    print("\n📊 Loading datasets...")
    train_dataset = KolamDataset(
        f"{CONFIG['data_dir']}/train_features.npy",
        f"{CONFIG['data_dir']}/train_metadata.json"
    )
    val_dataset = KolamDataset(
        f"{CONFIG['data_dir']}/val_features.npy",
        f"{CONFIG['data_dir']}/val_metadata.json"
    )
    test_dataset = KolamDataset(
        f"{CONFIG['data_dir']}/test_features.npy",
        f"{CONFIG['data_dir']}/test_metadata.json"
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val:   {len(val_dataset)} samples")
    print(f"   Test:  {len(test_dataset)} samples")
    
    # Analyze class distribution
    print("\n📈 Class distribution:")
    train_counts = Counter(train_dataset.labels.numpy())
    for i, name in enumerate(CLASS_NAMES):
        count = train_counts[i]
        print(f"   {name:15}: {count:4} samples ({100*count/len(train_dataset):.1f}%)")
    
    # Create data loaders
    print("\n⚖️  Creating balanced samplers...")
    if CONFIG['use_weighted_sampler']:
        train_sampler = get_weighted_sampler(train_dataset.labels)
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                 sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                 shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Create model
    print("\n🧠 Creating improved model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = train_dataset.features.shape[1]
    
    model = ImprovedKolamClassifier(
        input_dim=input_dim,
        num_classes=4,
        hidden_dims=CONFIG['hidden_dims'],
        dropout_rates=CONFIG['dropout_rates']
    ).to(device)
    
    print(f"   Architecture: {input_dim} → {' → '.join(map(str, CONFIG['hidden_dims']))} → 4")
    print(f"   Device: {device}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    if CONFIG['use_focal_loss']:
        criterion = FocalLoss(
            alpha=CONFIG['focal_alpha'],
            gamma=CONFIG['focal_gamma'],
            num_classes=4
        )
        print(f"   Loss: Focal Loss (α={CONFIG['focal_alpha']}, γ={CONFIG['focal_gamma']})")
    else:
        class_weights = get_class_weights(train_dataset.labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print(f"   Loss: Weighted Cross Entropy")
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    print("\n🚀 Starting training...")
    print("   Monitoring: Macro F1-Score (not accuracy!)")
    print("")
    
    best_val_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    for epoch in range(CONFIG['num_epochs']):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{CONFIG['num_epochs']}: "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"Acc: {train_acc:5.2f}%/{val_acc:5.2f}% | "
              f"F1: {train_f1:.4f}/{val_f1:.4f}")
        
        # Learning rate scheduling
        scheduler.step(val_f1)
        
        # Save best model based on Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, f"{CONFIG['output_dir']}/best_model_balanced.pth")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Load best model and evaluate on test set
    print("\n📊 Evaluating best model on test set...")
    checkpoint = torch.load(f"{CONFIG['output_dir']}/best_model_balanced.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion, device)
    
    # Get detailed test predictions
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate detailed metrics
    per_class = calculate_per_class_metrics(all_labels, all_preds)
    
    # Print results
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n📈 BEST VALIDATION:")
    print(f"   Macro F1: {best_val_f1:.4f}")
    print(f"   Accuracy: {checkpoint['val_acc']:.2f}%")
    
    print(f"\n🎯 TEST SET RESULTS:")
    print(f"   Macro F1:  {test_f1:.4f}")
    print(f"   Accuracy:  {test_acc:.2f}%")
    print(f"   Loss:      {test_loss:.4f}")
    
    print(f"\n📊 PER-CLASS PERFORMANCE:")
    print(f"   {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>8}")
    print("   " + "-" * 63)
    for class_name, metrics in per_class.items():
        print(f"   {class_name:<15} "
              f"{metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} "
              f"{metrics['f1_score']:>10.4f} "
              f"{metrics['support']:>8}")
    
    # Save results
    results = {
        'config': CONFIG,
        'best_epoch': checkpoint['epoch'],
        'validation': {
            'macro_f1': float(best_val_f1),
            'accuracy': float(checkpoint['val_acc'])
        },
        'test': {
            'macro_f1': float(test_f1),
            'accuracy': float(test_acc),
            'loss': float(test_loss),
            'per_class_metrics': per_class
        },
        'history': history,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(f"{CONFIG['output_dir']}/balanced_training_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {CONFIG['output_dir']}/")
    print(f"   • best_model_balanced.pth")
    print(f"   • balanced_training_results.json")
    
    # Compare with baseline
    print("\n📊 COMPARISON WITH BASELINE:")
    try:
        with open('kolam_dataset/05_trained_models/model_info.json', 'r') as f:
            baseline = json.load(f)
        
        baseline_f1 = baseline['test_results']['macro_f1']
        baseline_acc = baseline['test_results']['accuracy'] * 100
        
        print(f"   Baseline:  F1={baseline_f1:.4f}, Acc={baseline_acc:.2f}%")
        print(f"   Balanced:  F1={test_f1:.4f}, Acc={test_acc:.2f}%")
        print(f"   ")
        print(f"   Improvement: F1 {'+' if test_f1 > baseline_f1 else ''}{test_f1 - baseline_f1:.4f} "
              f"({100*(test_f1 - baseline_f1)/baseline_f1:+.1f}%)")
    except:
        print("   (Baseline results not found)")
    
    print("\n" + "=" * 70)
    print("✅ Option A Complete! Model is now balanced.")
    print("=" * 70)

if __name__ == '__main__':
    main()
