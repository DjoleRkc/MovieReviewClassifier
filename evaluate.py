"""
Evaluate the trained model on the full test dataset.
"""

import torch
from pathlib import Path
from tqdm import tqdm
from config import get_config, get_latest_weights
from train import get_dataset
from model import get_model


def evaluate_on_test_set():
    """
    Evaluate the model on the full IMDB test set.
    """
    print("\n" + "="*70)
    print("IMDB Classifier - Full Test Set Evaluation")
    print("="*70 + "\n")
    
    # Setup
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    _, _, test_dataloader, tokenizer = get_dataset(config)
    
    # Build and load model
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    # Use specific epoch or latest
    # Change to specific epoch like "09" to evaluate that model
    # Or use "best" to load the best model saved during training
    use_epoch = "best"  # Using best balanced model (auto-saved during training)
    
    if use_epoch:
        from config import get_weights_file_path
        model_filename = get_weights_file_path(config, use_epoch)
    else:
        model_filename = get_latest_weights(config)
    
    if not model_filename or not Path(model_filename).exists():
        print("Error: No trained model found! Train the model first.")
        return
    
    print(f"Loading model from: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # Show training info
    print(f"\nModel Info:")
    print(f"  Epoch: {state.get('epoch', 'unknown')}")
    if 'train_accuracy' in state:
        print(f"  Training Accuracy: {state['train_accuracy']:.2f}%")
    if 'val_accuracy' in state:
        print(f"  Validation Accuracy: {state['val_accuracy']:.2f}%")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    # Track per-class accuracy
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0
    
    print(f"\nEvaluating on {len(test_dataloader.dataset)} test samples...")
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
            
            # Overall accuracy
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for pred, label in zip(predictions, labels):
                if label == 1:  # Positive
                    total_pos += 1
                    if pred == label:
                        correct_pos += 1
                else:  # Negative
                    total_neg += 1
                    if pred == label:
                        correct_neg += 1
    
    # Calculate metrics
    overall_accuracy = 100.0 * correct / total
    pos_accuracy = 100.0 * correct_pos / total_pos if total_pos > 0 else 0
    neg_accuracy = 100.0 * correct_neg / total_neg if total_neg > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 70)
    print(f"{'Overall Test Accuracy':<30} {overall_accuracy:>9.2f}%")
    print(f"{'Correct Predictions':<30} {correct:>10,} / {total:,}")
    print()
    print(f"{'Positive Reviews Accuracy':<30} {pos_accuracy:>9.2f}%")
    print(f"{'Positive Correct':<30} {correct_pos:>10,} / {total_pos:,}")
    print()
    print(f"{'Negative Reviews Accuracy':<30} {neg_accuracy:>9.2f}%")
    print(f"{'Negative Correct':<30} {correct_neg:>10,} / {total_neg:,}")
    print("="*70 + "\n")
    
    return overall_accuracy


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    evaluate_on_test_set()
