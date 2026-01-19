"""
Test and inference script for IMDB movie review sentiment classification.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from tokenizers import Tokenizer
from model import get_model
from config import get_config, get_latest_weights


def load_trained_model(config):
    """
    Load a trained model from saved weights.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        tuple: (model, tokenizer, device)
    """
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load tokenizer
    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}. Train the model first!")
    
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.get_vocab_size()}")
    
    # Build model
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    # Load trained weights - Use best model (auto-saved during training)
    from config import get_weights_file_path
    best_epoch = "best"  # Best balanced model with class weights
    model_filename = get_weights_file_path(config, best_epoch)
    
    if not Path(model_filename).exists():
        raise FileNotFoundError(f"Best model not found at {model_filename}! Train the model first.")
    
    print(f"Loading BEST model from: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    
    # Print model info if available
    if 'val_accuracy' in state:
        print(f"Model validation accuracy: {state['val_accuracy']:.2f}%")
    
    model.eval()
    
    return model, tokenizer, device


def preprocess_text(text, tokenizer, context_size, device):
    """
    Preprocess a single text review for model input.
    
    Args:
        text (str): The review text.
        tokenizer: The tokenizer.
        context_size (int): Maximum sequence length.
        device: Device to put tensors on.
        
    Returns:
        tuple: (input_ids, attention_mask) tensors ready for the model
    """
    # Get special token IDs
    cls_token_id = tokenizer.token_to_id('[CLS]')
    eos_token_id = tokenizer.token_to_id('[EOS]')
    pad_token_id = tokenizer.token_to_id('[PAD]')
    
    # Tokenize the text
    tokens = tokenizer.encode(text).ids
    
    # Calculate padding
    num_padding = context_size - len(tokens) - 2
    
    # Truncate if too long
    if num_padding < 0:
        tokens = tokens[:context_size - 2]
        num_padding = 0
    
    # Build input: [CLS] tokens [EOS] [PAD]...[PAD]
    input_ids = [cls_token_id] + tokens + [eos_token_id] + [pad_token_id] * num_padding
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    # Create attention mask
    attention_mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(1).int()
    
    return input_ids, attention_mask


def classify_review(text, model, tokenizer, config, device):
    """
    Classify a single movie review as positive or negative.
    
    Args:
        text (str): The review text.
        model: The trained model.
        tokenizer: The tokenizer.
        config: Configuration dictionary.
        device: Device to run on.
        
    Returns:
        tuple: (prediction, confidence, probabilities)
            prediction: "Positive" or "Negative"
            confidence: float between 0 and 1
            probabilities: dict with 'negative' and 'positive' probabilities
    """
    model.eval()
    
    with torch.no_grad():
        # Preprocess
        input_ids, attention_mask = preprocess_text(text, tokenizer, config['context_size'], device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)[0]
        
        # Get prediction
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        
        prediction = "Positive" if pred_class == 1 else "Negative"
        
        probabilities = {
            'negative': probs[0].item(),
            'positive': probs[1].item()
        }
        
        return prediction, confidence, probabilities


def interactive_mode():
    """
    Interactive mode for testing individual reviews.
    """
    print("\n" + "="*70)
    print("IMDB Movie Review Sentiment Classifier - Interactive Mode")
    print("="*70)
    
    # Load model
    config = get_config()
    try:
        model, tokenizer, device = load_trained_model(config)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    print("\nModel loaded successfully!")
    print("Enter movie reviews to classify (or 'quit' to exit)\n")
    
    while True:
        print("-" * 70)
        review = input("Enter review: ").strip()
        
        if review.lower() in ['quit', 'exit', 'q']:
            print("\nExiting. Goodbye!")
            break
        
        if not review:
            print("Please enter a non-empty review.")
            continue
        
        # Classify
        prediction, confidence, probabilities = classify_review(
            review, model, tokenizer, config, device
        )
        
        # Display results
        print(f"\n{'='*70}")
        print(f"Review: {review[:100]}{'...' if len(review) > 100 else ''}")
        print(f"{'='*70}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilities:")
        print(f"  Negative: {probabilities['negative']:.2%}")
        print(f"  Positive: {probabilities['positive']:.2%}")
        print()


def test_examples():
    """
    Test the model on some example reviews.
    """
    print("\n" + "="*70)
    print("IMDB Movie Review Sentiment Classifier - Testing Examples")
    print("="*70)
    
    # Load model
    config = get_config()
    try:
        model, tokenizer, device = load_trained_model(config)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    
    # Example reviews
    examples = [
        "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout.",
        "Terrible waste of time. Poor acting, predictable plot, and boring dialogue.",
        "An absolute masterpiece! One of the best films I've ever seen.",
        "I hated every minute of it. Complete garbage.",
        "It was okay, nothing special but not terrible either.",
        "The cinematography was beautiful, but the story felt rushed and incomplete.",
        "I laughed, I cried, I was on the edge of my seat. Highly recommend!",
        "Save your money and skip this one. You'll regret watching it.",
    ]
    
    print("\nClassifying example reviews...\n")
    
    for i, review in enumerate(examples, 1):
        prediction, confidence, probabilities = classify_review(
            review, model, tokenizer, config, device
        )
        
        print(f"{i}. Review: {review}")
        print(f"   Prediction: {prediction} (confidence: {confidence:.2%})")
        print(f"   [Negative: {probabilities['negative']:.2%} | Positive: {probabilities['positive']:.2%}]")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # If arguments provided, classify them
        config = get_config()
        try:
            model, tokenizer, device = load_trained_model(config)
        except FileNotFoundError as e:
            print(f"\nError: {e}")
            sys.exit(1)
        
        review = " ".join(sys.argv[1:])
        prediction, confidence, probabilities = classify_review(
            review, model, tokenizer, config, device
        )
        
        print(f"\nReview: {review}")
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
        print(f"Probabilities: Negative={probabilities['negative']:.2%}, Positive={probabilities['positive']:.2%}")
    else:
        # Show menu
        print("\n" + "="*70)
        print("IMDB Movie Review Sentiment Classifier")
        print("="*70)
        print("\nChoose an option:")
        print("1. Interactive mode (enter reviews one by one)")
        print("2. Test on example reviews")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            interactive_mode()
        elif choice == "2":
            test_examples()
        elif choice == "3":
            print("Exiting. Goodbye!")
        else:
            print("Invalid choice. Exiting.")
