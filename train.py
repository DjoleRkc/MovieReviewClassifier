# Torch stuff
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Other files stuff
from dataset import IMDBDataset, load_imdb_data
from model import get_model
from config import get_weights_file_path, get_latest_weights, get_config

# HuggingFace stuff
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import Dataset as HFDataset

# Metrics stuff
import warnings

# Easy access stuff
from pathlib import Path
from tqdm import tqdm

# Set the random seed for this project, for reproducibility.
import random
SEED = get_config()["seed"]
torch.manual_seed(SEED)
random.seed(SEED)

            
def get_all_texts(
        dataset: HFDataset
    ):
    """
    Yields review texts from the provided IMDB dataset.

    Args:
        dataset (HFDataset): Dataset to iterate through with 'text' field.

    Yields:
        str: Review text from the dataset.
    """
    for item in dataset:
        yield item['text']
        

def get_or_build_tokenizer(
        config, 
        dataset: HFDataset, 
        force_rewrite: bool = False,
        min_frequency: int = 2,
        vocab_size: int = 30000
    ) -> Tokenizer:
    """ 
    If the path to tokenizer file is not specified in the config, or if
    we force rewrite, then build a tokenizer from scratch.
    Else, get the tokenizer from the specified file.

    Args:
        config: A config file.
        dataset (HFDataset): HuggingFace dataset of reviews to build the tokenizer from.
        force_rewrite (bool): If the function should disregard the config file.
        min_frequency (int): Minimum frequency of a word in the dataset to add it to the vocabulary.
        vocab_size (int): Maximum size of the vocabulary.

    Returns:
        Tokenizer: A tokenizer built from a vocabulary formed by review texts from the dataset.
    """
    # Get the path from config.
    tokenizer_path = Path(config['tokenizer_file'])

    # If such a path doesn't exist, or we force the rewrite, then build the tokenizer.
    if not Path.exists(tokenizer_path) or force_rewrite:

        # Initialize the tokenizer with the unknown token [UNK].
        # This tokenizer will consider full words to be tokens.
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()

        # Build a trainer with the specified special tokens
        # [CLS] for classification, [PAD] for padding, [EOS] for end of sequence
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[CLS]", "[EOS]"], 
            min_frequency=min_frequency, 
            vocab_size=vocab_size
        )
        
        # Train the tokenizer on the dataset
        tokenizer.train_from_iterator(get_all_texts(dataset), trainer=trainer)

        # Save the tokenizer to file.
        tokenizer.save(str(tokenizer_path))

    # Get the tokenizer from file.
    else: 
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    # Return the tokenizer and print the number of tokens in it.
    print(f"Tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


def get_dataset(config):
    """
    Initializes the training and validation datasets for IMDB classification.
    Initializes the tokenizer.

    Args:
        config: A config file.

    Returns:
        DataLoader: Training dataset dataloader.
        DataLoader: Validation dataset dataloader.
        DataLoader: Test dataset dataloader.
        Tokenizer: Tokenizer for the text.
    """
    # Load the IMDB data from HuggingFace
    train_dataset_raw, test_dataset_raw, _ = load_imdb_data()

    # Split training data into train and validation (90% train, 10% validation)
    dataset_size = len(train_dataset_raw)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size

    training_dataset_raw, validation_dataset_raw = random_split(
        train_dataset_raw, 
        [train_size, val_size]
    )

    # Build the tokenizer from the training data
    tokenizer = get_or_build_tokenizer(config, training_dataset_raw, force_rewrite=False)

    # Define the IMDBDataset objects for the training, validation and test datasets
    training_dataset = IMDBDataset(training_dataset_raw, tokenizer, config['context_size'])
    validation_dataset = IMDBDataset(validation_dataset_raw, tokenizer, config['context_size'])
    test_dataset = IMDBDataset(test_dataset_raw, tokenizer, config['context_size'])

    # Calculate the maximum length of reviews (for information)
    max_len = 0
    for item in train_dataset_raw:
        token_ids = tokenizer.encode(item['text']).ids
        max_len = max(max_len, len(token_ids))

    print(f"Max review length (in tokens): {max_len}")
    print(f"Training samples: {len(training_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Define the DataLoader objects for training, validation and test datasets
    training_dataloader = DataLoader(training_dataset, batch_size=config['batch_size'], shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return training_dataloader, validation_dataloader, test_dataloader, tokenizer


def load_pretrained_encoder(model, pretrained_path, device):
    """
    Load pre-trained encoder weights into the model.
    
    Args:
        model: The classifier model
        pretrained_path: Path to pre-trained weights
        device: Device to load on
    """
    if not Path(pretrained_path).exists():
        print(f"Pre-trained weights not found at {pretrained_path}")
        print("   Continuing without pre-training...")
        return model
    
    print(f"Loading pre-trained encoder from {pretrained_path}")
    pretrained = torch.load(pretrained_path, map_location=device)
    
    # Load encoder weights
    model.encoder.load_state_dict(pretrained['encoder_state_dict'])
    model.embed.load_state_dict(pretrained['embed_state_dict'])
    model.pos.load_state_dict(pretrained['pos_state_dict'])
    
    print("Pre-trained encoder loaded successfully!")
    return model


def train_model(config, use_pretrained=False):
    """
    Train the transformer classifier model with the given parameters.

    Args:
        config: A config file.
        use_pretrained: Whether to load pre-trained encoder weights.
    """
    # Use cuda if possible, otherwise use cpu.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')

    # Make the folder for the model weights.
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Get the datasets and define the model.
    training_dataloader, validation_dataloader, test_dataloader, tokenizer = get_dataset(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    # Load pre-trained weights if requested
    if use_pretrained:
        model = load_pretrained_encoder(model, 'pretrained/encoder_pretrained.pt', device)

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize the writer to visualize data.
    writer = SummaryWriter(config['experiment_name'])

    # Initialize the AdamW optimizer (better for transformers than Adam).
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    # Load a pretrained model if defined and if it exists.
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = get_latest_weights(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from the beginning.")

    # Define the loss function for binary classification.
    # Use class weights to balance the model's focus on both classes
    # Weight positive class more since model tends to be biased toward negative
    class_weights = torch.tensor([1.0, 1.3], device=device)  # [negative, positive]
    loss_function = nn.CrossEntropyLoss(weight=class_weights).to(device)
    print(f"Using class weights: negative={class_weights[0]:.1f}, positive={class_weights[1]:.1f}")

    # Early stopping setup
    best_val_accuracy = 0.0
    patience = 3  # Stop if no improvement for 3 epochs
    patience_counter = 0
    best_model_path = None

    # Run the epochs.
    for epoch in range(initial_epoch, config['num_epochs']):
        
        # Training phase
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        # Create a batch iterator and iterate through the batches.
        batch_iterator = tqdm(training_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        for batch in batch_iterator:

            # Move the tensors to the device.
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)

            # Calculate the loss
            loss = loss_function(logits, labels)
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            epoch_loss += loss.item()

            # Debug: Check class distribution every 50 batches
            if global_step % 50 == 0 and global_step > 0:
                pred_dist = torch.bincount(predictions, minlength=2)
                label_dist = torch.bincount(labels, minlength=2)
                print(f"\n[Debug Step {global_step}] Predictions: neg={pred_dist[0].item()}, pos={pred_dist[1].item()} | Labels: neg={label_dist[0].item()}, pos={label_dist[1].item()} | Loss: {loss.item():.4f}")

            # Update progress bar
            batch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct / total:.2f}%"
            })

            # Add loss to tensorboard.
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.add_scalar('train_accuracy', 100.0 * correct / total, global_step)
            writer.flush()

            # Backward pass and optimization
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()

            # Adjust the global step.
            global_step += 1

        # Calculate epoch statistics
        avg_epoch_loss = epoch_loss / len(training_dataloader)
        epoch_accuracy = 100.0 * correct / total
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase
        val_loss, val_accuracy = validate_model(model, validation_dataloader, loss_function, device)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")
        
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        writer.flush()

        # Save weights at certain 'milestone' epochs.
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        if epoch % 2 == 0 or epoch == config['num_epochs'] - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'train_accuracy': epoch_accuracy,
                'train_accuracy': epoch_accuracy
            }, model_filename)
            print(f"Model saved to {model_filename}")
        
        # Early stopping check
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            # Save best model
            best_model_path = get_weights_file_path(config, 'best')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'train_accuracy': epoch_accuracy,
                'train_accuracy': epoch_accuracy
            }, best_model_path)
            print(f"✅ New best model! Validation accuracy: {best_val_accuracy:.2f}% (saved to {best_model_path})")
        else:
            patience_counter += 1
            print(f"⚠️  No improvement for {patience_counter} epoch(s). Best: {best_val_accuracy:.2f}%")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {patience} epochs.")
                print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
                print(f"Loading best model from {best_model_path}")
                # Load best model for final test
                state = torch.load(best_model_path)
                model.load_state_dict(state['model_state_dict'])
                break
            
    # Final test evaluation
    print("\n" + "="*50)
    print("Running final test evaluation...")
    test_loss, test_accuracy = validate_model(model, test_dataloader, loss_function, device)
    print(f"Test - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    print("="*50)


def validate_model(model, dataloader, loss_function, device):
    """
    Validate the model on a dataset.
    
    Args:
        model: The transformer classifier model.
        dataloader: DataLoader for the validation/test set.
        loss_function: Loss function to use.
        device: Device to run on.
        
    Returns:
        tuple: (average_loss, accuracy_percentage)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    correct_pos = 0
    correct_neg = 0
    total_pos = 0
    total_neg = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = loss_function(logits, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            predictions = torch.argmax(logits, dim=1)
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
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    
    # Print per-class accuracy
    if total_pos > 0 and total_neg > 0:
        pos_acc = 100.0 * correct_pos / total_pos
        neg_acc = 100.0 * correct_neg / total_neg
        print(f"  Per-class: Positive={pos_acc:.2f}%, Negative={neg_acc:.2f}%")
    
    model.train()
    return avg_loss, accuracy


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    
    # Set use_pretrained=True to load pre-trained encoder
    # IMPORTANT: Set to False when using a different architecture!
    use_pretrained = False  # Set to True only if pre-training matches new architecture
    
    train_model(config, use_pretrained=use_pretrained)
