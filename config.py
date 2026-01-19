from pathlib import Path
from typing import Dict, Any

def get_config():
    """
    Returns:
        A static dictionary of model configuration variables:
            batch_size (int): batch size of the model
            num_epochs (int): number of epochs of the model
            learning_rate (float): learning rate of the model
            context_size (int): maximum allowed sentence length (in tokens)
            model_dimension (int): dimension of the embedding vector space
            num_classes (int): number of output classes (2 for binary classification)
            model_folder (str): folder in which the weights will be saved
            model_basename (str): name of the model
            preload (int | None): epoch from which to load the weights
            tokenizer_file: file where the tokenizer is stored
            experiment_name: tensorboard experiment name
            seed: (int | None): seed of the model
            num_encoder_blocks (int): number of encoder blocks in the transformer
            num_heads (int): number of attention heads
            dropout (float): dropout rate for regularization
            feed_forward_dimension (int): dimension of feed forward network
    """
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "learning_rate": 1 * 10**-4,
        "context_size": 256,
        "model_dimension": 256,
        "num_classes": 2,
        "model_folder": "weights",
        "model_basename": "imdb_balanced_",  # New name to keep old models
        "preload": None,  # Train from scratch with class weights
        "tokenizer_file": "tokenizer_imdb.json",
        "experiment_name": "runs/imdb_balanced",  # New experiment name
        "seed": 561,
        "num_encoder_blocks": 4,
        "num_heads": 8,
        "dropout": 0.1,
        "feed_forward_dimension": 1024
    }

def get_weights_file_path(
        config, 
        epoch: str
    ) -> str:
    """
    Get the saved model weights from a file.

    Args:
        config: Config file.
        epoch (str): Epoch from which to load the weights.

    Returns:
        str: Path to the saved weights of the model.
    """

    # Find the appropriate file.
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"

    return str(Path('.') / model_folder / model_filename)

def get_latest_weights(config) -> str:
    """
    Get the latest saved model weights from a folder.

    Args:
        config: Config file.

    Returns:
        str: Path to the latest saved weights of the model.
    """

    # Find all the files in the folder.
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}*"
    model_filenames = list(Path(model_folder).glob(model_filename))

    # If the folder is empty then there is nothing to return.
    if len(model_filenames) == 0:
        return None
    
    # Define a key for sorting. Extracts the epoch int from the filename.
    def extract_epoch(filename):
        return int(filename.stem.split('_')[-1])
    
    # Sort the files by their epoch.
    model_filenames.sort(key = extract_epoch)

    return str(model_filenames[-1])
