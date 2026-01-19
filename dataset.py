import torch 

from torch.utils.data import Dataset as TorchDataset
from datasets import Dataset as HFDataset, load_dataset
from tokenizers import Tokenizer

from typing import Any, Dict


class IMDBDataset(TorchDataset):
    """
    Dataset class for IMDB movie review classification.
    Returns text reviews with their sentiment labels (0=negative, 1=positive).
    """

    def __init__(
            self, 
            dataset: HFDataset, 
            tokenizer: Tokenizer, 
            context_size: int
        ) -> None:
        """Initializing the IMDBDataset object.

        Args:
            dataset (HFDataset): 
                HuggingFace dataset with 'text' and 'label' columns.
                label: 0 for negative, 1 for positive review.
                text: the movie review text.
            tokenizer (Tokenizer): Tokenizer for the text.
            context_size (int): Maximum allowed length of a review (in tokens).
        """
        super().__init__()

        # Initializing context size.
        self.context_size = context_size

        # Initializing the dataset.
        self.dataset = dataset

        # Initializing the tokenizer.
        self.tokenizer = tokenizer

        # Initializing special tokens
        # CLS token will be used for classification (at the beginning)
        self.cls_token = torch.tensor([tokenizer.token_to_id('[CLS]')], dtype=torch.int64)
        
        # End of sentence token signifies the end of a review.
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)

        # Padding token fills empty spaces for reviews shorter than context size.
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)


    def __len__(self) -> int:
        """
        Returns:
            int: Number of reviews in the dataset.
        """
        return len(self.dataset)
    
    
    def __getitem__(
            self, 
            index: int
        ) -> Dict[str, Any]:
        """Gets the review and label from the dataset at a specified index.

        Args:
            index (int): Index at which to return the element from the dataset.

        Returns:
            Dict[str, Any]: A dictionary with fields:
                input_ids: 
                    Input to be fed to the encoder. 
                    Tensor of dimension (context_size)
                attention_mask:
                    Mask for the encoder, that will mask padding tokens.
                    Tensor of dimension (1, 1, context_size)
                label:
                    Sentiment label (0 or 1).
                    Tensor (scalar)
                text:
                    Original review text.
        """
        # Get the index-th row of the dataset.
        item = self.dataset[index]

        # Get the review text and label.
        text = item['text']
        label = item['label']

        # Tokenize the text.
        input_tokens = self.tokenizer.encode(text).ids

        # Calculate number of padding tokens needed.
        # Input format: [CLS] token[0] token[1] ... token[N] [EOS] [PAD] ... [PAD]
        num_padding_tokens = self.context_size - len(input_tokens) - 2
        
        # If the review is too long, truncate it.
        if num_padding_tokens < 0:
            input_tokens = input_tokens[:self.context_size - 2]
            num_padding_tokens = 0
        
        # Build input: [CLS] tokens [EOS] [PAD] ... [PAD]
        input_ids = torch.cat(
            [
                self.cls_token,
                torch.tensor(input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        # Make sure the tensor dimensions are correct.
        assert input_ids.size(0) == self.context_size

        # Return the appropriate values.
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
            "text": text
        }


def load_imdb_data() -> tuple[HFDataset, HFDataset, HFDataset]:
    """
    Loads the IMDB dataset from HuggingFace.
    
    Returns:
        tuple: (train_dataset, test_dataset, unsupervised_dataset)
            Each dataset has 'text' and 'label' fields.
            label: 0 for negative, 1 for positive
            text: movie review text
    """
    # Load the IMDB dataset from HuggingFace
    dataset = load_dataset("stanfordnlp/imdb")
    
    # The IMDB dataset has 'train', 'test', and 'unsupervised' splits
    # For supervised binary classification, we'll use 'train' and 'test'
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    unsupervised_dataset = dataset['unsupervised'] if 'unsupervised' in dataset else None
    
    print(f"Loaded IMDB dataset:")
    print(f"  Train: {len(train_dataset)} reviews")
    print(f"  Test: {len(test_dataset)} reviews")
    if unsupervised_dataset:
        print(f"  Unsupervised: {len(unsupervised_dataset)} reviews")
    
    return train_dataset, test_dataset, unsupervised_dataset

