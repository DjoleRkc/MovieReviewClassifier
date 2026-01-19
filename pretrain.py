"""
Pre-training script using unsupervised IMDB reviews.
Uses Masked Language Modeling (MLM) like BERT.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import random

from tokenizers import Tokenizer
from datasets import Dataset as HFDataset
from model import build_transformer_classifier
from config import get_config
from dataset import load_imdb_data
from train import get_or_build_tokenizer


class MaskedLanguageModelingDataset(Dataset):
    """
    Dataset for Masked Language Modeling pre-training.
    Randomly masks 15% of tokens and trains model to predict them.
    """
    
    def __init__(self, dataset, tokenizer, context_size, mask_prob=0.15):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.mask_prob = mask_prob
        
        # Special tokens
        self.cls_token = tokenizer.token_to_id('[CLS]')
        self.eos_token = tokenizer.token_to_id('[EOS]')
        self.pad_token = tokenizer.token_to_id('[PAD]')
        self.unk_token = tokenizer.token_to_id('[UNK]')
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        # Get text
        text = self.dataset[index]['text']
        
        # Tokenize
        tokens = self.tokenizer.encode(text).ids
        
        # Truncate if needed
        if len(tokens) > self.context_size - 2:
            tokens = tokens[:self.context_size - 2]
        
        # Build sequence: [CLS] tokens [EOS] [PAD]...
        num_padding = self.context_size - len(tokens) - 2
        input_ids = [self.cls_token] + tokens + [self.eos_token] + [self.pad_token] * num_padding
        
        # Create labels (copy of input_ids)
        labels = input_ids.copy()
        
        # Mask tokens (15% random masking, but not special tokens)
        masked_indices = []
        for i in range(1, len(tokens) + 1):  # Skip [CLS], mask only actual tokens
            if random.random() < self.mask_prob:
                masked_indices.append(i)
                # 80% replace with [UNK], 10% random, 10% keep same
                rand = random.random()
                if rand < 0.8:
                    input_ids[i] = self.unk_token
                elif rand < 0.9:
                    input_ids[i] = random.randint(0, self.tokenizer.get_vocab_size() - 1)
                # else: keep original (10%)
        
        # Set non-masked positions to -100 (ignored by loss)
        for i in range(len(labels)):
            if i not in masked_indices:
                labels[i] = -100
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = (input_ids != self.pad_token).unsqueeze(0).unsqueeze(0).int()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class MLMTransformer(nn.Module):
    """
    Transformer for Masked Language Modeling.
    Predicts masked tokens instead of classification.
    """
    
    def __init__(self, encoder, embed, pos, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.embed = embed
        self.pos = pos
        # MLM head: project back to vocabulary
        self.mlm_head = nn.Linear(embed.model_dimension, vocab_size)
        
    def forward(self, input_ids, attention_mask):
        # Encode
        x = self.embed(input_ids)
        x = self.pos(x)
        encoder_output = self.encoder(x, attention_mask)
        
        # Predict tokens at all positions
        logits = self.mlm_head(encoder_output)
        return logits


def pretrain_mlm(config, num_epochs=5):
    """
    Pre-train the encoder using Masked Language Modeling.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')
    
    # Load unsupervised data
    _, _, unsupervised_dataset = load_imdb_data()
    
    if unsupervised_dataset is None:
        print("No unsupervised data available!")
        return
    
    print(f"Unsupervised reviews: {len(unsupervised_dataset)}")
    
    # Build tokenizer (or load existing)
    # We'll use a combined dataset for tokenizer building
    train_dataset, _, _ = load_imdb_data()
    from torch.utils.data import random_split
    train_subset, _ = random_split(train_dataset, [int(0.9 * len(train_dataset)), len(train_dataset) - int(0.9 * len(train_dataset))])
    tokenizer = get_or_build_tokenizer(config, train_subset, force_rewrite=False)
    
    # Create MLM dataset
    mlm_dataset = MaskedLanguageModelingDataset(
        unsupervised_dataset, 
        tokenizer, 
        config['context_size']
    )
    
    mlm_dataloader = DataLoader(
        mlm_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    # Build model (just encoder + MLM head)
    from model import InputEmbeddings, PositionalEncoding, Encoder, EncoderBlock, MultiHeadAttentionBlock, FeedForwardBlock
    
    embed = InputEmbeddings(config['model_dimension'], tokenizer.get_vocab_size())
    pos = PositionalEncoding(config['model_dimension'], config['context_size'], config['dropout'])
    
    encoder_blocks = []
    for _ in range(config['num_encoder_blocks']):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            config['model_dimension'], 
            config['num_heads'], 
            config['dropout']
        )
        feed_forward_block = FeedForwardBlock(
            config['model_dimension'], 
            config['feed_forward_dimension'], 
            config['dropout']
        )
        encoder_block = EncoderBlock(
            config['model_dimension'], 
            encoder_self_attention_block, 
            feed_forward_block, 
            config['dropout']
        )
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(config['model_dimension'], nn.ModuleList(encoder_blocks))
    
    model = MLMTransformer(encoder, embed, pos, tokenizer.get_vocab_size()).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], eps=1e-9)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100).to(device)
    
    writer = SummaryWriter(config['experiment_name'] + '_pretrain')
    
    global_step = 0
    
    # Pre-training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        batch_iterator = tqdm(mlm_dataloader, desc=f"Pre-train Epoch {epoch+1}/{num_epochs}")
        
        for batch in batch_iterator:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Calculate loss (only on masked positions)
            loss = loss_function(logits.view(-1, tokenizer.get_vocab_size()), labels.view(-1))
            
            # Calculate accuracy on masked tokens
            predictions = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct_predictions += ((predictions == labels) & mask).sum().item()
            total_predictions += mask.sum().item()
            
            epoch_loss += loss.item()
            
            batch_iterator.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct_predictions / max(total_predictions, 1):.2f}%"
            })
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            writer.add_scalar('pretrain_loss', loss.item(), global_step)
            writer.add_scalar('pretrain_accuracy', 100.0 * correct_predictions / max(total_predictions, 1), global_step)
            writer.flush()
            
            global_step += 1
        
        avg_loss = epoch_loss / len(mlm_dataloader)
        accuracy = 100.0 * correct_predictions / total_predictions
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}, MLM Accuracy: {accuracy:.2f}%")
    
    # Save pre-trained encoder
    Path('pretrained').mkdir(exist_ok=True)
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'embed_state_dict': model.embed.state_dict(),
        'pos_state_dict': model.pos.state_dict(),
        'vocab_size': tokenizer.get_vocab_size(),
        'config': config
    }, 'pretrained/encoder_pretrained.pt')
    
    print("\nPre-trained encoder saved to pretrained/encoder_pretrained.pt")
    print("Now you can fine-tune on labeled data using train.py with preload option!")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    config = get_config()
    pretrain_mlm(config, num_epochs=3)
