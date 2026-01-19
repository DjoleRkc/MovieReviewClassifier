# IMDB Movie Review Sentiment Classifier

A **transformer-based binary sentiment classifier** built from scratch for analyzing IMDB movie reviews. This project implements an encoder-only transformer architecture (similar to BERT) for classifying movie reviews as positive or negative sentiment.

## Project Overview

This sentiment analysis system uses a custom-built transformer model trained on the [Stanford IMDB dataset](https://huggingface.co/datasets/stanfordnlp/imdb) containing 50,000 movie reviews. The model learns to understand sentiment through self-attention mechanisms and achieves **~84% accuracy** on unseen test data.

### Key Features

- **Transformer architecture from scratch** - No pre-trained models, pure PyTorch implementation
- **Encoder-only design** - Optimized for classification tasks
- **Custom tokenizer** - 30,000 vocabulary size using BPE algorithm
- **Balanced training** - Class weighting to prevent prediction bias
- **Early stopping** - Intelligent training termination to prevent overfitting
- **TensorBoard integration** - Real-time training visualization
- **Per-class metrics** - Detailed accuracy breakdown for positive/negative classes

## Architecture

### Transformer Configuration
- **Model Type**: Encoder-only (BERT-style)
- **Embedding Dimension**: 256
- **Encoder Layers**: 4
- **Attention Heads**: 8 (multi-head self-attention)
- **Feed-Forward Dimension**: 1024
- **Context Length**: 256 tokens
- **Dropout Rate**: 0.1

### Key Components
1. **Input Embeddings** - Token-to-vector conversion
2. **Positional Encoding** - Sinusoidal position information
3. **Multi-Head Self-Attention** - Parallel attention mechanisms
4. **Feed-Forward Networks** - Non-linear transformations
5. **Layer Normalization** - Training stabilization
6. **Residual Connections** - Gradient flow optimization
7. **Classification Head** - `[CLS]` token pooling + linear layer

## Project Structure

```
MovieReviewClassificator/
├── train.py              # Training script with early stopping
├── evaluate.py           # Full test set evaluation
├── test.py              # Interactive testing interface
├── model.py             # Transformer architecture implementation
├── dataset.py           # IMDB dataset loading and preprocessing
├── config.py            # Hyperparameter configuration
├── pretrain.py          # Masked language modeling (optional)
├── requirements.txt     # Python dependencies
├── weights/             # Saved model checkpoints
│   ├── imdb_balanced_best.pt
│   └── ...
└── runs/                # TensorBoard logs
```

## 🧪 Technical Details

### Training Process
- **Optimizer**: AdamW (weight decay)
- **Loss Function**: Cross-Entropy with class weights [1.0, 1.3]
- **Learning Rate**: 1e-4
- **Batch Size**: 32
- **Training Time**: ~4-6 hours on CPU (6-8 epochs with early stopping)

### Dataset Split
- **Training**: 22,500 reviews (90%)
- **Validation**: 2,500 reviews (10%)
- **Testing**: 25,000 reviews (separate set)
- **Class Balance**: 50/50 positive/negative

### Special Tokens
- `[CLS]` - Classification token (aggregates sequence representation)
- `[PAD]` - Padding token for variable-length sequences
- `[EOS]` - End of sequence marker
- `[UNK]` - Unknown token for out-of-vocabulary words



## References

- Vaswani et al. (2017) - [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

**Note**: This is an educational implementation. For production use, consider using pre-trained models like BERT, RoBERTa, or DistilBERT which achieve 93-95% accuracy on this task.
