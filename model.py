import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Class for handling the input embeddings of tokens.
    """

    def __init__(
            self, 
            model_dimension: int, 
            vocab_size: int
        ) -> None:
        """Initializing the InputEmbeddings object."""
        super().__init__()

        # Initialize parameters.
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size

        # Initialize the embedding.
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x) -> torch.Tensor:
        """
        Translates the token into it's embedding.
        """
        return self.embedding(x) * math.sqrt(self.model_dimension)
    
    
class PositionalEncoding(nn.Module):
    """
    Class for handling the positional embeddings of tokens.
    """

    def __init__(
            self, 
            model_dimension: int, 
            context_size: int, 
            dropout: float
        ) -> None:
        """Initializing the PositionalEncoding object."""
        super().__init__()

        # Initialize parameters
        self.model_dimension = model_dimension
        self.context_size = context_size
        self.dropout = nn.Dropout(dropout)
        
        # Placeholder matrix for positional encodings
        positional_encodings = torch.zeros(context_size, model_dimension) # (context_size, model_dimension)
        # Vector [0, 1, 2, ..., context_size - 1]
        position = torch.arange(0, context_size, dtype = torch.float).unsqueeze(1) # (context_size, 1)
        # Division term from Attention is all you need, with weird math to improve stability
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension)) # (model_dimension / 2)
        # Apply sine to even indices
        positional_encodings[:, 0::2] = torch.sin(position * div_term) # sin(position * 10000 ^ (2i / model_dimension))
        # Apply cosine to odd indices
        positional_encodings[:, 1::2] = torch.cos(position * div_term) # cos(position * 10000 ^ (2i / model_dimension))

        # Add a dimension to support batches to positional encodings
        positional_encodings = positional_encodings.unsqueeze(0) # (1, context_size, model_dimension)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', positional_encodings)

    def forward(self, x):
        """
        Adds the positional encodings to input embeddings of a 
        given token, and applies dropout for regularization.
        """
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
        
class LayerNormalization(nn.Module):
    """
    Class for handling the normalization of vectors in a given layer.
    """

    def __init__(
            self, 
            features: int, 
            eps: float = 10**-6
        ) -> None:
        """Initializing the LayerNormalization object."""
        super().__init__()

        # Initialize parameters.
        # Eps is a small number that improves the numerical stability (of divisions with small numbers)
        self.eps = eps
        # Alpha and bias are learnable parameters that adjust the normalization as seen below.
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        Applies the normalization to a given embedding.
        """
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
    
class MultiHeadAttentionBlock(nn.Module):
    """
    Class for handling the multihead attention.
    """

    def __init__(
            self, 
            model_dimension: int, 
            heads: int, 
            dropout: float
        ) -> None:
        """Initializing the MultiHeadAttentionBlock object."""
        super().__init__()

        # Initialize parameters.
        self.model_dimension = model_dimension
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        # Ensure the model_dimension can be evenly split into heads.
        assert model_dimension % heads == 0, "model_dimension is not divisible by the number of heads."

        # Initialize the key and query vector dimension.
        self.head_dimension = model_dimension // heads

        # Initialize the key, query and value matrices.
        self.w_q = nn.Linear(model_dimension, model_dimension)
        self.w_k = nn.Linear(model_dimension, model_dimension)
        self.w_v = nn.Linear(model_dimension, model_dimension)

        # Initialize the output matrix.
        self.w_o = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(
            query, 
            key, 
            value, 
            mask, 
            dropout: nn.Dropout
        ):
        """
        Perform a masked multi head attention on the given matrices.
        Apply the formula from the "Attention is all you need".
        Attention(Q, K, V) = softmax(QK^T / sqrt(head_dimension))V
        head_i = Attention(QWi^Q, KWi^K, VWi^V)
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        """
        head_dimension = query.shape[-1]

        # Calculate the scores by multiplying the matrices.
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension) # (batch, heads, context_size, context_size)

        # If mask exists then mask the scores.
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # Apply softmax to the scores.
        attention_scores = attention_scores.softmax(dim = -1)

        # Apply dropout for regularization.
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Return the attention scores and values.
        return (attention_scores @ value), attention_scores # (batch, heads, context_size, head_dimension)

    def forward(self, q, k, v, mask):
        """
        Apply the multi-headed attention to the given inputs.
        Can be used for both encoder and decoder, the inputs determine which.
        """

        # Calculate the tensors. These serve as Q', K' and V'.
        query = self.w_q(q) # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)
        key = self.w_k(k) # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)
        value = self.w_v(v) # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)

        # Convert the tensors to the appropriate format.
        # (batch, context_size, model_dimension) --> (batch, context_size, heads, head_dimension) --> (batch, heads, context_size, head_dimension)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.head_dimension).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
         
        # Concatenate the heads.
        # (batch, heads, context_size, head_dimension) --> (batch, context_size, heads, head_dimension) -> (batch, context_size, model_dimension)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dimension)

        # Multiply by the output matrix and return.
        # (batch, context_size, model_dimension) --> (batch, context_size, model_dimension)
        return self.w_o(x)
    
    
class FeedForwardBlock(nn.Module):
    """
    Class for handling the feed forward neural networks.
    """

    def __init__(
            self, 
            model_dimension: int, 
            feed_forward_dimension: int, 
            dropout: float
        ) -> None:
        """Initializing the FeedForwardBlock object."""
        super().__init__()
        
        # Initialize the parameters
        self.linear_1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dimension, model_dimension)

    def forward(self, x):
        """
        Apply the feed forward to the given input.
        FNN(x) = ReLU(xW_1 + b_1)W_2 + b_2
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
    
class ResidualConnection(nn.Module):
    """
    Class for handling the residual connections in the model.
    Serves as a connection between the input and the LayerNormalization object.
    """

    def __init__(
            self, 
            features: int, 
            dropout: float
        ) -> None:
        """Initializing the ResidualConnection object."""
        super().__init__()

        # Initialize the parameters.
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """Apply the layer normalization to the input and input passed through the sublayer."""
        # The paper says to first apply the sublayer, and then the normalization, but in practice, this order works better.
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class EncoderBlock(nn.Module):
    """
    Class for handling one iteration of the encoder.
    """

    def __init__(
            self, 
            features: int, 
            self_attention_block: MultiHeadAttentionBlock, 
            feed_forward_block: FeedForwardBlock, 
            dropout: float
        ) -> None:
        """Initializing the EncoderBlock object."""
        super().__init__()

        # Initialize the building blocks of an encoder.
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Two residual connections, one for feed forward, and one for self attention.
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        """Generate the output of a single iteration of the encoder."""

        # Self attention, which means the inputs to the multiheaded attention block are just the inputs to the encoder block.
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))

        # Feed forward.
        x = self.residual_connections[1](x, self.feed_forward_block)

        return x
    
    
class Encoder(nn.Module):
    """
    Class for handling all the iterations of the encoder.
    """

    def __init__(
            self, 
            features: int, 
            layers: nn.ModuleList
        ) -> None:
        """Initializing the Encoder object."""
        super().__init__()

        # Initialize the EncoderBlock layers.
        self.layers = layers
        # Initialize the normalization layer.
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """Generate the output of the encoder."""

        # Iterate through all the layers (EncoderBlocks).
        for layer in self.layers:
            x = layer(x, mask)

        # Normalize the output.
        return self.norm(x)
    
    
class ClassificationHead(nn.Module):
    """
    Classification head that takes the [CLS] token embedding and outputs class probabilities.
    """

    def __init__(
            self, 
            model_dimension: int, 
            num_classes: int,
            dropout: float = 0.1
        ) -> None:
        """Initializing the ClassificationHead object."""
        super().__init__()

        # Use a feed-forward network for classification
        # [CLS] embedding -> hidden layer -> num_classes logits
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(model_dimension, num_classes)

    def forward(self, x):
        """
        Extract [CLS] token (first token) and classify.
        
        Args:
            x: Encoder output of shape (batch_size, context_size, model_dimension)
        
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Take the [CLS] token embedding (first token in sequence)
        cls_embedding = x[:, 0, :]  # (batch_size, model_dimension)
        
        # Apply dropout for regularization
        cls_embedding = self.dropout(cls_embedding)
        
        # Project to num_classes
        logits = self.linear(cls_embedding)  # (batch_size, num_classes)
        
        return logits
    
    
class TransformerClassifier(nn.Module):
    """
    Encoder-only Transformer for binary classification.
    Uses only the encoder part and a classification head.
    """
    
    def __init__(
            self, 
            encoder: Encoder, 
            embed: InputEmbeddings, 
            pos: PositionalEncoding, 
            classification_head: ClassificationHead
        ) -> None:
        """Initializing the TransformerClassifier object."""
        super().__init__()

        # Initialize the building blocks (encoder-only)
        self.encoder = encoder
        self.embed = embed
        self.pos = pos
        self.classification_head = classification_head

    def encode(self, input_ids, attention_mask):
        """Generate the output of the encoder.
        
        Args:
            input_ids: Token IDs of shape (batch_size, context_size)
            attention_mask: Attention mask of shape (batch_size, 1, 1, context_size)
            
        Returns:
            Encoder output of shape (batch_size, context_size, model_dimension)
        """
        x = self.embed(input_ids)
        x = self.pos(x)
        return self.encoder(x, attention_mask)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for classification.
        
        Args:
            input_ids: Token IDs of shape (batch_size, context_size)
            attention_mask: Attention mask of shape (batch_size, 1, 1, context_size)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Encode the input
        encoder_output = self.encode(input_ids, attention_mask)
        
        # Classify using [CLS] token
        logits = self.classification_head(encoder_output)
        
        return logits
    
    
def build_transformer_classifier(
        vocab_size: int, 
        context_size: int, 
        model_dimension: int = 256, 
        number_of_blocks: int = 4, 
        heads: int = 8, 
        dropout: float = 0.1, 
        feed_forward_dimension: int = 1024,
        num_classes: int = 2
    ) -> TransformerClassifier:
    """Build an encoder-only transformer for classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        context_size (int): Maximum allowed sentence size.
        model_dimension (int, optional): Dimension of the embedding space. Defaults to 256.
        number_of_blocks (int, optional): Number of encoder blocks. Defaults to 4.
        heads (int, optional): Number of heads for multihead attention. Defaults to 8.
        dropout (float, optional): Rate of dropout for regularization. Defaults to 0.1.
        feed_forward_dimension (int, optional): Dimension of the hidden layer in feed forward. Defaults to 1024.
        num_classes (int, optional): Number of output classes. Defaults to 2 (binary classification).

    Returns:
        TransformerClassifier: An initialized transformer classifier with the specified parameters.
    """
    # Get the input embeddings
    embed = InputEmbeddings(model_dimension, vocab_size)

    # Get the positional encodings
    pos = PositionalEncoding(model_dimension, context_size, dropout)

    # Build the encoder blocks
    encoder_blocks = []
    for _ in range(number_of_blocks):
        encoder_self_attention_block = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        encoder_block = EncoderBlock(model_dimension, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Build the encoder
    encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))

    # Build the classification head
    classification_head = ClassificationHead(model_dimension, num_classes, dropout)

    # Build the transformer classifier
    transformer_classifier = TransformerClassifier(encoder, embed, pos, classification_head)

    # Initialize transformer parameters with Xavier uniform initialization
    for p in transformer_classifier.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_classifier


def get_model(
        config, 
        vocab_size: int
    ) -> TransformerClassifier:
    """
    Build the transformer classifier from the config file with given vocabulary size,
    using the model.py::build_transformer_classifier function.

    Args:
        config: A config file.
        vocab_size (int): Vocabulary size of the tokenizer.

    Returns:
        TransformerClassifier: An initialized transformer classifier model.
    """
    model = build_transformer_classifier(
        vocab_size=vocab_size, 
        context_size=config['context_size'], 
        model_dimension=config['model_dimension'],
        number_of_blocks=config['num_encoder_blocks'],
        heads=config['num_heads'],
        dropout=config['dropout'],
        feed_forward_dimension=config['feed_forward_dimension'],
        num_classes=config['num_classes']
    )
    
    return model
