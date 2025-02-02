import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
#  PART 1: MULTI-HEAD SELF-ATTENTION LAYER (Uses Softmax for Attention Weights)
# ================================================================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Layer:
    - Computes self-attention for multiple "heads" in parallel.
    - Uses Softmax for probability-based attention weights.

    Parameters:
    - embed_dim: The dimension of the input embeddings.
    - num_heads: Number of attention heads.

    Operations:
    - Projects input embeddings into queries, keys, and values.
    - Computes scaled dot-product attention.
    - Applies Softmax to normalize attention scores.
    - Merges attention outputs and projects them back to embedding space.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads."
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Learnable projection matrices for Q, K, V
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        Forward pass of Multi-Head Self-Attention.

        Inputs:
        - x: Tensor of shape (batch_size, seq_length, embed_dim)
        - mask: Optional mask to prevent attending to certain positions.

        Returns:
        - Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        batch_size, seq_length, embed_dim = x.shape

        # Project input to Queries, Keys, and Values
        qkv = self.qkv_proj(x).reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim).permute(2, 0, 1, 3)
        queries, keys, values = qkv.chunk(3, dim=-1)

        # Compute attention scores (scaled dot-product)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Apply optional mask (for masked attention)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply Softmax to normalize attention scores
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Compute attention output
        context = torch.matmul(attention_weights, values).permute(1, 2, 0, 3).reshape(batch_size, seq_length, embed_dim)
        
        return self.out_proj(context)

# ================================================================
#  PART 2: FEEDFORWARD NETWORK (FFN) WITH GELU
# ================================================================

class FeedForwardNetwork(nn.Module):
    """
    FeedForward Network (FFN):
    - Applies a non-linear transformation to each token representation.
    - Uses GELU instead of standard ReLU.

    Parameters:
    - embed_dim: Input and output dimension.
    - hidden_dim: Expanded dimension inside the feedforward network.

    Operations:
    - Applies Linear → GELU → Linear transformations.
    """
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.gelu = nn.GELU()  # ✅ Using GELU instead of ReLU or Leaky ReLU
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        """
        Forward pass of FFN.

        Inputs:
        - x: Tensor of shape (batch_size, seq_length, embed_dim)

        Returns:
        - Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        return self.fc2(self.gelu(self.fc1(x)))

# ================================================================
#  PART 3: TRANSFORMER ENCODER BLOCK (Combining Attention & FFN)
# ================================================================

class TransformerBlock(nn.Module):
    """
    Transformer Encoder Block:
    - Combines Multi-Head Self-Attention and FeedForward Network (FFN).
    - Uses Layer Normalization for stability.
    - Applies residual connections to improve gradient flow.

    Parameters:
    - embed_dim: The embedding dimension.
    - num_heads: The number of attention heads.
    - hidden_dim: The size of the hidden layer in FFN.

    This block is stacked multiple times in the full Transformer.
    """
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()

        # Multi-Head Self-Attention Layer
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)

        # Layer Normalization for stable learning
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward Network
        self.ffn = FeedForwardNetwork(embed_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Forward pass through Transformer Encoder Block.

        Inputs:
        - x: Input tensor of shape (batch_size, seq_length, embed_dim)
        - mask: Optional attention mask

        Returns:
        - Output tensor of shape (batch_size, seq_length, embed_dim)
        """
        # Apply Self-Attention with Residual Connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)  

        # Apply FeedForward Network with Residual Connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  

        return x

# ================================================================
#  PART 4: FULL TRANSFORMER MODEL (25 Billion Parameters)
# ================================================================

class TransformerModel(nn.Module):
    """
    Full Transformer Model:
    - Consists of Token Embeddings, Positional Encoding, Transformer Blocks, and Output Layer.
    - Uses GELU for all activation functions.

    Parameters:
    - vocab_size: Number of unique tokens in vocabulary.
    - embed_dim: Size of the word embeddings.
    - num_heads: Number of attention heads per layer.
    - num_layers: Number of stacked Transformer layers.
    - hidden_dim: Size of the hidden layer in FFN.
    """
    def __init__(self, vocab_size, embed_dim=12288, num_heads=96, num_layers=96, hidden_dim=49152):
        super().__init__()

        # Token Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional Encoding (learnable)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 4096, embed_dim))

        # Transformer Layers (Stacked Encoder Blocks)
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

        # Final projection layer to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass through the full Transformer.

        Inputs:
        - input_ids: Tensor of tokenized input (batch_size, seq_length)
        - attention_mask: Optional mask to prevent attending to padding tokens.

        Returns:
        - Logits of shape (batch_size, seq_length, vocab_size)
        """
        # Token Embeddings
        x = self.embedding(input_ids)

        # Add Positional Encoding
        x = x + self.positional_encoding[:, :x.size(1), :]

        # Pass through all Transformer Encoder Blocks
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # Output projection
        logits = self.fc_out(x)
        return logits

# ================================================================
#  PART 5: MODEL CONFIGURATION & TESTING
# ================================================================
if __name__ == "__main__":
    vocab_size = 50000
    max_seq_length = 4096
    model = TransformerModel(vocab_size)

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Transformer Model with {total_params:,} Parameters")

    input_ids = torch.randint(0, vocab_size, (2, max_seq_length))
    output = model(input_ids)
    print("Output shape:", output.shape)  # Expected: (batch_size, seq_length, vocab_size)
