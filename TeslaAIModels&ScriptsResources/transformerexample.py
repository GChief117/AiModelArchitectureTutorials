import torch
import torch.nn as nn

# ------------------------------------------------------------------
# MODEL DESCRIPTION
# ------------------------------------------------------------------
"""
Vision Transformer (ViT) configured for an output of size (Batch, 1470),
corresponding to a 7×7 grid with 30 values each (bounding boxes, object
confidence, and class probabilities). This is a simplistic demonstration
of how a ViT could replicate the "YOLO-style" 7×7 output structure.

Main Components:
1. Patch + Position Embedding:
   - The image (3×224×224) is split into 14×14=196 patches (each 16×16).
   - Each patch is flattened and projected to a 768‐dim embedding.
   - A learnable [CLS] token is prepended.
   - A learnable positional embedding is added to each token.

2. Transformer Encoder (repeated L=10 times):
   - Multi‐Head Self‐Attention (768 hidden dim, 12 heads).
   - Feed‐Forward Network (MLP) inside each block (dimensions ~3072).
   - Layer Normalization before or after each sub‐layer.

3. Final Head:
   - We extract the [CLS] token embedding from the final Transformer block.
   - A linear layer projects from 768 → 1470 for bounding box + class outputs.

Parameter Count:
- Patch Embedding + Position Embedding: ~0.7M
- Each Transformer Block (768‐dim, 12‐head, MLP of size 3072):
  ~7.08M params per block.
- With 10 blocks: ~70–71M just in the encoder.
- Final 768→1470 linear head: ~1.13M
- Total is ~72–73M, slightly below 75M. 
  You can add 1 more block or increase the MLP size to reach ~75M.

Input Shape:
- (Batch_Size, 3, 224, 224)

Output Shape:
- (Batch_Size, 1470)
"""

# ------------------------------------------------------------------
# HELPER MODULES
# ------------------------------------------------------------------
class PatchEmbedding(nn.Module):
    """
    Splits the input image into non-overlapping patches and projects
    each patch into a D-dimensional embedding.
    
    Here we use a simple Conv2d layer with stride=patch_size to emulate
    patch extraction + linear projection.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        # Calculate how many patches we'll have: (img_size / patch_size) ^ 2
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        """
        x shape: (Batch, 3, 224, 224)
        Output shape: (Batch, Num_Patches, Embed_Dim)
        """
        x = self.proj(x)  # (Batch, Embed_Dim, #patches_row, #patches_col)
        # Flatten spatial dimensions
        x = x.flatten(2)  # (Batch, Embed_Dim, Num_Patches)
        x = x.transpose(1, 2)  # (Batch, Num_Patches, Embed_Dim)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer encoder block:
      1) LayerNorm
      2) Multi-Head Self-Attention
      3) LayerNorm
      4) Feed-Forward MLP
    Residual connections are added around the attention and MLP sub-layers.
    """
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        
    def forward(self, x):
        # x shape: (Batch, Tokens, Embed_Dim)
        
        # --- Multi-Head Self-Attention ---
        residual = x
        x = self.norm1(x)
        # nn.MultiheadAttention expects (Batch, Tokens, Embed_Dim); batch_first=True
        attn_out, _ = self.attn(x, x, x)
        x = residual + attn_out  # Residual connection

        # --- Feed-Forward / MLP ---
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x  # Residual connection
        
        return x


# ------------------------------------------------------------------
# VISION TRANSFORMER
# ------------------------------------------------------------------
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1470,   # 7×7×30 for YOLO-style
        embed_dim=768,
        depth=10,           # Number of Transformer blocks
        num_heads=12,
        mlp_ratio=4.0
    ):
        super().__init__()
        
        # ----------------------------------------------------------
        # PATCH + POSITION EMBEDDING
        # ----------------------------------------------------------
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim
        )
        # Add a learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional embedding for (num_patches + 1) tokens
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # ----------------------------------------------------------
        # TRANSFORMER ENCODER
        # ----------------------------------------------------------
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # ----------------------------------------------------------
        # FINAL HEAD
        # ----------------------------------------------------------
        # Instead of classification, we output 1470 for bounding boxes/class
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=1e-6)
        nn.init.normal_(self.pos_embed, std=1e-6)
        # You can also initialize linear layers more precisely here if desired.

    def forward(self, x):
        """
        Forward Pass:
        1) Convert image to patch embeddings.
        2) Add [CLS] token + positional embeddings.
        3) Pass through L Transformer blocks.
        4) Take the final [CLS] token and map to (Batch, 1470).
        """
        B = x.size(0)
        
        # (Batch, num_patches, embed_dim)
        x = self.patch_embed(x)
        
        # prepend the CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (Batch, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (Batch, num_patches+1, embed_dim)
        
        # add position embedding
        x = x + self.pos_embed[:, : x.size(1), :]
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        # Final norm
        x = self.norm(x)
        
        # Extract only the CLS token, shape: (Batch, embed_dim)
        cls_out = x[:, 0]
        
        # Final linear head
        out = self.head(cls_out)  # (Batch, 1470)
        return out


# ------------------------------------------------------------------
# MODEL CREATION & PARAMETER COUNT
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=7 * 7 * 30,  # 1470
        embed_dim=768,
        depth=10,                # 10 Transformer blocks
        num_heads=12,
        mlp_ratio=4.0
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"VisionTransformer Total Parameters: {total_params:,}")  
    # Expect ~72–73 million with these settings.

    # Example forward pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch_Size=2
    output = model(dummy_input)
    print("Output shape:", output.shape)  # (2, 1470)

    # ------------------------------------------------------------------
    # TRAINING CONFIGURATION
    # ------------------------------------------------------------------
    """
    - Batch Size: e.g., 16 or 32 for typical GPU memory.
    - Loss: Could combine bounding box regression (MSE or Smooth L1) 
      and classification (CrossEntropy or BCE) depending on how you 
      split the 1470 outputs.
    - Optimizer: Adam or AdamW commonly used for Transformers.
    - Learning Rate: 1e-4 or 1e-3 (tune as needed).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
