from torch import nn, Tensor
from zeta import (
    MultiQueryAttention,
    FeedForward,
    patch_img,
)


class GridAttention(nn.Module):
    """
    GridAttention module applies attention mechanism on a grid of input features.

    Args:
        dim (int): The dimension of the input features.
        heads (int, optional): The number of attention heads. Defaults to 8.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads

        # Projection
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the GridAttention module.

        Args:
            x (Tensor): The input tensor of shape (batch_size, sequence_length, dim).

        Returns:
            Tensor: The output tensor after applying attention mechanism.
        """
        b, s, d = x.shape

        k = self.proj(x)
        v = self.proj(x)

        # Average pool
        q = nn.AdaptiveAvgPool1d(d)(x)
        print(x.shape)

        out, _, _ = MultiQueryAttention(d, self.heads)(q + k + v)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ffn_dim: int = 4,
        dropout: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # ffn
        self.ffn = FeedForward(
            dim,
            dim,
            ffn_dim,
            swish=True,
        )

        # Attention
        self.attn = MultiQueryAttention(dim, heads, *args, **kwargs)

        # Norms
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.norm(x)

        # Multi-Query Attention
        out, _, _ = self.attn(x)

        # Add and Norm
        out = self.norm(out) + residual

        # 2nd path
        residual_two = out

        # FFN
        ffd = self.norm(self.ffn(out))

        return ffd + residual_two


class AdaptiveTokenMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout

        # grid attention
        self.attn = GridAttention(dim, heads)

        # Ffn
        self.ffn = FeedForward(dim, dim, 4, swish=True)

    def forward(self, x: Tensor) -> Tensor:
        # Grid Attention
        grid = self.attn(x)

        # Ffn
        ffn = self.ffn(grid)

        return ffn


class TransformerBlocks(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        depth: int = 9,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth

        # transformer Blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Loop through the blocks
        for block in self.blocks:
            x = block(x)

        return x


class Vitar(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.1,
        depth: int = 9,
        patch_size: int = 16,
        image_size: int = 224,
        channels: int = 3,
        ffn_dim: int = 4,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dropout = dropout
        self.depth = depth
        self.patch_size = patch_size
        self.image_size = image_size
        self.channels = channels
        self.ffn_dim = ffn_dim
        self.num_classes = num_classes

        # Transformer Blocks
        self.transformer_blocks = TransformerBlocks(
            dim,
            heads,
            dropout,
            depth,
        )

        # Norm
        self.to_latent = nn.Identity()
        self.linear_head = nn.Linear(dim, num_classes)

    # def classifier(self, x: Tensor) -> Tensor:
    #     x = x.mean(dim = 1)

    #     x = self.to_latent(x)

    #     return self.linear_head(x)

    def forward(self, x) -> Tensor:
        # Embed the image -> (B, S, D)
        x = patch_img(x, self.patch_size)
        print(x.shape)

        b, s, d = x.shape

        # Norm
        # norm = nn.LayerNorm(d)

        # Adaptive token merger
        out = AdaptiveTokenMerger(d, self.heads, self.dropout)(x)
        print(out.shape)

        # Transformer Blocks
        transformed = TransformerBlocks(
            d,
            self.heads,
            self.dropout,
            self.depth,
        )(out)
        print(transformed.shape)

        # Start of classifier
        x = transformed.mean(dim=1)

        x = self.to_latent(x)

        return nn.Linear(d, self.num_classes)(x)
