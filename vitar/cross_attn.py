import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttention(nn.Module):
    def __init__(
        self, dim_in: int, num_heads: int = 8, dim_out: int = None
    ):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out if dim_out is not None else dim_in

        self.query_conv = nn.Conv2d(
            dim_in, self.dim_out, kernel_size=1
        )
        self.key_conv = nn.Conv2d(dim_in, self.dim_out, kernel_size=1)
        self.value_conv = nn.Conv2d(
            dim_in, self.dim_out, kernel_size=1
        )
        self.output_conv = nn.Conv2d(
            self.dim_out, dim_in, kernel_size=1
        )

        self.scale = (self.dim_out // num_heads) ** -0.5

    def forward(
        self, q: Tensor, v: Tensor, k: Tensor
    ) -> torch.Tensor:
        B, C, H, W = q.shape

        # Compute query, key, value
        queries = self.query_conv(q)  # (B, dim_out, H, W)
        keys = self.key_conv(k)  # (B, dim_out, H, W)
        values = self.value_conv(v)  # (B, dim_out, H, W)

        # Reshape and transpose for multi-head attention
        queries = queries.view(
            B, self.num_heads, self.dim_out // self.num_heads, H * W
        )
        keys = keys.view(
            B, self.num_heads, self.dim_out // self.num_heads, H * W
        )
        values = values.view(
            B, self.num_heads, self.dim_out // self.num_heads, H * W
        )

        queries = queries.permute(
            0, 1, 3, 2
        )  # (B, num_heads, H*W, dim_out // num_heads)
        keys = keys.permute(
            0, 1, 3, 2
        )  # (B, num_heads, H*W, dim_out // num_heads)
        values = values.permute(
            0, 1, 3, 2
        )  # (B, num_heads, H*W, dim_out // num_heads)

        # Compute scaled dot-product attention
        attention_scores = (
            torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        )  # (B, num_heads, H*W, H*W)
        attention_probs = F.softmax(
            attention_scores, dim=-1
        )  # (B, num_heads, H*W, H*W)

        attention_output = torch.matmul(
            attention_probs, values
        )  # (B, num_heads, H*W, dim_out // num_heads)

        # Reshape and combine heads
        attention_output = attention_output.permute(
            0, 1, 3, 2
        ).contiguous()  # (B, num_heads, dim_out // num_heads, H*W)
        attention_output = attention_output.view(
            B, self.dim_out, H, W
        )  # (B, dim_out, H, W)

        # Apply output projection
        output = self.output_conv(
            attention_output
        )  # (B, dim_in, H, W)

        return output


# # Example usage
# if __name__ == "__main__":
#     input_tensor = torch.randn(8, 64, 32, 32)  # Example input tensor with shape (batch, channels, height, width)
#     cross_attention_layer = CrossAttention(dim_in=64, num_heads=8)
#     output_tensor = cross_attention_layer(input_tensor)
#     print(output_tensor.shape)  # Output tensor will have shape (8, 64, 32, 32)
