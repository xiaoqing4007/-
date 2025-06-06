import torch
import torch.nn as nn
import math

class SimpleTransformer(nn.Module):
    def __init__(self, embd_dim, num_heads, ff_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embd_dim, nhead=num_heads, dim_feedforward=ff_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

embed_dim = 64
num_heads = 4
ff_dim = 128
num_layers = 2
input_tensor = torch.randn(10, 32, embed_dim)

model = SimpleTransformer(embed_dim, num_heads, ff_dim, num_layers)
output = model(input_tensor)
print("Output:", output.shape)