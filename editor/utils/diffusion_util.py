import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        diffusion_step = diffusion_step.to(self.embedding.device)
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)
        table = steps * frequencies
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2, transformer_fn=None, text_mode="token"):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        if transformer_fn is None:
            transformer_fn = get_torch_trans

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    textemb=config["textemb"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step, text_emb=None):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, text_emb=text_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)
        x = F.relu(x)
        x = self.output_projection2(x)
        return x.reshape(B, K, L)


class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, textemb=384):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.text_proj = nn.Linear(textemb, channels)
        self.cross_attn = nn.MultiheadAttention(embed_dim=channels, num_heads=nheads, batch_first=True)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, C, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 2, 1, 3).reshape(B * K, C, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.reshape(B, K, C, L).permute(0, 2, 1, 3).reshape(B, C, K * L)

    def forward_feature(self, y, base_shape):
        B, C, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, C, K, L).permute(0, 3, 1, 2).reshape(B * L, C, K)
        y = y.permute(2, 0, 1)
        y = self.feature_layer(y)
        y = y.permute(1, 2, 0).reshape(B, L, C, K).permute(0, 2, 3, 1)
        return y.reshape(B, C, K * L)

    def forward(self, x, cond_info, diffusion_emb, text_emb=None):
        B, C, K, L = x.shape
        base_shape = x.shape

        y = x.reshape(B, C, K * L)
        y = y + self.diffusion_projection(diffusion_emb).unsqueeze(-1)

        if text_emb is not None:
            tok_emb, tok_mask = text_emb
            mem = self.text_proj(tok_emb)
            q = y.permute(0, 2, 1)
            attn_out, _ = self.cross_attn(q, mem, mem, key_padding_mask=~tok_mask.bool())
            y = q + attn_out
            y = y.permute(0, 2, 1)

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)

        y = self.mid_projection(y)
        cond_info = cond_info.reshape(B, cond_info.shape[1], K * L)
        y = y + self.cond_projection(cond_info)

        gate, filt = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filt)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual.reshape(base_shape)) / math.sqrt(2.0), skip.reshape(base_shape)
