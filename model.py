

import math
from typing import Dict

import torch
import torch.nn as nn

from config import Config
from dataset import Vocab


class Embedder(nn.Module):

    net: nn.Sequential
    hpos_cache: Dict[int, torch.Tensor] = {}

    @staticmethod
    def generate_sinusoidal_embeddings(length, dim):
        pos = torch.arange(length).unsqueeze(1)  # (length, 1)
        i = torch.arange(dim // 2).unsqueeze(0)  # (1, dim // 2)
        angle_rates = 1 / (10000 ** (2 * i / dim))
        embeddings = torch.zeros((length, dim))
        embeddings[:, 0::2] = torch.sin(pos * angle_rates)  # Even indices
        embeddings[:, 1::2] = torch.cos(pos * angle_rates)  # Odd indices
        return embeddings

    def __init__(self, max_width: int, input_height: int, reduction_dims: int, embed_size: int):
        super(Embedder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_height, reduction_dims),
            nn.GELU(),
            nn.Linear(reduction_dims, embed_size)
        )
        pe = torch.zeros(1, max_width, embed_size, dtype=torch.float)
        position = torch.arange(
            0, max_width, dtype=torch.float).unsqueeze(1)  # [1024, 1]
        div_term = torch.exp(torch.arange(0, embed_size, 2).float(
        ) * (-math.log(10000.0) / embed_size))  # [256]
        pe[:, :, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, :, 1::2] = torch.cos(position * div_term)  # Odd indices
        # Save as buffer        # Create a matrix of shape (max_len, d_model) to hold the positional encodings
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(torch.float)) + self.pe[:, :x.size(1), :]


class Translator(nn.Module):

    def __init__(self, config: Config):
        super(Translator, self).__init__()
        self.source_embedder = Embedder(
            config.ipad_shape[1],
            config.ipad_shape[0],
            config.image_reducer, config.embed_size
        )
        self.target_embedder = Embedder(
            config.spad_len - 1,
            config.max_chord,
            config.sequence_reducer, config.embed_size
        )
        self.transformer = nn.Transformer(
            d_model=config.embed_size,
            nhead=config.num_head,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dim_feedforward=config.dim_feed_forward,
            dropout=config.dropout,
            batch_first=True
        )
        self.generator = nn.Linear(
            config.embed_size,
            config.max_chord * config.vocab_size)

    def forward(self, source, target, attention_mask):
        source_embeds = self.source_embedder(source)
        target_embeds = self.target_embedder(target)
        source_mask = (source == Vocab.PAD)[:, :, 0]
        outs = self.transformer(
            src=source_embeds, tgt=target_embeds,
            tgt_mask=attention_mask,
            src_key_padding_mask=source_mask,
            tgt_key_padding_mask=(target == Vocab.PAD)[:, :, 0],
            memory_key_padding_mask=source_mask,
        )
        B, T, H = target.shape
        outs = self.generator(outs)
        outs = outs.view(B, T, H, -1)
        return outs

    def encode(self, source, src_key_padding_mask):
        return self.transformer.encoder(
            self.source_embedder(source),
            src_key_padding_mask=src_key_padding_mask)

    def decode(self, target, memory, target_mask):
        return self.transformer.decoder(self.target_embedder(target), memory, target_mask)
