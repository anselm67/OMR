

from dataclasses import dataclass

import torch
import torch.nn as nn

from grandpiano import GrandPiano


@dataclass
class Config:
    # Dataset related configuration.
    image_height: int = 256
    max_image_width: int = 2048
    max_sequence_height: int = 12
    max_sequence_width: int = 300
    vocab_size: int = -1

    # Image embedder config.
    image_reducer: int = 64
    embed_size: int = 128

    # Sequence embedder config.
    sequence_reducer: int = 64

    # Transformer config.
    num_head = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feed_forward = 1024
    dropout = 0.1


class SourceEmbedder(nn.Module):

    net: nn.Sequential
    pos: nn.Embedding

    def __init__(self, config: Config):
        super(SourceEmbedder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.image_height, config.image_reducer),
            nn.GELU(),
            nn.Linear(config.image_reducer, config.embed_size)
        )
        self.pos = nn.Embedding(config.max_image_width, config.embed_size)

    # TODO Check if it's ok to embed a padded sequence.
    def forward(self, x):
        width = x.shape[-2]
        tok = self.net(x)
        return tok + self.pos(torch.arange(width, device=x.device))


class TargetEmbedder(nn.Module):

    net: nn.Sequential
    hpos: nn.Embedding
    vpos: nn.Embedding

    def __init__(self, config: Config):
        super(TargetEmbedder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(config.max_sequence_height, config.sequence_reducer),
            nn.GELU(),
            nn.Linear(config.sequence_reducer, config.embed_size)
        )
        # Position is
        self.hpos = nn.Embedding(config.max_sequence_width, config.embed_size)

    # TODO Check if it's ok to embed a padded sequence.
    def forward(self, x):
        width, height = x.shape[-2], x.shape[-1]
        tok = self.net(x.to(torch.float))
        return (
            tok +
            self.hpos(torch.arange(width, device=x.device))
        )


class Translator(nn.Module):

    def __init__(self, config: Config):
        super(Translator, self).__init__()
        self.source_embedder = SourceEmbedder(config)
        self.target_embedded = TargetEmbedder(config)
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
            config.max_sequence_height * config.vocab_size)

    def forward(self, source, target, attention_mask):
        source_embeds = self.source_embedder(source)
        target_embeds = self.target_embedded(target)
        source_mask = (source == GrandPiano.PAD[0])[:, :, 0]
        outs = self.transformer(
            src=source_embeds, tgt=target_embeds,
            tgt_mask=attention_mask,
            src_key_padding_mask=source_mask,
            tgt_key_padding_mask=(target == GrandPiano.PAD[0])[:, :, 0],
            memory_key_padding_mask=source_mask,
        )
        B, T, H = target.shape
        outs = self.generator(outs)
        outs = outs.view(B, T, H, -1)
        return outs
