from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        num_users: int,
        num_articles: int,
        attribute_vocab_sizes: Sequence[int] = (),
        attribute_groups: Sequence[str] = (),
        article_embedding_dim: int = 64,
        user_embedding_dim: int = 32,
        attribute_embedding_dim: int = 8,
        hidden_dim: int = 128,
        output_dim: int = 64,
    ) -> None:
        super().__init__()
        if variant not in {"TT0", "TT1"}:
            raise ValueError("Two-Tower 版本只能是 TT0/TT1")
        if variant == "TT1" and (not attribute_vocab_sizes or len(attribute_vocab_sizes) != len(attribute_groups)):
            raise ValueError("TT1 必须提供完整 canonical attribute vocabulary")
        self.variant = variant
        self.output_dim = output_dim
        self.article_embedding = nn.Embedding(num_articles, article_embedding_dim)
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.attribute_groups = tuple(attribute_groups)
        self.group_names = tuple(dict.fromkeys(attribute_groups))
        self.attribute_embeddings = nn.ModuleList(
            nn.Embedding(size, attribute_embedding_dim) for size in attribute_vocab_sizes
        )
        base_dim = article_embedding_dim + (len(self.group_names) * attribute_embedding_dim if variant == "TT1" else 0)
        self.item_projection = nn.Linear(base_dim, output_dim)
        self.user_projection = nn.Sequential(
            nn.Linear(user_embedding_dim + base_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )
        self.register_buffer("attribute_index", torch.empty((0, 0), dtype=torch.long), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.article_embedding.weight, std=0.02)
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        for embedding in self.attribute_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.item_projection.weight)
        nn.init.zeros_(self.item_projection.bias)
        for layer in self.user_projection:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def set_attribute_index(self, values: torch.Tensor) -> None:
        if self.variant == "TT1" and values.shape[1] != len(self.attribute_embeddings):
            raise ValueError("attribute index 字段数不匹配")
        self.attribute_index = values.to(dtype=torch.long)

    def item_base(self, article_keys: torch.Tensor) -> torch.Tensor:
        identity = self.article_embedding(article_keys)
        if self.variant == "TT0":
            return identity
        if self.attribute_index.numel() == 0:
            raise RuntimeError("TT1 attribute index 尚未设置")
        indices = self.attribute_index[article_keys]
        field_vectors = [embedding(indices[:, i]) for i, embedding in enumerate(self.attribute_embeddings)]
        group_vectors = []
        for group in self.group_names:
            members = [field_vectors[i] for i, value in enumerate(self.attribute_groups) if value == group]
            # 先组内均值，再拼接三个等维 group，避免字段多的组自然获得更高先验权重。
            group_vectors.append(torch.stack(members, dim=0).mean(dim=0))
        return torch.cat([identity, *group_vectors], dim=-1)

    def encode_items(self, article_keys: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_projection(self.item_base(article_keys)), p=2, dim=-1)

    def encode_users(
        self,
        user_keys: torch.Tensor,
        history_items: torch.Tensor,
        history_weights: torch.Tensor,
    ) -> torch.Tensor:
        safe_items = history_items.clamp_min(0)
        shape = safe_items.shape
        history_base = self.item_base(safe_items.reshape(-1)).reshape(shape[0], shape[1], -1)
        mask = history_items.ge(0).unsqueeze(-1)
        weights = history_weights.unsqueeze(-1) * mask
        denominator = weights.sum(dim=1).clamp_min(1e-12)
        pooled = (history_base * weights).sum(dim=1) / denominator
        projected = self.user_projection(torch.cat([self.user_embedding(user_keys), pooled], dim=-1))
        return F.normalize(projected, p=2, dim=-1)

    def forward(
        self,
        user_keys: torch.Tensor,
        history_items: torch.Tensor,
        history_weights: torch.Tensor,
        target_items: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode_users(user_keys, history_items, history_weights), self.encode_items(target_items)


def in_batch_negative_mask(target_items: torch.Tensor, history_items: torch.Tensor) -> torch.Tensor:
    """True 表示该列可参与当前行 softmax；对角正类始终保留。"""
    targets = target_items.reshape(-1)
    allowed = torch.ones((len(targets), len(targets)), dtype=torch.bool, device=targets.device)
    duplicates = targets[:, None].eq(targets[None, :])
    allowed &= ~duplicates
    history_match = history_items[:, :, None].eq(targets[None, None, :]).any(dim=1)
    # 未观察项不是已知负反馈；已知同 target 与用户历史中的商品必须从负项集合移除。
    allowed &= ~history_match
    diagonal = torch.arange(len(targets), device=targets.device)
    allowed[diagonal, diagonal] = True
    return allowed


def in_batch_softmax_loss(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    target_items: torch.Tensor,
    history_items: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    logits = user_embeddings @ item_embeddings.T / temperature
    allowed = in_batch_negative_mask(target_items, history_items)
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
