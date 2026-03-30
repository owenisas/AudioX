import math
import typing as tp

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive for LoRALinear")

        self.base = linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        for param in self.base.parameters():
            param.requires_grad = False

        self.lora_a = nn.Linear(linear.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, linear.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling


def _matches_pattern(name: str, patterns: tp.Optional[tp.Iterable[str]]) -> bool:
    if not patterns:
        return False
    return any(pattern in name for pattern in patterns)


def _replace_linear_modules(
    module: nn.Module,
    prefix: str,
    rank: int,
    alpha: float,
    dropout: float,
    target_patterns: tp.Sequence[str],
    exclude_patterns: tp.Optional[tp.Sequence[str]],
    replaced_names: tp.List[str],
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, LoRALinear):
            continue

        if isinstance(child, nn.Linear):
            if _matches_pattern(full_name, target_patterns) and not _matches_pattern(full_name, exclude_patterns):
                setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
                replaced_names.append(full_name)
                continue

        _replace_linear_modules(
            child,
            full_name,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            target_patterns=target_patterns,
            exclude_patterns=exclude_patterns,
            replaced_names=replaced_names,
        )


def inject_lora(
    module: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_patterns: tp.Optional[tp.Sequence[str]] = None,
    exclude_patterns: tp.Optional[tp.Sequence[str]] = None,
    freeze_non_lora: bool = True,
) -> tp.List[str]:
    if freeze_non_lora:
        for param in module.parameters():
            param.requires_grad = False

    if not target_patterns:
        target_patterns = (
            "to_q",
            "to_kv",
            "to_qkv",
            "to_out",
            ".qkv",
            "proj_mm_tokens",
            "proj_mm_seq_len",
            "gating_network.0",
            "gating_network.2",
        )

    replaced_names: tp.List[str] = []
    _replace_linear_modules(
        module,
        prefix="",
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_patterns=target_patterns,
        exclude_patterns=exclude_patterns,
        replaced_names=replaced_names,
    )
    return replaced_names


def count_parameters(module: nn.Module) -> tp.Tuple[int, int]:
    trainable = 0
    total = 0
    for param in module.parameters():
        numel = param.numel()
        total += numel
        if param.requires_grad:
            trainable += numel
    return trainable, total
