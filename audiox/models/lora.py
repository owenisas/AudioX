import math
import typing as tp
from pathlib import Path

import torch
from torch import nn


DEFAULT_LORA_TARGET_PATTERNS = (
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
        target_patterns = DEFAULT_LORA_TARGET_PATTERNS

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


def is_lora_parameter_name(name: str) -> bool:
    return ".lora_a." in name or ".lora_b." in name


def extract_lora_state_dict(module: nn.Module) -> tp.Dict[str, torch.Tensor]:
    lora_state: tp.Dict[str, torch.Tensor] = {}
    for name, value in module.state_dict().items():
        if is_lora_parameter_name(name):
            lora_state[name] = value.detach().cpu()
    return lora_state


def extract_parameter_state_dict(
    module: nn.Module,
    parameter_names: tp.Sequence[str],
) -> tp.Dict[str, torch.Tensor]:
    module_state = module.state_dict()
    extracted: tp.Dict[str, torch.Tensor] = {}
    missing = [name for name in parameter_names if name not in module_state]
    if missing:
        raise ValueError(f"State dict extraction requested unknown parameter(s): {missing}")

    for name in parameter_names:
        extracted[name] = module_state[name].detach().cpu()
    return extracted


def load_parameter_state_dict(
    module: nn.Module,
    parameter_state_dict: tp.Mapping[str, torch.Tensor],
) -> None:
    parameter_lookup = dict(module.named_parameters())
    unknown = [name for name in parameter_state_dict if name not in parameter_lookup]
    if unknown:
        raise ValueError(f"Checkpoint contains unknown trainable-scope parameter(s): {unknown}")

    with torch.no_grad():
        for name, value in parameter_state_dict.items():
            parameter = parameter_lookup[name]
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _infer_rank_from_lora_state_dict(lora_state_dict: tp.Dict[str, torch.Tensor]) -> tp.Optional[int]:
    for key, value in lora_state_dict.items():
        if key.endswith(".lora_a.weight"):
            return int(value.shape[0])
    return None


def _infer_target_module_names_from_lora_state_dict(
    lora_state_dict: tp.Mapping[str, torch.Tensor]
) -> tp.List[str]:
    module_names = set()
    for key in lora_state_dict.keys():
        if key.endswith(".lora_a.weight"):
            module_names.add(key[: -len(".lora_a.weight")])
        elif key.endswith(".lora_b.weight"):
            module_names.add(key[: -len(".lora_b.weight")])
    return sorted(module_names)


def resolve_lora_config(lora_checkpoint: tp.Dict[str, tp.Any]) -> tp.Dict[str, tp.Any]:
    lora_state_dict = lora_checkpoint.get("lora_state_dict") or {}
    if not lora_state_dict:
        raise ValueError("LoRA checkpoint is missing lora_state_dict.")

    checkpoint_config = dict(lora_checkpoint.get("lora_config") or {})
    inferred_rank = _infer_rank_from_lora_state_dict(lora_state_dict)
    if inferred_rank is not None:
        checkpoint_config.setdefault("rank", inferred_rank)

    checkpoint_config.setdefault("rank", 8)
    checkpoint_config.setdefault("alpha", 16.0)
    checkpoint_config.setdefault("dropout", 0.0)
    checkpoint_config.setdefault("freeze_non_lora", True)

    target_patterns = checkpoint_config.get("target_patterns")
    if not target_patterns:
        inferred_targets = _infer_target_module_names_from_lora_state_dict(lora_state_dict)
        if not inferred_targets:
            raise ValueError(
                "LoRA checkpoint is missing target_patterns and exact target modules could not be inferred from lora_state_dict."
            )
        checkpoint_config["target_patterns"] = inferred_targets

    return checkpoint_config


def load_lora_checkpoint(module: nn.Module, checkpoint_path: tp.Union[str, Path]) -> tp.Dict[str, tp.Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    lora_config = resolve_lora_config(checkpoint)
    inject_lora(
        module,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config["dropout"],
        target_patterns=lora_config.get("target_patterns"),
        exclude_patterns=lora_config.get("exclude_patterns"),
        freeze_non_lora=lora_config.get("freeze_non_lora", True),
    )
    checkpoint_lora_keys = set(checkpoint["lora_state_dict"].keys())
    module_lora_keys = {name for name in module.state_dict().keys() if is_lora_parameter_name(name)}
    unknown_checkpoint_keys = sorted(checkpoint_lora_keys - module_lora_keys)
    if unknown_checkpoint_keys:
        raise ValueError(f"LoRA checkpoint contains unknown adapter keys: {unknown_checkpoint_keys}")

    missing_checkpoint_keys = sorted(module_lora_keys - checkpoint_lora_keys)
    if missing_checkpoint_keys:
        raise ValueError(
            "LoRA checkpoint does not fully cover the injected adapter set. "
            f"Missing key(s): {missing_checkpoint_keys}"
        )

    missing_keys, unexpected_keys = module.load_state_dict(checkpoint["lora_state_dict"], strict=False)
    missing_lora_keys = [name for name in missing_keys if is_lora_parameter_name(name)]
    unexpected_lora_keys = [name for name in unexpected_keys if is_lora_parameter_name(name)]
    if missing_lora_keys or unexpected_lora_keys:
        raise ValueError(
            "LoRA checkpoint load had adapter mismatches. "
            f"Missing: {missing_lora_keys}, Unexpected: {unexpected_lora_keys}"
        )

    trainable_scope_info = checkpoint.get("trainable_scope_info") or checkpoint.get("lora_info", {}).get("trainable_scope")
    if trainable_scope_info:
        scope_name = trainable_scope_info.get("scope")
        scope_state = checkpoint.get("trainable_scope_state_dict")
        if not scope_state:
            raise ValueError(
                f"Checkpoint declares trainable scope '{scope_name}' but is missing trainable_scope_state_dict."
            )

        expected_names = trainable_scope_info.get("non_lora_parameter_names")
        if expected_names:
            missing_scope_keys = [name for name in expected_names if name not in scope_state]
            if missing_scope_keys:
                raise ValueError(
                    "Checkpoint trainable_scope_state_dict is incomplete. "
                    f"Missing key(s): {missing_scope_keys}"
                )
        load_parameter_state_dict(module, scope_state)

    checkpoint["lora_config"] = lora_config
    return checkpoint
