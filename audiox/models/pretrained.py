import json
import copy

from .factory import create_model_from_config
from .lora import load_lora_checkpoint
from .utils import load_ckpt_state_dict

from huggingface_hub import hf_hub_download

def _download_optional_file(name: str, filename: str, cache_dir=None):
    try:
        return hf_hub_download(name, filename=filename, repo_type='model', cache_dir=cache_dir)
    except Exception:
        return None

def _patch_pretransform_ckpt_paths(node, vae_ckpt_path):
    if vae_ckpt_path is None:
        return

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "pretransform_ckpt_path" and isinstance(value, str) and value.endswith("VAE.ckpt"):
                node[key] = vae_ckpt_path
            else:
                _patch_pretransform_ckpt_paths(value, vae_ckpt_path)
    elif isinstance(node, list):
        for item in node:
            _patch_pretransform_ckpt_paths(item, vae_ckpt_path)

def download_pretrained_artifacts(name: str, cache_dir=None):
    model_ckpt_path = _download_optional_file(name, "model.safetensors", cache_dir=cache_dir)
    if model_ckpt_path is None:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model', cache_dir=cache_dir)

    return {
        "config_path": hf_hub_download(name, filename="config.json", repo_type='model', cache_dir=cache_dir),
        "model_ckpt_path": model_ckpt_path,
        "vae_ckpt_path": _download_optional_file(name, "VAE.ckpt", cache_dir=cache_dir),
        "synchformer_ckpt_path": _download_optional_file(name, "synchformer_state_dict.pth", cache_dir=cache_dir),
    }

def get_pretrained_model(name: str, cache_dir=None, lora_path=None):
    artifact_paths = download_pretrained_artifacts(name, cache_dir=cache_dir)

    with open(artifact_paths["config_path"]) as f:
        model_config = json.load(f)
    model_config = copy.deepcopy(model_config)
    _patch_pretransform_ckpt_paths(model_config, artifact_paths["vae_ckpt_path"])

    model = create_model_from_config(model_config)

    model.load_state_dict(load_ckpt_state_dict(artifact_paths["model_ckpt_path"]))
    if lora_path is not None:
        load_lora_checkpoint(model, lora_path)

    return model, model_config
