import json
import copy

from .factory import create_model_from_config
from .utils import load_ckpt_state_dict

from huggingface_hub import hf_hub_download

def _download_optional_file(name: str, filename: str):
    try:
        return hf_hub_download(name, filename=filename, repo_type='model')
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

def download_pretrained_artifacts(name: str):
    model_ckpt_path = _download_optional_file(name, "model.safetensors")
    if model_ckpt_path is None:
        model_ckpt_path = hf_hub_download(name, filename="model.ckpt", repo_type='model')

    return {
        "config_path": hf_hub_download(name, filename="config.json", repo_type='model'),
        "model_ckpt_path": model_ckpt_path,
        "vae_ckpt_path": _download_optional_file(name, "VAE.ckpt"),
        "synchformer_ckpt_path": _download_optional_file(name, "synchformer_state_dict.pth"),
    }

def get_pretrained_model(name: str):
    artifact_paths = download_pretrained_artifacts(name)

    with open(artifact_paths["config_path"]) as f:
        model_config = json.load(f)
    model_config = copy.deepcopy(model_config)
    _patch_pretransform_ckpt_paths(model_config, artifact_paths["vae_ckpt_path"])

    model = create_model_from_config(model_config)

    model.load_state_dict(load_ckpt_state_dict(artifact_paths["model_ckpt_path"]))

    return model, model_config
