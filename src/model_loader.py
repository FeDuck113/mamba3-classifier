import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from .mambaClf import CLF_HEADS, MambaClassifier


def _get_device_and_dtype(device):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    return device, dtype


def _build_model(config, tokenizer, weights_path, device, dtype):
    backbone = MambaLMHeadModel.from_pretrained(
        config["base_model_name_or_path"],
        device=str(device),
        dtype=dtype,
    )
    model = MambaClassifier(
        mamba_lm=backbone,
        num_labels=config["num_labels"],
        problem_type=config["problem_type"],
        classifier_head=CLF_HEADS[config["classifier_head"]],
        pooling=config["pooling"],
        dropout=config["dropout"],
    ).to(device=device, dtype=dtype)

    model.load_state_dict(load_file(str(weights_path), device=str(device)))
    model.id2label = {int(key): value for key, value in config["id2label"].items()}
    model.threshold = config.get("threshold", 0.5)
    model.max_length = config.get("max_length", 256)
    model.tokenizer = tokenizer
    model.eval()
    return model


def load_model(model_id: str, device: str | torch.device | None = None):
    device, dtype = _get_device_and_dtype(device)
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    weights_path = hf_hub_download(repo_id=model_id, filename="model.safetensors")

    return _build_model(config, tokenizer, weights_path, device, dtype)


def load_model_from_dir(model_dir: str | Path, device: str | torch.device | None = None):
    model_dir = Path(model_dir)
    device, dtype = _get_device_and_dtype(device)
    with (model_dir / "config.json").open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return _build_model(
        config,
        tokenizer,
        model_dir / "model.safetensors",
        device,
        dtype,
    )
