# Copyright (c) FLM Team, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Retrieves the pretrained models for Mimi."""

from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_model
import torch

from ..third_party.moshi.models import MimiModel
from ..third_party.moshi.modules import SEANetEncoder, SEANetDecoder, transformer
from ..third_party.moshi.quantization import SplitResidualVectorQuantizer

SAMPLE_RATE = 24000
FRAME_RATE = 12.5

MIMI_NAME = 'tokenizer-e351c8d8-checkpoint125.safetensors'


_seanet_kwargs = {
    "channels": 1,
    "dimension": 512,
    "causal": True,
    "n_filters": 64,
    "n_residual_layers": 1,
    "activation": "ELU",
    "compress": 2,
    "dilation_base": 2,
    "disable_norm_outer_blocks": 0,
    "kernel_size": 7,
    "residual_kernel_size": 3,
    "last_kernel_size": 3,
    "norm": "none",
    "pad_mode": "constant",
    "ratios": [8, 6, 5, 4],
    "true_skip": True,
}

_quantizer_kwargs = {
    "dimension": 256,
    "n_q": 32,
    "bins": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimension": _seanet_kwargs["dimension"],
}

_transformer_kwargs = {
    "d_model": _seanet_kwargs["dimension"],
    "num_heads": 8,
    "num_layers": 8,
    "causal": True,
    "layer_scale": 0.01,
    "context": 250,
    "conv_layout": True,
    "max_period": 10000,
    "gating": "none",
    "norm": "layer_norm",
    "positional_embedding": "rope",
    "dim_feedforward": 2048,
    "input_dimension": _seanet_kwargs["dimension"],
    "output_dimensions": [_seanet_kwargs["dimension"]],
}


def hf_get(filename: str | Path, hf_repo: str | None = None,
           check_local_file_exists: bool = False) -> Path:
    if isinstance(filename, Path):
        return filename
    if filename.startswith("hf://"):
        parts = filename.removeprefix("hf://").split("/")
        repo_name = parts[0] + "/" + parts[1]
        filename = "/".join(parts[2:])
        return Path(hf_hub_download(repo_name, filename))
    elif filename.startswith("file://"):
        # Provide a way to force the read of a local file.
        filename = filename.removeprefix("file://")
        return Path(filename)
    elif hf_repo is not None:
        if check_local_file_exists:
            if Path(filename).exists():
                return Path(filename)
        return Path(hf_hub_download(hf_repo, filename))
    else:
        return Path(filename)


def get_mimi(mimi_path: str, device: torch.device | str = 'cpu') -> MimiModel:
    model_path = Path(mimi_path) / MIMI_NAME
    if not model_path.exists():
        model_path = hf_get(MIMI_NAME, hf_repo=mimi_path)
    return _get_mimi(model_path, device=device)


def _is_safetensors(path: Path | str) -> bool:
    return Path(path).suffix in (".safetensors", ".sft", ".sfts")


def _get_mimi(
        filename: str | Path,
        device: torch.device | str = 'cpu',
        num_codebooks: int = 8,
    ) -> MimiModel:
    """Return a pretrained Mimi model."""
    encoder = SEANetEncoder(**_seanet_kwargs)
    decoder = SEANetDecoder(**_seanet_kwargs)
    encoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    decoder_transformer = transformer.ProjectedTransformer(
        device=device, **_transformer_kwargs
    )
    quantizer = SplitResidualVectorQuantizer(
        **_quantizer_kwargs,
    )
    model = MimiModel(
        encoder,
        decoder,
        quantizer,
        channels=1,
        sample_rate=SAMPLE_RATE,
        frame_rate=FRAME_RATE,
        encoder_frame_rate=SAMPLE_RATE / encoder.hop_length,
        causal=True,
        resample_method="conv",
        encoder_transformer=encoder_transformer,
        decoder_transformer=decoder_transformer,
    ).to(device=device)
    model.eval()
    if _is_safetensors(filename):
        load_model(model, filename, device=str(device))
    else:
        pkg = torch.load(filename, "cpu")
        model.load_state_dict(pkg["model"])
    model.set_num_codebooks(num_codebooks)
    return model
