import re
from pathlib import Path

import torch


_SCALE_DATA = None


def load_static_scales(
    path: str,
):
    global _SCALE_DATA

    _SCALE_DATA = torch.load(
        path,
        map_location="cpu",
    )

    print(
        "[INT8] static scale loaded:",
        path,
    )

    print(
        "[INT8] k_scale:",
        _SCALE_DATA["k_scale"].shape,
    )

    print(
        "[INT8] v_scale:",
        _SCALE_DATA["v_scale"].shape,
    )


def extract_layer_idx(
    layer_name: str,
) -> int:
    """
    例如：

    model.layers.0.self_attn.attn
    ->
    0
    """

    match = re.search(
        r"layers\.(\d+)",
        layer_name,
    )

    if match is None:
        raise RuntimeError(f"Cannot extract layer index from {layer_name}")

    return int(match.group(1))


def get_layer_scales(
    layer_name: str,
    device,
):
    if _SCALE_DATA is None:
        raise RuntimeError("Call load_static_scales() first.")

    idx = extract_layer_idx(layer_name)

    k_scale = (
        _SCALE_DATA["k_scale"][idx]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .contiguous()
    )

    v_scale = (
        _SCALE_DATA["v_scale"][idx]
        .to(
            device=device,
            dtype=torch.float32,
        )
        .contiguous()
    )

    return (
        k_scale,
        v_scale,
    )
