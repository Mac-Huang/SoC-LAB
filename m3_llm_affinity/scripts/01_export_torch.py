#!/usr/bin/env python3
"""Export fixed-shape TorchScript wrappers for prefill/decode stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]


class PrefillWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

        logits_last = outputs.logits[:, -1, :].to(torch.float16)

        key_layers = []
        value_layers = []
        for key, value in outputs.past_key_values:
            key_layers.append(key.squeeze(0).to(torch.float16))
            value_layers.append(value.squeeze(0).to(torch.float16))

        past_key = torch.stack(key_layers, dim=0)
        past_value = torch.stack(value_layers, dim=0)
        return logits_last, past_key, past_value


class DecodeWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, n_layers: int):
        super().__init__()
        self.model = model
        self.n_layers = int(n_layers)

    def forward(
        self,
        input_id: torch.Tensor,
        attention_mask: torch.Tensor,
        position_id: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ):
        past_key_values = []
        for layer_idx in range(self.n_layers):
            key = past_key[layer_idx].unsqueeze(0)
            value = past_value[layer_idx].unsqueeze(0)
            past_key_values.append((key, value))

        outputs = self.model(
            input_ids=input_id,
            attention_mask=attention_mask,
            position_ids=position_id,
            past_key_values=tuple(past_key_values),
            use_cache=True,
            return_dict=True,
        )

        logits = outputs.logits[:, -1, :].to(torch.float16)

        present_keys = []
        present_values = []
        for key, value in outputs.past_key_values:
            present_keys.append(key.squeeze(0).to(torch.float16))
            present_values.append(value.squeeze(0).to(torch.float16))

        present_key = torch.stack(present_keys, dim=0)
        present_value = torch.stack(present_values, dim=0)
        return logits, present_key, present_value


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def read_model_meta(cfg: Any) -> Dict[str, int]:
    n_layers = getattr(cfg, "n_layer", None)
    if n_layers is None:
        n_layers = getattr(cfg, "num_hidden_layers")

    n_heads = getattr(cfg, "n_head", None)
    if n_heads is None:
        n_heads = getattr(cfg, "num_attention_heads")

    hidden_size = getattr(cfg, "n_embd", None)
    if hidden_size is None:
        hidden_size = getattr(cfg, "hidden_size")

    intermediate_size = getattr(cfg, "n_inner", None)
    if intermediate_size is None:
        intermediate_size = getattr(cfg, "intermediate_size", None)
    if intermediate_size is None:
        intermediate_size = 4 * int(hidden_size)

    vocab_size = int(getattr(cfg, "vocab_size"))
    head_dim = int(hidden_size) // int(n_heads)

    return {
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "hidden_size": int(hidden_size),
        "head_dim": int(head_dim),
        "vocab_size": vocab_size,
        "intermediate_size": int(intermediate_size),
    }


def trace_or_script(
    module: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    out_path: Path,
) -> str:
    module.eval()
    with torch.no_grad():
        try:
            traced = torch.jit.trace(
                module,
                example_inputs,
                strict=False,
                check_trace=False,
            )
            _ = traced(*example_inputs)
            traced.save(str(out_path))
            return "trace"
        except Exception as trace_exc:
            print(
                "WARNING: torch.jit.trace failed; falling back to script. "
                f"reason={type(trace_exc).__name__}: {trace_exc}"
            )
            scripted = torch.jit.script(module)
            _ = scripted(*example_inputs)
            scripted.save(str(out_path))
            return "script"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    cfg = load_yaml(cfg_path)

    hf_model_id = str(cfg["hf_model_id"])
    context_len = int(cfg["context_len"])
    prefill_len = int(cfg["prefill_len"])
    batch_size = int(cfg["batch_size"])
    seed = int(cfg["seed"])

    if batch_size != 1:
        raise ValueError("This suite currently supports batch_size=1 only.")
    if prefill_len != context_len - 1:
        raise ValueError("prefill_len must equal context_len - 1.")

    artifacts_torch = ROOT / "artifacts" / "torch"
    artifacts_torch.mkdir(parents=True, exist_ok=True)

    model_cfg = AutoConfig.from_pretrained(hf_model_id)
    meta = read_model_meta(model_cfg)

    rng = np.random.default_rng(seed)
    example_prefill_ids = torch.from_numpy(
        rng.integers(0, meta["vocab_size"], size=(1, prefill_len), dtype=np.int64)
    )
    example_prefill_mask = torch.ones((1, prefill_len), dtype=torch.int64)

    model = AutoModelForCausalLM.from_pretrained(hf_model_id)
    model.eval()
    # Keep export weights in fp32 for robust Torch->Core ML conversion.
    # Wrapper outputs and converted compute precision are still float16.
    model = model.float()
    with torch.no_grad():
        _ = model(
            input_ids=example_prefill_ids,
            attention_mask=example_prefill_mask,
            use_cache=True,
            return_dict=True,
        )
    weight_dtype = "float32"
    print(f"model_weight_dtype_for_export: {weight_dtype}")

    prefill_wrapper = PrefillWrapper(model).eval()
    decode_wrapper = DecodeWrapper(model, n_layers=meta["n_layers"]).eval()

    past_key = torch.randn(
        (
            meta["n_layers"],
            meta["n_heads"],
            prefill_len,
            meta["head_dim"],
        ),
        dtype=torch.float16,
    )
    past_value = torch.randn(
        (
            meta["n_layers"],
            meta["n_heads"],
            prefill_len,
            meta["head_dim"],
        ),
        dtype=torch.float16,
    )

    example_decode_id = torch.from_numpy(
        rng.integers(0, meta["vocab_size"], size=(1, 1), dtype=np.int64)
    )
    example_decode_mask = torch.ones((1, prefill_len + 1), dtype=torch.int64)
    example_position_id = torch.tensor([[prefill_len]], dtype=torch.int64)

    prefill_path = artifacts_torch / "prefill.pt"
    decode_path = artifacts_torch / "decode.pt"

    prefill_method = trace_or_script(
        prefill_wrapper,
        (example_prefill_ids, example_prefill_mask),
        prefill_path,
    )
    decode_method = trace_or_script(
        decode_wrapper,
        (
            example_decode_id,
            example_decode_mask,
            example_position_id,
            past_key,
            past_value,
        ),
        decode_path,
    )

    meta_payload: Dict[str, Any] = {
        **meta,
        "hf_model_id": hf_model_id,
        "context_len": context_len,
        "prefill_len": prefill_len,
        "batch_size": batch_size,
        "torchscript_prefill_method": prefill_method,
        "torchscript_decode_method": decode_method,
        "export_weight_dtype": weight_dtype,
    }

    meta_path = ROOT / "artifacts" / "model_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2, sort_keys=True)

    print(f"saved: {prefill_path}")
    print(f"saved: {decode_path}")
    print(f"saved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
