#!/usr/bin/env python3
"""Export fixed-shape TorchScript wrappers for prefill/decode stages."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM

from lib_paths import default_model_alias, torch_paths

ROOT = Path(__file__).resolve().parents[1]


class PrefillWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

        logits_last = outputs.logits[:, -1, :].to(torch.float16)

        key_layers: List[torch.Tensor] = []
        value_layers: List[torch.Tensor] = []
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

        present_keys: List[torch.Tensor] = []
        present_values: List[torch.Tensor] = []
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

    n_kv_heads = getattr(cfg, "num_key_value_heads", None)
    if n_kv_heads is None:
        n_kv_heads = n_heads

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
        "n_kv_heads": int(n_kv_heads),
        "hidden_size": int(hidden_size),
        "head_dim": int(head_dim),
        "vocab_size": vocab_size,
        "intermediate_size": int(intermediate_size),
    }


def get_max_positions(cfg: Any) -> int:
    candidates = [
        getattr(cfg, "n_positions", None),
        getattr(cfg, "max_position_embeddings", None),
        getattr(cfg, "max_sequence_length", None),
    ]
    for value in candidates:
        if value is not None:
            return int(value)
    return 0


def trace_or_script(
    module: torch.nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    out_path: Path,
    prefer_script: bool = False,
) -> str:
    def save_module_atomic(ts_module: torch.jit.ScriptModule, path: Path) -> None:
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            payload = ts_module.save_to_buffer()
            with tmp_path.open("wb") as handle:
                handle.write(payload)
        except Exception as buffer_exc:
            print(
                "WARNING: save_to_buffer failed; falling back to direct TorchScript save. "
                f"reason={type(buffer_exc).__name__}: {buffer_exc}"
            )
            ts_module.save(str(tmp_path))
        tmp_path.replace(path)

    module.eval()
    with torch.no_grad():
        if prefer_script:
            try:
                scripted = torch.jit.script(module)
                save_module_atomic(scripted, out_path)
                return "script"
            except Exception as script_exc:
                print(
                    "WARNING: torch.jit.script failed; retrying with trace. "
                    f"reason={type(script_exc).__name__}: {script_exc}"
                )

        try:
            with torch.jit.optimized_execution(False):
                traced = torch.jit.trace(
                    module,
                    example_inputs,
                    strict=False,
                    check_trace=False,
                )
            save_module_atomic(traced, out_path)
            return "trace"
        except Exception as trace_exc:
            print(
                "WARNING: torch.jit.trace failed; falling back to script. "
                f"reason={type(trace_exc).__name__}: {trace_exc}"
            )
            scripted = torch.jit.script(module)
            save_module_atomic(scripted, out_path)
            return "script"


def resolve_context_len(raw_cfg: Dict[str, Any], override: int | None) -> int:
    if override is not None:
        return int(override)

    context_len = int(raw_cfg["context_len"])
    cfg_prefill = raw_cfg.get("prefill_len")
    if cfg_prefill is not None and int(cfg_prefill) != context_len - 1:
        raise ValueError("prefill_len must equal context_len - 1")
    return context_len


def configure_torchscript_export_runtime() -> None:
    # Disable JIT profiling passes for lower peak memory during large-model tracing.
    try:
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
    except Exception:
        pass


def resolve_model_id(raw_cfg: Dict[str, Any], override: str | None) -> str:
    if override:
        return str(override)
    return str(raw_cfg["hf_model_id"])


def resolve_hf_token(hf_token_env: str | None) -> str | None:
    if hf_token_env:
        token = os.getenv(hf_token_env)
        if token:
            return token
    return os.getenv("HF_TOKEN")


def is_large_decoder_lm(model_id: str, cfg_obj: Any) -> bool:
    model_id_l = str(model_id).lower()
    if "llama" in model_id_l or "qwen" in model_id_l:
        return True
    hidden = getattr(cfg_obj, "hidden_size", None)
    layers = getattr(cfg_obj, "num_hidden_layers", None)
    if hidden is not None and layers is not None:
        return int(hidden) >= 4096 and int(layers) >= 24
    return False


def load_model(
    hf_model_id: str,
    token: Optional[str],
    model_cfg: Any,
) -> Tuple[torch.nn.Module, str, Dict[str, Any]]:
    load_kwargs: Dict[str, Any] = {"token": token}
    model_id_l = str(hf_model_id).lower()
    if "qwen" in model_id_l or "llama" in model_id_l:
        # Avoid Torch SDPA graph patterns that can fail Core ML conversion on cache decode.
        load_kwargs["attn_implementation"] = "eager"
    large_model = is_large_decoder_lm(hf_model_id, model_cfg)
    if large_model:
        # low_cpu_mem_usage requires accelerate; gracefully disable when unavailable.
        if importlib.util.find_spec("accelerate") is not None:
            load_kwargs["low_cpu_mem_usage"] = True
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(hf_model_id, **load_kwargs)
    model.eval()

    model_dtype = str(next(model.parameters()).dtype)

    return model, model_dtype, {
        "low_cpu_mem_usage": bool(load_kwargs.get("low_cpu_mem_usage", False)),
        "torch_dtype": str(load_kwargs.get("torch_dtype", torch.float32)),
    }


def ensure_forward_works(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    allow_upcast: bool,
) -> Tuple[torch.nn.Module, str]:
    with torch.no_grad():
        try:
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            return model, str(next(model.parameters()).dtype)
        except Exception as exc:
            if not allow_upcast:
                raise
            print(
                "WARNING: initial forward failed; retrying with float32 weights. "
                f"reason={type(exc).__name__}: {exc}"
            )
            model = model.float()
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            return model, str(next(model.parameters()).dtype)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--context-len", type=int)
    parser.add_argument("--model-id")
    parser.add_argument("--variant-dir")
    parser.add_argument("--hf-token-env")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()
    configure_torchscript_export_runtime()

    cfg_path = (ROOT / args.config).resolve()
    cfg = load_yaml(cfg_path)

    hf_model_id = resolve_model_id(cfg, args.model_id)
    context_len = resolve_context_len(cfg, args.context_len)
    prefill_len = int(context_len - 1)
    batch_size = int(args.batch_size if args.batch_size is not None else cfg.get("batch_size", 1))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 1337))

    if batch_size != 1:
        raise ValueError("This suite currently supports batch_size=1 only.")

    use_legacy_layout = args.variant_dir is None
    path_map = torch_paths(
        model_id=hf_model_id,
        context_len=context_len,
        variant_dir=args.variant_dir,
        legacy=use_legacy_layout,
    )

    path_map["prefill_pt"].parent.mkdir(parents=True, exist_ok=True)
    path_map["decode_pt"].parent.mkdir(parents=True, exist_ok=True)
    path_map["model_meta_json"].parent.mkdir(parents=True, exist_ok=True)

    hf_token_env = args.hf_token_env or cfg.get("hf_token_env")
    token = resolve_hf_token(hf_token_env)
    if hf_token_env and token is None:
        raise RuntimeError(
            f"Missing Hugging Face token in env var '{hf_token_env}' for model '{hf_model_id}'."
        )

    model_cfg = AutoConfig.from_pretrained(hf_model_id, token=token)
    max_positions = get_max_positions(model_cfg)
    if max_positions and context_len > max_positions:
        raise ValueError(
            f"context_len={context_len} exceeds model max positions={max_positions} for {hf_model_id}"
        )

    meta = read_model_meta(model_cfg)

    rng = np.random.default_rng(seed)
    example_prefill_ids = torch.from_numpy(
        rng.integers(0, meta["vocab_size"], size=(1, prefill_len), dtype=np.int64)
    )
    example_prefill_mask = torch.ones((1, prefill_len), dtype=torch.int64)
    example_prefill_pos = torch.arange(prefill_len, dtype=torch.int64).unsqueeze(0)

    model, loaded_weight_dtype, load_settings = load_model(
        hf_model_id=hf_model_id,
        token=token,
        model_cfg=model_cfg,
    )
    model, effective_dtype = ensure_forward_works(
        model,
        input_ids=example_prefill_ids,
        attention_mask=example_prefill_mask,
        position_ids=example_prefill_pos,
        allow_upcast=True,
    )

    print(f"model_weight_dtype_loaded: {loaded_weight_dtype}")
    print(f"model_weight_dtype_effective: {effective_dtype}")

    prefill_wrapper = PrefillWrapper(model).eval()
    decode_wrapper = DecodeWrapper(model, n_layers=meta["n_layers"]).eval()

    past_key = torch.randn(
        (
            meta["n_layers"],
            meta["n_kv_heads"],
            prefill_len,
            meta["head_dim"],
        ),
        dtype=torch.float16,
    )
    past_value = torch.randn(
        (
            meta["n_layers"],
            meta["n_kv_heads"],
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

    prefill_path = path_map["prefill_pt"]
    decode_path = path_map["decode_pt"]
    prefer_script = is_large_decoder_lm(hf_model_id, model_cfg) and context_len >= 256

    prefill_method = trace_or_script(
        prefill_wrapper,
        (example_prefill_ids, example_prefill_mask, example_prefill_pos),
        prefill_path,
        prefer_script=prefer_script,
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
        prefer_script=prefer_script,
    )

    meta_payload: Dict[str, Any] = {
        **meta,
        "hf_model_id": hf_model_id,
        "model_alias": default_model_alias(hf_model_id),
        "context_len": context_len,
        "prefill_len": prefill_len,
        "batch_size": batch_size,
        "max_positions": max_positions,
        "torchscript_prefill_method": prefill_method,
        "torchscript_decode_method": decode_method,
        "export_weight_dtype_loaded": loaded_weight_dtype,
        "export_weight_dtype_effective": effective_dtype,
        "load_settings": load_settings,
        "variant_dir": str(path_map["model_meta_json"].parent),
    }

    meta_path = path_map["model_meta_json"]
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_payload, handle, indent=2, sort_keys=True)

    if use_legacy_layout:
        legacy_meta = ROOT / "artifacts" / "model_meta.json"
        legacy_meta.parent.mkdir(parents=True, exist_ok=True)
        with legacy_meta.open("w", encoding="utf-8") as handle:
            json.dump(meta_payload, handle, indent=2, sort_keys=True)

    print(f"saved: {prefill_path}")
    print(f"saved: {decode_path}")
    print(f"saved: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
