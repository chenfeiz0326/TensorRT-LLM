from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path
from random import choices, shuffle
from typing import Dict, List, Tuple, Union

import yaml

from tensorrt_llm._torch.pyexecutor.model_engine import \
    validate_and_set_kv_cache_quant
from tensorrt_llm.bench.build.build import (get_benchmark_engine_settings,
                                            get_model_config)
from tensorrt_llm.bench.dataclasses.general import (DatasetMetadata,
                                                    InferenceRequest)
from tensorrt_llm.logger import logger
from tensorrt_llm.quantization.mode import QuantAlgo

_KV_CACHE_MAP = {
    QuantAlgo.FP8.value: "fp8",
    QuantAlgo.NVFP4.value: "fp8",
}

ALL_SUPPORTED_BACKENDS = ["pytorch", "_autodeploy", "tensorrt"]


def get_settings_from_engine(
    engine_path: Path
) -> Tuple[Dict[str, Union[str, int]], Dict[str, Union[str, int]]]:
    """Retrieve basic engine information.

    Args:
        engine_path (Path): Path to a TRT-LLM engine directory.

    Returns:
        Tuple[Dict[str, Union[str, int]], Dict[str, Union[str, int]]]: Engine
        properties parsed from the engine at engine_path.
    """
    config_path = engine_path / "config.json"
    runtime_config = {}

    with open(config_path, "r") as config_json:
        config = json.load(config_json)

    engine_world_map = config["pretrained_config"]["mapping"]
    engine_build_cfg = config["build_config"]
    engine_parallel_map = engine_build_cfg["auto_parallel_config"]

    world_config = {
        "pp_size": engine_world_map["pp_size"],
        "tp_size": engine_world_map["tp_size"],
        "world_size": engine_world_map["world_size"],
        "gpus_per_node": engine_parallel_map["gpus_per_node"],
    }

    executor_settings = {
        "max_batch_size": engine_build_cfg["max_batch_size"],
        "max_num_tokens": engine_build_cfg["max_num_tokens"],
    }

    runtime_config.update({
        "sw_version": config["version"],
        "engine_dir": str(engine_path.absolute()),
        "settings_config": executor_settings,
        "world_config": world_config,
    })

    runtime_config["performance_options"] = {}
    runtime_config["decoding_config"] = {
        "decoding_mode": engine_build_cfg["speculative_decoding_mode"]
    }
    return runtime_config, engine_build_cfg


def get_settings(params: dict, dataset_metadata: DatasetMetadata, model: str,
                 model_path: Union[Path, None]) -> Dict[str, Union[str, int]]:
    """Retrieve basic runtime config for pytorch backend path

    Args:
        params (dict): Configuration parameters.
        model (str): Model name.
        model_path (Union[Path, None]): Path to the model.
    Returns:
        Dict[str, Union[str, int]]: Properties for runtime config.
    """
    extra_llm_api_options = params.get("extra_llm_api_options")
    enable_chunked_prefill = params.get("enable_chunked_prefill", False)

    kv_cache_dtype = "auto"
    if extra_llm_api_options:
        with open(extra_llm_api_options, 'r') as f:
            llm_args_dict = yaml.safe_load(f)

        if "kv_cache_dtype" in llm_args_dict:
            kv_cache_dtype = llm_args_dict["kv_cache_dtype"]

        enable_chunked_prefill = llm_args_dict.get("enable_chunked_prefill",
                                                   enable_chunked_prefill)

    world_config = {
        "pp_size": params.get("pp"),
        "tp_size": params.get("tp"),
        "world_size": params.get("pp") * params.get("tp"),
        "ep_size": params.get("ep"),
        "cluster_size": params.get("cluster_size"),
    }

    if params.get("max_batch_size") and params.get("max_num_tokens"):
        logger.info("Use user-provided max batch size and max num tokens.")
        max_batch_size, max_num_tokens = params.get(
            "max_batch_size"), params.get("max_num_tokens")
    else:
        model_config = get_model_config(model, model_path)

        from tensorrt_llm._torch.model_config import ModelConfig
        model = model_path or model
        tllm_model_config = ModelConfig.from_pretrained(model,
                                                        trust_remote_code=True)

        if (kv_cache_dtype is None
                and tllm_model_config.quant_config.kv_cache_quant_algo is None):
            kv_cache_dtype = _KV_CACHE_MAP.get(
                tllm_model_config.quant_config.quant_algo, "auto")

        validate_and_set_kv_cache_quant(tllm_model_config, kv_cache_dtype)

        max_batch_size, max_num_tokens = get_benchmark_engine_settings(
            model_config,
            tllm_model_config.quant_config,
            params.get("tp"),
            params.get("pp"),
            dataset_metadata.avg_isl,
            dataset_metadata.avg_osl,
            params.get("kv_cache_free_gpu_mem_fraction"),
        )

        logger.info(
            f"Max batch size and max num tokens not provided. "
            f"Using heuristics or pre-defined settings: max_batch_size={max_batch_size}, max_num_tokens={max_num_tokens}."
        )

        # If chunked prefill is disabled, we need to ensure that the max_num_tokens is at least the max_isl
        if not enable_chunked_prefill:
            logger.warning(
                f"Chunked prefill is disabled, but max_num_tokens ({max_num_tokens}) is less than the max ISL ({dataset_metadata.max_isl}). "
                f"Forcing max_num_tokens to {dataset_metadata.max_isl + max_batch_size}."
            )
            max_num_tokens = max(max_num_tokens,
                                 dataset_metadata.max_isl + max_batch_size)
        else:
            # TODO: Figure out how to handle chunked block size.
            # Expecting this to be the max of chunk block and max_num_tokens.
            pass

    cuda_graph_config = {
        "enable_padding": True,
        "max_batch_size": max_batch_size
    }

    pyt_options = {
        "cuda_graph_config": cuda_graph_config,
        "kv_cache_dtype": kv_cache_dtype,
    }

    backend = params.get("backend", "pytorch")
    return {
        "sw_version": version("tensorrt_llm"),
        "model_path": model_path,
        "settings_config": {
            "max_batch_size": int(max_batch_size),
            "max_num_tokens": int(max_num_tokens),
            "chunking": enable_chunked_prefill,
        },
        "world_config": world_config,
        "backend": backend,
        "decoding_config": {},
        "performance_options": {
            "cuda_graphs": True,
            "pytorch_config": pyt_options,
        }
    }


def generate_warmup_dataset(requests, steps) -> List[InferenceRequest]:
    """Warm up the benchmarker."""
    warm_up_dataset = choices(requests, k=steps)
    shuffle(warm_up_dataset)
    return warm_up_dataset
