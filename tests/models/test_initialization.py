# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest
from transformers import PretrainedConfig

from vllm import LLM

from ..conftest import MODELS_ON_S3
from .registry import HF_EXAMPLE_MODELS


@pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
def test_can_initialize(model_arch):
    model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)
    model_info.check_available_online(on_fail="skip")
    model_info.check_transformers_version(on_fail="skip")

    # Avoid OOM
    def hf_overrides(hf_config: PretrainedConfig) -> PretrainedConfig:
        hf_config.update(model_info.hf_overrides)

        if hasattr(hf_config, "text_config"):
            text_config: PretrainedConfig = hf_config.text_config
        else:
            text_config = hf_config

        text_config.update({
            "num_layers": 1,
            "num_hidden_layers": 1,
            "num_experts": 2,
            "num_experts_per_tok": 2,
            "num_local_experts": 2,
        })

        return hf_config

    # Avoid calling model.forward()
    def _initialize_kv_caches(self) -> None:
        self.cache_config.num_gpu_blocks = 0
        self.cache_config.num_cpu_blocks = 0

    with patch.object(LLM.get_engine_class(), "_initialize_kv_caches",
                      _initialize_kv_caches):
        model_name = model_info.default
        if model_name in MODELS_ON_S3:
            model_name = f"s3://vllm-ci-model-weights/{model_name.split('/')[-1]}"
        LLM(
            model_name,
            tokenizer=model_info.tokenizer,
            tokenizer_mode=model_info.tokenizer_mode,
            speculative_model=model_info.speculative_model,
            num_speculative_tokens=1 if model_info.speculative_model else None,
            trust_remote_code=model_info.trust_remote_code,
            load_format="dummy",
            hf_overrides=hf_overrides,
        )
