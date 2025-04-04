# SPDX-License-Identifier: Apache-2.0
"""Compare the outputs of HF and vLLM when using beam search.

Run `pytest tests/samplers/test_beam_search.py`.
"""

import pytest

# FIXME(zhuohan): The test can not pass if we:
#   1. Increase max_tokens to 256.
#   2. Increase beam_width to 8.
#   3. Use the model "huggyllama/llama-7b".
MAX_TOKENS = [64]
BEAM_WIDTHS = [4]
MODELS = ["TinyLlama/TinyLlama-1.1B-Chat-v1.0"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_tokens", MAX_TOKENS)
@pytest.mark.parametrize("beam_width", BEAM_WIDTHS)
def test_beam_search_single_input(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_tokens: int,
    beam_width: int,
) -> None:
    example_prompts = example_prompts[:1]
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_beam_search(example_prompts, beam_width,
                                                   max_tokens)

    with vllm_runner(model, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_beam_search(example_prompts,
                                                       beam_width, max_tokens)

    for i in range(len(example_prompts)):
        hf_output_ids, hf_output_texts = hf_outputs[i]
        vllm_output_ids, vllm_output_texts = vllm_outputs[i]
        for i, (hf_text,
                vllm_text) in enumerate(zip(hf_output_texts,
                                            vllm_output_texts)):
            print(f">>>{i}-th hf output:")
            print(hf_text)
            print(f">>>{i}-th vllm output:")
            print(vllm_text)
        assert len(hf_output_ids) == len(vllm_output_ids)
        for j in range(len(hf_output_ids)):
            assert hf_output_ids[j] == vllm_output_ids[j], (
                f"Test{i} output{j}:\nHF: {hf_output_ids}\n"
                f"vLLM: {vllm_output_ids}")
