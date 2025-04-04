# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm import LLM
from vllm.config import LoadFormat


@pytest.fixture(autouse=True)
def v1(run_with_both_engines):
    # Simple autouse wrapper to run both engines for each test
    # This can be promoted up to conftest.py to run for every
    # test in a package
    pass


def test_empty_prompt():
    llm = LLM(model="s3://vllm-ci-model-weights/gpt2",
              load_format=LoadFormat.RUNAI_STREAMER,
              enforce_eager=True)
    with pytest.raises(ValueError, match='Prompt cannot be empty'):
        llm.generate([""])


@pytest.mark.skip_v1
def test_out_of_vocab_token():
    llm = LLM(model="s3://vllm-ci-model-weights/gpt2",
              load_format=LoadFormat.RUNAI_STREAMER,
              enforce_eager=True)
    with pytest.raises(ValueError, match='out of vocabulary'):
        llm.generate({"prompt_token_ids": [999999]})
