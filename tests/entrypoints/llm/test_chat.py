# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest

from vllm import LLM
from vllm.config import LoadFormat

from ...conftest import MODEL_WEIGHTS_S3_BUCKET
from ..openai.test_vision import TEST_IMAGE_URLS

RUNAI_STREAMER_LOAD_FORMAT = LoadFormat.RUNAI_STREAMER


def test_chat():
    llm = LLM(model=f"{MODEL_WEIGHTS_S3_BUCKET}/Llama-3.2-1B-Instruct",
              load_format=RUNAI_STREAMER_LOAD_FORMAT)

    prompt1 = "Explain the concept of entropy."
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt1
        },
    ]
    outputs = llm.chat(messages)
    assert len(outputs) == 1


def test_multi_chat():
    llm = LLM(model=f"{MODEL_WEIGHTS_S3_BUCKET}/Llama-3.2-1B-Instruct",
              load_format=RUNAI_STREAMER_LOAD_FORMAT)

    prompt1 = "Explain the concept of entropy."
    prompt2 = "Explain what among us is."

    conversation1 = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt1
        },
    ]

    conversation2 = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": prompt2
        },
    ]

    messages = [conversation1, conversation2]

    outputs = llm.chat(messages)
    assert len(outputs) == 2


@pytest.mark.parametrize("image_urls",
                         [[TEST_IMAGE_URLS[0], TEST_IMAGE_URLS[1]]])
def test_chat_multi_image(image_urls: List[str]):
    llm = LLM(
        model=f"{MODEL_WEIGHTS_S3_BUCKET}/Phi-3.5-vision-instruct",
        load_format=RUNAI_STREAMER_LOAD_FORMAT,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=5,
        enforce_eager=True,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 2},
    )

    messages = [{
        "role":
        "user",
        "content": [
            *({
                "type": "image_url",
                "image_url": {
                    "url": image_url
                }
            } for image_url in image_urls),
            {
                "type": "text",
                "text": "What's in this image?"
            },
        ],
    }]
    outputs = llm.chat(messages)
    assert len(outputs) >= 0
