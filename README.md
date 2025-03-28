# AdaLLM

## About

AdaLLM aims to develop an adaptive batching scheduling base on vLLM 0.7.3.

## Getting Started

### Install adaLLM

```bash
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

### Inference with vllm engine

```bash
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  --api-key 123 --dtype half --enable-chunked-prefill False --enforce-eager
```

## Mention

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit).
