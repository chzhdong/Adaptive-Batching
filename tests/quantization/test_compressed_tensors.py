# SPDX-License-Identifier: Apache-2.0
"""Test model set-up and weight loading for llmcompressor-quantized models.

Run `pytest tests/quantization/test_compressed_tensors.py`.
"""

from typing import Optional

import pytest
import torch
from compressed_tensors.quantization import QuantizationType

from tests.models.utils import check_logprobs_close
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensors24, CompressedTensorsLinearMethod,
    CompressedTensorsW4A16Sparse24, CompressedTensorsW8A8Fp8,
    CompressedTensorsW8A8Int8, CompressedTensorsW8A16Fp8,
    CompressedTensorsWNA16)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    sparse_cutlass_supported)
from vllm.platforms import current_platform


@pytest.mark.parametrize(
    "model_args",
    [
        (
            "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change",
            "tensor",
            QuantizationType.INT,
            2560,
            True,
        ),
        (
            "nm-testing/tinyllama-oneshot-w8-channel-a8-tensor",
            "channel",
            QuantizationType.INT,
            2560,
            True,
        ),
        (
            "nm-testing/asym-w8w8-int8-static-per-tensor-tiny-llama",
            "tensor",
            QuantizationType.INT,
            2560,
            False,
        ),
    ],
)
def test_compressed_tensors_w8a8_static_setup(vllm_runner, model_args):
    model_path, strategy, quant_type, shape_0, is_symmetric = model_args
    with vllm_runner(model_path, enforce_eager=True) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            o_proj = layer.self_attn.o_proj
            gate_up_proj = layer.mlp.gate_up_proj
            down_proj = layer.mlp.down_proj

            # assert zp for symmetric and asymmetric cases
            def zp_valid(zp: Optional[torch.Tensor]):
                if is_symmetric:
                    return zp is None

                return zp is not None and zp.dtype is torch.int32

            assert zp_valid(qkv_proj.input_zero_point)
            assert zp_valid(o_proj.input_zero_point)
            assert zp_valid(gate_up_proj.input_zero_point)
            assert zp_valid(down_proj.input_zero_point)

            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(o_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(gate_up_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(down_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8Int8)

            assert qkv_proj.scheme.strategy == strategy
            assert qkv_proj.scheme.is_static_input_scheme
            expected_type = torch.int8

            assert qkv_proj.weight.dtype is expected_type
            assert o_proj.weight.dtype is expected_type
            assert gate_up_proj.weight.dtype is expected_type

            if qkv_proj.scheme.strategy == "tensor":
                # Make sure it is a channelwise buffer
                # After running process_weights_after_loading
                assert len(qkv_proj.weight_scale.shape) == 2
                assert qkv_proj.weight_scale.shape[0] == shape_0
                assert qkv_proj.weight_scale.shape[1] == 1
            assert qkv_proj.weight_scale.dtype is torch.float32
            assert qkv_proj.input_scale.dtype is torch.float32

        llm.apply_model(check_model)

        output = llm.generate_greedy(["Hello my name is"], max_tokens=20)
        assert output


@pytest.mark.parametrize(
    "model_path",
    [
        "neuralmagic/Llama-3.2-1B-quantized.w8a8",
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Dynamic-Asym",
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Sym",
        "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Asym",
    ],
)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("num_logprobs", [10])
def test_compressed_tensors_w8a8_logprobs(
    hf_runner,
    vllm_runner,
    example_prompts,
    model_path,
    max_tokens,
    num_logprobs,
):
    dtype = "bfloat16"

    # skip language translation prompt for the static per tensor asym model
    if (model_path ==
            "nm-testing/Meta-Llama-3-8B-Instruct-W8A8-Static-Per-Tensor-Asym"
        ):  # noqa: E501
        example_prompts = example_prompts[0:-1]

    with hf_runner(model_path, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs_limit(
            example_prompts, max_tokens, num_logprobs)

    with vllm_runner(model_path, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy_logprobs(
            example_prompts, max_tokens, num_logprobs)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
    )


def test_compressed_tensors_no_enforce_eager(vllm_runner):
    model_path = "nm-testing/tinyllama-oneshot-w8w8-test-static-shape-change"
    with vllm_runner(model_path) as llm:
        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


@pytest.mark.parametrize(
    "model_args",
    [
        ("nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2", "tensor"),
        ("nm-testing/tinyllama-oneshot-w8a8-dynamic-token-v2-asym", "tensor"),
        (
            "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2",
            "channel",
        ),
        (
            "nm-testing/tinyllama-oneshot-w8a8-channel-dynamic-token-v2-asym",
            "channel",
        ),
    ],
)
def test_compressed_tensors_w8a8_dynamic_per_token(vllm_runner, model_args):
    model_path, strategy = model_args
    with vllm_runner(model_path, dtype=torch.float16) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensorsW8A8Int8)
            assert not qkv_proj.scheme.is_static_input_scheme
            assert qkv_proj.scheme.strategy == strategy
            assert qkv_proj.weight.dtype is torch.int8

        llm.apply_model(check_model)

        output = llm.generate_greedy(["Hello my name is"], max_tokens=20)
        assert output


@pytest.mark.parametrize(
    "wNa16_args",
    [
        ("nm-testing/tinyllama-oneshot-w4a16-channel-v2", "channel", None, 8),
        ("nm-testing/tinyllama-oneshot-w4a16-group128-v2", "group", 128, 8),
        ("nm-testing/tinyllama-oneshot-w8a16-per-channel", "channel", None, 4),
    ],
)
def test_compressed_tensors_wNa16(vllm_runner, wNa16_args):
    model, strategy, group, pack_factor = wNa16_args
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensorsWNA16)

            assert qkv_proj.scheme.strategy == strategy
            assert qkv_proj.scheme.group_size == (-1
                                                  if group is None else group)

            assert qkv_proj.weight_packed.dtype is torch.int32
            assert qkv_proj.weight_scale.dtype is torch.float16
            assert qkv_proj.scheme.pack_factor == pack_factor

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_w4a16_marlin24(vllm_runner):
    model_path = "nm-testing/llama7b-one-shot-2_4-w4a16-marlin24-t"
    with vllm_runner(model_path) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensorsW4A16Sparse24)
            assert qkv_proj.weight_packed.dtype is torch.int32

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_fp8(vllm_runner):
    model_path = "nm-testing/Meta-Llama-3-8B-FP8-compressed-tensors-test"
    with vllm_runner(model_path) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj

            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(
                qkv_proj.scheme,
                (CompressedTensorsW8A8Fp8, CompressedTensorsW8A16Fp8),
            )

            assert qkv_proj.input_scale.dtype is torch.float32

            if isinstance(qkv_proj.scheme, CompressedTensorsW8A8Fp8):
                assert len(qkv_proj.input_scale.shape) == 0
                assert qkv_proj.weight.dtype is torch.float8_e4m3fn
                assert qkv_proj.weight_scale.dtype is torch.float32
                assert len(qkv_proj.weight_scale.shape) == 0

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        assert output


def test_compressed_tensors_kv_cache(vllm_runner):
    model_path = "nm-testing/TinyLlama-1.1B-compressed-tensors-kv-cache-scheme"
    with vllm_runner(model_path, kv_cache_dtype="fp8") as llm:
        output = llm.generate_greedy("Hello world!", max_tokens=20)
        assert output


@pytest.mark.skipif(
    not sparse_cutlass_supported(),
    reason="Sparse FP8 is not yet supported on this GPU type.",
)
def _test_2of4_quant_models(qkv_proj,
                            weight_strategy,
                            input_strategy,
                            format="dense"):
    assert isinstance(qkv_proj.quant_method, CompressedTensorsLinearMethod)
    assert isinstance(qkv_proj.scheme, CompressedTensors24)

    assert qkv_proj.scheme.weight_quant.strategy == weight_strategy
    assert qkv_proj.scheme.input_quant.strategy == input_strategy
    assert qkv_proj.scheme.quantized
    assert qkv_proj.quant_method.quantization_config.sparsity_scheme_map
    sparsity_map = qkv_proj.quant_method.quantization_config.sparsity_scheme_map  # noqa: E501
    assert sparsity_map.get("Linear").format == format
    assert sparsity_map.get("Linear").sparsity_structure == "2:4"


@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Sparse FP8 is not yet supported on this GPU type.",
)
@pytest.mark.parametrize(
    "args_2of4",
    [
        (
            "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Dynamic-2of4-testing",
            "channel",
            "token",
        ),
        (
            "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Static-Per-Tensor-testing",
            "channel",
            "tensor",
        ),
        (
            "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Static-testing",
            "tensor",
            "tensor",
        ),
        (
            "nm-testing/Meta-Llama-3-8B-Instruct-FP8-Dynamic-IA-Per-Tensor-Weight-testing",
            "tensor",
            "token",
        ),
    ],
)
def test_compressed_tensors_2of4_quant_fp8(vllm_runner, args_2of4):
    model, weight_strategy, input_strategy = args_2of4
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert qkv_proj.scheme.weights_dtype == torch.float8_e4m3fn
            _test_2of4_quant_models(qkv_proj, weight_strategy, input_strategy)

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        print(output)
        assert output


@pytest.mark.skipif(
    not current_platform.has_device_capability(90),
    reason="Sparse FP8 is not yet supported on this GPU type.",
)
@pytest.mark.parametrize(
    "args_2of4",
    [
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_per_tok_dyn_act_fp8-BitM",
            "channel",
            "token",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_tensor_act_fp8-BitM",
            "channel",
            "tensor",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_per_tok_dyn_act_fp8-BitM",
            "tensor",
            "token",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_tensor_act_fp8-BitM",
            "tensor",
            "tensor",
        ),
    ],
)
def test_compressed_tensors_2of4_quant_fp8_compressed(vllm_runner, args_2of4):
    model, weight_strategy, input_strategy = args_2of4
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert qkv_proj.scheme.weights_dtype == torch.float8_e4m3fn
            _test_2of4_quant_models(
                qkv_proj,
                weight_strategy,
                input_strategy,
                format="sparse-24-bitmask",
            )

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        print(output)
        assert output


@pytest.mark.skipif(
    not sparse_cutlass_supported(),
    reason="cutlass is not yet supported on this GPU type.",
)
@pytest.mark.parametrize(
    "args_2of4",
    [
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_per_tok_dyn_act_int8-BitM",
            "channel",
            "token",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-chnl_wts_tensor_act_int8-BitM",
            "channel",
            "tensor",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_per_tok_dyn_act_int8-BitM",
            "tensor",
            "token",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-gsm8k-pruned.2of4-tensor_wts_tensor_act_int8-BitM",
            "tensor",
            "tensor",
        ),
    ],
)
def test_compressed_tensors_2of4_quant_int8_compressed(vllm_runner, args_2of4):
    model, weight_strategy, input_strategy = args_2of4
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert qkv_proj.scheme.weights_dtype == torch.int8
            _test_2of4_quant_models(
                qkv_proj,
                weight_strategy,
                input_strategy,
                format="sparse-24-bitmask",
            )

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        print(output)
        assert output


@pytest.mark.skipif(
    not sparse_cutlass_supported(),
    reason="Sparse FP8 is not yet supported on this GPU type.",
)
@pytest.mark.parametrize(
    "args_2of4",
    [
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Dynamic-IA-Per-Channel-Weight-testing",
            "channel",
            "token",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Static-testing",
            "tensor",
            "tensor",
        ),
        (
            "nm-testing/TinyLlama-1.1B-Chat-v1.0-INT8-Dynamic-IA-Per-Tensor-Weight-testing",
            "tensor",
            "token",
        ),
    ],
)
def test_compressed_tensors_2of4_quant_int8(vllm_runner, args_2of4):
    model, weight_strategy, input_strategy = args_2of4
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert qkv_proj.scheme.weights_dtype == torch.int8
            _test_2of4_quant_models(qkv_proj, weight_strategy, input_strategy)

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        print(output)
        assert output


@pytest.mark.skipif(
    not sparse_cutlass_supported(),
    reason="2of4 Sparse is not yet supported on this GPU type.",
)
@pytest.mark.parametrize(
    "args_2of4",
    [("nm-testing/TinyLlama-1.1B-Chat-v1.0-2of4-Sparse-Dense-Compressor")],
)
def test_compressed_tensors_2of4_sparse(vllm_runner, args_2of4):
    model = args_2of4
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensors24)

            assert qkv_proj.scheme.weight_quant is None
            assert qkv_proj.scheme.input_quant is None
            assert not qkv_proj.scheme.quantized
            assert qkv_proj.quant_method.quantization_config.sparsity_scheme_map
            sparsity_map = (
                qkv_proj.quant_method.quantization_config.sparsity_scheme_map
            )  # noqa: E501
            assert sparsity_map.get("Linear").format == "dense"
            assert sparsity_map.get("Linear").sparsity_structure == "2:4"

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        print(output)
        assert output


@pytest.mark.skipif(
    not sparse_cutlass_supported(),
    reason="Cutlass is not yet supported on this GPU type.",
)
@pytest.mark.parametrize(
    "args_2of4", [("nm-testing/llama2.c-stories42M-pruned2.4-compressed")])
def test_compressed_tensors_2of4_sparse_compressed(vllm_runner, args_2of4):
    model = args_2of4
    with vllm_runner(model) as llm:

        def check_model(model):
            layer = model.model.layers[0]

            qkv_proj = layer.self_attn.qkv_proj
            assert isinstance(qkv_proj.quant_method,
                              CompressedTensorsLinearMethod)
            assert isinstance(qkv_proj.scheme, CompressedTensors24)

            assert qkv_proj.scheme.weight_quant is None
            assert qkv_proj.scheme.input_quant is None
            assert not qkv_proj.scheme.quantized
            assert qkv_proj.quant_method.quantization_config.sparsity_scheme_map
            sparsity_map = (
                qkv_proj.quant_method.quantization_config.sparsity_scheme_map
            )  # noqa: E501
            assert sparsity_map.get("Linear").format == "sparse-24-bitmask"
            assert sparsity_map.get("Linear").sparsity_structure == "2:4"

        llm.apply_model(check_model)

        output = llm.generate_greedy("Hello my name is", max_tokens=20)
        print(output)
        assert output
