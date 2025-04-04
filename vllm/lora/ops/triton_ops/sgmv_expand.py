# SPDX-License-Identifier: Apache-2.0
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023).
Punica: Multi-Tenant LoRA Serving.
https://arxiv.org/abs/2310.18547
"""

from typing import List

import torch
import triton
import triton.language as tl

from vllm.utils import direct_register_custom_op

from .kernel_utils import do_expand_kernel
from .utils import _get_lora_b_ptr


@triton.jit
def _sgmv_expand_kernel(
        input_ptr,
        lora_ptr,
        out_ptr,
        N,
        K,
        b_seq_start_loc,
        seq_lens,
        lora_indices,
        slice_start_loc,
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,  # 1
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,  # 1
        output_d0_stride,
        output_d1_stride,  # 1
        output_hs_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        ADD_INPUTS: tl.constexpr,
        CAST_TYPE: tl.constexpr,
        SLICE_NUM: tl.constexpr,
        SAME_STRIDE: tl.constexpr):
    """

    Similar to the 'sgmv_expand' operator, but with an added parameter
    'slice_offset'. The reason for not reusing the 'sgmv_expand' operator
    might be that in the future, we could implement a fusion operator to
    achieve the current functionality instead of having to call it multiple
    times.
    """
    pid = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    slice_id = tl.program_id(axis=2)
    cta_n_num = tl.cdiv(N, BLOCK_N)
    # When the output dimensions of each slice are the same,cur_n=N, otherwise
    # cur_n=tl.load(output_hs_ptr + slice_id), this situation exists in GQA's
    # qkv linear.
    curr_N = N if SAME_STRIDE else tl.load(output_hs_ptr + slice_id)
    pid_m = pid // cta_n_num
    pid_n = pid % cta_n_num

    M = tl.load(seq_lens + cur_batch)
    if pid_m * BLOCK_M >= M:
        return
    if pid_n * BLOCK_N >= curr_N:
        return
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return

    m_offset = tl.load(b_seq_start_loc + cur_batch)

    cta_m_len = min(BLOCK_M, M - (pid_m * BLOCK_M))
    cta_m_offset = m_offset + (pid_m * BLOCK_M)
    offset_m = tl.arange(0, BLOCK_M)
    ram = cta_m_offset + tl.max_contiguous(
        tl.multiple_of(offset_m % cta_m_len, BLOCK_M), BLOCK_M)
    do_expand_kernel(
        pid_n,
        lora_index,
        slice_id,
        input_ptr,
        lora_ptr,
        out_ptr,
        curr_N,
        K,
        cta_m_len,
        ram,  # array identifying the rows of Input ptr to operate on
        slice_start_loc,
        # input ptr strides
        input_d0_stride,
        input_d1_stride,
        input_d2_stride,
        # lora ptr strides
        ls_d0_ptr,
        ls_d1_ptr,
        ls_d2_ptr,
        # out ptr strides
        output_d0_stride,
        output_d1_stride,
        # constants
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SAME_STRIDE,
        SLICE_NUM,
        EVEN_K,
        CAST_TYPE,
        ADD_INPUTS,
    )


@torch.inference_mode()
def _sgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    """
    Args:
        inputs (torch.Tensor): input tensor
        lora_b_weights (List[torch.Tensor]): lora'b weight
        output_tensor (torch.Tensor): output tensor
        b_seq_start_loc (torch.Tensor): (batch_size,). The cumulative
            sequence lengths of the sequences in the batch, used to index
            into sequence. E.g., if the sequence length is [4, 6], it is
            [0, 4].
        seq_len_tensor (torch.Tensor): (batch_size,). Record the sequence
            length of the sequences in the batch.
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        batches (int): batch size
        max_seq_length (int): The max sequence lengths of the sequences in the 
            batch.
        token_nums (int): The token numbers in the batch. Used to verify if the 
            token numbers in the inputs matches the one in the metadata.
        offset_start (int, optional): Offset start for output_tensor. 
            Defaults to 0.
        add_inputs (bool, optional): Whether to add the input tensor to the 
            output tensor. Defaults to False.
    """
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    for weight in lora_b_weights:
        assert weight.dtype in [torch.float16, torch.bfloat16]

    assert inputs.size(1) == token_nums
    assert inputs.size(0) == len(lora_b_weights)

    assert b_seq_start_loc.size(0) == batches
    assert lora_indices_tensor.size(0) == batches
    assert output_tensor.is_contiguous()
    (slice_start_tensor, lora_ptr_tensor, lora_strides_d0_tensor,
     lora_strides_d1_tensor, lora_strides_d2_tensor, hidden_sizes_tensor,
     same_stride, MAX_N) = _get_lora_b_ptr(lora_b_weights, offset_start,
                                           b_seq_start_loc.device)

    # TODO tuning this config
    K = lora_b_weights[0].shape[-1]  # K= rank

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 16
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = False

    if inputs.dtype == torch.float32 and lora_b_weights[0].dtype in [
            torch.float16,
            torch.bfloat16,
    ]:
        CAST_TYPE = True
    grid = (
        triton.cdiv(max_seq_length, BLOCK_M) * triton.cdiv(MAX_N, BLOCK_N),
        batches,
        len(lora_b_weights),
    )
    _sgmv_expand_kernel[grid](
        inputs,
        lora_ptr_tensor,
        output_tensor,
        MAX_N,
        K,
        b_seq_start_loc,
        seq_len_tensor,
        lora_indices_tensor,
        slice_start_tensor,
        inputs.stride(0),
        inputs.stride(1),
        inputs.stride(2),
        lora_strides_d0_tensor,
        lora_strides_d1_tensor,
        lora_strides_d2_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        hidden_sizes_tensor,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        EVEN_K,
        ADD_INPUTS,
        CAST_TYPE,
        len(lora_b_weights),
        same_stride,
    )
    return


def _sgmv_expand_fake(
    inputs: torch.Tensor,
    lora_b_weights: List[torch.Tensor],
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    offset_start: int = 0,
    add_inputs: bool = False,
) -> None:
    return


try:
    direct_register_custom_op(
        op_name="sgmv_expand",
        op_func=_sgmv_expand,
        mutates_args=["output_tensor"],
        fake_impl=_sgmv_expand_fake,
    )
    sgmv_expand = torch.ops.vllm.sgmv_expand

except AttributeError:
    sgmv_expand = _sgmv_expand
