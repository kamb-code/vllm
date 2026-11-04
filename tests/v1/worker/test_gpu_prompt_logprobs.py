# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import HAS_TRITON
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

NUM_DRAFTS = 4


def _batch(computed, prompt_len, query_len, device):
    buffers = InputBuffers(1, query_len, device)
    batch = InputBatch.make_dummy(1, query_len, buffers, is_padding=False)
    computed_tokens = np.array([computed], dtype=np.int32)
    return replace(
        batch,
        req_ids=["0"],
        num_computed_tokens_np=computed_tokens,
        num_computed_prefill_tokens_np=computed_tokens,
        prefill_len_np=np.array([prompt_len], dtype=np.int32),
        is_prefilling_np=np.array([computed < prompt_len]),
        has_prefill=computed < prompt_len,
    )


@pytest.mark.skipif(not HAS_TRITON, reason="Requires Triton kernels")
class TestPromptLogprobs:
    @pytest.mark.parametrize(
        "prompt_len, computed, num_drafts, prompt_logprob_start",
        [
            (8, 0, 0, 0),
            (8, 7, NUM_DRAFTS, 0),
            (1, 0, NUM_DRAFTS, 0),
            (8, 0, NUM_DRAFTS, 2),
        ],
    )
    def test_scores_stop_at_the_end_of_the_prompt(
        self, prompt_len, computed, num_drafts, prompt_logprob_start
    ):
        device = torch.device(current_platform.device_type)
        worker = PromptLogprobsWorker(1, device)
        worker.add_request(
            "0",
            0,
            SamplingParams(
                prompt_logprobs=0, prompt_logprob_start=prompt_logprob_start
            ),
        )
        query_len = prompt_len - computed + num_drafts
        result = worker.compute_prompt_logprobs(
            lambda hidden: torch.zeros(len(hidden), 32, device=device),
            torch.zeros(query_len, 4, device=device),
            _batch(computed, prompt_len, query_len, device),
            torch.zeros(1, 32, dtype=torch.int32, device=device),
            torch.tensor([computed], dtype=torch.int32, device=device),
            np.array([prompt_len]),
        )
        num_rows = result["0"].logprobs.shape[0] if result else 0
        assert num_rows == max(prompt_len - max(computed, prompt_logprob_start) - 1, 0)

    @pytest.mark.parametrize("first_chunk_len", [3, 7])
    def test_chunked_scores_preserved_at_prompt_end(self, first_chunk_len):
        """Keep earlier scores when the final chunk adds some or no prompt rows."""
        device = torch.device(current_platform.device_type)
        prompt_len = 8
        vocab_size = 32
        all_token_ids = torch.arange(
            vocab_size, dtype=torch.int32, device=device
        ).unsqueeze(0)
        hidden_states = torch.arange(
            prompt_len + NUM_DRAFTS, dtype=torch.float32, device=device
        ).unsqueeze(1)
        vocabulary = torch.arange(vocab_size, device=device).unsqueeze(0)

        def logits_fn(hidden):
            # Distinct scores per position expose lost, repeated or reordered rows.
            return -(vocabulary - hidden[:, :1]).square()

        def compute(worker, computed, query_len):
            return worker.compute_prompt_logprobs(
                logits_fn,
                hidden_states[computed : computed + query_len],
                _batch(computed, prompt_len, query_len, device),
                all_token_ids,
                torch.tensor([computed], dtype=torch.int32, device=device),
                np.array([prompt_len]),
            )

        reference_worker = PromptLogprobsWorker(1, device)
        reference_worker.add_request("0", 0, SamplingParams(prompt_logprobs=0))
        # A whole prompt without placeholders is correct even before this fix.
        expected = compute(reference_worker, 0, prompt_len)["0"]
        assert expected.logprobs.shape == (prompt_len - 1, 1)

        worker = PromptLogprobsWorker(1, device)
        worker.add_request("0", 0, SamplingParams(prompt_logprobs=0))
        assert compute(worker, 0, first_chunk_len) == {}
        result = compute(
            worker, first_chunk_len, prompt_len - first_chunk_len + NUM_DRAFTS
        )

        assert set(result) == {"0"}
        actual = result["0"]
        for field in ("logprob_token_ids", "logprobs", "selected_token_ranks"):
            torch.testing.assert_close(
                getattr(actual, field), getattr(expected, field), rtol=0, atol=0
            )
