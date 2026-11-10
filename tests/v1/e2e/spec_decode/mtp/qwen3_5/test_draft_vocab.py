# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import pytest
import torch

from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k_offline
from tests.utils import single_gpu_only
from vllm import SamplingParams
from vllm.config import CompilationConfig

from ...utils import compute_acceptance_len

MODEL = "Qwen/Qwen3.5-0.8B-Base"
# Low ids of a BPE vocabulary are roughly the most frequent tokens.
NUM_DRAFT_TOKENS = 32768
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"


def _check_draft_vocab_after_reload(worker):
    """A real weight reload must refresh storage already used by a head graph."""
    from dataclasses import fields, is_dataclass

    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.draft_vocab import DraftVocab
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        use_workspace_lane,
    )

    runner = worker.model_runner
    draft = runner.get_draft_model()
    head = draft.lm_head
    vocab = draft.draft_vocab
    parameter_name = next(
        name
        for name, parameter in runner.get_model().named_parameters()
        if parameter is head.weight
    )

    def addresses():
        tensors = {"head": head.weight, "token_ids": vocab.token_ids}

        def add_packed(name, value):
            if isinstance(value, torch.Tensor):
                tensors[name] = value
            elif is_dataclass(value):
                for field in fields(value):
                    add_packed(f"{name}.{field.name}", getattr(value, field.name))

        for name in ("basis", "candidate_ids"):
            value = getattr(vocab, name, None)
            if value is not None:
                tensors[name] = value
        for name in ("rows", "scorer"):
            module = getattr(vocab, name, None)
            if module is not None:
                tensors.update(
                    (f"{name}.{key}", value) for key, value in module.named_parameters()
                )
                tensors.update(
                    (f"{name}.{key}", value) for key, value in module.named_buffers()
                )
                add_packed(
                    f"{name}.packed", getattr(module, "b12x_nvfp4_packed_weight", None)
                )
        tensors.update(
            (f"gather_rows.{i}", value)
            for i, value in enumerate(getattr(vocab, "gather_rows", ()))
        )
        return {name: tensor.data_ptr() for name, tensor in tensors.items()}

    # FULL draft-prefill capture already warmed this stream's kernel resources.
    manager = runner.speculator.prefill_cudagraph_manager
    assert manager is not None
    stream = manager.capture_stream
    assert stream is not None and stream.cuda_stream != 0
    caller_stream = torch.cuda.current_stream()
    stream.wait_stream(caller_stream)
    try:
        with (
            torch.cuda.stream(stream),
            use_workspace_lane(runner._draft_workspace_lane),
            torch.inference_mode(),
            set_current_vllm_config(worker.vllm_config),
        ):
            original = head.weight.detach().clone()
            hidden = original[:2].clone()
            before_addresses = addresses()
            workspace = current_workspace_manager()
            assert workspace.is_locked()
            for _ in range(3):
                vocab.compute_logits(head, hidden)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                captured_logits = vocab.compute_logits(head, hidden)

            def check(phase):
                assert workspace.is_locked()
                fresh = DraftVocab(vocab.logits_processor, vocab.vocab_size)
                fresh.load_token_map(
                    head,
                    vocab.token_ids,
                    vocab.num_dynamic_rows,
                    worker.vllm_config.speculative_config.draft_lm_head_quantization,
                )
                expected = fresh.compute_logits(head, hidden)
                graph.replay()
                try:
                    torch.testing.assert_close(
                        captured_logits, expected, rtol=0, atol=0
                    )
                    assert addresses() == before_addresses
                except AssertionError as error:
                    raise AssertionError(f"{phase}: {error}") from error

            def reload(weight):
                worker.reload_weights(
                    weights_iterator=[(parameter_name, weight)],
                    is_checkpoint_format=False,
                )

            check("initial")
            reload(original)
            check("no_op")
            # Nonuniform scaling changes relative logits and the dynamic projection.
            scale = torch.linspace(
                0.5,
                1.5,
                original.shape[0],
                device=original.device,
                dtype=torch.float32,
            ).to(original.dtype)
            updated = original * scale[:, None]
            assert not torch.equal(updated, original)
            try:
                reload(updated)
                check("updated")
            finally:
                reload(original)
            check("restored")
    finally:
        caller_stream.wait_stream(stream)


@single_gpu_only
@pytest.mark.parametrize("dynamic_rows", [0, 16])
def test_mtp_draft_token_map_refreshes_after_weight_reload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    vllm_runner,
    enable_pickle,
    dynamic_rows: int,
):
    """Reloaded target weights reach sparse drafting without replacing graph storage."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    token_map = tmp_path / "draft_vocab.json"
    token_map.write_text(json.dumps(list(range(NUM_DRAFT_TOKENS))))
    with vllm_runner(
        MODEL,
        revision=MODEL_REVISION,
        tokenizer_revision=MODEL_REVISION,
        dtype="bfloat16",
        max_model_len=2048,
        max_num_seqs=2,
        # Reserve space for saved weights and fresh reference layers in the fixture.
        gpu_memory_utilization=0.65,
        block_size=None,
        enable_chunked_prefill=None,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(),
        limit_mm_per_prompt={"image": 0, "video": 0},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 1,
            "draft_token_map": str(token_map),
            "draft_token_map_dynamic_rows": dynamic_rows,
        },
    ) as runner:
        params = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
        prompt = "The capital of France is"
        runner.llm.generate(prompt, params)
        runner.llm.collective_rpc(_check_draft_vocab_after_reload)
        outputs = runner.llm.generate(prompt, params)
        assert len(outputs[0].outputs[0].token_ids) == 8


@single_gpu_only
def test_mtp_draft_token_map(monkeypatch: pytest.MonkeyPatch, tmp_path, vllm_runner):
    """MTP with a draft token map keeps GSM8K above the stock-MTP reference
    threshold, and its acceptance length within 10% of stock MTP."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    token_map = tmp_path / "draft_vocab.json"
    token_map.write_text(json.dumps(list(range(NUM_DRAFT_TOKENS))))

    acceptance_lens = []
    for draft_token_map in (None, str(token_map)):
        with vllm_runner(
            MODEL,
            max_model_len=2048,
            block_size=None,
            enable_chunked_prefill=None,
            compilation_config=CompilationConfig(),
            limit_mm_per_prompt={"image": 0, "video": 0},
            disable_log_stats=False,
            speculative_config={
                "method": "mtp",
                "num_speculative_tokens": 1,
                "draft_token_map": draft_token_map,
                "draft_token_map_dynamic_rows": 1024 if draft_token_map else 0,
            },
        ) as runner:
            accuracy = evaluate_gsm8k_offline(runner.llm, num_questions=200)["accuracy"]
            acceptance_lens.append(compute_acceptance_len(runner.llm.get_metrics()))
        # Stock GSM8K for this model is about 34% (see test_mtp.py).
        assert accuracy >= 0.20, f"{draft_token_map=}: GSM8K {accuracy:.3f}"

    stock, reduced = acceptance_lens
    print(f"Acceptance length: stock {stock:.3f}, draft_token_map {reduced:.3f}")
    assert reduced > 1.0
    assert reduced >= 0.9 * stock
