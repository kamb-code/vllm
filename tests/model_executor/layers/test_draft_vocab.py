# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.draft_vocab import (
    DraftVocab,
    load_draft_token_ids,
    refresh_draft_token_maps,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.online.lm_head import (
    refresh_quantized_lm_heads,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.v1.worker.gpu.spec_decode.eagle.utils import share_target_lm_head

VOCAB = 200
HIDDEN = 32
DRAFT_IDS = [0, 3, 7, 50, 51, 128, 130, 199]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _head(vocab: int = VOCAB, device: str = "cpu") -> ParallelLMHead:
    head = ParallelLMHead(vocab, HIDDEN, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    head.weight.data.copy_(torch.randn(head.weight.shape, generator=gen))
    head = head.to(device)
    head.quant_method.process_weights_after_loading(head)
    return head


def _hidden(n: int = 64, device: str = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(1)
    return torch.randn(n, HIDDEN, generator=gen).to(device)


@pytest.mark.cpu_test
def test_load_draft_token_ids(tmp_path):
    path = tmp_path / "map.pt"
    torch.save(torch.tensor([9, 2, 9, 4]), path)
    assert load_draft_token_ids(str(path), extra_ids=[7, 2]).tolist() == [2, 4, 7, 9]


@pytest.mark.cpu_test
@pytest.mark.parametrize("refresh", [False, True])
def test_pruned_head_maps_draft_ids_to_target_ids(default_vllm_config, refresh):
    """EAGLE3/DFlash pruned heads: logits land on their d2t target ids."""
    head = _head(len(DRAFT_IDS))
    draft_vocab = DraftVocab(LogitsProcessor(len(DRAFT_IDS)), VOCAB)
    d2t = draft_vocab.draft_id_to_target_id
    assert d2t is not None
    d2t.data.copy_(torch.tensor(DRAFT_IDS) - torch.arange(len(DRAFT_IDS)))
    mapping = d2t.detach().clone()
    mapping_pointer = d2t.data_ptr()
    if refresh:
        drafter = torch.nn.Module()
        drafter.lm_head = head
        drafter.draft_vocab = draft_vocab
        refresh_draft_token_maps(drafter)
        assert draft_vocab.token_ids is None and draft_vocab.rows is None
        assert d2t.data_ptr() == mapping_pointer
        assert torch.equal(d2t, mapping)

    hidden = _hidden()
    logits = draft_vocab.compute_logits(head, hidden)
    assert torch.equal(logits[:, DRAFT_IDS], draft_vocab.logits_processor(head, hidden))
    assert torch.isinf(logits).sum() == hidden.shape[0] * (VOCAB - len(DRAFT_IDS))
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("num_rows", [0, 16])
def test_token_map_drafts_over_listed_and_picked_rows(
    default_vllm_config, dist_init, device, num_rows
):
    """Each token drafts over the list plus `num_rows` picked rows, with the
    exact logits of the full head on all of them."""
    head = _head(device=device)
    hidden = _hidden(device=device)
    draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    full = draft_vocab.compute_logits(head, hidden).float()
    draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS), num_rows)

    logits = draft_vocab.compute_logits(head, hidden).float()
    kept = logits.isfinite()
    assert torch.all(kept[:, DRAFT_IDS])
    assert torch.all(kept.sum(dim=-1) == len(DRAFT_IDS) + num_rows)
    torch.testing.assert_close(logits[kept], full[kept], atol=1e-4, rtol=1e-4)
    assert torch.equal(draft_vocab.get_top_tokens(head, hidden), logits.argmax(-1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="quantized rows need CUDA")
@pytest.mark.parametrize(
    ("quantization", "max_rel_err"), [("fp8", 0.05), ("nvfp4", 0.15)]
)
@pytest.mark.parametrize("extra_listed_id", [False, True])
def test_quantized_rows_track_bf16_rows(
    default_vllm_config, dist_init, quantization, max_rel_err, extra_listed_id
):
    """Quantized rows preserve logical token IDs, including unaligned maps."""
    vocab, hidden_size = 4000, 512
    head = ParallelLMHead(vocab, hidden_size, disable_tp=True)
    gen = torch.Generator().manual_seed(0)
    weight = torch.randn(head.weight.shape, generator=gen) * 0.02
    token_ids = torch.arange(0, vocab, 4)
    if extra_listed_id:
        token_ids = torch.cat((token_ids, torch.tensor([1]))).sort().values
        weight = weight.abs()
    head.weight.data.copy_(weight)
    head = head.to("cuda", torch.bfloat16)
    hidden = torch.randn(
        512,
        hidden_size,
        generator=torch.Generator(device="cuda").manual_seed(1),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if extra_listed_id:
        # Dummy zero rows must not win when every real logit is negative.
        hidden[-1].fill_(-1)

    draft_vocab = DraftVocab(LogitsProcessor(vocab), vocab)
    draft_vocab.load_token_map(head, token_ids, 256, quantization)
    out = draft_vocab.compute_logits(head, hidden).float()
    full = LogitsProcessor(vocab)(head, hidden).float()
    kept = out.isfinite()
    assert out.shape == (hidden.shape[0], vocab)
    assert torch.all(kept[:, token_ids])
    assert torch.all(kept.sum(dim=-1) == len(token_ids) + 256)
    if extra_listed_id:
        assert torch.all(out[-1, kept[-1]] < 0)
    assert (out[kept] - full[kept]).norm() / full[kept].norm() < max_rel_err


@pytest.mark.cpu_test
@pytest.mark.parametrize("use_token_map", [False, True])
def test_draft_logits_follow_shared_head_reload(
    default_vllm_config, dist_init, use_token_map
):
    """Rows derived from a shared head must reflect a completed weight reload."""
    head = _head()
    record_metadata_for_reloading(head)
    drafter = torch.nn.Module()
    assert share_target_lm_head(drafter, head, None) is head
    drafter.draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    if use_token_map:
        drafter.draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS))
    hidden = _hidden(n=2)
    before = drafter.draft_vocab.compute_logits(drafter.lm_head, hidden)
    new_weight = head.weight[:VOCAB].detach().clone() * 2

    initialize_layerwise_reload(head)
    head.weight.weight_loader(head.weight, new_weight)
    finalize_layerwise_reload(head, None)
    refresh_quantized_lm_heads(drafter)
    refresh_draft_token_maps(drafter)

    assert drafter.lm_head is head
    assert torch.equal(head.weight[:VOCAB], new_weight)
    expected = LogitsProcessor(VOCAB)(head, hidden)[:, DRAFT_IDS]
    assert not torch.equal(before[:, DRAFT_IDS], expected)
    actual = drafter.draft_vocab.compute_logits(drafter.lm_head, hidden)
    torch.testing.assert_close(actual[:, DRAFT_IDS], expected, atol=1e-4, rtol=1e-4)


def _draft_vocab_tensors(draft_vocab: DraftVocab) -> dict[str, torch.Tensor]:
    """Collect the derived storage that an already-captured graph can read."""
    assert draft_vocab.token_ids is not None
    tensors = {"token_ids": draft_vocab.token_ids}
    for name in ("rows", "scorer"):
        module = getattr(draft_vocab, name, None)
        if module is not None:
            tensors.update(
                (f"{name}.{key}", tensor)
                for key, tensor in (
                    dict(module.named_parameters()) | dict(module.named_buffers())
                ).items()
            )
    if draft_vocab.num_dynamic_rows:
        tensors["candidate_ids"] = draft_vocab.candidate_ids
        tensors["basis"] = draft_vocab.basis
        tensors.update(
            (f"gather_rows.{index}", tensor)
            for index, tensor in enumerate(draft_vocab.gather_rows)
        )
    return tensors


@pytest.mark.cpu_test
@pytest.mark.parametrize("dynamic_rows", [0, 16])
def test_token_map_refresh_matches_fresh_after_repeated_reload(
    default_vllm_config, dist_init, dynamic_rows
):
    """No-op and changed checkpoints refresh all sparse state at graph addresses."""
    head = _head()
    record_metadata_for_reloading(head)
    drafter = torch.nn.Module()
    share_target_lm_head(drafter, head, None)
    drafter.draft_vocab = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
    drafter.draft_vocab.load_token_map(head, torch.tensor(DRAFT_IDS), dynamic_rows)
    hidden = _hidden(n=4)
    pointers = {
        name: tensor.data_ptr()
        for name, tensor in _draft_vocab_tensors(drafter.draft_vocab).items()
    }
    head_pointer = head.weight.data_ptr()
    previous = drafter.draft_vocab.compute_logits(head, hidden)

    # Exercise an unchanged checkpoint, two independent updates, and another
    # unchanged checkpoint after the stored copies have already been refreshed.
    for seed in (None, 17, 29, None):
        weight = (
            head.weight[:VOCAB].detach().clone()
            if seed is None
            else torch.randn(
                VOCAB, HIDDEN, generator=torch.Generator().manual_seed(seed)
            )
        )
        initialize_layerwise_reload(head)
        head.weight.weight_loader(head.weight, weight)
        finalize_layerwise_reload(head, None)
        refresh_quantized_lm_heads(drafter)
        refresh_draft_token_maps(drafter)

        fresh = DraftVocab(LogitsProcessor(VOCAB), VOCAB)
        fresh.load_token_map(head, torch.tensor(DRAFT_IDS), dynamic_rows)
        expected = fresh.compute_logits(head, hidden)
        actual = drafter.draft_vocab.compute_logits(head, hidden)
        assert torch.equal(actual, expected)
        assert torch.equal(
            drafter.draft_vocab.get_top_tokens(head, hidden),
            fresh.get_top_tokens(head, hidden),
        )
        assert head.weight.data_ptr() == head_pointer
        assert {
            name: tensor.data_ptr()
            for name, tensor in _draft_vocab_tensors(drafter.draft_vocab).items()
        } == pointers
        if dynamic_rows:
            assert drafter.draft_vocab.gather_rows[0].data_ptr() == head_pointer
        if seed is None:
            assert torch.equal(actual, previous)
        else:
            assert not torch.equal(actual[:, DRAFT_IDS], previous[:, DRAFT_IDS])
            if dynamic_rows:
                assert not torch.equal(actual.isfinite(), previous.isfinite())
        previous = actual
