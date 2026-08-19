# SPDX-License-Identifier: Apache-2.0
"""Sparsity matrix for the #48459 dialogue (MEASURE-BEFORE-PREDICT).

Scenario: annotated ephemeral drafter group whose STORED chunks are sparse
([M,M,H,M,M] per #47890's trace), target full-attention group fully stored.
Measures tokens served by OffloadingConnectorScheduler._lookup for a fresh
request, per code arm. Stores are injected directly (the sparsity is the
GIVEN, reproducing the reporter's observed storage pattern; how it arises on
their rig is out of scope).

Rows: (a) drafter group on the sliding-window lookup path;
      (b) drafter group on the full-attention prefix lookup path.
"""
import sys
from unittest.mock import MagicMock

import torch

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockOffloadingSpec,
)
from vllm.config import KVEventsConfig
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import LookupResult

BLOCK = 4
PROMPT_TOKENS = 20  # 5 chunks per group at block_size 4


def make_scheduler(drafter_kind: str):
    vllm_config = _make_vllm_config(extra_config={"self_describing_kv_events": True})
    vllm_config.cache_config.block_size = BLOCK
    vllm_config.cache_config.prefix_match_unit = BLOCK
    vllm_config.speculative_config = None
    vllm_config.kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True, publisher="null"
    )
    if drafter_kind == "sw":
        drafter_spec = SlidingWindowSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1,
            dtype=torch.float32, sliding_window=8,
        )
    else:
        drafter_spec = FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32,
        )
    kv_cache_config = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["target_layer"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=2, head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["drafter_layer"], drafter_spec, is_eagle_group=True
            ),
        ],
    )
    spec = MockOffloadingSpec(build_offloading_config(vllm_config, kv_cache_config))
    return OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config)


def served_tokens(drafter_kind: str, drafter_stored_chunks: set[int]) -> int:
    scheduler = make_scheduler(drafter_kind)
    request = MagicMock()
    request.request_id = "R"
    request.kv_transfer_params = None
    request.num_prompt_tokens = PROMPT_TOKENS
    request.num_tokens = PROMPT_TOKENS
    request.num_computed_tokens = 0
    request.block_hashes = [
        BlockHash(f"h{i}".encode()) for i in range(PROMPT_TOKENS // BLOCK)
    ]
    request.all_token_ids = list(range(PROMPT_TOKENS))
    request.lora_request = None
    request.is_finished.return_value = False
    request.status = None
    scheduler.on_new_request(request)
    rs = scheduler._req_status["R"]
    rs.num_locally_computed_tokens = 0
    rs.update_offload_keys()

    stored = set(rs.group_states[0].offload_keys)  # target: fully stored
    for c in drafter_stored_chunks:
        stored.add(rs.group_states[1].offload_keys[c])
    scheduler.manager.lookup.side_effect = lambda key, ctx: (
        LookupResult.HIT if key in stored else LookupResult.MISS
    )
    served = scheduler._lookup(rs)
    return served if served is not None else 0


arm = sys.argv[1]
for kind in ("sw", "fa"):
    runs = [served_tokens(kind, {2}) for _ in range(3)]
    ctrl = served_tokens(kind, {0, 1, 2, 3, 4})
    print(f"ARM={arm} drafter={kind} sparse[M,M,H,M,M] served={runs} "
          f"dense-control served={ctrl} (prompt={PROMPT_TOKENS})")
