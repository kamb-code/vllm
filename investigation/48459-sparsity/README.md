# Sparsity matrix for vllm-project/vllm#48459 discussion

Measures tokens served by `OffloadingConnectorScheduler._lookup` when an
annotated ephemeral drafter group has sparse stored chunks ([M,M,H,M,M], per
vllm-project/vllm#47890's trace), across four code arms (main `41f179b`,
+#52771, +#48459, both). Run from a vLLM checkout of the desired arm:

    VLLM_TARGET_DEVICE=cpu PYTHONPATH=. python sparsity_matrix.py <arm-label>

Result (2026-08-19): 0 tokens served in all four arms on the sparse pattern;
dense-storage control serves 16/16 in all arms.
