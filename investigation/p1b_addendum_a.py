#!/usr/bin/env python3
"""Addendum-A instrumentation for the P1b zeroing probe (#39146 investigation).

Adds, all env-gated by VLLM_P1B_CANARY=1:
  1. per-path zero-record counters in SingleTypeKVCacheManager
     (paths: allocate_new_blocks -> "new", allocate_new_computed_blocks ->
     "computed", CoW redirect -> "cow")
  2. scheduler-side log of drained-to-zero ids per step + per-path totals
  3. worker-side (V2 runner) canary: per zeroing call, sample up to 4 block
     slabs, record whether they held nonzero (stale) data BEFORE zeroing and
     verify they read back all-zero AFTER zeroing; log call/block counters.
"""

# ---- 1. per-path counters in single_type_kv_cache_manager -------------------
path = "/workspace/vllm/vllm/v1/core/single_type_kv_cache_manager.py"
src = open(path).read()

old = """        self._record_new_block_ids = needs_kv_cache_zeroing and isinstance(
            kv_cache_spec, AttentionSpec
        )
        self.new_block_ids: list[int] = []"""
new = """        self._record_new_block_ids = needs_kv_cache_zeroing and isinstance(
            kv_cache_spec, AttentionSpec
        )
        self.new_block_ids: list[int] = []
        # P1b Addendum-A: per-allocation-path zero-record counters
        self.p1b_path_counts = {"new": 0, "computed": 0, "cow": 0}"""
assert src.count(old) == 1, "anchor1"
src = src.replace(old, new)

old = """        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks("""
new = """        req_blocks.extend(allocated_blocks)
        if self._record_new_block_ids:
            self.new_block_ids.extend(b.block_id for b in allocated_blocks)
            self.p1b_path_counts["computed"] += len(allocated_blocks)

    def allocate_new_blocks("""
assert src.count(old) == 1, "anchor2"
src = src.replace(old, new)

old = """            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)"""
new = """            self._apply_cow(request_id, block_idx, source_block, cow_block)
            self.new_block_ids.append(cow_block.block_id)
            self.p1b_path_counts["cow"] += 1"""
assert src.count(old) == 1, "anchor3"
src = src.replace(old, new)

old = """            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return cow_blocks + new_blocks"""
new = """            req_blocks.extend(new_blocks)
            if self._record_new_block_ids:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
                self.p1b_path_counts["new"] += len(new_blocks)
            return cow_blocks + new_blocks"""
assert src.count(old) == 1, "anchor4"
src = src.replace(old, new)
open(path, "w").write(src)
print("patched single_type_kv_cache_manager.py")

# ---- 2. scheduler-side drain log --------------------------------------------
path = "/workspace/vllm/vllm/v1/core/sched/scheduler.py"
src = open(path).read()
old = """        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        return new_block_ids_to_zero or None"""
new = """        if self._skip_zero_block_ids:
            skip = self._skip_zero_block_ids
            new_block_ids_to_zero = [b for b in new_block_ids_to_zero if b not in skip]
            skip.clear()

        import os as _os
        if new_block_ids_to_zero and _os.environ.get("VLLM_P1B_CANARY"):
            totals = {"new": 0, "computed": 0, "cow": 0}
            for mgr in self.kv_cache_manager.coordinator.single_type_managers:
                for k, v in getattr(mgr, "p1b_path_counts", {}).items():
                    totals[k] += v
            logger.info(
                "P1B-SCHED drained=%d path_totals=%s", len(new_block_ids_to_zero), totals
            )

        return new_block_ids_to_zero or None"""
assert src.count(old) == 1, "anchor5"
src = src.replace(old, new)
open(path, "w").write(src)
print("patched scheduler.py")

# ---- 3. V2 runner canary ----------------------------------------------------
path = "/workspace/vllm/vllm/v1/worker/gpu/model_runner.py"
src = open(path).read()
old = """        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            if not getattr(self, "_p1b_zero_logged", False):
                self._p1b_zero_logged = True
                logger.info("P1B-PROBE: KV block zeroing ACTIVE (V2 runner), first batch of %d block ids", len(scheduler_output.new_block_ids_to_zero))
            self.kv_block_zeroer.zero_block_ids(scheduler_output.new_block_ids_to_zero)"""
new = """        if scheduler_output.new_block_ids_to_zero:
            assert self.kv_block_zeroer is not None
            import os as _os
            _canary = _os.environ.get("VLLM_P1B_CANARY")
            _ids = scheduler_output.new_block_ids_to_zero
            _kv = None
            if _canary:
                self._p1b_calls = getattr(self, "_p1b_calls", 0) + 1
                self._p1b_blocks = getattr(self, "_p1b_blocks", 0) + len(_ids)
                for _layer in self.compilation_config.static_forward_context.values():
                    if isinstance(getattr(_layer, "kv_cache", None), torch.Tensor):
                        _kv = _layer.kv_cache
                        break
                if _kv is not None:
                    _bd = max(range(_kv.ndim), key=lambda d: _kv.shape[d])
                    _sample = list(_ids[:4])
                    _pre_dirty = [bool(_kv.select(_bd, i).any()) for i in _sample]
            self.kv_block_zeroer.zero_block_ids(_ids)
            if _canary and _kv is not None:
                _post_zero_ok = [not bool(_kv.select(_bd, i).any()) for i in _sample]
                logger.info(
                    "P1B-ZERO call=%d +%d total_blocks=%d sample=%s pre_dirty=%s post_zero_ok=%s",
                    self._p1b_calls, len(_ids), self._p1b_blocks,
                    _sample, _pre_dirty, _post_zero_ok,
                )"""
assert src.count(old) == 1, "anchor6"
src = src.replace(old, new)
open(path, "w").write(src)
print("patched gpu/model_runner.py")
print("ADDENDUM-A APPLIED")
