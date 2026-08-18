#!/usr/bin/env python3
"""Apply P1b: enable KV block zeroing for FullAttentionSpec (replicates #43741 intent on current main)."""
import re, sys
path = "/workspace/vllm/vllm/v1/kv_cache_interface.py"
src = open(path).read()
old = "        return self.has_mamba_layers or self.has_mixed_precision_kv_cache"
new = (
    "        return (\n"
    "            self.has_mamba_layers\n"
    "            or self.has_mixed_precision_kv_cache\n"
    "            # P1b probe (#39146): zero recycled blocks for full attention too\n"
    "            or any(\n"
    "                isinstance(g.kv_cache_spec, FullAttentionSpec)\n"
    "                for g in self.kv_cache_groups\n"
    "            )\n"
    "        )"
)
assert src.count(old) == 1, "anchor not found"
open(path, "w").write(src.replace(old, new))
print("P1b patch applied")
