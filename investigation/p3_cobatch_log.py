#!/usr/bin/env python3
"""Apply P3: log co-batched request ids per scheduler step (temporary probe)."""
path = "/workspace/vllm/vllm/v1/core/sched/scheduler.py"
src = open(path).read()
old = "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n            self._update_after_schedule(scheduler_output)\n        return scheduler_output"
new = (
    "        with record_function_or_nullcontext(\"schedule: update_after_schedule\"):\n"
    "            self._update_after_schedule(scheduler_output)\n"
    "        if scheduler_output.total_num_scheduled_tokens > 0:\n"
    "            import os as _os\n"
    "            if _os.environ.get(\"VLLM_COBATCH_LOG\"):\n"
    "                logger.info(\n"
    "                    \"COBATCH %s\",\n"
    "                    {r: n for r, n in scheduler_output.num_scheduled_tokens.items()},\n"
    "                )\n"
    "        return scheduler_output"
)
assert src.count(old) == 1, "anchor not found"
open(path, "w").write(src.replace(old, new))
print("P3 patch applied")
