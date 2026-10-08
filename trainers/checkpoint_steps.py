# Copyright 2026 RouteWeaver.
"""Save checkpoints at an EXPLICIT list of steps instead of a fixed period.

verl offers `trainer.save_freq` only: one period, plus the last step
(ray_trainer.py -- `is_last_step or global_steps % save_freq == 0`). The RouteWeaver
baseline is asked for steps 1, 5, 10, 20, 30, 50, 75, 100 -- dense early, sparse
late, because the interesting part of a collapse is the first ten updates. With
save_freq=5 that costs 20 checkpoints; a 4B FSDP2 checkpoint (weights +
optimizer + extra state) is ~30 GB, so the period form would write 600 GB to
hold 8 wanted steps.

MECHANISM: run with `trainer.save_freq=1` so verl offers to save every step, and
gate the offer here. `_save_checkpoint` is wrapped, not replaced -- the wanted
steps go through verl's own save unchanged, so `latest_checkpointed_iteration.txt`
and therefore `resume_mode=auto` keep working exactly as before.

Enabled only when ROUTEWEAVER_SAVE_STEPS is set (comma separated, e.g.
"1,5,10,20,30,50,75,100"); absent, this module does nothing at all and every
other run behaves as it always did.
"""
import os


def wanted_steps():
    raw = os.environ.get("ROUTEWEAVER_SAVE_STEPS", "").strip()
    if not raw:
        return None
    return {int(part) for part in raw.replace(" ", "").split(",") if part}


def install_save_step_gate(ray_trainer_module) -> bool:
    steps = wanted_steps()
    if steps is None:
        return False
    trainer_cls = getattr(ray_trainer_module, "RayPPOTrainer", None)
    if trainer_cls is None:
        return False
    original = getattr(trainer_cls, "_save_checkpoint", None)
    if original is None or getattr(original, "_routeweaver_save_gate", False):
        return False

    import functools

    @functools.wraps(original)
    def _save_checkpoint(self, *args, **kwargs):
        step = int(getattr(self, "global_steps", 0))
        if step not in steps:
            print(f"[checkpoint_steps] step {step} not in {sorted(steps)}; "
                  f"skipping checkpoint")
            return None
        return original(self, *args, **kwargs)

    _save_checkpoint._routeweaver_save_gate = True
    trainer_cls._save_checkpoint = _save_checkpoint
    return True
