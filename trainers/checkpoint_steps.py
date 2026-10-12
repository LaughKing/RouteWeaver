# Copyright 2026 RouteWeaver.
"""Save checkpoints at an EXPLICIT list of steps instead of a fixed period."""
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
