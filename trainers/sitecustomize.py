# Copyright 2026 RouteWeaver.
"""Auto-loaded by every Python process that has trainers/ on sys.path (PYTHONPATH).

WHY: in the router-r1-the env, importing verl.workers.engine_workers segfaults
inside pyarrow's native library when pyarrow is loaded LATE, after the large
torch/vllm/flash-attn .so pile (deterministic A/B: early `import pyarrow` ->
clean import; late -> SIGSEGV at pyarrow.lib load. Loading pyarrow first,
while static TLS / relocation space is plentiful, sidesteps the crash. Ray
workers inherit this repository's PYTHONPATH via the job runtime_env, so this
covers the trainer driver, WorkerDict actors,
AgentLoopWorkers, reward-loop workers and vLLM server processes alike.
"""
try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover — never block interpreter start
    pass

# ---------------------------------------------------------------------------
# Post-import hook: register this repository's advantage estimators the moment verl's
# estimator registry module is imported. compute_advantage resolves
# algorithm.adv_estimator BY NAME in the trainer driver, and nothing in the
# driver imports these modules on its own -- the reward manager loads in reward
# workers, the agent loop in rollout workers. Importing grpo_gated eagerly
# here is not an option either: it pulls verl -> torch into every interpreter
# that merely has trainers/ on PYTHONPATH (tests are CPU-only and import no verl).
# So: intercept the first import of verl.trainer.ppo.core_algos, let it
# complete, then import grpo_gated (whose decorators register into the
# just-created registry). Idempotent; never blocks interpreter start.
# ---------------------------------------------------------------------------
import sys as _sys
from importlib.abc import Loader as _Loader, MetaPathFinder as _MetaPathFinder
from importlib.machinery import PathFinder as _PathFinder

_ESTIMATOR_HOST = "verl.trainer.ppo.core_algos"


class _RegisterEstimatorsAfter(_Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        try:
            import grpo_gated  # noqa: F401  (grpo_gated.py)
            import comet_grpo        # noqa: F401  registers "comet_grpo"
            import comet_gdpo        # noqa: F401  registers "comet_gdpo"
        except Exception:  # pragma: no cover — registration must never kill training
            import traceback
            traceback.print_exc()


class _CoreAlgosImportHook(_MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ESTIMATOR_HOST:
            return None
        # PathFinder only, never the meta path: no re-entry into this hook.
        spec = _PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _RegisterEstimatorsAfter(spec.loader)
        return spec


_sys.meta_path.insert(0, _CoreAlgosImportHook())

# ---------------------------------------------------------------------------
# Second post-import hook, same shape and the same reason: the dispatch gate in
# grpo_gated needs the per-row `dispatch_failed_calls` column, and verl's
# registry branch hands a registered estimator only tensors -- non_tensor_batch
# goes to GDPO and nothing else (ray_trainer.py:258-260). So wrap
# ray_trainer.compute_advantage, which fit() resolves as a module global at
# call time, and let it pass the column through. Nothing in verl is edited.
#
# Why a hook rather than an import in the launcher: the driver imports
# ray_trainer through verl.trainer.main_ppo before any of these modules is reachable
# by name, so there is no earlier moment at which to patch it.
# ---------------------------------------------------------------------------
_TRAINER_HOST = "verl.trainer.ppo.ray_trainer"


class _InstallDispatchGateAfter(_Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        try:
            import grpo_gated  # grpo_gated.py

            if grpo_gated.install_compute_advantage_gate(module):
                print("[sitecustomize] dispatch gate installed on "
                      "ray_trainer.compute_advantage")
            import comet_grpo  # comet_grpo.py (ROUTEWEAVER_COMET=1)

            if comet_grpo.install(module):
                print("[sitecustomize] hierarchical-advantage wrapper installed on "
                      "ray_trainer.compute_advantage")
            # Cost-decoupled advantage (ADV_ESTIMATOR=comet_gdpo): hands the
            # estimator the second reward component. Wraps OUTSIDE comet_grpo so
            # the segment masks and the clean-row column are still published by
            # it; at COST_ALPHA=0 the estimator delegates back to comet_grpo.
            import comet_gdpo  # comet_gdpo.py

            if comet_gdpo.install(module):
                print("[sitecustomize] cost-decoupled wrapper installed on "
                      "ray_trainer.compute_advantage")
        except Exception:  # pragma: no cover — the gate must never kill training
            import traceback
            traceback.print_exc()
        # Same hook, unrelated concern: an explicit checkpoint-step list. Inert
        # unless ROUTEWEAVER_SAVE_STEPS is set, so no other run is affected.
        try:
            import checkpoint_steps  # checkpoint_steps.py

            if checkpoint_steps.install_save_step_gate(module):
                print("[sitecustomize] checkpoint step gate installed: "
                      f"{sorted(checkpoint_steps.wanted_steps())}")
            import train_metrics  # train_metrics.py

            if train_metrics.install_metrics_hook(module):
                print("[sitecustomize] routeweaver/* step metrics installed on "
                      "ray_trainer.compute_data_metrics")
        except Exception:  # pragma: no cover — the gate must never kill training
            import traceback
            traceback.print_exc()


class _RayTrainerImportHook(_MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TRAINER_HOST:
            return None
        spec = _PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _InstallDispatchGateAfter(spec.loader)
        return spec


_sys.meta_path.insert(0, _RayTrainerImportHook())


# ---------------------------------------------------------------------------
# Third post-import hook, same shape. The minimum-support auxiliary loss needs
# its forward/backward to land in the SAME optimizer step as GRPO's, and
# verl's BaseEngine.train_batch is
#     optimizer_zero_grad -> forward_backward_batch -> optimizer_step
# with backward only accumulating. So wrap optimizer_step: the auxiliary pass
# runs immediately before the real step, its gradient adds to the same .grad,
# and the auxiliary sequences never enter forward_backward_batch -- so they
# cannot reach the rollout, the advantage, the PPO ratio or loss normalisation.
#
# Inert unless ROUTEWEAVER_SUPPORT=1, so every other run is
# byte-for-byte unaffected.
# ---------------------------------------------------------------------------
# Segment-wise loss reduction (ROUTEWEAVER_LOSS_REDUCTION=segment). engine_workers
# binds `ppo_loss` into its own namespace at import and freezes it into a
# partial when the worker is built, so the swap must happen right after that
# module executes.
_LOSS_HOST = "verl.workers.engine_workers"


class _InstallSegmentLossAfter(_Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        try:
            import segment_loss  # segment_loss.py

            segment_loss.install(module)
        except Exception:  # pragma: no cover — never kill training
            import traceback
            traceback.print_exc()


class _EngineWorkersImportHook(_MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _LOSS_HOST:
            return None
        spec = _PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _InstallSegmentLossAfter(spec.loader)
        return spec


_sys.meta_path.insert(0, _EngineWorkersImportHook())


# Fourth post-import hook, same shape. verl computes each rollout's ordinal
# inside its group but never hands it to the concrete agent loop; the 4x2
# worker rotation cannot exist without it. Inert unless a loop reads the key.
_AGENTLOOP_HOST = "verl.experimental.agent_loop.agent_loop"


class _InstallRolloutOrdinalAfter(_Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        try:
            import rollout_ordinal_hook

            if rollout_ordinal_hook.install(module):
                print("[sitecustomize] rollout ordinal forwarded to the agent loop")
        except Exception:  # pragma: no cover -- never kill training
            import traceback
            traceback.print_exc()


class _AgentLoopImportHook(_MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _AGENTLOOP_HOST:
            return None
        spec = _PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _InstallRolloutOrdinalAfter(spec.loader)
        return spec


_sys.meta_path.insert(0, _AgentLoopImportHook())


_ENGINE_HOST = "verl.workers.engine.fsdp.transformer_impl"


class _InstallSupportHookAfter(_Loader):
    def __init__(self, wrapped):
        self._wrapped = wrapped

    def create_module(self, spec):
        return self._wrapped.create_module(spec)

    def exec_module(self, module):
        self._wrapped.exec_module(module)
        try:
            import support_hook  # support_hook.py

            support_hook.install(module)
        except Exception:  # pragma: no cover — never kill training
            import traceback
            traceback.print_exc()
        try:
            # publishes the mini-batch-global mode-token count that the segment
            # reduction uses as its denominator (segment_loss.py)
            import segment_loss

            segment_loss.install_engine(module)
        except Exception:  # pragma: no cover — never kill training
            import traceback
            traceback.print_exc()


class _EngineBaseImportHook(_MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _ENGINE_HOST:
            return None
        spec = _PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _InstallSupportHookAfter(spec.loader)
        return spec


_sys.meta_path.insert(0, _EngineBaseImportHook())
