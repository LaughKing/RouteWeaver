# Copyright 2026 RouteWeaver.
"""Auto-loaded via PYTHONPATH: imports pyarrow early, then registers RouteWeaver hooks in verl."""
try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover — never block interpreter start
    pass

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
        spec = _PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _RegisterEstimatorsAfter(spec.loader)
        return spec


_sys.meta_path.insert(0, _CoreAlgosImportHook())

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
            import comet_gdpo  # comet_gdpo.py

            if comet_gdpo.install(module):
                print("[sitecustomize] cost-decoupled wrapper installed on "
                      "ray_trainer.compute_advantage")
        except Exception:  # pragma: no cover — the gate must never kill training
            import traceback
            traceback.print_exc()
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
