"""Heuristic policies and utilities for the OWF_Sens environment.

This module is a Python reimplementation of the exploratory notebook located at
``imp_marl/imp_marl/environments/heur_test.ipynb``.  It collects the heuristic
rules, agent wrapper, evaluation helper, and a basic hyper-parameter grid search
utility so they can be reused outside of the notebook context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Callable, Dict, List, Sequence, Tuple

import importlib
import numpy as np
import os
import time

HeuristicFn = Callable[[int, np.ndarray, np.ndarray, Dict[str, Any]], np.ndarray]
RuleSpec = Tuple[HeuristicFn, Dict[str, Any]]


def actions_to_dict(actions: np.ndarray, agent_list: Sequence[str]) -> Dict[str, int]:
    """Map a vector of integer actions to the environment's agent ordering."""

    if len(actions) != len(agent_list):
        raise ValueError(
            f"actions length {len(actions)} != len(agent_list) {len(agent_list)}"
        )
    return {agent: int(actions[i]) for i, agent in enumerate(agent_list)}


# ---------------------------------------------------------------------------
# Rule definitions
# ---------------------------------------------------------------------------

def heuristic_inspect_topk(
    timestep: int,
    inspections: np.ndarray,
    pf_comp: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Inspect the components with the highest failure probability."""

    interval = int(params.get("inspection_interval", 1))
    k = int(params.get("n_components", 1))
    actions = np.zeros_like(pf_comp, dtype=int)

    if timestep > 0 and (timestep % interval == 0):
        idx = np.argsort(pf_comp)[::-1][:k]
        actions[idx] = 2  # Inspect
    return actions


def heuristic_repair_on_recent_inspection(
    timestep: int,
    inspections: np.ndarray,
    pf_comp: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Repair components that were just inspected on the previous step."""

    actions = np.zeros_like(pf_comp, dtype=int)
    actions[inspections == 1] = 4
    return actions


def heuristic_monitoring_install_topk(
    timestep: int,
    inspections: np.ndarray,
    pf_comp: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Install monitoring on the top-k failure probabilities at a fixed cadence."""

    interval = int(params.get("inspection_interval", 1))
    k = int(params.get("n_components", 1))
    actions = np.zeros_like(pf_comp, dtype=int)

    if timestep > 0 and (timestep % interval == 0):
        idx = np.argsort(pf_comp)[::-1][:k]
        actions[idx] = 1  # Install sensor
    return actions


def heuristic_repair_on_window(
    timestep: int,
    inspections: np.ndarray,
    pf_comp: np.ndarray,
    params: Dict[str, Any],
) -> np.ndarray:
    """Repair when the time since last inspection falls inside a target window."""

    min_age = int(params.get("min_age", 16))
    max_age = int(params.get("max_age", 60))
    actions = np.zeros_like(pf_comp, dtype=int)

    mask = (inspections > min_age) & (inspections < max_age)
    actions[mask] = 4
    return actions


# ---------------------------------------------------------------------------
# Agent wrapper
# ---------------------------------------------------------------------------


@dataclass
class HeuristicAgent:
    """Compose multiple heuristic rules; later rules override earlier ones."""

    rules: List[RuleSpec] = field(default_factory=list)

    def act(
        self,
        timestep: int,
        inspections: np.ndarray,
        pf_comp: np.ndarray,
    ) -> np.ndarray:
        actions = np.zeros_like(pf_comp, dtype=int)
        for rule_fn, params in self.rules:
            proposal = rule_fn(timestep, inspections, pf_comp, params)
            overwrite = proposal != 0
            actions[overwrite] = proposal[overwrite]
        return actions


# ---------------------------------------------------------------------------
# Environment runner
# ---------------------------------------------------------------------------


@dataclass
class EnvRunner:
    """Evaluate heuristics on an environment matching OWF_Sens."""

    env: Any
    agent: HeuristicAgent
    n_agents: int | None = None
    pf_slice: slice | None = None

    def _pf_components(self) -> np.ndarray:
        # damage_proba has shape (lev, stress_conditions, crack_conditions)
        pf = self.env.damage_proba[0, :].reshape(
            (
                self.env.lev,
                self.env.stress_conditions,
                self.env.crack_conditions,
            )
        ).sum(axis=1)[:, -1]
        if self.pf_slice is not None:
            pf = pf[self.pf_slice]
        if self.n_agents is not None:
            pf = pf[: self.n_agents]
        return pf

    def run_episode(self, init_inspection_age: int = 61, log: bool = False) -> float:
        _ = self.env.reset()
        agent_list = list(self.env.agent_list)
        n = len(agent_list) if self.n_agents is None else int(self.n_agents)
        inspections = np.ones(n, dtype=int) * int(init_inspection_age)

        done = False
        ep_return = 0.0

        while not done:
            pf = self._pf_components()
            actions_vec = self.agent.act(self.env.time_step, inspections, pf)
            actions = actions_to_dict(actions_vec, agent_list[:n])

            _, reward, done, info = self.env.step(actions)
            inspections = np.asarray(info["inspections"][0, :n])
            ep_return += float(reward[agent_list[0]])

            if log:
                print(
                    f"t={self.env.time_step} pf={np.round(pf,3)} "
                    f"ins={inspections} act={actions_vec} r={reward}"
                )

        return ep_return

    def evaluate(
            self, 
            episodes: int = 100, 
            init_inspection_age: int = 61,
            boot_iters: int = 1,
            ci: float = 0.95,
    ) -> Dict[str, Any]:
        returns: List[float] = []
        t0 = time.time()
        for _ in range(episodes):
            ret = self.run_episode(init_inspection_age=init_inspection_age, log=False)
            returns.append(ret)
        elapsed = time.time() - t0

        if returns:
            arr = np.asarray(returns, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))

            if boot_iters > 0:
                n = arr.size
                stats = np.empty(boot_iters, dtype=float)
                for b in range(boot_iters):
                    sample = np.random.choice(arr, size=n, replace=True)
                    stats[b] = np.mean(sample)
                alpha = 1.0 - ci
                lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
                mean_ci = (float(lo), float(hi))
            else:
                mean_ci = (mean_val, mean_val)
        else:
            mean_val = std_val = min_val = max_val = 0.0
            mean_ci = (0.0, 0.0)

        return {
            "episodes": episodes,
            "mean_return": mean_val,
            "mean_return_ci": mean_ci,   # <-- added
            "std_return": std_val,
            "min_return": min_val,
            "max_return": max_val,
            "elapsed_sec": elapsed,
            "ci_level": ci,              # <-- added (metadata)
            "bootstrap_iters": boot_iters,
        }


# ---------------------------------------------------------------------------
# Hyper-parameter sweep
# ---------------------------------------------------------------------------

def _load_env(import_path: str, kwargs: Dict[str, Any]) -> Any:
    module_path, _, attr = import_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path '{import_path}'")
    module = importlib.import_module(module_path)
    env_cls = getattr(module, attr)
    return env_cls(**kwargs)


def _slice_to_tuple(pf_slice: slice | None) -> Tuple[Any, Any, Any] | None:
    if pf_slice is None:
        return None
    return (pf_slice.start, pf_slice.stop, pf_slice.step)


def _tuple_to_slice(pf_slice_tuple: Tuple[Any, Any, Any] | None) -> slice | None:
    if pf_slice_tuple is None:
        return None
    start, stop, step = pf_slice_tuple
    return slice(start, stop, step)


def _eval_one_combo(
    keys: List[str],
    values: Sequence[Any],
    base_rules: List["RuleSpec"],
    env_import: str,
    env_kwargs: Dict[str, Any],
    n_agents: int | None,
    pf_slice_tuple: Tuple[Any, Any, Any] | None,
    episodes: int,
    init_inspection_age: int,
    boot_iters: int,
    ci: float,
) -> Dict[str, Any]:
    override = dict(zip(keys, values))
    rules: List["RuleSpec"] = [(fn, {**params, **override}) for fn, params in base_rules]

    env = _load_env(env_import, env_kwargs)
    agent = HeuristicAgent(rules)
    runner = EnvRunner(
        env=env,
        agent=agent,
        n_agents=n_agents,
        pf_slice=_tuple_to_slice(pf_slice_tuple),
    )
    metrics = runner.evaluate(
        episodes=episodes,
        init_inspection_age=init_inspection_age,
        boot_iters=boot_iters,
        ci=ci,
    )
    metrics.update({"params": override})
    return metrics


def grid_search(
    base_rules: List[RuleSpec],
    sweep: Dict[str, Sequence[Any]],
    env_import: str,
    env_kwargs: Dict[str, Any] | None = None,
    *,
    n_agents: int | None = None,
    pf_slice: slice | None = None,
    episodes: int = 100,
    init_inspection_age: int = 61,
    boot_iters: int = 2000,
    ci: float = 0.95,
    max_workers: int | None = None,
) -> List[Dict[str, Any]]:
    """Grid search over heuristic parameter combinations.

    Args:
        base_rules: Baseline rule list (function, params).
        sweep: Dict mapping parameter names to value sequences.
        env_import: Dotted path to environment class.
        env_kwargs: kwargs passed to the environment constructor per worker.
        n_agents: Optional number of agents to include.
        pf_slice: Optional slice of the probability-of-failure vector.
        episodes: Evaluation episodes per combination.
        init_inspection_age: Initial inspection age fed to the runner.
        max_workers: Process pool size (defaults to CPU count / len combos heuristics).
    """

    env_kwargs = dict(env_kwargs or {})
    keys = list(sweep.keys())
    combos = list(product(*[sweep[k] for k in keys]))
    if not combos:
        return []

    pf_slice_tuple = _slice_to_tuple(pf_slice)

    if max_workers is not None:
        worker_count = max(1, min(max_workers, len(combos)))
    else:
        cpu = os.cpu_count() or 1
        worker_count = max(1, min(cpu, len(combos)))

    if worker_count == 1:
        results = [
            _eval_one_combo(
                keys,
                values,
                base_rules,
                env_import,
                env_kwargs,
                n_agents,
                pf_slice_tuple,
                episodes,
                init_inspection_age,
                boot_iters,
                ci,
            )
            for values in combos
        ]
    else:
        results: List[Dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futures = [
                ex.submit(
                    _eval_one_combo,
                    keys,
                    values,
                    base_rules,
                    env_import,
                    env_kwargs,
                    n_agents,
                    pf_slice_tuple,
                    episodes,
                    init_inspection_age,
                    boot_iters,
                    ci,
                )
                for values in combos
            ]
            for fut in futures:
                results.append(fut.result())

    results.sort(key=lambda x: x["mean_return"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Convenience rule bundles
# ---------------------------------------------------------------------------


def make_inspection_rules(
    inspection_interval: int = 1,
    n_components: int = 2,
) -> List[RuleSpec]:
    """Inspect top-k components and repair them immediately after inspection."""

    return [
        (heuristic_inspect_topk, {
            "inspection_interval": inspection_interval,
            "n_components": n_components,
        }),
        (heuristic_repair_on_recent_inspection, {}),
    ]


def make_monitoring_rules(
    inspection_interval: int = 1,
    n_components: int = 2,
    min_age: int = 15,
    max_age: int = 61,
) -> List[RuleSpec]:
    """Install monitoring on high-risk components and repair within an age window."""

    return [
        (heuristic_monitoring_install_topk, {
            "inspection_interval": inspection_interval,
            "n_components": n_components,
        }),
        (heuristic_repair_on_window, {
            "min_age": min_age,
            "max_age": max_age,
        }),
    ]


__all__ = [
    "HeuristicFn",
    "RuleSpec",
    "actions_to_dict",
    "heuristic_inspect_topk",
    "heuristic_repair_on_recent_inspection",
    "heuristic_monitoring_install_topk",
    "heuristic_repair_on_window",
    "HeuristicAgent",
    "EnvRunner",
    "grid_search",
    "make_inspection_rules",
    "make_monitoring_rules",
]
