"""Heuristic policies and utilities for the OWF_Sens environment.

This module is a Python reimplementation of the exploratory notebook located at
``imp_marl/imp_marl/environments/heur_test.ipynb``.  It collects the heuristic
rules, agent wrapper, evaluation helper, and a basic hyper-parameter grid search
utility so they can be reused outside of the notebook context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
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
            boot_iters: int = 2000,
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
            std_val  = float(np.std(arr))
            min_val  = float(np.min(arr))
            max_val  = float(np.max(arr))

            # --- bootstrap for mean (percentile CI) ---
            n = arr.size
            stats = np.empty(boot_iters, dtype=float)
            for b in range(boot_iters):
                sample = np.random.choice(arr, size=n, replace=True)
                stats[b] = np.mean(sample)
            alpha = 1.0 - ci
            lo, hi = np.quantile(stats, [alpha / 2, 1.0 - alpha / 2])
            mean_ci = (float(lo), float(hi))
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


def grid_search(
    base_rules: List[RuleSpec],
    sweep: Dict[str, Sequence[Any]],
    make_runner: Callable[[List[RuleSpec]], EnvRunner],
    episodes: int = 100,
    init_inspection_age: int = 61,
) -> List[Dict[str, Any]]:
    """Grid search over heuristic parameter combinations."""

    keys = list(sweep.keys())
    combos = list(product(*[sweep[k] for k in keys]))

    results: List[Dict[str, Any]] = []
    for values in combos:
        rules: List[RuleSpec] = []
        override = {k: v for k, v in zip(keys, values)}
        for fn, params in base_rules:
            merged = {**params, **override}
            rules.append((fn, merged))
        runner = make_runner(rules)
        metrics = runner.evaluate(
            episodes=episodes,
            init_inspection_age=init_inspection_age,
        )
        results.append({"params": override, **metrics})

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
