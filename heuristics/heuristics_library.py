from dataclasses import dataclass, field
# from concurrent.futures import ProcessPoolExecutor
from itertools import product
from typing import Any, Callable, Dict, List, Sequence, Tuple, Optional
from copy import deepcopy
from itertools import product

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


def _summarize_returns(
    *,
    returns: Sequence[float],
    episodes: int,
    elapsed: float,
    boot_iters: int,
    ci: float,
) -> Dict[str, Any]:
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
        "mean_return_ci": mean_ci,
        "std_return": std_val,
        "min_return": min_val,
        "max_return": max_val,
        "elapsed_sec": elapsed,
        "ci_level": ci,
        "bootstrap_iters": boot_iters,
    }


# ---------------------------------------------------------------------------
# Environment runner
# ---------------------------------------------------------------------------


@dataclass
class EnvRunner:
    """Evaluate heuristics on an environment matching OWF_Sens."""

    env: Any
    agent: HeuristicAgent

    def _pf_components(self) -> np.ndarray:
    # damage_proba has shape (n_owt, lev * stress * crack)
        pf_all = []
        for i in range(self.env.n_owt):  # loop over all turbines
            pf_i = self.env.damage_proba[i, :].reshape(
                (self.env.lev, self.env.stress_conditions, self.env.crack_conditions)
            ).sum(axis=1)[:, -1]  # shape: (lev,)
            pf_all.append(pf_i[:-1])
        return np.concatenate(pf_all)  # shape: (n_agents * lev,)

    def run_episode(self, init_inspection_age: int = 61, log: bool = False) -> float:
        _ = self.env.reset()
        agent_list = list(self.env.agent_list)
        n = len(agent_list) 
        inspections = np.ones(n, dtype=int) * int(init_inspection_age)

        done = False
        ep_return = 0.0

        while not done:
            pf = self._pf_components()
            actions_vec = self.agent.act(self.env.time_step, inspections, pf)
            actions = actions_to_dict(actions_vec, agent_list[:n])

            _, reward, done, info = self.env.step(actions)
            
            inspections = np.asarray(info["inspections"], dtype=int)
            inspections = inspections[:, :-1].flatten()  # shape: (n_agents * (lev-1),)
            ep_return += float(reward[agent_list[0]])

            if log:
                print(
                    f"t={self.env.time_step} pf={np.round(pf,7)} "
                    f"ins={inspections} act={actions_vec} r={reward}"
                )

        return ep_return

    def evaluate(
            self, 
            episodes: int = 100, 
            init_inspection_age: int = 61,
            boot_iters: int = 1,
            ci: float = 0.95,
            log: bool = False, 
    ) -> Dict[str, Any]:
        returns: List[float] = []
        t0 = time.time()
        for _ in range(episodes):
            ret = self.run_episode(init_inspection_age=init_inspection_age, log=log)
            returns.append(ret)
        elapsed = time.time() - t0

        return _summarize_returns(
            returns=returns,
            episodes=episodes,
            elapsed=elapsed,
            boot_iters=boot_iters,
            ci=ci,
        )
    
    def optimize(
        self,
        param_spaces: Sequence[Dict[str, Sequence[Any]]],
        episodes: int = 100,
        init_inspection_age: int = 61,
        boot_iters: int = 1,
        ci: float = 0.95,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Grid search over heuristic parameters."""
        return grid_search_params(
            runner=self,
            param_spaces=param_spaces,
            episodes=episodes,
            init_inspection_age=init_inspection_age,
            boot_iters=boot_iters,
            ci=ci,
            top_k=top_k,
        )

def _clone_agent_with_params(agent: HeuristicAgent,
                             per_rule_overrides: Sequence[Dict[str, Any]]) -> HeuristicAgent:
    """
    Create a new HeuristicAgent with params overridden per rule (same order as agent.rules).
    Existing params are kept unless overridden.
    """
    if len(per_rule_overrides) != len(agent.rules):
        raise ValueError("per_rule_overrides length must match number of rules in agent.")
    new_rules: List[RuleSpec] = []
    for (rule_fn, base_params), override in zip(agent.rules, per_rule_overrides):
        merged = deepcopy(base_params)
        merged.update(override or {})
        new_rules.append((rule_fn, merged))
    return HeuristicAgent(rules=new_rules)


def grid_search_params(
    runner: EnvRunner,
    param_spaces: Sequence[Dict[str, Sequence[Any]]],
    *,
    episodes: int = 100,
    init_inspection_age: int = 61,
    boot_iters: int = 1,
    ci: float = 0.95,
    top_k: int = 1,
) -> Dict[str, Any]:
    """
    Exhaustive grid search over parameter spaces for each rule in runner.agent.

    param_spaces: list (len == number of rules) of dicts mapping param_name -> iterable of values.
                  Example for 2 rules:
                  [
                    {"inspection_interval": [6, 12], "n_components": [1, 2, 3]},
                    {"min_age": [12, 16], "max_age": [48, 60]},
                  ]

    Returns a dict with:
      - "best": {"params": per_rule_param_dicts, "result": eval_result}
      - "top": list of top results sorted by mean_return (length <= top_k)
      - "trials": number of evaluated combinations
    """
    # Build per-rule grids of dictionaries (cartesian per rule)
    per_rule_grids: List[List[Dict[str, Any]]] = []
    for space in param_spaces:
        if not space:
            # No search for this rule -> keep single empty override
            per_rule_grids.append([{}])
            continue
        keys = list(space.keys())
        vals = [list(space[k]) for k in keys]
        combos = [dict(zip(keys, v)) for v in product(*vals)]
        per_rule_grids.append(combos)

    results: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
    trials = 0

    # Global cartesian product across rules
    for combo in product(*per_rule_grids):
        trials += 1
        # Build a fresh agent with these params
        agent_try = _clone_agent_with_params(runner.agent, list(combo))

        # Fresh runner instance sharing env reference (keeps everything else identical)
        try_runner = EnvRunner(
            env=runner.env,
            agent=agent_try,
        )

        res = try_runner.evaluate(
            episodes=episodes,
            init_inspection_age=init_inspection_age,
            boot_iters=boot_iters,
            ci=ci,
        )
        results.append((list(combo), res))

    # Sort by mean_return descending
    results.sort(key=lambda x: x[1]["mean_return"], reverse=True)

    best_params, best_res = results[0]
    top_list = [{"params": p, "result": r} for p, r in results[:max(1, top_k)]]

    # All trials (sorted by mean_return desc)
    all_list = [{"params": p, "result": r} for p, r in results]

    return {
        "best": {"params": best_params, "result": best_res},
        "top": top_list,
        "all": all_list,
        "trials": trials,
    }