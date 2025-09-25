#!/usr/bin/env python3
import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import yaml

from imp_marl.environments.owf_sens_env import OWF_Sens


# ---- import your existing code ----
# Adjust this to the module/file where you put the code you shared.
from heuristics_library import (
    HeuristicAgent,
    EnvRunner,
    heuristic_inspect_topk,
    heuristic_repair_on_recent_inspection,
    heuristic_monitoring_install_topk,
    heuristic_repair_on_window,
)

# ---------- Registry for rule names -> functions ----------
RULE_REGISTRY = {
    "inspect_topk": heuristic_inspect_topk,
    "repair_on_recent_inspection": heuristic_repair_on_recent_inspection,
    "monitoring_install_topk": heuristic_monitoring_install_topk,
    "repair_on_window": heuristic_repair_on_window,
}

# ---------- Utilities ----------
def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}

def numpy_to_jsonable(obj):
    """Recursively convert NumPy types to vanilla Python for json.dump."""
    if isinstance(obj, dict):
        return {k: numpy_to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj

def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(numpy_to_jsonable(obj), f, indent=2)

# ---------- Environment factory ----------
def make_env_from_config(env_cfg: Dict[str, Any]):
    return OWF_Sens(config=env_cfg)

def _expand_range_or_list(spec: Any) -> Sequence[Any]:
    """
    Accepts either:
      - a list of values (e.g., [1, 2, 3])
      - a dict of the form {"range": [start, stop]} or {"range": [start, stop, step]}
        with all integers, inclusive of stop
    Returns a list of values.
    """
    if isinstance(spec, (list, tuple)):
        return list(spec)

    if isinstance(spec, dict) and "range" in spec:
        vals = spec["range"]
        if not isinstance(vals, (list, tuple)) or len(vals) not in {2, 3}:
            raise ValueError(f"Invalid range format: {vals}")
        start, stop = vals[0], vals[1]
        step = vals[2] if len(vals) == 3 else 1
        return list(range(int(start), int(stop) + 1, int(step)))

    raise TypeError(f"Unsupported search_space spec: {spec}")

# ---------- Build agent + optional search spaces ----------
def build_agent_and_spaces(h_cfg: Dict[str, Any]):
    """
    heuristics.yml schema:
      rules:
        - name: inspect_topk
          params: {inspection_interval: 12, n_components: 2}
          search_space: {inspection_interval: [6,12,24], n_components: [1,2,3]}   # optional

        - name: repair_on_window
          params: {min_age: 16, max_age: 60}
          # search_space: {min_age: [12,16,20], max_age: [48,60,72]}
    """
    rule_specs = []
    param_spaces: List[Dict[str, Sequence[Any]]] = []

    for r in h_cfg.get("rules", []):
        name = r["name"]
        fn = RULE_REGISTRY.get(name)
        if fn is None:
            raise KeyError(f"Unknown rule name '{name}'. Known: {list(RULE_REGISTRY)}")
        params = dict(r.get("params") or {})
        rule_specs.append((fn, params))

        # optional search space; if absent, keep empty dict to skip optimization for this rule
        raw_space = dict(r.get("search_space") or {})
        expanded_space: Dict[str, Sequence[Any]] = {
            k: _expand_range_or_list(v) for k, v in raw_space.items()
        }
        param_spaces.append(expanded_space)

    agent = HeuristicAgent(rules=rule_specs)
    return agent, param_spaces

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", type=Path, required=True, help="Path to env.yml")
    ap.add_argument("--heuristics", type=Path, required=True, help="Path to heuristics.yml")
    ap.add_argument("--mode", choices=["eval", "optimize"], required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/out.json"), help="Output JSON path")
    args = ap.parse_args()

    env_cfg = load_yaml(args.env)
    heur_cfg = load_yaml(args.heuristics)

    # Build env
    env = make_env_from_config(env_cfg)

    # Build agent and (optional) param spaces
    agent, param_spaces = build_agent_and_spaces(heur_cfg)

    # Build runner 
    runner = EnvRunner(env=env, agent=agent)

    # Common eval args
    eval_cfg = heur_cfg.get("evaluate", {})
    episodes = int(eval_cfg.get("episodes", 100))
    init_inspection_age = int(eval_cfg.get("init_inspection_age", 61))
    boot_iters = int(eval_cfg.get("boot_iters", 1))
    ci = float(eval_cfg.get("ci", 0.95))
    log_flag = bool(eval_cfg.get("log", False))   # <--- added

    if args.mode == "eval":
        result = runner.evaluate(
            episodes=episodes,
            init_inspection_age=init_inspection_age,
            boot_iters=boot_iters,
            ci=ci,
            log=log_flag,
        )
        payload = {
        "mode": "eval",
        "result": result,
        "configs": {"env": env_cfg, "heuristics": heur_cfg},  
    }

    else:  # optimize
        opt_cfg = heur_cfg.get("optimize", {})
        top_k = int(opt_cfg.get("top_k", 1))

        # If no search_space given for any rule, this will behave like a single trial.
        result = runner.optimize(
            param_spaces=param_spaces,
            episodes=episodes,
            init_inspection_age=init_inspection_age,
            boot_iters=boot_iters,
            ci=ci,
            top_k=top_k,
        )
        payload = {
        "mode": "optimize",
        "result": result,
        "configs": {"env": env_cfg, "heuristics": heur_cfg},  
    }

    # Save outputs
    save_json(payload, args.out)

    print(f"Wrote results to {args.out}")

if __name__ == "__main__":
    main()
