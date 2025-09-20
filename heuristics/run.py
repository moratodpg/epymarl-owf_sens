"""Simple CLI to evaluate heuristic policies or run parameter sweeps.

Usage:
    python -m heuristics.run --config heuristics/configs/inspection_run.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import importlib
import numpy as np
import yaml

try:  # allow execution as script or module
    from .heuristic_library import (
        EnvRunner,
        HeuristicAgent,
        RuleSpec,
        grid_search,
        heuristic_inspect_topk,
        heuristic_monitoring_install_topk,
        heuristic_repair_on_recent_inspection,
        heuristic_repair_on_window,
        make_inspection_rules,
        make_monitoring_rules,
    )
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(CURRENT_DIR.parent))
    from heuristics.heuristic_library import (
        EnvRunner,
        HeuristicAgent,
        RuleSpec,
        grid_search,
        heuristic_inspect_topk,
        heuristic_monitoring_install_topk,
        heuristic_repair_on_recent_inspection,
        heuristic_repair_on_window,
        make_inspection_rules,
        make_monitoring_rules,
    )

AVAILABLE_RULES: Dict[str, Any] = {
    "heuristic_inspect_topk": heuristic_inspect_topk,
    "heuristic_repair_on_recent_inspection": heuristic_repair_on_recent_inspection,
    "heuristic_monitoring_install_topk": heuristic_monitoring_install_topk,
    "heuristic_repair_on_window": heuristic_repair_on_window,
}
PRESETS: Dict[str, Any] = {
    "inspection": make_inspection_rules,
    "monitoring": make_monitoring_rules,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return data


def _resolve_path(base: Path, maybe_path: Any) -> Any:
    if isinstance(maybe_path, str):
        p = Path(maybe_path)
        if not p.is_absolute():
            p = (base.parent / p).resolve()
        return str(p)
    return maybe_path


def _load_env_kwargs(cfg_base: Path, env_cfg: Dict[str, Any]) -> Dict[str, Any]:
    kwargs = dict(env_cfg.get("kwargs", {}))
    if "config" in env_cfg:
        kwargs["config"] = env_cfg["config"]
    config_file = env_cfg.get("config_file")
    if config_file:
        file_path = Path(_resolve_path(cfg_base, config_file))
        with file_path.open("r", encoding="utf-8") as fh:
            if file_path.suffix.lower() in {".yaml", ".yml"}:
                kwargs["config"] = yaml.safe_load(fh)
            else:
                kwargs["config"] = json.load(fh)
    return kwargs


def _parse_rules(agent_cfg: Dict[str, Any]) -> List[RuleSpec]:
    if "preset" in agent_cfg:
        preset = agent_cfg["preset"]
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available presets: {sorted(PRESETS)}")
        params = agent_cfg.get("params", {})
        return PRESETS[preset](**params)

    rules_cfg = agent_cfg.get("rules")
    if not rules_cfg:
        raise ValueError("Agent config must provide 'preset' or explicit 'rules'.")

    rules: List[RuleSpec] = []
    for rule in rules_cfg:
        name = rule.get("fn")
        if name not in AVAILABLE_RULES:
            raise ValueError(f"Unknown rule '{name}'. Available rules: {sorted(AVAILABLE_RULES)}")
        params = rule.get("params", {})
        rules.append((AVAILABLE_RULES[name], params))
    return rules


def _parse_slice(value: Any) -> slice | None:
    if value is None:
        return None
    if isinstance(value, slice):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) == 2:
            return slice(value[0], value[1])
        if len(value) == 3:
            return slice(value[0], value[1], value[2])
    raise ValueError("pf_slice must be a list/tuple of length 2 or 3.")


def _expand_sweep_values(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        if not raw:
            raise ValueError("sweep parameter lists must contain at least one value")
        return raw
    if isinstance(raw, dict):
        if "values" in raw:
            values = raw["values"]
            if not isinstance(values, list) or not values:
                raise ValueError("sweep.values must be a non-empty list")
            return values
        if "range" in raw or "start" in raw:
            range_vals = raw.get("range")
            if range_vals is not None:
                if not isinstance(range_vals, list) or len(range_vals) < 2:
                    raise ValueError("range must look like [start, stop] or [start, stop, step]")
                start = range_vals[0]
                stop = range_vals[1]
                step = raw.get("step", range_vals[2] if len(range_vals) >= 3 else 1)
            else:
                start = raw.get("start")
                stop = raw.get("stop")
                step = raw.get("step", 1)
            inclusive = bool(raw.get("inclusive", False))
            stop_adj = stop + step if inclusive else stop
            values = list(np.arange(start, stop_adj, step))
            if not values:
                raise ValueError("range definition produced no values")
            if np.allclose(values, np.array(values, dtype=float).astype(int)):
                values = [int(v) for v in values]
            return values
    raise TypeError("Unsupported sweep value definition; use list, {values: [...]}, or range spec")


def _load_env(import_path: str, kwargs: Dict[str, Any]) -> Any:
    module_path, _, attr = import_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid env import path '{import_path}'")
    module = importlib.import_module(module_path)
    cls = getattr(module, attr)
    return cls(**kwargs)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_single(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    runner_cfg = cfg.get("runner", {})
    eval_cfg = cfg.get("evaluation", {})

    env_import = env_cfg["import_path"]
    env_kwargs = _load_env_kwargs(cfg_path, env_cfg)

    rules = _parse_rules(agent_cfg)
    n_agents = runner_cfg.get("n_agents")
    pf_slice = _parse_slice(runner_cfg.get("pf_slice"))
    init_age = int(runner_cfg.get("init_inspection_age", 61))
    episodes = int(eval_cfg.get("episodes", 100))
    boot_iters = int(eval_cfg.get("boot_iters", 2000))
    ci = float(eval_cfg.get("ci", 0.95))
    log_episode = bool(eval_cfg.get("log", False))

    env = _load_env(env_import, env_kwargs)
    agent = HeuristicAgent(rules)
    runner = EnvRunner(env=env, agent=agent, n_agents=n_agents, pf_slice=pf_slice)

    if log_episode:
        runner.run_episode(init_inspection_age=init_age, log=True)

    results = runner.evaluate(
        episodes=episodes,
        init_inspection_age=init_age,
        boot_iters=boot_iters,
        ci=ci,
    )
    results.update(
        {
            "config_file": cfg_path.name,
            "init_inspection_age": init_age,
            "episodes": episodes,
            "boot_iters": boot_iters,
            "ci": ci,
        }
    )
    return results


def run_sweep(cfg: Dict[str, Any], cfg_path: Path) -> Dict[str, Any]:
    env_cfg = cfg["env"]
    agent_cfg = cfg["agent"]
    runner_cfg = cfg.get("runner", {})
    sweep_cfg = cfg["sweep"]

    env_import = env_cfg["import_path"]
    env_kwargs = _load_env_kwargs(cfg_path, env_cfg)

    base_rules = _parse_rules(agent_cfg)
    n_agents = runner_cfg.get("n_agents")
    pf_slice = _parse_slice(runner_cfg.get("pf_slice"))
    init_age = int(runner_cfg.get("init_inspection_age", 61))

    sweep_params = sweep_cfg["params"]
    expanded_params = {key: _expand_sweep_values(value) for key, value in sweep_params.items()}
    eval_cfg = cfg.get("evaluation", {})
    sweep_episodes = int(sweep_cfg.get("episodes", eval_cfg.get("episodes", 100)))
    sweep_boot_iters = int(sweep_cfg.get("boot_iters", eval_cfg.get("boot_iters", 2000)))
    sweep_ci = float(sweep_cfg.get("ci", eval_cfg.get("ci", 0.95)))
    max_workers = sweep_cfg.get("max_workers")

    results = grid_search(
        base_rules=base_rules,
        sweep=expanded_params,
        env_import=env_import,
        env_kwargs=env_kwargs,
        n_agents=n_agents,
        pf_slice=pf_slice,
        episodes=sweep_episodes,
        init_inspection_age=init_age,
        boot_iters=sweep_boot_iters,
        ci=sweep_ci,
        max_workers=max_workers,
    )

    best = results[0] if results else None
    payload = {
        "metadata": {
            "config_file": cfg_path.name,
            "init_inspection_age": init_age,
            "episodes": sweep_episodes,
            "sweep": expanded_params,
            "best_by_mean_return": best,
            "max_workers": max_workers,
            "boot_iters": sweep_boot_iters,
            "ci": sweep_ci,
        },
        "results": results,
    }
    return payload


def save_output(content: Dict[str, Any], output_path: Path) -> None:
    _ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(content, fh, indent=2, sort_keys=True)


def resolve_output(cfg: Dict[str, Any], cfg_path: Path, default_name: str) -> Path:
    output_cfg = cfg.get("output", {})
    path_value = output_cfg.get("path", default_name)
    out_path = Path(path_value)
    if not out_path.is_absolute():
        out_path = (cfg_path.parent / out_path).resolve()
    return out_path


def run_from_config(cfg_path: Path) -> Path:
    cfg = _load_yaml(cfg_path)
    is_sweep = "sweep" in cfg
    default_name = f"{cfg_path.stem}_{'sweep' if is_sweep else 'results'}.json"
    output_path = resolve_output(cfg, cfg_path, default_name)

    if is_sweep:
        payload = run_sweep(cfg, cfg_path)
    else:
        payload = run_single(cfg, cfg_path)

    save_output(payload, output_path)
    return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run heuristic agent or sweep from YAML config.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg_path = args.config.resolve()
    output_path = run_from_config(cfg_path)
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
