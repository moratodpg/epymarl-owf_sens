"""Command-line utility for running heuristics against OWF-style environments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence

import importlib
import sys

import numpy as np

try:  # allow execution both as module and as script
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


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHONPATH = [ROOT.parent.parent / "imp_marl"]
for default_path in DEFAULT_PYTHONPATH:
    if default_path.exists() and str(default_path) not in sys.path:
        sys.path.insert(0, str(default_path))
AVAILABLE_RULES = {
    "heuristic_inspect_topk": heuristic_inspect_topk,
    "heuristic_repair_on_recent_inspection": heuristic_repair_on_recent_inspection,
    "heuristic_monitoring_install_topk": heuristic_monitoring_install_topk,
    "heuristic_repair_on_window": heuristic_repair_on_window,
}
PRESETS = {
    "inspection": make_inspection_rules,
    "monitoring": make_monitoring_rules,
}


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def import_env(import_path: str, kwargs: Dict[str, Any]) -> Any:
    module_path, _, attr = import_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid import path '{import_path}'")
    module = importlib.import_module(module_path)
    env_cls = getattr(module, attr)
    return env_cls(**kwargs)


def build_rules(agent_cfg: Dict[str, Any]) -> List[RuleSpec]:
    if "preset" in agent_cfg:
        preset = agent_cfg["preset"]
        if preset not in PRESETS:
            raise ValueError(f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}")
        params = agent_cfg.get("params", {})
        return PRESETS[preset](**params)

    rules_cfg = agent_cfg.get("rules")
    if not rules_cfg:
        raise ValueError("Agent configuration must define 'preset' or 'rules'.")

    rules: List[RuleSpec] = []
    for rule_item in rules_cfg:
        fn_name = rule_item.get("fn")
        if fn_name not in AVAILABLE_RULES:
            raise ValueError(
                f"Unknown rule '{fn_name}'. Available: {sorted(AVAILABLE_RULES)}"
            )
        params = rule_item.get("params", {})
        rules.append((AVAILABLE_RULES[fn_name], params))
    return rules


def parse_slice(pf_slice: Sequence[int] | None) -> slice | None:
    if pf_slice is None:
        return None
    if not pf_slice:
        return slice(None)
    if len(pf_slice) == 2:
        return slice(pf_slice[0], pf_slice[1])
    if len(pf_slice) == 3:
        return slice(pf_slice[0], pf_slice[1], pf_slice[2])
    raise ValueError("pf_slice must contain 2 or 3 integers if provided.")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def expand_sweep_values(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        if not raw:
            raise ValueError("Sweep parameter lists must contain at least one value.")
        return raw

    if isinstance(raw, dict):
        if "values" in raw:
            values = raw["values"]
            if not isinstance(values, list) or not values:
                raise ValueError("sweep.values must be a non-empty list.")
            return values

        if "range" in raw or "start" in raw:
            range_vals = raw.get("range")
            if range_vals is not None:
                if not isinstance(range_vals, list) or len(range_vals) < 2:
                    raise ValueError("range must be a list like [start, stop] or [start, stop, step].")
                start = range_vals[0]
                stop = range_vals[1]
                step = raw.get("step", range_vals[2] if len(range_vals) >= 3 else 1)
            else:
                start = raw.get("start")
                stop = raw.get("stop")
                step = raw.get("step", 1)
            if step == 0:
                raise ValueError("range step must be non-zero.")
            inclusive = bool(raw.get("inclusive", False))
            stop_adj = stop + step if inclusive else stop
            values = np.arange(start, stop_adj, step)
            if values.size == 0:
                raise ValueError("Computed range for sweep produced no values.")
            if np.allclose(values, values.astype(int)):
                values = values.astype(int)
            return values.tolist()

    raise TypeError(
        "Sweep parameter values must be a list, a dict with 'values', or a dict specifying a range."
    )


def resolve_output_path(
    config_path: Path,
    output_override: Path | None,
    output_cfg: Dict[str, Any],
    default_filename: str,
) -> Path:
    output_path = output_override or output_cfg.get("path")
    if not output_path:
        output_path = Path("results") / default_filename
    if isinstance(output_path, str):
        output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    return output_path


def run_from_config(config_path: Path, output_override: Path | None = None) -> Path:
    cfg = load_config(config_path)

    for entry in cfg.get("pythonpath", []):
        extra = Path(entry)
        if not extra.is_absolute():
            extra = (config_path.parent / extra).resolve()
        if str(extra) not in sys.path:
            sys.path.insert(0, str(extra))

    env_cfg = cfg.get("env", {})
    agent_cfg = cfg.get("agent", {})
    runner_cfg = cfg.get("runner", {})
    eval_cfg = cfg.get("evaluation", {})
    output_cfg = cfg.get("output", {})

    env_import = env_cfg.get("import_path")
    env_kwargs = env_cfg.get("kwargs", {})
    if not env_import:
        raise ValueError("Config must provide env.import_path")

    rules = build_rules(agent_cfg)

    pf_slice = parse_slice(runner_cfg.get("pf_slice"))
    n_agents = runner_cfg.get("n_agents")
    init_age = runner_cfg.get("init_inspection_age", 61)
    episodes = int(eval_cfg.get("episodes", 100))

    def make_env() -> Any:
        return import_env(env_import, env_kwargs)

    sweep_cfg = cfg.get("sweep")
    if sweep_cfg:
        return run_sweep(
            config_path=config_path,
            sweep_cfg=sweep_cfg,
            output_cfg=output_cfg,
            output_override=output_override,
            base_rules=rules,
            make_env=make_env,
            n_agents=n_agents,
            pf_slice=pf_slice,
            init_age=init_age,
            default_eval_episodes=episodes,
        )

    env = make_env()
    agent = HeuristicAgent(rules)
    runner = EnvRunner(
        env=env,
        agent=agent,
        n_agents=n_agents,
        pf_slice=pf_slice,
    )

    log_episode = bool(eval_cfg.get("log", False))

    if log_episode:
        runner.run_episode(init_inspection_age=init_age, log=True)

    results = runner.evaluate(episodes=episodes, init_inspection_age=init_age)
    results.update(
        {
            "config": config_path.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episodes": episodes,
            "init_inspection_age": init_age,
        }
    )

    output_path = resolve_output_path(
        config_path=config_path,
        output_override=output_override,
        output_cfg=output_cfg,
        default_filename=f"{config_path.stem}_results.json",
    )

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)

    return output_path


def run_sweep(
    *,
    config_path: Path,
    sweep_cfg: Dict[str, Any],
    output_cfg: Dict[str, Any],
    output_override: Path | None,
    base_rules: List[RuleSpec],
    make_env: Callable[[], Any],
    n_agents: int | None,
    pf_slice: slice | None,
    init_age: int,
    default_eval_episodes: int,
) -> Path:
    sweep_params = sweep_cfg.get("params")
    if not sweep_params:
        raise ValueError("Sweep configuration must include a 'params' mapping.")
    resolved_params: Dict[str, List[Any]] = {}
    for key, value in sweep_params.items():
        resolved_params[key] = expand_sweep_values(value)

    sweep_episodes = int(sweep_cfg.get("episodes", default_eval_episodes))
    sweep_init_age = int(sweep_cfg.get("init_inspection_age", init_age))

    def make_runner(rule_specs: List[RuleSpec]) -> EnvRunner:
        env = make_env()
        return EnvRunner(
            env=env,
            agent=HeuristicAgent(rule_specs),
            n_agents=n_agents,
            pf_slice=pf_slice,
        )

    base_rules_copy = [(fn, dict(params)) for fn, params in base_rules]
    sweep_results = grid_search(
        base_rules=base_rules_copy,
        sweep=resolved_params,
        make_runner=make_runner,
        episodes=sweep_episodes,
        init_inspection_age=sweep_init_age,
    )

    best_result = sweep_results[0] if sweep_results else None
    payload = {
        "metadata": {
            "config": config_path.name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "episodes": sweep_episodes,
            "init_inspection_age": sweep_init_age,
            "sweep": resolved_params,
            "best_by_mean_return": best_result,
        },
        "results": sweep_results,
    }

    output_path = resolve_output_path(
        config_path=config_path,
        output_override=output_override,
        output_cfg=output_cfg,
        default_filename=f"{config_path.stem}_sweep.json",
    )

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)

    return output_path


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "inspection_run.json",
        help="Path to the JSON config file. Relative paths are resolved from heuristics/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path for results JSON.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = args.config
    if not config_path.is_absolute():
        config_path = (ROOT / config_path).resolve()
    output_override = args.output
    if output_override is not None and not output_override.is_absolute():
        output_override = (ROOT / output_override).resolve()

    output_path = run_from_config(config_path, output_override=output_override)
    rel = output_path.relative_to(ROOT)
    print(f"Results written to {rel}")


if __name__ == "__main__":  # pragma: no cover
    main()
