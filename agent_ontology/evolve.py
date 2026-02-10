#!/usr/bin/env python3
"""
Evolutionary Search over Agent Architectures

Connects mutate.py + instantiate.py + test_agents.py into an evolutionary loop:
  1. Start with a base spec
  2. Generate mutations (field-level or pattern-level)
  3. Optionally crossover between parents
  4. Validate and instantiate surviving specs
  5. Score fitness (pass/fail, LLM calls, duration, or benchmark)
  6. Select top-K, repeat with lineage tracking

Usage:
    python3 evolve.py specs/react.yaml --generations 3 --population 5
    python3 evolve.py specs/react.yaml --generations 5 --population 8 --benchmark gsm8k
    python3 evolve.py specs/react.yaml --generations 5 --crossover --crossover-rate 0.3
    python3 evolve.py specs/react.yaml --generations 1 --population 3 --json
"""

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
import tempfile
import importlib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from . import mutate
from . import validate as validator_module


def validate_spec_text(yaml_text):
    """Write yaml_text to a temp file and validate it. Returns (ok, output)."""
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir=SCRIPT_DIR) as f:
        f.write(yaml_text)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "validate.py"), tmp_path],
            capture_output=True, text=True, cwd=SCRIPT_DIR
        )
        has_errors = "ERROR" in (result.stdout + result.stderr) or result.returncode != 0
        return not has_errors, (result.stdout + result.stderr).strip()
    finally:
        os.unlink(tmp_path)


def validate_spec_file(path):
    """Validate a spec file. Returns (ok, output)."""
    path = os.path.abspath(path)
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "validate.py"), path],
        capture_output=True, text=True, cwd=SCRIPT_DIR
    )
    has_errors = "ERROR" in (result.stdout + result.stderr) or result.returncode != 0
    return not has_errors, (result.stdout + result.stderr).strip()


def instantiate_spec(spec_path, agent_path):
    """Instantiate a spec to a Python agent file. Returns True on success."""
    spec_path = os.path.abspath(spec_path)
    agent_path = os.path.abspath(agent_path)
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "instantiate.py"), spec_path, "-o", agent_path],
        capture_output=True, text=True, cwd=SCRIPT_DIR
    )
    return result.returncode == 0


def run_agent_test(agent_module_name, test_inputs, timeout_sec=60):
    """Run an agent with test inputs and return results dict."""
    import signal

    result = {
        "status": "unknown",
        "duration_ms": 0,
        "llm_calls": 0,
        "issues": [],
        "error": None,
    }

    try:
        # Force reimport
        if agent_module_name in sys.modules:
            del sys.modules[agent_module_name]
        mod = importlib.import_module(agent_module_name)
    except Exception as e:
        result["status"] = "IMPORT_ERROR"
        result["error"] = str(e)
        return result

    # Reset trace
    if hasattr(mod, 'TRACE'):
        mod.TRACE.clear()

    # Patch input
    import builtins
    original_input = builtins.input
    builtins.input = lambda prompt="": "skip"

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {timeout_sec}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    t0 = time.time()
    try:
        state = mod.run(test_inputs.copy())
        result["duration_ms"] = int((time.time() - t0) * 1000)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, 'TRACE') else 0
        result["status"] = "PASS"
    except TimeoutError as e:
        result["status"] = "TIMEOUT"
        result["error"] = str(e)
        result["duration_ms"] = int((time.time() - t0) * 1000)
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, 'TRACE') else 0
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
        result["duration_ms"] = int((time.time() - t0) * 1000)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        builtins.input = original_input

    return result


def compute_fitness(test_result):
    """Compute a fitness score from test results. Higher is better."""
    if test_result["status"] != "PASS":
        return 0.0

    # Base score for passing
    score = 100.0

    # Reward fewer LLM calls (efficiency)
    calls = test_result["llm_calls"]
    if calls > 0:
        score += max(0, 50 - calls * 2)  # Up to +50 for very few calls

    # Reward faster execution
    duration_s = test_result["duration_ms"] / 1000
    if duration_s > 0:
        score += max(0, 30 - duration_s * 0.5)  # Up to +30 for fast

    return round(score, 1)


def compute_fitness_benchmark(agent_module_name, suite, examples=5, timeout_sec=120,
                              base_agent_type=None):
    """Compute fitness via benchmark suite inline. Returns (score, test_result_dict).

    Loads agent module directly (must be importable via sys.path),
    runs it on benchmark examples, and scores the results.
    base_agent_type: the original agent type (e.g. "self_refine") for input formatting.
    """
    import signal
    import builtins

    # Lazy imports from benchmarks subpackage
    from .benchmarks.scoring import (
        extract_answer, score_hotpotqa, score_gsm8k,
        score_arc, score_humaneval,
    )

    # Use base_agent_type for formatting inputs (evolved agents share base schema)
    agent_type = base_agent_type or agent_module_name.replace("_agent", "")

    # Load dataset
    datasets_dir = os.path.join(SCRIPT_DIR, "benchmarks", "datasets")
    dataset_path = os.path.join(datasets_dir, f"{suite}.json")
    try:
        with open(dataset_path) as f:
            dataset_meta = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return 0.0, {"status": "BENCH_FAIL", "llm_calls": 0, "duration_ms": 0}

    dataset_examples = dataset_meta["examples"][:examples]

    # Import module
    try:
        if agent_module_name in sys.modules:
            del sys.modules[agent_module_name]
        mod = importlib.import_module(agent_module_name)
    except Exception as e:
        return 0.0, {"status": "IMPORT_ERROR", "llm_calls": 0, "duration_ms": 0, "error": str(e)}

    # Score each example
    scores_em = []
    scores_f1 = []
    total_calls = 0
    total_duration_ms = 0

    original_input = builtins.input
    builtins.input = lambda prompt="": "skip"

    def timeout_handler(signum, frame):
        raise TimeoutError(f"Timed out after {timeout_sec}s")

    try:
        for example in dataset_examples:
            # Format input — try the base agent type for formatting
            inputs = _format_benchmark_input(agent_type, example, suite)

            # Reset trace
            if hasattr(mod, 'TRACE'):
                mod.TRACE.clear()

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_sec)
            t0 = time.time()
            try:
                state = mod.run(inputs)
                elapsed_ms = int((time.time() - t0) * 1000)
                calls = len(mod.TRACE) if hasattr(mod, 'TRACE') else 0
                total_calls += calls
                total_duration_ms += elapsed_ms

                # Extract answer
                state_data = state.data if hasattr(state, "data") else {}
                predicted = extract_answer(state_data)

                # Score
                expected = example["answer"]
                if suite == "hotpotqa":
                    s = score_hotpotqa(predicted, str(expected))
                elif suite == "gsm8k":
                    s = score_gsm8k(predicted, expected)
                elif suite == "arc":
                    s = score_arc(predicted, expected)
                elif suite == "humaneval":
                    s = score_humaneval(predicted, example)
                else:
                    s = {"em": 1.0 if str(expected).lower() in predicted.lower() else 0.0}

                scores_em.append(s.get("em", 0.0))
                if "f1" in s:
                    scores_f1.append(s["f1"])

            except TimeoutError:
                scores_em.append(0.0)
                total_duration_ms += int((time.time() - t0) * 1000)
            except Exception:
                scores_em.append(0.0)
                total_duration_ms += int((time.time() - t0) * 1000)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
    finally:
        builtins.input = original_input

    if not scores_em:
        return 0.0, {"status": "BENCH_FAIL", "llm_calls": 0, "duration_ms": 0}

    # Compute fitness
    em_mean = sum(scores_em) / len(scores_em)
    f1_mean = sum(scores_f1) / len(scores_f1) if scores_f1 else 0.0

    # Combine: EM weighted higher, scale to 200
    score = (em_mean * 0.7 + f1_mean * 0.3) * 200

    # Efficiency bonus: fewer avg calls = better
    avg_calls = total_calls / len(dataset_examples) if dataset_examples else 0
    if avg_calls > 0:
        score += max(0, 20 - avg_calls)

    test_result = {
        "status": "PASS" if score > 0 else "BENCH_FAIL",
        "llm_calls": total_calls,
        "duration_ms": total_duration_ms,
    }

    return round(score, 1), test_result


def _format_benchmark_input(agent_type, example, dataset_name):
    """Format a benchmark example into agent input.

    Tries format_input from compatibility module first, falls back to generic.
    """
    try:
        from .benchmarks.compatibility import format_input
        return format_input(agent_type, example, dataset_name)
    except Exception:
        pass
    # Generic fallback
    question = example["question"]
    return {"query": question, "task": question, "problem": question}


# ════════════════════════════════════════════════════════════════════
# Lineage tracking
# ════════════════════════════════════════════════════════════════════

def _init_lineage(spec, generation=0):
    """Initialize lineage metadata on a spec."""
    spec.setdefault("metadata", {})
    spec["metadata"]["lineage"] = {
        "parents": [],
        "generation": generation,
        "mutations": [],
        "patterns": _detect_pattern_names(spec),
    }


def _update_lineage(spec, parent_names, generation, mutation_ops):
    """Update lineage after mutation/crossover."""
    spec.setdefault("metadata", {})
    spec["metadata"]["lineage"] = {
        "parents": parent_names,
        "generation": generation,
        "mutations": mutation_ops,
        "patterns": _detect_pattern_names(spec),
    }


def _detect_pattern_names(spec):
    """Detect pattern names in a spec (lazy import to avoid circular)."""
    try:
        from . import patterns as pat_mod
        detected = pat_mod.detect_patterns(spec)
        return [pname for pname, _, _ in detected]
    except Exception:
        return []


def _format_lineage_tree(history):
    """Format a lineage tree from evolution history."""
    lines = []
    for gen_data in history:
        gen = gen_data["generation"]
        for entry in gen_data.get("results", []):
            if entry.get("fitness", 0) > 0:
                parents = entry.get("lineage", {}).get("parents", [])
                parent_str = f" <- {', '.join(parents)}" if parents else ""
                patterns = entry.get("lineage", {}).get("patterns", [])
                pat_str = f" [{', '.join(patterns)}]" if patterns else ""
                lines.append(
                    f"  Gen {gen}: {entry['name']} "
                    f"(fitness={entry['fitness']:.1f}){parent_str}{pat_str}"
                )
    return "\n".join(lines) if lines else "  (no lineage data)"


# ════════════════════════════════════════════════════════════════════
# Evolution loop
# ════════════════════════════════════════════════════════════════════

def evolve(base_spec_path, test_inputs, generations=3, population=5,
           timeout_sec=60, verbose=True, benchmark_suite=None,
           benchmark_examples=5, crossover_enabled=False,
           crossover_rate=0.3):
    """Run evolutionary search. Returns list of generation results."""
    import yaml

    base_spec_path = os.path.abspath(base_spec_path)
    with open(base_spec_path) as f:
        base_spec = yaml.safe_load(f)

    _init_lineage(base_spec, generation=0)

    base_name = base_spec.get("name", "agent").lower().replace(" ", "_")
    # Derive base agent type from spec filename for benchmark input formatting
    base_agent_type = os.path.basename(base_spec_path).replace(".yaml", "")
    work_dir = tempfile.mkdtemp(prefix="evolve_")

    history = []
    current_population = [("base", base_spec_path, copy.deepcopy(base_spec))]

    for gen in range(generations):
        gen_results = []
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Generation {gen + 1}/{generations} — {len(current_population)} parent(s)")
            if benchmark_suite:
                print(f"  Fitness: benchmark ({benchmark_suite}, {benchmark_examples} examples)")
            if crossover_enabled:
                print(f"  Crossover rate: {crossover_rate:.0%}")
            print(f"{'='*60}")

        # Generate mutations and crossovers from parents
        candidates = []
        for parent_name, parent_path, parent_spec in current_population:
            # Keep parent as-is
            candidates.append((parent_name, parent_path, parent_spec, [], [parent_name]))

            # Generate mutations
            mutations_per_parent = max(1, population // len(current_population))
            for i in range(mutations_per_parent):
                # Decide: crossover or mutation
                use_crossover = (crossover_enabled and
                                 len(current_population) >= 2 and
                                 random.random() < crossover_rate)

                try:
                    if use_crossover:
                        # Pick a different parent for crossover
                        other_parents = [(n, p, s) for n, p, s in current_population
                                         if n != parent_name]
                        if other_parents:
                            other_name, _, other_spec = random.choice(other_parents)
                            result = mutate.crossover(parent_spec, other_spec)
                            mut_name = f"gen{gen+1}_{parent_name}_cx_{other_name}_{i+1}"
                            mut_ops = ["crossover"]
                            parent_list = [parent_name, other_name]
                        else:
                            use_crossover = False

                    if not use_crossover:
                        result, op_name = mutate.apply_random_mutation(parent_spec)
                        mut_name = f"gen{gen+1}_{parent_name}_mut{i+1}"
                        mut_ops = [op_name]
                        parent_list = [parent_name]

                    # Track lineage
                    _update_lineage(result, parent_list, gen + 1, mut_ops)

                    # Write to temp file
                    mut_path = os.path.join(work_dir, f"{mut_name}.yaml")
                    with open(mut_path, "w") as f:
                        yaml.dump(result, f, default_flow_style=False)
                    candidates.append((mut_name, mut_path, result, mut_ops, parent_list))

                except (ValueError, Exception) as e:
                    if verbose:
                        print(f"  Warning: mutation/crossover failed: {e}")

        if verbose:
            print(f"  {len(candidates)} candidates (parents + offspring)")

        # Validate, instantiate, and test each candidate
        scored = []
        for name, spec_path, spec, mutations, parents in candidates:
            # Validate
            ok, output = validate_spec_file(spec_path)
            if not ok:
                if verbose:
                    print(f"  {name:40s}  INVALID")
                gen_results.append({
                    "name": name, "status": "INVALID", "fitness": 0,
                    "mutations": mutations,
                    "lineage": spec.get("metadata", {}).get("lineage", {}),
                })
                continue

            # Instantiate
            agent_path = os.path.join(work_dir, f"{name}_agent.py")
            ok = instantiate_spec(spec_path, agent_path)
            if not ok:
                if verbose:
                    print(f"  {name:40s}  GEN_FAIL")
                gen_results.append({
                    "name": name, "status": "GEN_FAIL", "fitness": 0,
                    "mutations": mutations,
                    "lineage": spec.get("metadata", {}).get("lineage", {}),
                })
                continue

            # Add work_dir to path for import
            if work_dir not in sys.path:
                sys.path.insert(0, work_dir)

            # Score fitness
            if benchmark_suite:
                module_name = f"{name}_agent"
                fitness, test_result = compute_fitness_benchmark(
                    module_name, benchmark_suite, benchmark_examples,
                    timeout_sec=timeout_sec,
                    base_agent_type=base_agent_type,
                )
            else:
                module_name = f"{name}_agent"
                test_result = run_agent_test(module_name, test_inputs, timeout_sec)
                fitness = compute_fitness(test_result)

            entry = {
                "name": name,
                "status": test_result["status"],
                "fitness": fitness,
                "llm_calls": test_result.get("llm_calls", 0),
                "duration_ms": test_result.get("duration_ms", 0),
                "mutations": mutations,
                "error": test_result.get("error"),
                "lineage": spec.get("metadata", {}).get("lineage", {}),
            }
            gen_results.append(entry)
            scored.append((fitness, name, spec_path, spec))

            if verbose:
                icon = "+" if fitness > 0 else "-"
                mut_str = f" [{', '.join(mutations)}]" if mutations else " [base]"
                print(f"  {icon} {name:40s}  {test_result['status']:8s}  "
                      f"fitness={fitness:6.1f}{mut_str}")

        # Select top-K for next generation
        scored.sort(key=lambda x: x[0], reverse=True)
        top_k = max(2, population // 2)
        current_population = [
            (name, path, spec) for fitness, name, path, spec in scored[:top_k]
            if fitness > 0
        ]

        if not current_population:
            if verbose:
                print("\n  No surviving candidates! Reverting to base.")
            current_population = [("base", base_spec_path, copy.deepcopy(base_spec))]

        history.append({
            "generation": gen + 1,
            "candidates": len(candidates),
            "results": gen_results,
            "survivors": [name for fitness, name, _, _ in scored[:top_k] if fitness > 0],
            "best_fitness": scored[0][0] if scored else 0,
            "best_name": scored[0][1] if scored else "none",
        })

    # Final summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Evolution Complete — {generations} generation(s)")
        print(f"{'='*60}")
        for gen_data in history:
            best = gen_data["best_name"]
            fit = gen_data["best_fitness"]
            print(f"  Gen {gen_data['generation']}: {gen_data['candidates']} candidates, "
                  f"best={best} (fitness={fit})")
        if current_population:
            print(f"\n  Final best: {current_population[0][0]}")

        # Print lineage tree
        print(f"\n  Lineage:")
        print(_format_lineage_tree(history))

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return history


# ── Test inputs for common agents ──
EVOLVE_INPUTS = {
    "react": {"query": "What year was the Eiffel Tower built?"},
    "self_refine": {"task": "Explain photosynthesis in one paragraph.", "max_rounds": 2},
    "tree_of_thought": {"problem": "What is 17 * 23?", "branching_factor": 2, "max_depth": 2, "beam_width": 2},
    "plan_and_solve": {"problem": "How to make a sandwich?", "max_retries": 1},
    "debate": {"topic": "Should AI be regulated?"},
    "rag": {"query": "What is machine learning?"},
}


def detect_agent_name(spec_path):
    """Extract agent name from spec filename."""
    import yaml
    with open(spec_path) as f:
        spec = yaml.safe_load(f)
    name = os.path.basename(spec_path).replace(".yaml", "")
    return name


def main():
    parser = argparse.ArgumentParser(
        description="Evolutionary Search over Agent Architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 evolve.py specs/react.yaml --generations 3 --population 5\n"
            "  python3 evolve.py specs/react.yaml --generations 5 --benchmark gsm8k\n"
            "  python3 evolve.py specs/react.yaml --crossover --crossover-rate 0.3\n"
        ),
    )
    parser.add_argument("spec", help="Base spec YAML file")
    parser.add_argument("--generations", "-g", type=int, default=3,
                        help="Number of generations")
    parser.add_argument("--population", "-p", type=int, default=5,
                        help="Population size per generation")
    parser.add_argument("--timeout", "-t", type=int, default=60,
                        help="Per-agent test timeout (seconds)")
    parser.add_argument("--benchmark", metavar="SUITE",
                        help="Use benchmark suite for fitness (e.g., gsm8k, hotpotqa)")
    parser.add_argument("--benchmark-examples", type=int, default=5,
                        help="Number of benchmark examples per evaluation")
    parser.add_argument("--crossover", action="store_true",
                        help="Enable crossover between parents")
    parser.add_argument("--crossover-rate", type=float, default=0.3,
                        help="Fraction of offspring produced by crossover (0-1)")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress verbose output")
    args = parser.parse_args()

    agent_name = detect_agent_name(args.spec)
    test_inputs = EVOLVE_INPUTS.get(agent_name)
    if not test_inputs and not args.benchmark:
        print(f"No test inputs defined for '{agent_name}'. "
              f"Add to EVOLVE_INPUTS or use --benchmark.")
        print(f"Available: {', '.join(EVOLVE_INPUTS.keys())}")
        sys.exit(1)

    history = evolve(
        args.spec,
        test_inputs or {},
        generations=args.generations,
        population=args.population,
        timeout_sec=args.timeout,
        verbose=not args.quiet and not args.json,
        benchmark_suite=args.benchmark,
        benchmark_examples=args.benchmark_examples,
        crossover_enabled=args.crossover,
        crossover_rate=args.crossover_rate,
    )

    if args.json:
        print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
