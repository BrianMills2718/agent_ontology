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
from .knowledge_store import KnowledgeStore


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
        "score_em": round(em_mean, 4),
        "score_f1": round(f1_mean, 4),
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
# LLM-guided mutation (Flash) and deep analysis (Mini)
# ════════════════════════════════════════════════════════════════════

FLASH_MODEL = os.environ.get("AGENT_ONTOLOGY_FLASH_MODEL", "gemini-3-flash-preview")
ANALYST_MODEL = os.environ.get("AGENT_ONTOLOGY_ANALYST_MODEL", "gpt-5-mini")

MUTATION_SYSTEM_PROMPT = """\
You are an agent architecture mutation engine. You receive a YAML agent spec and context
about its performance, then produce a MUTATED version of the spec.

Rules:
- Output ONLY the complete mutated YAML spec (no explanation, no markdown fences)
- Make exactly ONE structural change (add/remove/modify a process, edge, gate condition, or schema)
- Keep the spec valid: every process referenced in edges must exist, entry_point must exist,
  every gate must have branches, every invoke edge needs input/output schemas
- Preserve the name field but append a short suffix describing your change
- Do NOT change the model fields on entities
- Focus on architecture changes that could improve benchmark performance

Common effective mutations:
- Add a review/critique step after generation (improves quality)
- Add a retrieval step before reasoning (adds knowledge)
- Change gate conditions to be more/less strict
- Add a reflection loop (retry with self-evaluation)
- Simplify by removing unnecessary intermediate steps
- Add parallel execution paths (fan-out + aggregation)
"""

MUTATION_USER_TEMPLATE = """\
## Parent Spec
```yaml
{spec_yaml}
```

## Context
- Detected patterns: {patterns}
- Lint warnings: {lint_warnings}
- Verify issues: {verify_issues}
{benchmark_context}
{previous_analysis}
{knowledge_context}

## Task
Produce a mutated version of this spec that addresses the issues above and could score higher
on the benchmark. Output ONLY the complete YAML spec.
"""

ANALYSIS_SYSTEM_PROMPT = """\
You are a deep analyst for agent architecture evolution. You examine the top-performing
agent specs from this generation and explain WHY they succeeded or failed.

Your output must be valid JSON with exactly these fields:
{
  "diagnosis": "2-3 sentence explanation of what made the top candidates score well",
  "failure_patterns": "What went wrong with candidates that scored poorly",
  "lessons": "Reusable insight for future generations (1-2 sentences)",
  "suggested_mutations": ["list of 2-3 specific structural changes to try next"]
}
"""

ANALYSIS_USER_TEMPLATE = """\
## Generation {generation} Results

### Top candidates (by fitness):
{top_candidates}

### Failed/low-scoring candidates:
{failed_candidates}

### Benchmark: {benchmark}

### Knowledge store context:
{knowledge_context}

Analyze these results and produce your JSON output.
"""


def _get_lint_warnings(spec):
    """Get lint warnings for a spec."""
    try:
        from . import lint as lint_mod
        issues = lint_mod.lint_spec(spec)
        return "; ".join(f"{i.code}: {i.message}" for i in issues[:5])
    except Exception:
        return "none"


def _get_verify_issues(spec):
    """Get verify failures for a spec."""
    try:
        from . import verify as verify_mod
        passes, failures = verify_mod.verify_spec(spec)
        return "; ".join(failures[:5]) if failures else "none"
    except Exception:
        return "none"


def _get_knowledge_context(store, benchmark, limit=5):
    """Get relevant context from the knowledge store."""
    if not store:
        return "No knowledge store available."

    lines = []
    best = store.best_genotypes(benchmark, limit=limit) if benchmark else []
    if best:
        lines.append("Best known genotypes:")
        for r in best:
            lines.append(f"  - {r['spec_name']}: fitness={r['fitness']}, patterns={r['detected_patterns']}")

    mut_eff = store.mutation_effectiveness()[:5]
    if mut_eff:
        lines.append("Most effective mutations:")
        for r in mut_eff:
            lines.append(f"  - {r['mutation_description']}: avg_fitness={r['avg_fitness']:.1f} (n={r['count']})")

    failures = store.failure_lessons(benchmark, limit=3) if benchmark else []
    if failures:
        lines.append("Recent failure lessons:")
        for r in failures:
            err = (r.get('error_details') or '')[:100]
            lines.append(f"  - [{r['status']}] {r['spec_name']}: {err}")

    return "\n".join(lines) if lines else "No prior data."


def llm_mutate(parent_spec, parent_yaml, benchmark_suite=None,
               benchmark_results=None, previous_analysis=None,
               knowledge_store=None, model=None, verbose=False):
    """Generate a mutation using an LLM (Flash model).

    Returns (mutated_spec_dict, mutation_description) or raises on failure.
    """
    import yaml
    from .specgen import call_llm, extract_yaml

    model = model or FLASH_MODEL
    patterns = _detect_pattern_names(parent_spec)

    # Build context
    benchmark_ctx = ""
    if benchmark_results:
        benchmark_ctx = f"- Benchmark ({benchmark_suite}): fitness={benchmark_results.get('fitness', 0)}, " \
                        f"status={benchmark_results.get('status', '?')}, " \
                        f"llm_calls={benchmark_results.get('llm_calls', 0)}"
        if benchmark_results.get('error'):
            benchmark_ctx += f"\n- Error: {benchmark_results['error'][:200]}"

    analysis_ctx = ""
    if previous_analysis:
        analysis_ctx = f"- Previous generation analysis:\n  {previous_analysis.get('diagnosis', '')}\n" \
                        f"  Suggested mutations: {previous_analysis.get('suggested_mutations', [])}"

    knowledge_ctx = _get_knowledge_context(knowledge_store, benchmark_suite)

    user_prompt = MUTATION_USER_TEMPLATE.format(
        spec_yaml=parent_yaml[:3000],  # truncate very large specs
        patterns=", ".join(patterns) if patterns else "none detected",
        lint_warnings=_get_lint_warnings(parent_spec),
        verify_issues=_get_verify_issues(parent_spec),
        benchmark_context=benchmark_ctx,
        previous_analysis=analysis_ctx,
        knowledge_context=knowledge_ctx,
    )

    if verbose:
        print(f"    [Flash] Generating mutation with {model}...")

    # Retry up to 2 times with error feedback
    last_error = None
    for attempt in range(2):
        prompt = user_prompt
        if last_error and attempt > 0:
            prompt += f"\n\n## IMPORTANT: Your previous attempt failed with this error:\n{last_error}\nPlease fix the issue and output ONLY valid YAML."

        response = call_llm(model, MUTATION_SYSTEM_PROMPT, prompt,
                             temperature=0.7, max_tokens=8192)
        yaml_text = extract_yaml(response)

        try:
            mutated_spec = yaml.safe_load(yaml_text)
            if isinstance(mutated_spec, dict) and "processes" in mutated_spec:
                break
            last_error = "Output is a dict but missing 'processes' key. Include ALL sections: entities, processes, edges, schemas."
        except yaml.YAMLError as e:
            last_error = f"YAML parse error: {e}"
            continue
    else:
        raise ValueError(last_error or "LLM output is not a valid spec dict")

    # Compute a description of what changed
    orig_procs = {p["id"] for p in parent_spec.get("processes", [])}
    new_procs = {p["id"] for p in mutated_spec.get("processes", [])}
    added = new_procs - orig_procs
    removed = orig_procs - new_procs

    desc_parts = []
    if added:
        desc_parts.append(f"added {', '.join(sorted(added))}")
    if removed:
        desc_parts.append(f"removed {', '.join(sorted(removed))}")
    if not desc_parts:
        # Check edge count changes
        orig_edges = len(parent_spec.get("edges", []))
        new_edges = len(mutated_spec.get("edges", []))
        if new_edges != orig_edges:
            desc_parts.append(f"edges {orig_edges}→{new_edges}")
        else:
            desc_parts.append("modified structure")

    mutation_desc = "llm: " + "; ".join(desc_parts)

    return mutated_spec, mutation_desc, yaml_text


def llm_analyze(generation, gen_results, benchmark_suite=None,
                knowledge_store=None, model=None, verbose=False):
    """Deep analysis of generation results using analyst model (Mini).

    Returns analysis dict with diagnosis, failure_patterns, lessons, suggested_mutations.
    """
    from .specgen import call_llm

    model = model or ANALYST_MODEL

    # Format top candidates
    sorted_results = sorted(gen_results, key=lambda x: x.get("fitness", 0), reverse=True)
    top = sorted_results[:4]
    failed = [r for r in sorted_results if r.get("fitness", 0) == 0][:4]

    top_lines = []
    for r in top:
        patterns = r.get("lineage", {}).get("patterns", [])
        top_lines.append(
            f"  - {r['name']}: fitness={r.get('fitness', 0):.1f}, "
            f"status={r.get('status')}, llm_calls={r.get('llm_calls', 0)}, "
            f"mutations={r.get('mutations', [])}, patterns={patterns}"
        )

    failed_lines = []
    for r in failed:
        err = (r.get("error") or "")[:150]
        failed_lines.append(
            f"  - {r['name']}: status={r.get('status')}, "
            f"mutations={r.get('mutations', [])}, error={err}"
        )

    knowledge_ctx = _get_knowledge_context(knowledge_store, benchmark_suite)

    user_prompt = ANALYSIS_USER_TEMPLATE.format(
        generation=generation,
        top_candidates="\n".join(top_lines) if top_lines else "  (none)",
        failed_candidates="\n".join(failed_lines) if failed_lines else "  (none)",
        benchmark=benchmark_suite or "standard",
        knowledge_context=knowledge_ctx,
    )

    if verbose:
        print(f"    [Mini] Analyzing generation {generation} with {model}...")

    response = call_llm(model, ANALYSIS_SYSTEM_PROMPT, user_prompt,
                         temperature=0.3, max_tokens=1024)

    # Parse JSON response
    try:
        # Strip markdown fences if present
        text = response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        analysis = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        analysis = {
            "diagnosis": response[:300],
            "failure_patterns": "",
            "lessons": "",
            "suggested_mutations": [],
        }

    return analysis


# ════════════════════════════════════════════════════════════════════
# Evolution loop
# ════════════════════════════════════════════════════════════════════

def evolve(base_spec_path, test_inputs, generations=3, population=5,
           timeout_sec=60, verbose=True, benchmark_suite=None,
           benchmark_examples=5, crossover_enabled=False,
           crossover_rate=0.3, knowledge_store=None,
           llm_guided=False, programmatic_ratio=0.2,
           flash_model=None, analyst_model=None):
    """Run evolutionary search. Returns list of generation results.

    If knowledge_store is provided (a KnowledgeStore instance), every candidate
    result is persisted for cross-run learning.

    If llm_guided=True, uses Flash/Mini cascade:
    - Flash generates mutations (sees spec + errors + previous analysis + knowledge)
    - Mini analyzes top-K after each generation (diagnosis + lessons)
    - programmatic_ratio controls what fraction of mutations use random mutate.py (diversity)
    """
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
    previous_analysis = None  # Mini's analysis from previous generation

    for gen in range(generations):
        gen_results = []
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Generation {gen + 1}/{generations} — {len(current_population)} parent(s)")
            if benchmark_suite:
                print(f"  Fitness: benchmark ({benchmark_suite}, {benchmark_examples} examples)")
            if llm_guided:
                print(f"  Mutations: LLM-guided ({flash_model or FLASH_MODEL}) "
                      f"+ {programmatic_ratio:.0%} programmatic diversity")
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
            parent_yaml = yaml.dump(parent_spec, default_flow_style=False)

            for i in range(mutations_per_parent):
                # Decide mutation method: LLM-guided, crossover, or programmatic
                use_crossover = (crossover_enabled and
                                 len(current_population) >= 2 and
                                 random.random() < crossover_rate)
                use_programmatic = (not use_crossover and llm_guided and
                                    random.random() < programmatic_ratio)
                use_llm = llm_guided and not use_crossover and not use_programmatic

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
                            use_llm = llm_guided  # fallback

                    if use_llm and not use_crossover:
                        # LLM-guided mutation via Flash
                        result, mut_desc, _ = llm_mutate(
                            parent_spec, parent_yaml,
                            benchmark_suite=benchmark_suite,
                            previous_analysis=previous_analysis,
                            knowledge_store=knowledge_store,
                            model=flash_model,
                            verbose=verbose,
                        )
                        mut_name = f"gen{gen+1}_{parent_name}_llm{i+1}"
                        mut_ops = [mut_desc]
                        parent_list = [parent_name]

                    elif not use_crossover:
                        # Programmatic mutation (diversity mechanism)
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
                        print(f"  Warning: mutation failed: {e}")

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
                if knowledge_store:
                    try:
                        knowledge_store.record_candidate(
                            spec_name=name, spec=spec, base_spec=base_name,
                            generation=gen + 1, parents=parents,
                            mutation_description=", ".join(mutations) if mutations else None,
                            benchmark=benchmark_suite, fitness=0.0,
                            status="INVALID", error_details=output[:500],
                        )
                    except Exception:
                        pass
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
                if knowledge_store:
                    try:
                        knowledge_store.record_candidate(
                            spec_name=name, spec=spec, base_spec=base_name,
                            generation=gen + 1, parents=parents,
                            mutation_description=", ".join(mutations) if mutations else None,
                            benchmark=benchmark_suite, fitness=0.0,
                            status="GEN_FAIL",
                        )
                    except Exception:
                        pass
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

            # Persist to knowledge store
            if knowledge_store:
                try:
                    spec_yaml_text = yaml.dump(spec, default_flow_style=False)
                    knowledge_store.record_candidate(
                        spec_name=name,
                        spec=spec,
                        spec_yaml=spec_yaml_text if fitness > 0 else None,
                        base_spec=base_name,
                        generation=gen + 1,
                        parents=parents,
                        mutation_description=", ".join(mutations) if mutations else None,
                        benchmark=benchmark_suite,
                        score_em=test_result.get("score_em"),
                        score_f1=test_result.get("score_f1"),
                        fitness=fitness,
                        llm_calls=test_result.get("llm_calls", 0),
                        duration_ms=test_result.get("duration_ms", 0),
                        status=test_result["status"],
                        error_details=test_result.get("error"),
                    )
                except Exception as e:
                    if verbose:
                        print(f"  Warning: failed to record to knowledge store: {e}")

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

        # Deep analysis via analyst model (Mini) — for LLM-guided mode
        gen_analysis = None
        if llm_guided and gen_results:
            try:
                gen_analysis = llm_analyze(
                    generation=gen + 1,
                    gen_results=gen_results,
                    benchmark_suite=benchmark_suite,
                    knowledge_store=knowledge_store,
                    model=analyst_model,
                    verbose=verbose,
                )
                previous_analysis = gen_analysis  # feed to next generation

                if verbose and gen_analysis:
                    print(f"\n  [Analysis] {gen_analysis.get('diagnosis', '')[:200]}")
                    suggestions = gen_analysis.get('suggested_mutations', [])
                    if suggestions:
                        print(f"  [Suggestions] {', '.join(suggestions[:3])}")

                # Store analysis + lessons in knowledge store for top candidates
                if knowledge_store and gen_analysis:
                    for fitness, name, _, _ in scored[:top_k]:
                        if fitness > 0:
                            try:
                                # Update the most recent record for this candidate
                                knowledge_store.conn.execute(
                                    """UPDATE evolution_results
                                       SET analysis = ?, lessons = ?
                                       WHERE spec_name = ? AND generation = ?
                                       ORDER BY id DESC LIMIT 1""",
                                    (
                                        gen_analysis.get("diagnosis", ""),
                                        gen_analysis.get("lessons", ""),
                                        name,
                                        gen + 1,
                                    ),
                                )
                                knowledge_store.conn.commit()
                            except Exception:
                                pass

            except Exception as e:
                if verbose:
                    print(f"  Warning: analysis failed: {e}")

        history.append({
            "generation": gen + 1,
            "candidates": len(candidates),
            "results": gen_results,
            "survivors": [name for fitness, name, _, _ in scored[:top_k] if fitness > 0],
            "best_fitness": scored[0][0] if scored else 0,
            "best_name": scored[0][1] if scored else "none",
            "analysis": gen_analysis,
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
    parser.add_argument("--llm-guided", action="store_true",
                        help="Use LLM-guided mutations (Flash/Mini cascade)")
    parser.add_argument("--flash-model", default=None,
                        help=f"Model for mutation generation (default: {FLASH_MODEL})")
    parser.add_argument("--analyst-model", default=None,
                        help=f"Model for deep analysis (default: {ANALYST_MODEL})")
    parser.add_argument("--programmatic-ratio", type=float, default=0.2,
                        help="Fraction of mutations that are programmatic (0-1, default: 0.2)")
    parser.add_argument("--no-store", action="store_true",
                        help="Disable knowledge store persistence")
    parser.add_argument("--db", metavar="PATH",
                        help="Knowledge store database path (default: ~/.agent_ontology/evolution.db)")
    args = parser.parse_args()

    agent_name = detect_agent_name(args.spec)
    test_inputs = EVOLVE_INPUTS.get(agent_name)
    if not test_inputs and not args.benchmark:
        print(f"No test inputs defined for '{agent_name}'. "
              f"Add to EVOLVE_INPUTS or use --benchmark.")
        print(f"Available: {', '.join(EVOLVE_INPUTS.keys())}")
        sys.exit(1)

    # Knowledge store
    store = None
    if not args.no_store:
        try:
            store = KnowledgeStore(args.db)
            if not args.quiet and not args.json:
                print(f"  Knowledge store: {store.db_path} ({store.count()} existing records)")
        except Exception as e:
            print(f"  Warning: could not open knowledge store: {e}")

    try:
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
            knowledge_store=store,
            llm_guided=args.llm_guided,
            programmatic_ratio=args.programmatic_ratio,
            flash_model=args.flash_model,
            analyst_model=args.analyst_model,
        )
    finally:
        if store:
            store.close()

    if args.json:
        print(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
