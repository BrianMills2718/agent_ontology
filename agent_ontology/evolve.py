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
import re
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

    # Accuracy component: EM weighted higher, scale to 100
    accuracy_score = (em_mean * 0.7 + f1_mean * 0.3) * 100

    # Efficiency component (matters more when accuracy is saturated)
    # Fewer LLM calls = better (scale: 1 call = 50, 5 calls = 30, 10+ = 10)
    avg_calls = total_calls / len(dataset_examples) if dataset_examples else 0
    call_efficiency = max(0, 50 - avg_calls * 4) if avg_calls > 0 else 50

    # Speed bonus: faster execution = better (scale: 1s = 30, 10s = 15, 60s = 0)
    avg_duration_s = (total_duration_ms / len(dataset_examples) / 1000) if dataset_examples else 0
    speed_bonus = max(0, 30 - avg_duration_s * 0.5)

    # Topology simplicity: fewer processes/edges = more elegant solution
    # This is a small tiebreaker (0-20) that rewards parsimony
    # Passed in via module introspection if available
    simplicity_bonus = 0
    try:
        if hasattr(mod, 'PROCESSES') and hasattr(mod, 'TRANSITIONS'):
            n_procs = len(mod.PROCESSES)
            n_trans = sum(len(v) if isinstance(v, list) else 1
                         for v in mod.TRANSITIONS.values() if v)
            # Fewer = better. Baseline: 8 procs + 8 transitions = 0 bonus
            simplicity_bonus = max(0, 20 - (n_procs + n_trans - 8) * 1.5)
    except Exception:
        pass

    # Total fitness: accuracy dominates, but efficiency differentiates ties
    score = accuracy_score + call_efficiency + speed_bonus + simplicity_bonus

    test_result = {
        "status": "PASS" if score > 0 else "BENCH_FAIL",
        "llm_calls": total_calls,
        "duration_ms": total_duration_ms,
        "score_em": round(em_mean, 4),
        "score_f1": round(f1_mean, 4),
        "accuracy_component": round(accuracy_score, 1),
        "efficiency_component": round(call_efficiency, 1),
        "speed_component": round(speed_bonus, 1),
        "simplicity_component": round(simplicity_bonus, 1),
    }

    return round(score, 1), test_result


def benchmark_candidate(spec, benchmark_suite="gsm8k", benchmark_examples=5,
                        timeout_sec=120, base_agent_type=None, verbose=False):
    """End-to-end benchmark evaluation of a spec dict.

    Takes a spec dict → validates → instantiates → benchmarks → returns results.
    Designed to be called from both evolve() and self_improver/design pipelines.

    Returns dict with:
        ok: bool, fitness: float, score_em: float, score_f1: float,
        llm_calls: int, duration_ms: int, status: str, error: str|None
    """
    import yaml

    result = {
        "ok": False, "fitness": 0.0, "score_em": 0.0, "score_f1": 0.0,
        "llm_calls": 0, "duration_ms": 0, "status": "unknown", "error": None,
    }

    work_dir = tempfile.mkdtemp(prefix="bench_candidate_")
    try:
        # Write spec to temp file
        spec_name = spec.get("name", "candidate").lower().replace(" ", "_")
        spec_path = os.path.join(work_dir, f"{spec_name}.yaml")
        with open(spec_path, "w") as f:
            yaml.dump(spec, f, default_flow_style=False)

        # Validate
        ok, output = validate_spec_file(spec_path)
        if not ok:
            result["status"] = "INVALID"
            result["error"] = output[:500]
            return result

        # Instantiate
        agent_path = os.path.join(work_dir, f"{spec_name}_agent.py")
        ok = instantiate_spec(spec_path, agent_path)
        if not ok:
            result["status"] = "GEN_FAIL"
            result["error"] = "Code generation failed"
            return result

        # Add to import path
        if work_dir not in sys.path:
            sys.path.insert(0, work_dir)

        # Run benchmark
        module_name = f"{spec_name}_agent"
        agent_type = base_agent_type or spec_name
        fitness, test_result = compute_fitness_benchmark(
            module_name, benchmark_suite, benchmark_examples,
            timeout_sec=timeout_sec, base_agent_type=agent_type,
        )

        result["ok"] = fitness > 0
        result["fitness"] = fitness
        result["score_em"] = test_result.get("score_em", 0.0)
        result["score_f1"] = test_result.get("score_f1", 0.0)
        result["llm_calls"] = test_result.get("llm_calls", 0)
        result["duration_ms"] = test_result.get("duration_ms", 0)
        result["status"] = test_result.get("status", "unknown")

        if verbose:
            print(f"  Benchmark result: fitness={fitness:.1f}, "
                  f"EM={result['score_em']:.3f}, F1={result['score_f1']:.3f}, "
                  f"calls={result['llm_calls']}, status={result['status']}")

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = f"{type(e).__name__}: {e}"
    finally:
        # Cleanup
        if work_dir in sys.path:
            sys.path.remove(work_dir)
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)

    return result


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
You are an agent architecture mutation engine. You receive an agent spec and context
about its performance, then propose a SINGLE structural mutation as a JSON instruction.

Your output must be valid JSON with exactly ONE of these mutation types:

1. Add a process:
{"action": "add_process", "process": {"id": "verify_answer", "type": "step", "label": "Verify Answer", "description": "Check the answer against known constraints"}, "add_edges": [{"type": "flow", "from": "generate", "to": "verify_answer"}, {"type": "flow", "from": "verify_answer", "to": "emit_answer"}], "remove_edges": [{"type": "flow", "from": "generate", "to": "emit_answer"}], "description": "Add verification step between generate and emit"}

2. Remove a process:
{"action": "remove_process", "process_id": "unnecessary_step", "rewire": {"from": "step_before", "to": "step_after"}, "description": "Remove redundant step, connect its neighbors"}

3. Modify a gate condition:
{"action": "modify_gate", "process_id": "quality_gate", "new_condition": "score >= 8", "description": "Raise quality threshold from 7 to 8"}

4. Add an entity (agent, tool, store):
{"action": "add_entity", "entity": {"id": "verifier", "type": "agent", "label": "Verifier", "model": "gemini-3-flash-preview", "system_prompt": "You verify answers for correctness.", "input_schema": "VerifyInput", "output_schema": "VerifyOutput"}, "add_schemas": [{"name": "VerifyInput", "fields": [{"name": "answer", "type": "string"}, {"name": "question", "type": "string"}]}, {"name": "VerifyOutput", "fields": [{"name": "is_correct", "type": "boolean"}, {"name": "feedback", "type": "string"}]}], "description": "Add verifier agent"}

5. Add an edge:
{"action": "add_edge", "edge": {"type": "invoke", "from": "verify_step", "to": "verifier", "input": "VerifyInput", "output": "VerifyOutput"}, "description": "Wire verify step to verifier agent"}

6. Modify process logic:
{"action": "modify_logic", "process_id": "receive_task", "append_logic": "state.data['max_rounds'] = 5", "description": "Increase max refinement rounds"}

Rules:
- Output ONLY valid JSON, no explanation or markdown fences
- Make exactly ONE mutation (you can add multiple edges to support it)
- Ensure all process IDs referenced in edges exist in the spec
- Do NOT change model fields
- Focus on changes that could improve benchmark performance
- Keep all string values on a single line (no literal newlines inside strings)
- Your entire response must be a single JSON object starting with { and ending with }
"""

MUTATION_USER_TEMPLATE = """\
## Spec Summary
Name: {spec_name}
Entry point: {entry_point}
Entities ({entity_count}): {entity_ids}
Processes ({process_count}): {process_ids}
Edges ({edge_count}): {edge_summary}
Detected patterns: {patterns}

## Process Details
{process_details}

## Context
- Lint warnings: {lint_warnings}
- Verify issues: {verify_issues}
{benchmark_context}
{previous_analysis}
{knowledge_context}

## Task
Propose ONE structural mutation as JSON that could improve this agent's benchmark performance.
Output ONLY the JSON instruction.
"""

# ════════════════════════════════════════════════════════════════════
# Structured LLM mutation selection (menu-based)
# ════════════════════════════════════════════════════════════════════

STRUCTURED_MUTATION_SYSTEM = """\
You are an agent architecture optimizer. You receive a menu of available mutations \
for an agent spec, plus context about performance. Select 1-3 mutations to apply \
in sequence to improve the agent.

Rules:
- Output ONLY a JSON array of selected mutations
- Each element must have "operator" plus the exact parameter fields shown in the menu
- Order matters: mutations are applied sequentially, each building on the previous result
- 1 mutation for minor tweaks, 2-3 for larger architectural changes
- Prefer combinations that work together (e.g., insert a pattern + modify its agent's prompt)
- Do NOT select mutations that would conflict (e.g., removing a process then modifying it)
- Do NOT select change_model mutations — model is controlled externally
- Your entire response must be a JSON array starting with [ and ending with ]
"""

STRUCTURED_MUTATION_USER = """\
## Agent: {spec_name}
Entry point: {entry_point}
Patterns detected: {patterns}

## Process Details
{process_details}

## Performance Context
{benchmark_context}
{previous_analysis}
{knowledge_context}

## Available Mutations ({option_count} options)
{mutation_menu}

## Task
Select 1-3 mutations from the menu above. Output a JSON array of selected mutations.
Each must have "operator" plus the parameter fields shown. Example:
[{{"operator": "modify_prompt", "agent": "solver", "transform": "add_chain_of_thought"}}]
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


def _summarize_spec(spec):
    """Create a concise summary of a spec for the LLM prompt."""
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])

    entity_ids = ", ".join(f"{e['id']}({e.get('type','')})" for e in entities)
    process_ids = ", ".join(f"{p['id']}({p.get('type','')})" for p in processes)

    # Summarize edges by type
    edge_counts = {}
    for e in edges:
        t = e.get("type", "?")
        edge_counts[t] = edge_counts.get(t, 0) + 1
    edge_summary = ", ".join(f"{c}x {t}" for t, c in sorted(edge_counts.items()))

    # Process details (id, type, label, key fields)
    proc_lines = []
    for p in processes:
        parts = [f"  - {p['id']} (type={p.get('type','step')})"]
        if p.get("label"):
            parts.append(f"label=\"{p['label']}\"")
        if p.get("condition"):
            parts.append(f"condition=\"{p['condition']}\"")
        if p.get("branches"):
            targets = [b.get("target", "?") for b in p["branches"]]
            parts.append(f"branches→{','.join(targets)}")
        if p.get("data_in"):
            parts.append(f"in={p['data_in']}")
        if p.get("data_out"):
            parts.append(f"out={p['data_out']}")
        proc_lines.append(" ".join(parts))

    return {
        "spec_name": spec.get("name", "?"),
        "entry_point": spec.get("entry_point", "?"),
        "entity_count": len(entities),
        "entity_ids": entity_ids,
        "process_count": len(processes),
        "process_ids": process_ids,
        "edge_count": len(edges),
        "edge_summary": edge_summary,
        "process_details": "\n".join(proc_lines),
    }


def _apply_mutation_instruction(spec, instruction):
    """Apply a structured mutation instruction (JSON dict) to a spec.

    Returns (mutated_spec, description).
    """
    result = copy.deepcopy(spec)
    action = instruction.get("action", "")
    desc = instruction.get("description", action)

    if action == "add_process":
        new_proc = instruction["process"]
        result.setdefault("processes", []).append(new_proc)
        for edge in instruction.get("add_edges", []):
            result.setdefault("edges", []).append(edge)
        for edge_spec in instruction.get("remove_edges", []):
            result["edges"] = [
                e for e in result.get("edges", [])
                if not (e.get("type") == edge_spec.get("type") and
                        e.get("from") == edge_spec.get("from") and
                        e.get("to") == edge_spec.get("to"))
            ]

    elif action == "remove_process":
        pid = instruction["process_id"]
        result["processes"] = [p for p in result.get("processes", []) if p["id"] != pid]
        # Remove edges referencing the removed process
        result["edges"] = [
            e for e in result.get("edges", [])
            if e.get("from") != pid and e.get("to") != pid
        ]
        # Add rewire edge if specified
        rewire = instruction.get("rewire")
        if rewire:
            result["edges"].append({
                "type": "flow",
                "from": rewire["from"],
                "to": rewire["to"],
                "label": f"rewired (was through {pid})",
            })

    elif action == "modify_gate":
        pid = instruction["process_id"]
        for proc in result.get("processes", []):
            if proc["id"] == pid:
                if "new_condition" in instruction:
                    proc["condition"] = instruction["new_condition"]
                if "new_branches" in instruction:
                    proc["branches"] = instruction["new_branches"]
                break

    elif action == "add_entity":
        new_entity = instruction["entity"]
        result.setdefault("entities", []).append(new_entity)
        for schema in instruction.get("add_schemas", []):
            result.setdefault("schemas", []).append(schema)

    elif action == "add_edge":
        result.setdefault("edges", []).append(instruction["edge"])

    elif action == "modify_logic":
        pid = instruction["process_id"]
        for proc in result.get("processes", []):
            if proc["id"] == pid:
                if "append_logic" in instruction:
                    existing = proc.get("logic", "")
                    proc["logic"] = existing + "\n" + instruction["append_logic"]
                if "replace_logic" in instruction:
                    proc["logic"] = instruction["replace_logic"]
                break

    else:
        raise ValueError(f"Unknown mutation action: {action}")

    # Update spec name
    result["name"] = result.get("name", "Agent") + " (mutated)"

    return result, f"llm: {desc}"


def _extract_json(text):
    """Robustly extract a JSON object from LLM output.

    Handles: markdown fences, leading/trailing text, literal newlines
    in string values, and truncated output.
    """
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse first (fast path)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost {...} using brace counting
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    in_string = False
    escape = False
    end = len(text)  # default: take everything from start
    matched = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                matched = True
                break

    candidate = text[start:end]

    # Try parsing the extracted object
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Replace literal newlines inside string values with \\n
    # Walk through char-by-char, tracking whether we're in a string
    sanitized = []
    in_str = False
    esc = False
    for c in candidate:
        if esc:
            sanitized.append(c)
            esc = False
            continue
        if c == "\\":
            sanitized.append(c)
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            sanitized.append(c)
            continue
        if in_str and c == "\n":
            sanitized.append("\\n")
            continue
        if in_str and c == "\t":
            sanitized.append("\\t")
            continue
        sanitized.append(c)
    sanitized_text = "".join(sanitized)

    try:
        return json.loads(sanitized_text)
    except json.JSONDecodeError:
        pass

    # Last resort: try to close truncated JSON
    if not matched and depth > 0:
        # Close any open string, then close braces
        suffix = ""
        if in_string:
            suffix += '"'
        # Close any open arrays we might be inside
        open_brackets = sanitized_text.count("[") - sanitized_text.count("]")
        if open_brackets > 0:
            suffix += "]" * open_brackets
        suffix += "}" * depth
        try:
            return json.loads(sanitized_text + suffix)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(
        f"Could not extract valid JSON (depth={depth})", candidate[:200], 0
    )


def _format_mutation_menu(options, max_per_operator=5):
    """Format enumerated mutations as a compact menu for the LLM prompt.

    Groups by operator, shows up to max_per_operator examples per type,
    then indicates how many more are available.
    """
    from collections import defaultdict
    by_op = defaultdict(list)
    for opt in options:
        # Skip change_model — model is controlled externally
        if opt["operator"] == "change_model":
            continue
        by_op[opt["operator"]].append(opt)

    lines = []
    for op_name, op_options in sorted(by_op.items()):
        lines.append(f"\n### {op_name} ({len(op_options)} options)")
        shown = op_options[:max_per_operator]
        for opt in shown:
            # Show the selection fields (everything except 'description')
            params = {k: v for k, v in opt.items() if k != "description"}
            lines.append(f"  {json.dumps(params)}")
            lines.append(f"    → {opt['description']}")
        if len(op_options) > max_per_operator:
            remaining = len(op_options) - max_per_operator
            # Show the parameter space for remaining options
            example = op_options[max_per_operator]
            param_keys = [k for k in example if k not in ("operator", "description")]
            lines.append(f"  ... and {remaining} more (vary: {', '.join(param_keys)})")
    return "\n".join(lines)


def llm_select_mutations(parent_spec, parent_yaml, benchmark_suite=None,
                         benchmark_results=None, previous_analysis=None,
                         knowledge_store=None, model=None, verbose=False):
    """LLM selects 1-3 mutations from the enumerated menu of valid options.

    Unlike llm_mutate (freeform), this guarantees structural validity because
    every selected mutation goes through battle-tested mutate.py operators.

    Returns (mutated_spec_dict, mutation_description, yaml_text).
    """
    import yaml
    from .specgen import call_llm

    model = model or FLASH_MODEL
    patterns = _detect_pattern_names(parent_spec)
    summary = _summarize_spec(parent_spec)

    # Enumerate all valid mutations for this spec
    options = mutate.enumerate_mutations(parent_spec)
    if not options:
        raise ValueError("No mutations available for this spec")

    menu_text = _format_mutation_menu(options)

    # Build context
    benchmark_ctx = ""
    if benchmark_results:
        benchmark_ctx = f"Benchmark ({benchmark_suite}): fitness={benchmark_results.get('fitness', 0)}, " \
                        f"status={benchmark_results.get('status', '?')}, " \
                        f"llm_calls={benchmark_results.get('llm_calls', 0)}"
        if benchmark_results.get('error'):
            benchmark_ctx += f"\nError: {benchmark_results['error'][:200]}"

    analysis_ctx = ""
    if previous_analysis:
        analysis_ctx = f"Previous analysis: {previous_analysis.get('diagnosis', '')}\n" \
                       f"Suggested: {previous_analysis.get('suggested_mutations', [])}"

    knowledge_ctx = _get_knowledge_context(knowledge_store, benchmark_suite)

    user_prompt = STRUCTURED_MUTATION_USER.format(
        spec_name=summary["spec_name"],
        entry_point=summary["entry_point"],
        patterns=", ".join(patterns) if patterns else "none detected",
        process_details=summary["process_details"],
        benchmark_context=benchmark_ctx or "No benchmark data yet.",
        previous_analysis=analysis_ctx or "First generation.",
        knowledge_context=knowledge_ctx,
        option_count=len(options),
        mutation_menu=menu_text,
    )

    if verbose:
        print(f"    [Flash] Selecting from {len(options)} mutation options with {model}...")

    # Retry up to 3 times
    last_error = None
    for attempt in range(3):
        prompt = user_prompt
        if last_error and attempt > 0:
            prompt += f"\n\n## IMPORTANT: Previous attempt failed:\n{last_error}\n\nOutput ONLY a valid JSON array. No markdown, no explanation. Example:\n[{{\"operator\": \"modify_prompt\", \"agent\": \"solver\", \"transform\": \"add_chain_of_thought\"}}]"

        response = call_llm(model, STRUCTURED_MUTATION_SYSTEM, prompt,
                            temperature=0.7, max_tokens=8192)

        try:
            selections = _extract_json_array(response)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = f"JSON parse error: {e}. Output ONLY a JSON array like [{{'operator': ...}}]"
            continue

        if not isinstance(selections, list) or not selections:
            last_error = "Must output a non-empty JSON array of mutations."
            continue

        if len(selections) > 3:
            selections = selections[:3]  # Silently truncate

        # Validate each selection has required fields
        valid = True
        for i, sel in enumerate(selections):
            if not isinstance(sel, dict) or "operator" not in sel:
                last_error = f"Selection {i} missing 'operator' field."
                valid = False
                break
            if sel["operator"] not in mutate.MUTATIONS and sel["operator"] != "change_model":
                last_error = f"Unknown operator '{sel['operator']}'. Must be one of: {sorted(mutate.MUTATIONS.keys())}"
                valid = False
                break
        if not valid:
            continue

        # Apply mutations sequentially
        try:
            current_spec = copy.deepcopy(parent_spec)
            applied = []
            for sel in selections:
                current_spec = mutate.apply_selected_mutation(current_spec, sel)
                applied.append(f"{sel['operator']}({', '.join(f'{k}={v}' for k, v in sel.items() if k != 'operator')})")

            mutation_desc = "structured: " + " → ".join(applied)
            yaml_text = yaml.dump(current_spec, default_flow_style=False)
            return current_spec, mutation_desc, yaml_text

        except (KeyError, ValueError, TypeError) as e:
            last_error = f"Failed to apply mutations: {e}. Try different selections."
            continue

    raise ValueError(last_error or "LLM failed to select valid mutations")


def _extract_json_array(text):
    """Extract a JSON array from LLM output. Handles markdown fences and surrounding text."""
    text = text.strip()

    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        # Find content between first ``` and last ```
        parts = text.split("```")
        # parts[0] is before first fence, parts[1] is inside, parts[2+] after
        if len(parts) >= 3:
            inner = parts[1]
            # Strip optional language tag (json, JSON, etc.)
            if inner.startswith(("json", "JSON")):
                inner = inner[4:]
            text = inner.strip()
        elif len(parts) == 2:
            inner = parts[1]
            if inner.startswith(("json", "JSON")):
                inner = inner[4:]
            text = inner.strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]  # Single mutation, wrap
    except json.JSONDecodeError:
        pass

    # Find outermost [...]
    start = text.find("[")
    if start == -1:
        # Maybe they returned a single object — wrap it
        try:
            obj = _extract_json(text)
            return [obj]
        except json.JSONDecodeError:
            raise ValueError("No JSON array found in response")

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    raise ValueError(f"Found array brackets but invalid JSON")
                break

    raise ValueError("Unclosed JSON array in response")


def llm_mutate(parent_spec, parent_yaml, benchmark_suite=None,
               benchmark_results=None, previous_analysis=None,
               knowledge_store=None, model=None, verbose=False):
    """Generate a mutation using an LLM (Flash model).

    Flash outputs a JSON mutation instruction, which is applied programmatically
    to the parent spec. This avoids the YAML generation reliability problem.

    Returns (mutated_spec_dict, mutation_description, yaml_text) or raises on failure.
    """
    import yaml
    from .specgen import call_llm

    model = model or FLASH_MODEL
    patterns = _detect_pattern_names(parent_spec)
    summary = _summarize_spec(parent_spec)

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
        spec_name=summary["spec_name"],
        entry_point=summary["entry_point"],
        entity_count=summary["entity_count"],
        entity_ids=summary["entity_ids"],
        process_count=summary["process_count"],
        process_ids=summary["process_ids"],
        edge_count=summary["edge_count"],
        edge_summary=summary["edge_summary"],
        patterns=", ".join(patterns) if patterns else "none detected",
        process_details=summary["process_details"],
        lint_warnings=_get_lint_warnings(parent_spec),
        verify_issues=_get_verify_issues(parent_spec),
        benchmark_context=benchmark_ctx,
        previous_analysis=analysis_ctx,
        knowledge_context=knowledge_ctx,
    )

    if verbose:
        print(f"    [Flash] Generating mutation with {model}...")

    # Retry up to 3 times with error feedback
    last_error = None
    for attempt in range(3):
        prompt = user_prompt
        if last_error and attempt > 0:
            prompt += f"\n\n## IMPORTANT: Your previous attempt failed:\n{last_error}\nPlease output ONLY valid JSON."

        response = call_llm(model, MUTATION_SYSTEM_PROMPT, prompt,
                             temperature=0.7, max_tokens=8192)

        try:
            instruction = _extract_json(response)
        except json.JSONDecodeError as e:
            last_error = f"JSON parse error: {e}. Output ONLY valid JSON, no explanations."
            continue

        if not isinstance(instruction, dict) or "action" not in instruction:
            last_error = "Missing 'action' field. Must be one of: add_process, remove_process, modify_gate, add_entity, add_edge, modify_logic"
            continue

        try:
            mutated_spec, mutation_desc = _apply_mutation_instruction(parent_spec, instruction)
            yaml_text = yaml.dump(mutated_spec, default_flow_style=False)
            return mutated_spec, mutation_desc, yaml_text
        except (KeyError, ValueError, TypeError) as e:
            last_error = f"Failed to apply mutation: {e}"
            continue

    raise ValueError(last_error or "LLM failed to produce a valid mutation instruction")


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
                         temperature=0.3, max_tokens=8192)

    # Parse JSON response
    try:
        analysis = _extract_json(response)
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
                        # Structured LLM mutation selection (menu-based)
                        result, mut_desc, _ = llm_select_mutations(
                            parent_spec, parent_yaml,
                            benchmark_suite=benchmark_suite,
                            benchmark_results=None,
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
