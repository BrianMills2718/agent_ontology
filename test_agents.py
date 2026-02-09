#!/usr/bin/env python3
"""
Agent Test Harness — Non-interactive test runner for all generated agents.

Runs each agent with canned inputs, captures traces, and reports pass/fail.
Patches input() to avoid blocking on checkpoint processes.

Usage:
    python3 test_agents.py                    # Run all agents (requires API keys)
    python3 test_agents.py --agent react      # Run specific agent
    python3 test_agents.py --dry-run          # Check imports only, no API calls
    python3 test_agents.py --timeout 60       # Set per-agent timeout (seconds)
    python3 test_agents.py --json             # Output results as JSON
"""

import argparse
import importlib
import json
import os
import sys
import time
import signal
import traceback
from contextlib import contextmanager
from io import StringIO

# ── Test fixtures: canned inputs for each agent ──

TEST_INPUTS = {
    "react": {
        "query": "What year was the Eiffel Tower built?",
    },
    "debate": {
        "topic": "Should remote work be mandatory?",
    },
    "babyagi": {
        "objective": "Plan a birthday party",
        "max_tasks": 2,
        "tasks": [
            {"id": 1, "description": "Choose a venue", "status": "pending", "priority": 1}
        ],
    },
    "babyagi_autogen": {
        "objective": "Plan a birthday party",
        "max_tasks": 2,
        "tasks": [
            {"id": 1, "description": "Choose a venue", "status": "pending", "priority": 1}
        ],
    },
    "autogpt": {
        "goal": "Write a haiku about the moon",
    },
    "rag": {
        "query": "What is the tallest building in the world?",
    },
    "code_reviewer": {
        "diff": '--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,5 @@\n+import os\n+PASSWORD = "secret123"\n def hello():\n     return "world"',
        "files": ["app.py"],
        "description": "Add password constant",
    },
    "crew": {
        "objective": "Write a short blog post about AI safety",
    },
}

# ── Validation criteria per agent ──

def validate_react(state):
    """ReAct should produce a final answer."""
    issues = []
    if not state.data.get("answer") and not state.data.get("final_answer"):
        issues.append("No answer produced")
    trajectory = state.data.get("trajectory", [])
    if len(trajectory) < 1:
        issues.append("No reasoning steps recorded")
    return issues

def validate_debate(state):
    """Debate should complete 3 rounds with a winner."""
    issues = []
    if not state.data.get("winner"):
        issues.append("No winner determined")
    history = state.data.get("debate_history", [])
    if len(history) < 4:  # at least 2 pro + 2 con
        issues.append(f"Only {len(history)} debate turns (expected >= 4)")
    if state.data.get("pro_score") is None:
        issues.append("No pro_score from judge")
    return issues

def validate_babyagi(state):
    """BabyAGI should complete at least 1 task loop."""
    issues = []
    if state.data.get("completed_count", 0) < 1:
        issues.append("No tasks completed")
    return issues

def validate_babyagi_autogen(state):
    """BabyAGI Autogen should complete at least 1 full loop."""
    issues = []
    if not state.data.get("result"):
        issues.append("No execution result produced")
    return issues

def validate_autogpt(state):
    """AutoGPT should produce output through its think/execute loop."""
    issues = []
    results = state.data.get("results", [])
    if not results and not state.data.get("result") and not state.data.get("output"):
        issues.append("No result produced")
    return issues

def validate_rag(state):
    """RAG should produce an answer (or explicit no-answer)."""
    issues = []
    if not state.data.get("answer"):
        issues.append("No answer produced")
    return issues

def validate_code_reviewer(state):
    """Code Reviewer should produce a synthesized review with quality score."""
    issues = []
    if state.data.get("quality_score") is None:
        issues.append("No quality_score produced")
    if not state.data.get("security_review"):
        issues.append("No security_review (namespacing may be broken)")
    if not state.data.get("recommendation"):
        issues.append("No recommendation")
    return issues

def validate_crew(state):
    """Crew should produce assignments and execute them."""
    issues = []
    if not state.data.get("assignments"):
        issues.append("No assignments created")
    return issues

VALIDATORS = {
    "react": validate_react,
    "debate": validate_debate,
    "babyagi": validate_babyagi,
    "babyagi_autogen": validate_babyagi_autogen,
    "autogpt": validate_autogpt,
    "rag": validate_rag,
    "code_reviewer": validate_code_reviewer,
    "crew": validate_crew,
}

# ── Timeout context manager ──

class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

# ── Monkey-patch input() to avoid blocking ──

def fake_input(prompt=""):
    """Auto-respond to checkpoint prompts."""
    print(f"  [AUTO-INPUT] {prompt} → 'skip'")
    return "skip"

# ── Run a single agent test ──

def run_agent_test(agent_name, timeout_sec=120, dry_run=False):
    """Run a single agent test. Returns a result dict."""
    module_name = f"agents.{agent_name}_agent"
    result = {
        "agent": agent_name,
        "status": "unknown",
        "duration_ms": 0,
        "llm_calls": 0,
        "iterations": 0,
        "issues": [],
        "error": None,
    }

    # Import check
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        result["status"] = "IMPORT_ERROR"
        result["error"] = str(e)
        return result

    if dry_run:
        result["status"] = "DRY_RUN_OK"
        return result

    # Get test inputs
    inputs = TEST_INPUTS.get(agent_name, {})
    if not inputs:
        result["status"] = "NO_TEST_DATA"
        result["issues"].append("No test inputs defined")
        return result

    # Patch input() and capture output
    import builtins
    original_input = builtins.input
    builtins.input = fake_input

    # Reset the module's TRACE
    if hasattr(mod, 'TRACE'):
        mod.TRACE.clear()

    t0 = time.time()
    try:
        with timeout(timeout_sec):
            state = mod.run(inputs.copy())

        result["duration_ms"] = int((time.time() - t0) * 1000)
        result["iterations"] = state.iteration if hasattr(state, 'iteration') else 0
        result["llm_calls"] = len(mod.TRACE) if hasattr(mod, 'TRACE') else 0

        # Validate
        validator = VALIDATORS.get(agent_name)
        if validator:
            result["issues"] = validator(state)

        if result["issues"]:
            result["status"] = "ISSUES"
        else:
            result["status"] = "PASS"

        # Save trace
        trace_path = f"traces/{agent_name}_trace.json"
        os.makedirs("traces", exist_ok=True)
        if hasattr(mod, 'TRACE'):
            with open(trace_path, "w") as f:
                json.dump(mod.TRACE, f, indent=2)

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
        builtins.input = original_input

    return result


# ── Main ──

AGENT_NAMES = [
    "react", "debate", "babyagi", "babyagi_autogen",
    "autogpt", "rag", "code_reviewer", "crew",
]

def main():
    parser = argparse.ArgumentParser(
        description="Agent Test Harness — run agents with canned inputs and validate outputs"
    )
    parser.add_argument("--agent", "-a", choices=AGENT_NAMES, help="Run specific agent only")
    parser.add_argument("--dry-run", action="store_true", help="Check imports only, no API calls")
    parser.add_argument("--timeout", "-t", type=int, default=120, help="Per-agent timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    agents_to_test = [args.agent] if args.agent else AGENT_NAMES
    results = []

    if not args.json:
        print(f"\n{'='*60}")
        print(f"  Agent Test Harness")
        print(f"  Testing {len(agents_to_test)} agent(s), timeout={args.timeout}s")
        print(f"{'='*60}\n")

    for name in agents_to_test:
        if not args.json:
            print(f"[TEST] {name}...", end=" ", flush=True)

        result = run_agent_test(name, timeout_sec=args.timeout, dry_run=args.dry_run)
        results.append(result)

        if not args.json:
            status = result["status"]
            duration = result["duration_ms"]
            calls = result["llm_calls"]
            icon = {"PASS": "✓", "ISSUES": "⚠", "ERROR": "✗", "TIMEOUT": "⏱",
                    "DRY_RUN_OK": "○", "IMPORT_ERROR": "✗", "NO_TEST_DATA": "○"}.get(status, "?")
            print(f"{icon} {status} ({duration}ms, {calls} LLM calls)")
            if result["issues"]:
                for issue in result["issues"]:
                    print(f"    - {issue}")
            if result["error"]:
                print(f"    ERROR: {result['error']}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        # Summary
        passed = sum(1 for r in results if r["status"] == "PASS")
        total = len(results)
        total_calls = sum(r["llm_calls"] for r in results)
        total_ms = sum(r["duration_ms"] for r in results)
        print(f"\n{'='*60}")
        print(f"  Results: {passed}/{total} passed")
        print(f"  Total: {total_calls} LLM calls, {total_ms/1000:.1f}s")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
