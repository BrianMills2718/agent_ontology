#!/usr/bin/env python3
"""
OpenClaw Spec Generator (docs → spec)
Takes a natural language description of an agent architecture and generates
a valid OpenClaw YAML spec.

Usage:
  python3 specgen.py description.md -o specs/my_agent.yaml
  echo "A simple chatbot that..." | python3 specgen.py - -o specs/chatbot.yaml
  python3 specgen.py description.md --validate --model gpt-4.1-mini
"""

import sys
import os
import json
import argparse
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_ontology():
    """Load the ONTOLOGY.yaml as text to include in the prompt."""
    path = os.path.join(SCRIPT_DIR, "ONTOLOGY.yaml")
    with open(path) as f:
        return f.read()


def load_examples():
    """Load 1-2 example specs to include as few-shot examples."""
    examples = []
    for name in ["babyagi.yaml", "react.yaml"]:
        path = os.path.join(SCRIPT_DIR, "specs", name)
        if os.path.exists(path):
            with open(path) as f:
                examples.append((name, f.read()))
    return examples


def build_prompt(description, ontology, examples):
    """Build the system and user prompts for spec generation."""

    example_text = ""
    for name, content in examples[:2]:
        example_text += f"\n--- Example: {name} ---\n{content}\n"

    system = f"""You are an expert agent architecture analyst. You convert natural language
descriptions of AI agent systems into formal OpenClaw YAML specifications.

An OpenClaw spec defines an agent architecture using:
- entities: agents (LLM-based), stores (persistence), tools (capabilities), humans, configs
- processes: steps (computation), gates (decisions), checkpoints (human approval), spawns (sub-agents)
- edges: flow (sequence), invoke (agent/tool call), loop (cycle), branch (conditional), read/write (store access)
- schemas: data shapes flowing between components

RULES:
1. Every entity needs: id, type, label. Agents also need: model, system_prompt.
2. Every process needs: id, type, label. Steps can have logic (Python code).
3. Gates need: condition (human-readable), branches (2+ with condition and target).
4. Edges connect processes to processes (flow/loop) or processes to entities (invoke/read/write).
5. Schemas define the data contracts between components.
6. The spec needs: name, version, description, entry_point.
7. Tools need: tool_type (api, function, shell, browser, mcp).
8. Stores need: store_type (vector, file, kv, queue, relational, blackboard).
9. Gate conditions should reference flat state.data fields, not nested paths.
10. Use gpt-4.1-mini as the default model unless the description specifies otherwise.
11. Add logic fields to processes that need data preparation before agent invocations.

Output ONLY valid YAML. No markdown code fences. No explanatory text before or after.

Here is the OpenClaw ontology (type system):

{ontology}

Here are example specs for reference:
{example_text}"""

    user = f"""Convert the following agent architecture description into an OpenClaw YAML spec:

---
{description}
---

Generate a complete, valid OpenClaw spec YAML. Include all entities, processes, edges, and schemas.
Make sure every entity referenced in an edge exists. Make sure the entry_point process exists.
Add logic fields to processes that need data preparation."""

    return system, user


def call_llm(model, system, user, temperature=0.3, max_tokens=8192):
    """Call the LLM to generate the spec."""
    if model.startswith("claude") or model.startswith("anthropic"):
        return _call_anthropic(model, system, user, temperature, max_tokens)
    else:
        return _call_openai(model, system, user, temperature, max_tokens)


def _call_openai(model, system, user, temperature, max_tokens):
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(model, system, user, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text


def extract_yaml(response):
    """Extract YAML from the LLM response, handling code fences."""
    text = response.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```yaml
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text


def validate_spec(yaml_text):
    """Run the validator on the generated spec."""
    import yaml
    import tempfile
    import subprocess

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, dir=SCRIPT_DIR) as f:
        f.write(yaml_text)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, os.path.join(SCRIPT_DIR, "validate.py"), tmp_path],
            capture_output=True, text=True, cwd=SCRIPT_DIR
        )
        return result.returncode == 0, result.stdout + result.stderr
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="OpenClaw Spec Generator (docs → spec)")
    parser.add_argument("input", help="Path to description file, or '-' for stdin")
    parser.add_argument("-o", "--output", help="Output YAML file path")
    parser.add_argument("--model", default="gpt-4.1-mini", help="LLM model to use")
    parser.add_argument("--validate", action="store_true", help="Validate generated spec")
    parser.add_argument("--fix", action="store_true", help="Auto-fix validation errors (re-prompt)")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt but don't call LLM")
    args = parser.parse_args()

    # Read input
    if args.input == "-":
        description = sys.stdin.read()
    else:
        with open(args.input) as f:
            description = f.read()

    if not description.strip():
        print("Error: empty description", file=sys.stderr)
        sys.exit(1)

    print(f"Generating spec from description ({len(description)} chars)...")
    print(f"Model: {args.model}")

    # Load ontology and examples
    ontology = load_ontology()
    examples = load_examples()

    # Build prompt
    system, user = build_prompt(description, ontology, examples)

    if args.dry_run:
        print("\n=== SYSTEM PROMPT ===")
        print(system[:2000] + "..." if len(system) > 2000 else system)
        print("\n=== USER PROMPT ===")
        print(user)
        return

    # Generate spec
    t0 = time.time()
    response = call_llm(args.model, system, user)
    duration = time.time() - t0
    print(f"Generated in {duration:.1f}s")

    yaml_text = extract_yaml(response)

    # Validate if requested
    if args.validate or args.fix:
        valid, output = validate_spec(yaml_text)
        print(f"\nValidation: {'PASS' if valid else 'FAIL'}")
        if not valid:
            print(output)

            if args.fix:
                # Re-prompt with validation errors
                print("\nAttempting auto-fix...")
                fix_user = f"""The spec you generated has validation errors:

{output}

Here is the spec you generated:

{yaml_text}

Fix ALL validation errors and return the corrected YAML. Output ONLY valid YAML, no markdown fences."""

                response2 = call_llm(args.model, system, fix_user)
                yaml_text = extract_yaml(response2)

                valid2, output2 = validate_spec(yaml_text)
                print(f"After fix: {'PASS' if valid2 else 'FAIL'}")
                if not valid2:
                    print(output2)

    # Output
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(yaml_text)
        print(f"\nWritten to: {args.output}")
    else:
        print("\n" + yaml_text)


if __name__ == "__main__":
    main()
