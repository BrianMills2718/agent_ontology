#!/usr/bin/env python3
"""
Agent-dataset compatibility matrix and input formatters.

Defines which agents can run which benchmark suites, and how to format
dataset examples into the agent's expected input schema.
"""

# Agent -> list of compatible datasets
COMPATIBILITY = {
    "react":             ["hotpotqa", "gsm8k", "arc"],
    "rag":               ["hotpotqa"],
    "plan_and_solve":    ["gsm8k", "arc"],
    "self_refine":       ["gsm8k", "humaneval"],
    "tree_of_thought":   ["gsm8k", "arc"],
    "lats":              ["hotpotqa", "gsm8k"],
    "reflexion":         ["gsm8k"],
    "multi_agent_codegen": ["humaneval"],
}

# Dataset -> list of compatible agents (derived from COMPATIBILITY)
DATASET_AGENTS = {}
for agent, datasets in COMPATIBILITY.items():
    for ds in datasets:
        DATASET_AGENTS.setdefault(ds, []).append(agent)


def get_compatible_agents(dataset_name):
    """Return list of agent names compatible with the given dataset."""
    return DATASET_AGENTS.get(dataset_name, [])


def get_compatible_datasets(agent_name):
    """Return list of dataset names compatible with the given agent."""
    return COMPATIBILITY.get(agent_name, [])


def format_input(agent_name, example, dataset_name):
    """Format a benchmark example into the agent's expected input dict.

    Each agent has different input schema expectations. This function
    maps the generic example format to what each agent needs.
    """
    question = example["question"]

    if agent_name == "react":
        return {"query": question}

    elif agent_name == "rag":
        return {"query": question}

    elif agent_name == "plan_and_solve":
        return {"problem": question}

    elif agent_name == "self_refine":
        return {"task": f"Solve this math problem step by step and give a final numeric answer: {question}"}

    elif agent_name == "tree_of_thought":
        return {"problem": question}

    elif agent_name == "lats":
        if dataset_name == "hotpotqa":
            return {"query": question}
        else:
            return {"query": f"Solve: {question}"}

    elif agent_name == "reflexion":
        return {"task": question}

    elif agent_name == "multi_agent_codegen":
        return {
            "project_description": question,
            "language": "Python",
        }

    else:
        # Generic fallback
        return {"query": question}
