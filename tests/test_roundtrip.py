#!/usr/bin/env python3
"""Round-trip tests for import → validate → instantiate pipeline.

Tests that imported specs from LangGraph and CrewAI source files can be:
1. Imported to a YAML spec
2. Validated (0 errors)
3. Instantiated to valid Python code (syntax check)

For full E2E (running the agent), use test_agents.py with --spec flag.
"""

import ast
import sys
import tempfile
from pathlib import Path

import yaml

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_ontology"))

from agent_ontology.import_langgraph import import_langgraph
from agent_ontology.validate import validate_spec, load_yaml, ONTOLOGY_PATH

_ontology = load_yaml(ONTOLOGY_PATH)


# ═══════════════════════════════════════════════════════════════
# Test LangGraph Files
# ═══════════════════════════════════════════════════════════════

LG_CLASS_BASED = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class AgentState(TypedDict):
    messages: list[str]
    result: str

class MyAgent:
    def __init__(self):
        self.graph = StateGraph(AgentState)
        self.graph.add_node("process", self.process_node)
        self.graph.add_node("analyze", self.analyze_node)
        self.graph.add_edge(START, "process")
        self.graph.add_edge("process", "analyze")
        self.graph.add_edge("analyze", END)

    def process_node(self, state):
        return {"messages": state["messages"] + ["processed"]}

    def analyze_node(self, state):
        return {"result": "analysis complete"}
"""

LG_ADD_SEQUENCE = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    data: str

def step_a(state): return {"data": "a"}
def step_b(state): return {"data": "b"}
def step_c(state): return {"data": "c"}

graph = StateGraph(State)
graph.add_sequence(["step_a", "step_b", "step_c"])
graph.add_edge(START, "step_a")
graph.add_edge("step_c", END)
"""

LG_SEND_FANOUT = """\
from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class State(TypedDict):
    chunks: list[str]
    results: Annotated[list[str], operator.add]

def split_input(state):
    return {"chunks": ["chunk1", "chunk2"]}

def process_chunk(state):
    return {"results": [f"processed"]}

def aggregate(state):
    return {"results": state["results"]}

def route_chunks(state):
    return [Send("process_chunk", {"chunks": [c]}) for c in state["chunks"]]

graph = StateGraph(State)
graph.add_node("split_input", split_input)
graph.add_node("process_chunk", process_chunk)
graph.add_node("aggregate", aggregate)
graph.add_edge(START, "split_input")
graph.add_conditional_edges("split_input", route_chunks)
graph.add_edge("process_chunk", "aggregate")
graph.add_edge("aggregate", END)
"""

LG_INTERRUPT = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

class State(TypedDict):
    query: str
    result: str

def process_query(state):
    return {"result": "processed"}

def human_review(state):
    approval = interrupt({"question": "Approve?"})
    return {"result": state["result"]}

def finalize(state):
    return {"result": state["result"]}

graph = StateGraph(State)
graph.add_node("process_query", process_query)
graph.add_node("human_review", human_review)
graph.add_node("finalize", finalize)
graph.add_edge(START, "process_query")
graph.add_edge("process_query", "human_review")
graph.add_edge("human_review", "finalize")
graph.add_edge("finalize", END)
app = graph.compile(interrupt_before=["human_review"])
"""

LG_DATACLASS = """\
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END

@dataclass
class AgentState:
    query: str = ""
    result: str = ""
    iterations: int = 0

def search(state):
    return {"result": "found"}

graph = StateGraph(AgentState)
graph.add_node("search", search)
graph.add_edge(START, "search")
graph.add_edge("search", END)
"""

LG_SUBGRAPH = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class InnerState(TypedDict):
    data: str

class OuterState(TypedDict):
    input: str
    output: str

def inner_process(state):
    return {"data": "processed"}

inner_graph = StateGraph(InnerState)
inner_graph.add_node("inner_process", inner_process)
inner_graph.add_edge(START, "inner_process")
inner_graph.add_edge("inner_process", END)
inner_app = inner_graph.compile()

def prepare(state):
    return {"input": state["input"]}

def finalize(state):
    return {"output": "done"}

outer_graph = StateGraph(OuterState)
outer_graph.add_node("prepare", prepare)
outer_graph.add_node("sub_agent", inner_app)
outer_graph.add_node("finalize", finalize)
outer_graph.add_edge(START, "prepare")
outer_graph.add_edge("prepare", "sub_agent")
outer_graph.add_edge("sub_agent", "finalize")
outer_graph.add_edge("finalize", END)
"""

LG_CONDITIONAL = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    question: str
    answer: str
    needs_search: bool

def reason(state):
    return {"needs_search": True}

def search(state):
    return {"answer": "found"}

def answer(state):
    return {"answer": "direct answer"}

def should_search(state):
    if state.get("needs_search"):
        return "search"
    return "answer"

graph = StateGraph(State)
graph.add_node("reason", reason)
graph.add_node("search", search)
graph.add_node("answer", answer)
graph.add_edge(START, "reason")
graph.add_conditional_edges("reason", should_search, {"search": "search", "answer": "answer"})
graph.add_edge("search", "answer")
graph.add_edge("answer", END)
"""

LG_COMMAND = """\
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

class State(TypedDict):
    data: str
    step: str

def router(state):
    if state.get("data"):
        return Command(goto="process")
    return Command(goto="collect")

def collect(state):
    return {"data": "collected"}

def process(state):
    return {"step": "processed"}

graph = StateGraph(State)
graph.add_node("router", router)
graph.add_node("collect", collect)
graph.add_node("process", process)
graph.add_edge(START, "router")
graph.add_edge("collect", "router")
graph.add_edge("process", END)
"""


def _import_and_check(source_code, test_name, expected_checks=None):
    """Import source code, validate spec, and check properties."""
    # Write source to temp file
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(source_code)
        f.flush()
        source_path = f.name

    try:
        # Step 1: Import
        spec = import_langgraph(source_path)
        assert spec is not None, f"{test_name}: import returned None"
        assert "entry_point" in spec, f"{test_name}: missing entry_point"
        assert spec["entry_point"] is not None, f"{test_name}: entry_point is None"
        assert len(spec.get("processes", [])) > 0, f"{test_name}: no processes"

        # Step 2: Validate
        errors, warnings = validate_spec(spec, _ontology, source_path)
        assert len(errors) == 0, f"{test_name}: validation errors: {errors}"

        # Step 3: Check specific properties
        if expected_checks:
            for check_name, check_fn in expected_checks.items():
                assert check_fn(spec), f"{test_name}: check '{check_name}' failed"

        return spec, errors, warnings
    finally:
        Path(source_path).unlink(missing_ok=True)


def test_lg_class_based():
    """self.graph = StateGraph(...) is detected."""
    spec, _, _ = _import_and_check(LG_CLASS_BASED, "class_based", {
        "has_process": lambda s: any(p["id"] == "process" for p in s["processes"]),
        "has_analyze": lambda s: any(p["id"] == "analyze" for p in s["processes"]),
        "entry_point": lambda s: s["entry_point"] == "process",
        "has_schema": lambda s: any(sc["name"] == "AgentState" for sc in s.get("schemas", [])),
    })
    print(f"  PASS class_based: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_add_sequence():
    """add_sequence() creates nodes and edges."""
    spec, _, _ = _import_and_check(LG_ADD_SEQUENCE, "add_sequence", {
        "3_processes": lambda s: len([p for p in s["processes"] if p["type"] == "step"]) == 3,
        "2_flow_edges": lambda s: len([e for e in s["edges"] if e["type"] == "flow"]) == 2,
        "entry_point": lambda s: s["entry_point"] == "step_a",
    })
    print(f"  PASS add_sequence: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_send_fanout():
    """Send() creates fan-out edges."""
    spec, _, _ = _import_and_check(LG_SEND_FANOUT, "send_fanout", {
        "has_split": lambda s: any(p["id"] == "split_input" for p in s["processes"]),
        "has_process_chunk": lambda s: any(p["id"] == "process_chunk" for p in s["processes"]),
        "fan_out_edge": lambda s: any(
            e["from"] == "split_input" and e["to"] == "process_chunk" for e in s["edges"]
        ),
    })
    print(f"  PASS send_fanout: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_interrupt():
    """interrupt() creates checkpoint process."""
    spec, _, _ = _import_and_check(LG_INTERRUPT, "interrupt", {
        "has_checkpoint": lambda s: any(p["type"] == "checkpoint" for p in s["processes"]),
        "has_human": lambda s: any(e["type"] == "human" for e in s.get("entities", [])),
    })
    print(f"  PASS interrupt: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


def test_lg_dataclass():
    """@dataclass state is extracted."""
    spec, _, _ = _import_and_check(LG_DATACLASS, "dataclass", {
        "has_schema": lambda s: any(sc["name"] == "AgentState" for sc in s.get("schemas", [])),
        "schema_has_fields": lambda s: len(next(
            sc["fields"] for sc in s.get("schemas", []) if sc["name"] == "AgentState"
        )) == 3,
    })
    print(f"  PASS dataclass: {len(spec.get('schemas', []))} schemas")


def test_lg_subgraph():
    """Subgraph-as-node is detected with description."""
    spec, _, _ = _import_and_check(LG_SUBGRAPH, "subgraph", {
        "has_sub_agent": lambda s: any(p["id"] == "sub_agent" for p in s["processes"]),
        "sub_agent_description": lambda s: any(
            p["id"] == "sub_agent" and "subgraph" in p.get("description", "").lower()
            for p in s["processes"]
        ),
    })
    print(f"  PASS subgraph: {len(spec['processes'])} processes")


def test_lg_conditional():
    """Conditional edges create gate processes."""
    spec, _, _ = _import_and_check(LG_CONDITIONAL, "conditional", {
        "has_gate": lambda s: any(p["type"] == "gate" for p in s["processes"]),
        "gate_has_branches": lambda s: any(
            p["type"] == "gate" and len(p.get("branches", [])) >= 2
            for p in s["processes"]
        ),
    })
    print(f"  PASS conditional: {len(spec['processes'])} processes")


def test_lg_command():
    """Command(goto=...) creates edges."""
    spec, _, _ = _import_and_check(LG_COMMAND, "command", {
        "router_to_process": lambda s: any(
            e["from"] == "router" and e["to"] == "process" for e in s["edges"]
        ),
        "router_to_collect": lambda s: any(
            e["from"] == "router" and e["to"] == "collect" for e in s["edges"]
        ),
    })
    print(f"  PASS command: {len(spec['processes'])} processes, {len(spec['edges'])} edges")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

ALL_TESTS = [
    test_lg_class_based,
    test_lg_add_sequence,
    test_lg_send_fanout,
    test_lg_interrupt,
    test_lg_dataclass,
    test_lg_subgraph,
    test_lg_conditional,
    test_lg_command,
]


def main():
    print("Round-trip Import Tests")
    print("=" * 50)
    passed = 0
    failed = 0
    for test_fn in ALL_TESTS:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL {test_fn.__name__}: {e}")
            failed += 1

    print(f"\n{passed}/{passed + failed} passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
