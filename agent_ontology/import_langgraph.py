#!/usr/bin/env python3
"""
LangGraph Importer — Parse LangGraph StateGraph Python files into Agent Ontology YAML specs.

Uses Python AST to extract:
  - TypedDict state class → schemas
  - add_node() calls → processes (step)
  - add_edge() calls → edges (flow)
  - add_conditional_edges() calls → processes (gate) + edges (branch)
  - set_entry_point() → entry_point
  - LLM calls in node functions → entities (agent) + edges (invoke)

Usage:
    python3 import_langgraph.py agent_lg.py
    python3 import_langgraph.py agent_lg.py -o specs/imported.yaml
    python3 import_langgraph.py agent_lg.py --validate
"""

import argparse
import ast
import re
import sys
import textwrap
from pathlib import Path

import yaml


# ═══════════════════════════════════════════════════════════════
# AST Extraction
# ═══════════════════════════════════════════════════════════════

def _get_string(node):
    """Extract string value from AST Constant or JoinedStr."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _get_dict_literal(node):
    """Extract a dict literal from AST Dict node. Handles END variable as '__END__'."""
    if not isinstance(node, ast.Dict):
        return None
    result = {}
    for k, v in zip(node.keys, node.values):
        key = _get_string(k)
        val = _get_string(v)
        # Handle END variable reference
        if val is None and isinstance(v, ast.Name) and v.id == "END":
            val = "__END__"
        if key is not None and val is not None:
            result[key] = val
    return result


def _find_typed_dicts(tree):
    """Find TypedDict class definitions -> list of (name, fields)."""
    results = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Check if base class is TypedDict
        is_typed_dict = False
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "TypedDict":
                is_typed_dict = True
            elif isinstance(base, ast.Attribute) and base.attr == "TypedDict":
                is_typed_dict = True
        for kw in node.keywords:
            pass  # total=False etc
        if not is_typed_dict:
            continue

        fields = []
        for item in node.body:
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                fname = item.target.id
                ftype = _annotation_to_type(item.annotation)
                fields.append({"name": fname, "type": ftype})
        results.append((node.name, fields))
    return results


def _annotation_to_type(ann):
    """Convert a Python type annotation AST to a spec type string."""
    if isinstance(ann, ast.Name):
        mapping = {
            "str": "string", "int": "integer", "float": "float",
            "bool": "boolean", "dict": "object", "list": "list",
        }
        return mapping.get(ann.id, ann.id)
    if isinstance(ann, ast.Constant):
        return str(ann.value)
    if isinstance(ann, ast.Subscript):
        base = _annotation_to_type(ann.value)
        if base == "list":
            inner = _annotation_to_type(ann.slice)
            return f"list<{inner}>"
        if base == "dict":
            return "object"
        if base == "Annotated":
            # Annotated[type, reducer] — extract base type
            if isinstance(ann.slice, ast.Tuple) and ann.slice.elts:
                return _annotation_to_type(ann.slice.elts[0])
        return base
    if isinstance(ann, ast.Attribute):
        return ann.attr
    if isinstance(ann, ast.BinOp) and isinstance(ann.op, ast.BitOr):
        # X | Y union type
        left = _annotation_to_type(ann.left)
        right = _annotation_to_type(ann.right)
        return f"{left} | {right}"
    return "object"


def _find_graph_calls(tree, source_lines):
    """Find StateGraph construction and all graph builder API calls.

    Returns dict with:
      state_class: str
      entry_point: str
      nodes: [(name, func_name)]
      edges: [(src, dst)]
      conditional_edges: [(src, router_func, mapping)]
    """
    result = {
        "state_class": None,
        "entry_point": None,
        "nodes": [],
        "edges": [],
        "conditional_edges": [],
        "end_edges": [],
    }

    # Track which variable names are the graph builder
    graph_vars = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) and not isinstance(node, ast.Expr):
            # Check for StateGraph(...) assignments
            if isinstance(node, ast.Assign):
                pass  # handled below
            continue

        # Handle: graph = StateGraph(AgentState)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            val = node.value
            if isinstance(val, ast.Call):
                func = val.func
                if isinstance(func, ast.Name) and func.id == "StateGraph":
                    if isinstance(target, ast.Name):
                        graph_vars.add(target.id)
                    if val.args:
                        arg = val.args[0]
                        if isinstance(arg, ast.Name):
                            result["state_class"] = arg.id

        # Handle method calls on graph builder
        call = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value

        if call is None:
            continue

        func = call.func
        if not isinstance(func, ast.Attribute):
            continue

        method = func.attr
        # Check if the object is a known graph variable
        obj = func.value
        obj_name = None
        if isinstance(obj, ast.Name):
            obj_name = obj.id

        # set_entry_point
        if method == "set_entry_point" and call.args:
            ep = _get_string(call.args[0])
            if ep:
                result["entry_point"] = ep

        # add_node
        elif method == "add_node" and len(call.args) >= 2:
            name = _get_string(call.args[0])
            func_ref = call.args[1]
            func_name = None
            if isinstance(func_ref, ast.Name):
                func_name = func_ref.id
            elif isinstance(func_ref, ast.Lambda):
                func_name = "__lambda__"
            if name:
                result["nodes"].append((name, func_name))

        # add_edge
        elif method == "add_edge" and len(call.args) >= 2:
            src = _get_string(call.args[0])
            dst = call.args[1]
            dst_str = _get_string(dst)
            # Check for END
            if dst_str is None and isinstance(dst, ast.Name) and dst.id == "END":
                dst_str = "__END__"
            if src and dst_str:
                if dst_str == "__END__":
                    result["end_edges"].append(src)
                else:
                    result["edges"].append((src, dst_str))

        # add_conditional_edges
        elif method == "add_conditional_edges" and len(call.args) >= 3:
            src = _get_string(call.args[0])
            router = call.args[1]
            router_name = None
            if isinstance(router, ast.Name):
                router_name = router.id
            mapping = _get_dict_literal(call.args[2])
            if src and mapping:
                result["conditional_edges"].append((src, router_name, mapping))

    return result


def _find_node_functions(tree, source_lines):
    """Extract node function bodies for logic/LLM analysis.

    Returns dict: func_name -> {body_source, calls_llm, llm_model, system_prompt_var}
    """
    functions = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        fn_name = node.name
        # Get source of function body
        if node.body:
            start = node.body[0].lineno - 1
            end = node.end_lineno
            body_lines = source_lines[start:end]
            body_source = "\n".join(body_lines)
        else:
            body_source = ""

        info = {
            "body_source": body_source,
            "calls_llm": False,
            "model": None,
            "system_prompt": None,
            "invokes_func": None,
        }

        # Walk function body for LLM call patterns
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                cfunc = child.func
                cname = None
                if isinstance(cfunc, ast.Name):
                    cname = cfunc.id
                elif isinstance(cfunc, ast.Attribute):
                    cname = cfunc.attr

                if cname and ("call_llm" in cname or "invoke" in cname.lower()):
                    info["calls_llm"] = True
                    # Try to extract function name being called
                    if cname.startswith("invoke_"):
                        info["invokes_func"] = cname

        functions[fn_name] = info
    return functions


def _find_invoke_functions(tree):
    """Find invoke_* wrapper functions that call call_llm with a specific agent.

    Returns dict: func_name -> {model, agent_entity_id, system_prompt_snippet}
    """
    results = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if not node.name.startswith("invoke_"):
            continue

        info = {"model": None, "system_prompt": None}

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                cfunc = child.func
                cname = None
                if isinstance(cfunc, ast.Name):
                    cname = cfunc.id
                if cname == "call_llm":
                    # call_llm(model, system_prompt, user_msg, ...)
                    if len(child.args) >= 2:
                        model_arg = child.args[0]
                        if isinstance(model_arg, ast.Constant):
                            info["model"] = model_arg.value
                        prompt_arg = child.args[1]
                        if isinstance(prompt_arg, ast.Constant):
                            info["system_prompt"] = prompt_arg.value
                        elif isinstance(prompt_arg, ast.Name):
                            info["system_prompt"] = f"<variable:{prompt_arg.id}>"

        results[node.name] = info
    return results


def _find_string_constants(tree):
    """Find module-level string constant assignments.

    Returns dict: var_name -> value
    """
    results = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str) and len(node.value.value) > 20:
                    results[target.id] = node.value.value
    return results


# ═══════════════════════════════════════════════════════════════
# Routing Function Analysis
# ═══════════════════════════════════════════════════════════════

def _analyze_router(tree, router_name, mapping):
    """Analyze a routing function to extract gate condition and branch conditions.

    Returns: (condition_str, branches)
    """
    func_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == router_name:
            func_node = node
            break

    if not func_node:
        branches = []
        has_end = False
        for k, v in mapping.items():
            if v == "__END__" or v == "END":
                has_end = True
                branches.append({"condition": k, "target": "__terminal__"})
            else:
                branches.append({"condition": k, "target": v})
        return "condition unknown", branches, has_end

    # Extract condition from docstring or first if-statement
    condition = ""
    docstring = ast.get_docstring(func_node)
    if docstring:
        # Often format: "Gate: Label — condition"
        if "—" in docstring:
            condition = docstring.split("—", 1)[1].strip()
        elif "-" in docstring:
            condition = docstring.split("-", 1)[1].strip()
        else:
            condition = docstring.strip()

    # Fallback: analyze if-statements for condition
    if not condition:
        for child in ast.walk(func_node):
            if isinstance(child, ast.If):
                condition = _if_condition_to_str(child.test)
                break

    branches = []
    has_end = False
    for route_key, target in mapping.items():
        if target == "__END__" or target == "END":
            has_end = True
            branches.append({
                "condition": route_key,
                "target": "__terminal__",
            })
        else:
            branches.append({
                "condition": route_key,
                "target": target,
            })

    return condition, branches, has_end


def _if_condition_to_str(test):
    """Convert an if-test AST node to a human-readable condition string."""
    if isinstance(test, ast.Compare):
        left = _expr_to_str(test.left)
        ops = test.ops
        comparators = test.comparators
        if ops and comparators:
            op_str = _cmpop_to_str(ops[0])
            right = _expr_to_str(comparators[0])
            return f"{left} {op_str} {right}"
    if isinstance(test, ast.Call):
        return _expr_to_str(test)
    if isinstance(test, ast.BoolOp):
        op = "and" if isinstance(test.op, ast.And) else "or"
        parts = [_if_condition_to_str(v) for v in test.values]
        return f" {op} ".join(parts)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return f"not {_if_condition_to_str(test.operand)}"
    return _expr_to_str(test)


def _expr_to_str(node):
    """Simple AST expression to string."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Attribute):
        return f"{_expr_to_str(node.value)}.{node.attr}"
    if isinstance(node, ast.Call):
        func = _expr_to_str(node.func)
        args = ", ".join(_expr_to_str(a) for a in node.args)
        return f"{func}({args})"
    if isinstance(node, ast.Subscript):
        val = _expr_to_str(node.value)
        sl = _expr_to_str(node.slice)
        return f"{val}[{sl}]"
    return "..."


def _cmpop_to_str(op):
    mapping = {
        ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=",
        ast.Gt: ">", ast.GtE: ">=", ast.Is: "is", ast.IsNot: "is not",
        ast.In: "in", ast.NotIn: "not in",
    }
    return mapping.get(type(op), "?")


# ═══════════════════════════════════════════════════════════════
# Spec Assembly
# ═══════════════════════════════════════════════════════════════

def _snake_to_label(s):
    """Convert snake_case ID to human-readable label."""
    return s.replace("_", " ").title()


def import_langgraph(source_path):
    """Parse a LangGraph Python file and produce an Agent Ontology spec dict."""
    path = Path(source_path)
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    tree = ast.parse(source)

    # Extract all components
    typed_dicts = _find_typed_dicts(tree)
    graph_info = _find_graph_calls(tree, source_lines)
    node_funcs = _find_node_functions(tree, source_lines)
    invoke_funcs = _find_invoke_functions(tree)
    string_consts = _find_string_constants(tree)

    # Build spec
    spec = {
        "name": path.stem.replace("_agent_lg", "").replace("_agent", "").replace("_", " ").title(),
        "version": "1.0",
        "description": f"Imported from LangGraph file: {path.name}",
        "entry_point": graph_info["entry_point"] or (graph_info["nodes"][0][0] if graph_info["nodes"] else None),
    }

    entities = []
    processes = []
    edges = []
    schemas = []
    seen_agents = set()
    gate_nodes = set()  # Nodes that are actually gates (conditional edge sources)

    # Identify which nodes are gate sources
    for src, router, mapping in graph_info["conditional_edges"]:
        gate_nodes.add(src)

    # --- Schemas from TypedDicts ---
    for td_name, td_fields in typed_dicts:
        # Filter out control fields
        user_fields = [f for f in td_fields if not f["name"].startswith("_")]
        if user_fields:
            schemas.append({
                "name": td_name,
                "description": f"State schema from TypedDict {td_name}",
                "fields": user_fields,
            })

    # --- Agents from invoke_* functions ---
    for func_name, info in invoke_funcs.items():
        agent_id = func_name.replace("invoke_", "")
        if agent_id in seen_agents:
            continue
        seen_agents.add(agent_id)

        agent = {
            "id": agent_id,
            "type": "agent",
            "label": _snake_to_label(agent_id),
        }
        if info["model"]:
            agent["model"] = info["model"]
        else:
            agent["model"] = "unknown"

        # Resolve system prompt
        prompt = info.get("system_prompt", "")
        if prompt and prompt.startswith("<variable:"):
            var_name = prompt[10:-1]
            prompt = string_consts.get(var_name, f"<see {var_name}>")
        if prompt:
            agent["system_prompt"] = prompt

        entities.append(agent)

    # --- Processes from nodes ---
    for node_name, func_name in graph_info["nodes"]:
        if node_name in gate_nodes:
            # This node is a conditional edge source — check if it's a pass-through
            is_passthrough = func_name == "__lambda__"
            # For pass-through gates, we'll create a gate process from
            # conditional_edges info later
            if is_passthrough:
                continue
            # Non-passthrough: it's a step that also feeds into a gate
            # Create a step process for it
            proc = {
                "id": node_name,
                "type": "step",
                "label": _snake_to_label(node_name),
            }

            # Check if this function calls an LLM
            func_info = node_funcs.get(func_name, {}) if func_name else {}
            if func_info.get("calls_llm"):
                invoked = func_info.get("invokes_func")
                if invoked and invoked.replace("invoke_", "") in seen_agents:
                    agent_id = invoked.replace("invoke_", "")
                    # Add invoke edge
                    edges.append({
                        "type": "invoke",
                        "from": node_name,
                        "to": agent_id,
                        "label": f"Call {_snake_to_label(agent_id)}",
                    })
            processes.append(proc)
        else:
            proc = {
                "id": node_name,
                "type": "step",
                "label": _snake_to_label(node_name),
            }

            func_info = node_funcs.get(func_name, {}) if func_name else {}
            if func_info.get("calls_llm"):
                invoked = func_info.get("invokes_func")
                if invoked and invoked.replace("invoke_", "") in seen_agents:
                    agent_id = invoked.replace("invoke_", "")
                    edges.append({
                        "type": "invoke",
                        "from": node_name,
                        "to": agent_id,
                        "label": f"Call {_snake_to_label(agent_id)}",
                    })

            processes.append(proc)

    # --- Create terminal step if any gate branches go to END ---
    needs_terminal = False
    for src, router_name, mapping in graph_info["conditional_edges"]:
        for target in mapping.values():
            if target == "__END__" or target == "END":
                needs_terminal = True
                break

    terminal_id = "__done__"
    if needs_terminal:
        # Check if a terminal step already exists
        if not any(p["id"] == terminal_id for p in processes):
            processes.append({
                "id": terminal_id,
                "type": "step",
                "label": "Done",
                "description": "Terminal step (imported from LangGraph END)",
                "logic": 'state.data["_done"] = True\n',
            })

    # --- Gate processes from conditional edges ---
    for src, router_name, mapping in graph_info["conditional_edges"]:
        condition, branches, has_end = _analyze_router(tree, router_name, mapping)

        # Replace __terminal__ targets with actual terminal step
        for b in branches:
            if b["target"] == "__terminal__":
                b["target"] = terminal_id

        # Check if src already has a step process (non-passthrough)
        has_step = any(p["id"] == src for p in processes)

        if has_step:
            # Create a separate gate after the step
            gate_id = f"{src}_gate"
            gate_proc = {
                "id": gate_id,
                "type": "gate",
                "label": f"{_snake_to_label(src)} Decision",
                "condition": condition,
                "branches": branches,
            }
            processes.append(gate_proc)
            # Add flow edge from step to gate
            edges.append({
                "type": "flow",
                "from": src,
                "to": gate_id,
                "label": "Decide",
            })
        else:
            # Pass-through gate — src IS the gate
            gate_proc = {
                "id": src,
                "type": "gate",
                "label": _snake_to_label(src),
                "condition": condition,
                "branches": branches,
            }
            processes.append(gate_proc)

    # --- Flow edges ---
    for src, dst in graph_info["edges"]:
        # Skip edges that duplicate gate branches
        edges.append({
            "type": "flow",
            "from": src,
            "to": dst,
            "label": _snake_to_label(dst),
        })

    # --- End edges: mark terminal steps ---
    for src in graph_info["end_edges"]:
        # Find the process and add _done logic
        for proc in processes:
            if proc["id"] == src:
                if proc["type"] == "step":
                    proc["logic"] = proc.get("logic", "") + 'state.data["_done"] = True\n'
                break

    # --- Detect loop edges (back-edges where target appears earlier) ---
    process_order = {p["id"]: i for i, p in enumerate(processes)}
    flow_edges = [e for e in edges if e["type"] == "flow"]
    for e in flow_edges:
        src_idx = process_order.get(e["from"], -1)
        dst_idx = process_order.get(e["to"], -1)
        if dst_idx >= 0 and src_idx > dst_idx:
            e["type"] = "loop"

    spec["entities"] = entities if entities else [
        {"id": "llm", "type": "agent", "label": "LLM", "model": "unknown"}
    ]
    spec["processes"] = processes
    spec["edges"] = edges
    if schemas:
        spec["schemas"] = schemas

    return spec


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Import a LangGraph StateGraph Python file into an Agent Ontology YAML spec"
    )
    parser.add_argument("input", help="Path to LangGraph Python file")
    parser.add_argument("-o", "--output", help="Output YAML file path (default: stdout)")
    parser.add_argument("--validate", action="store_true", help="Validate the output spec")
    parser.add_argument("--quiet", action="store_true", help="Suppress info messages")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    spec = import_langgraph(input_path)

    # Count what we found
    n_ent = len(spec.get("entities", []))
    n_proc = len(spec.get("processes", []))
    n_edge = len(spec.get("edges", []))
    n_schema = len(spec.get("schemas", []))

    if not args.quiet:
        print(f"Imported: {n_ent} entities, {n_proc} processes, {n_edge} edges, {n_schema} schemas",
              file=sys.stderr)

    # Serialize
    yaml_str = yaml.dump(spec, default_flow_style=False, sort_keys=False, allow_unicode=True, width=120)

    if args.output:
        Path(args.output).write_text(yaml_str, encoding="utf-8")
        if not args.quiet:
            print(f"Written to: {args.output}", file=sys.stderr)
    else:
        print(yaml_str)

    # Validate
    if args.validate:
        try:
            from .validate import validate_spec, load_ontology
            ontology = load_ontology()
            errors, warnings = validate_spec(spec, ontology)
            if errors:
                print(f"\n  {len(errors)} validation error(s):", file=sys.stderr)
                for e in errors:
                    print(f"    ✗ {e}", file=sys.stderr)
            if warnings:
                print(f"\n  {len(warnings)} warning(s):", file=sys.stderr)
                for w in warnings:
                    print(f"    ⚠ {w}", file=sys.stderr)
            if not errors:
                print(f"  ✓ Valid spec", file=sys.stderr)
        except ImportError:
            print("Warning: validate.py not found, skipping validation", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
