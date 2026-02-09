#!/usr/bin/env python3
"""
OpenClaw Spec Validator
Validates agent specs against ONTOLOGY.yaml.

Usage: python3 validate.py specs/claude-code.yaml
       python3 validate.py specs/*.yaml
"""

import sys
import os
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ONTOLOGY_PATH = os.path.join(SCRIPT_DIR, "ONTOLOGY.yaml")


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def validate_spec(spec, ontology, filepath):
    errors = []
    warnings = []

    def err(msg):
        errors.append(msg)

    def warn(msg):
        warnings.append(msg)

    # ── Extract ontology type sets ──
    valid_entity_types = set(ontology.get("entity_types", {}).keys())
    valid_process_types = set(ontology.get("process_types", {}).keys())
    valid_edge_types = set(ontology.get("edge_types", {}).keys())

    entity_type_defs = ontology.get("entity_types", {})
    process_type_defs = ontology.get("process_types", {})
    edge_type_defs = ontology.get("edge_types", {})

    # ── Extract spec contents ──
    entities = spec.get("entities", [])
    processes = spec.get("processes", [])
    edges = spec.get("edges", [])
    schemas = spec.get("schemas", [])
    entry_point = spec.get("entry_point")

    # Build ID maps
    all_nodes = {}  # id -> {type, category, data}
    for e in entities:
        eid = e.get("id")
        if not eid:
            err("Entity missing 'id'")
            continue
        if eid in all_nodes:
            err(f"Duplicate node id: '{eid}'")
        all_nodes[eid] = {"type": e.get("type"), "category": "entity", "data": e}

    for p in processes:
        pid = p.get("id")
        if not pid:
            err("Process missing 'id'")
            continue
        if pid in all_nodes:
            err(f"Duplicate node id: '{pid}'")
        all_nodes[pid] = {"type": p.get("type"), "category": "process", "data": p}

    schema_names = {s.get("name") for s in schemas if s.get("name")}

    # ════════════════════════════════════════════════════
    # 1. Spec-level required fields
    # ════════════════════════════════════════════════════
    if not spec.get("name"):
        err("Spec missing required field: 'name'")
    if not spec.get("version"):
        err("Spec missing required field: 'version'")
    if not entities:
        err("Spec has no entities")
    if not processes:
        err("Spec has no processes")

    # ════════════════════════════════════════════════════
    # 2. Entity type validation
    # ════════════════════════════════════════════════════
    for e in entities:
        eid = e.get("id", "?")
        etype = e.get("type")
        if not etype:
            err(f"Entity '{eid}' missing 'type'")
        elif etype not in valid_entity_types:
            err(f"Entity '{eid}' has invalid type '{etype}'. Valid: {sorted(valid_entity_types)}")
        else:
            # Check required properties
            typedef = entity_type_defs[etype]
            for prop in typedef.get("required", {}):
                if prop not in e:
                    err(f"Entity '{eid}' (type={etype}) missing required property: '{prop}'")

    # ════════════════════════════════════════════════════
    # 3. Process type validation
    # ════════════════════════════════════════════════════
    for p in processes:
        pid = p.get("id", "?")
        ptype = p.get("type")
        if not ptype:
            err(f"Process '{pid}' missing 'type'")
        elif ptype not in valid_process_types:
            err(f"Process '{pid}' has invalid type '{ptype}'. Valid: {sorted(valid_process_types)}")
        else:
            typedef = process_type_defs[ptype]
            for prop in typedef.get("required", {}):
                if prop not in p:
                    err(f"Process '{pid}' (type={ptype}) missing required property: '{prop}'")

    # ════════════════════════════════════════════════════
    # 4. Edge type + reference validation
    # ════════════════════════════════════════════════════
    for i, e in enumerate(edges):
        etype = e.get("type")
        efrom = e.get("from")
        eto = e.get("to")
        elabel = e.get("label", f"edge[{i}]")

        if not etype:
            err(f"Edge '{elabel}' missing 'type'")
        elif etype not in valid_edge_types:
            err(f"Edge '{elabel}' has invalid type '{etype}'. Valid: {sorted(valid_edge_types)}")

        if not efrom:
            err(f"Edge '{elabel}' missing 'from'")
        elif efrom not in all_nodes:
            err(f"Edge '{elabel}': 'from' references unknown node '{efrom}'")

        if not eto:
            err(f"Edge '{elabel}' missing 'to'")
        elif eto not in all_nodes:
            err(f"Edge '{elabel}': 'to' references unknown node '{eto}'")

        # Type constraint checks (from ontology edge_type definitions)
        if etype and efrom in all_nodes and eto in all_nodes:
            from_node = all_nodes[efrom]
            to_node = all_nodes[eto]
            _check_edge_constraints(etype, efrom, eto, from_node, to_node, elabel, edge_type_defs, err, warn)

        # Schema reference checks on edges
        for field in ("data", "input", "output"):
            ref = e.get(field)
            if ref and ref not in schema_names:
                err(f"Edge '{elabel}': {field} references unknown schema '{ref}'")

    # ════════════════════════════════════════════════════
    # 5. Ontology validation rules
    # ════════════════════════════════════════════════════

    # Rule: at least one agent entity
    agent_entities = [e for e in entities if e.get("type") == "agent"]
    if not agent_entities:
        err("Spec must have at least one agent entity")

    # Rule: entry_point exists
    if entry_point:
        if entry_point not in all_nodes:
            err(f"entry_point '{entry_point}' references unknown node")
    else:
        # Check for exactly one node with no incoming flow/invoke
        incoming = set()
        for e in edges:
            if e.get("type") in ("flow", "invoke"):
                incoming.add(e.get("to"))
        process_ids = {p.get("id") for p in processes}
        roots = process_ids - incoming
        if len(roots) == 0:
            warn("No entry_point defined and no root process found (all processes have incoming edges)")
        elif len(roots) > 1:
            warn(f"No entry_point defined and multiple root processes found: {sorted(roots)}")

    # Rule: every gate has at least 2 branches
    for p in processes:
        if p.get("type") == "gate":
            branches = p.get("branches", [])
            if len(branches) < 2:
                err(f"Gate '{p.get('id')}' must have at least 2 branches, has {len(branches)}")
            # Also check branch targets resolve
            for b in branches:
                target = b.get("target")
                if target and target not in all_nodes:
                    err(f"Gate '{p.get('id')}' branch target '{target}' references unknown node")

    # Rule: every loop targets an earlier node in the flow
    # (We approximate "earlier" by checking the loop target exists — full
    # topological ordering would require building the flow graph)
    for e in edges:
        if e.get("type") == "loop":
            if e.get("to") not in all_nodes:
                err(f"Loop edge targets unknown node '{e.get('to')}'")

    # Rule: every schema_ref resolves
    _check_schema_refs(entities, processes, schema_names, err)

    # Rule: no orphan nodes
    connected = set()
    for e in edges:
        connected.add(e.get("from"))
        connected.add(e.get("to"))
    for nid in all_nodes:
        if nid not in connected:
            warn(f"Orphan node: '{nid}' has no edges")

    # Rule: agent tools[] should be defined as tool entities
    tool_entity_ids = {e.get("id") for e in entities if e.get("type") == "tool"}
    for e in entities:
        if e.get("type") == "agent" and e.get("tools"):
            for tool_ref in e["tools"]:
                if tool_ref not in tool_entity_ids:
                    warn(f"Agent '{e.get('id')}' references tool '{tool_ref}' not defined as a tool entity")

    # Rule: spawn templates valid
    for p in processes:
        if p.get("type") == "spawn":
            template = p.get("template")
            if not template:
                err(f"Spawn '{p.get('id')}' missing required 'template'")
            elif template == "self":
                pass  # valid recursive reference
            elif template not in all_nodes:
                # Could be a spec_ref (external file) — warn, don't error
                warn(f"Spawn '{p.get('id')}' template '{template}' is not a node in this spec (may be external spec_ref)")

    # Rule: recursive spawns should document bounds
    for p in processes:
        if p.get("type") == "spawn" and (p.get("recursive") or p.get("template") == "self"):
            if not p.get("max_depth"):
                warn(f"Recursive spawn '{p.get('id')}' should specify max_depth or document natural bounds")

    # Rule: protocol participants must reference valid entities
    for p in processes:
        if p.get("type") == "protocol":
            participants = p.get("participants", [])
            if len(participants) < 2:
                err(f"Protocol '{p.get('id')}' must have at least 2 participants")
            for part in participants:
                entity_ref = part.get("entity")
                if entity_ref and entity_ref not in all_nodes:
                    err(f"Protocol '{p.get('id')}' participant references unknown entity '{entity_ref}'")

    return errors, warnings


def _check_edge_constraints(etype, efrom, eto, from_node, to_node, elabel, edge_type_defs, err, warn):
    """Check that edge from/to types match ontology constraints."""
    typedef = edge_type_defs.get(etype, {})
    required = typedef.get("required", {})

    from_constraint = required.get("from", {}).get("type", "")
    to_constraint = required.get("to", {}).get("type", "")

    if from_constraint and from_constraint != "node_ref":
        allowed_from = _parse_type_constraint(from_constraint)
        actual_from = from_node["type"]
        if allowed_from and actual_from not in allowed_from:
            warn(f"Edge '{elabel}' ({etype}): 'from' node '{efrom}' is type '{actual_from}', "
                 f"but {etype} expects from to be {from_constraint}")

    if to_constraint and to_constraint != "node_ref":
        allowed_to = _parse_type_constraint(to_constraint)
        actual_to = to_node["type"]
        if allowed_to and actual_to not in allowed_to:
            warn(f"Edge '{elabel}' ({etype}): 'to' node '{eto}' is type '{actual_to}', "
                 f"but {etype} expects to to be {to_constraint}")


def _parse_type_constraint(constraint):
    """Parse type constraints like 'step | gate | spawn' or 'agent | tool' or 'store'."""
    constraint = constraint.strip('"').strip("'")
    if "|" in constraint:
        return {t.strip() for t in constraint.split("|")}
    return {constraint}


def _check_schema_refs(entities, processes, schema_names, err):
    """Check that all schema references in entities and processes resolve."""
    schema_ref_fields = ["input_schema", "output_schema", "data_in", "data_out", "schema"]
    for e in entities:
        eid = e.get("id", "?")
        for field in schema_ref_fields:
            ref = e.get(field)
            if ref and ref not in schema_names:
                err(f"Entity '{eid}': {field} references unknown schema '{ref}'")
    for p in processes:
        pid = p.get("id", "?")
        for field in schema_ref_fields:
            ref = p.get(field)
            if ref and ref not in schema_names:
                err(f"Process '{pid}': {field} references unknown schema '{ref}'")


def print_results(filepath, errors, warnings):
    name = os.path.basename(filepath)
    total = len(errors) + len(warnings)

    if total == 0:
        print(f"\033[32m✓ {name}: VALID (0 errors, 0 warnings)\033[0m")
        return True

    print(f"\n{'═' * 60}")
    print(f"  {name}")
    print(f"{'═' * 60}")

    if errors:
        print(f"\n  \033[31m{len(errors)} ERROR{'S' if len(errors) != 1 else ''}:\033[0m")
        for e in errors:
            print(f"    \033[31m✗\033[0m {e}")

    if warnings:
        print(f"\n  \033[33m{len(warnings)} WARNING{'S' if len(warnings) != 1 else ''}:\033[0m")
        for w in warnings:
            print(f"    \033[33m⚠\033[0m {w}")

    print()
    return len(errors) == 0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <spec.yaml> [spec2.yaml ...]")
        sys.exit(1)

    # Load ontology
    try:
        ontology = load_yaml(ONTOLOGY_PATH)
    except Exception as e:
        print(f"\033[31mFailed to load ontology: {e}\033[0m")
        sys.exit(1)

    all_valid = True
    for filepath in sys.argv[1:]:
        try:
            spec = load_yaml(filepath)
        except Exception as e:
            print(f"\033[31m✗ {filepath}: Failed to parse YAML: {e}\033[0m")
            all_valid = False
            continue

        errors, warnings = validate_spec(spec, ontology, filepath)
        valid = print_results(filepath, errors, warnings)
        if not valid:
            all_valid = False

    sys.exit(0 if all_valid else 1)


if __name__ == "__main__":
    main()
