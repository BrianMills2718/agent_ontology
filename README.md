# OpenClaw

**OpenAPI for Agents** — a universal format for describing AI agent architectures.

```
docs  -->  spec  -->  { diagram, running agent }
```

Given any agent's documentation, OpenClaw lets you:
1. **Decompose** it into a structured YAML spec
2. **Validate** the spec against a formal ontology
3. **Visualize** it as an interactive flowchart
4. **Instantiate** it as a runnable Python agent with real LLM calls

Nobody does all four today. This is the gap.

## Quick Start

```bash
# 1. Validate a spec
python3 validate.py specs/babyagi.yaml

# 2. Generate a runnable agent
python3 instantiate.py specs/babyagi.yaml -o agents/babyagi_agent.py

# 3. Run it (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python3 agents/babyagi_agent.py

# 4. View specs in the browser
python3 -m http.server 8000
# Open http://localhost:8000/spec-viewer.html
```

## The Pipeline

### Step 1: Write or generate a spec

Specs are YAML files following the OpenClaw ontology. You can write one by hand or generate one from documentation:

```bash
# Auto-generate from a natural language description
python3 specgen.py description.md -o specs/my_agent.yaml --validate --fix
```

The `specgen.py` pipeline loads the ontology + example specs as context for an LLM, generates a spec, validates it, and optionally auto-fixes errors.

### Step 2: Validate

```bash
python3 validate.py specs/my_agent.yaml
# Output: errors (must fix) and warnings (informational)
```

Validates against `ONTOLOGY.yaml` — checks entity types, process types, edge types, required fields, schema references, graph connectivity.

### Step 3: Visualize

Open `spec-viewer.html` in a browser (via HTTP server). Four views:

- **Graph** — Interactive canvas flowchart. Drag nodes, click for details, hover for tooltips.
- **State Machine** — Linear process flow with gates, branches, loops, and agent invocations.
- **Schemas** — All data schemas with field types and cross-references.
- **Compare All** — Side-by-side comparison table across all 9 agent specs.

### Step 4: Instantiate

```bash
python3 instantiate.py specs/babyagi.yaml -o agents/babyagi_agent.py
```

Generates a complete Python agent with:
- State machine (PROCESSES dict + TRANSITIONS dict)
- Schema-aware LLM calls (input extraction, output parsing, structured prompts)
- Gate condition evaluation (parsed from human-readable conditions)
- Trace logging (every LLM call logged to `trace.json`)
- Store abstractions (queue, vector, buffer, log)

## Agent Catalog

9 agent specs, all validated and instantiable:

| Spec | Type | Entities | Processes | Schemas | Notes |
|------|------|----------|-----------|---------|-------|
| `claude-code` | Tool-use agent | 14 | 13 | 8 | Most complex: permissions, spawning, compaction |
| `babyagi` | Task-driven autonomous | 5 | 5 | 9 | Fully functional with real API calls |
| `react` | Reason+Act | 6 | 8 | 5 | Think/Act/Observe loop with tools |
| `rag` | Retrieval-augmented | 6 | 8 | 7 | Query rewriting + relevance judging |
| `autogpt` | Goal-driven | 7 | 12 | 7 | Planning + criticism loop |
| `crew` | Multi-agent | 6 | 15 | 7 | Coordinator dispatches to specialists |
| `code_reviewer` | Parallel analysis | 6 | 9 | 9 | Auto-generated from description |
| `babyagi_autogen` | Task-driven | 5 | 5 | 9 | Auto-generated; matches hand-written |
| `debate` | Multi-agent debate | 5 | 13 | 7 | Auto-generated; pro/con/judge |

## Spec Format

A spec has four sections:

```yaml
name: "My Agent"
version: "1.0"
description: "What this agent does"
entry_point: first_step

entities:       # Things that exist (agents, stores, tools, humans)
processes:      # Things that happen (steps, gates, checkpoints)
edges:          # Connections (flow, invoke, loop, branch, read, write)
schemas:        # Data shapes flowing between components
```

### Entity types
- `agent` — LLM-backed component with system prompt, model, I/O schemas
- `store` — Persistent state (queue, vector, buffer, log)
- `tool` — External capability (API, function, system)
- `human` — Human-in-the-loop participant
- `config` — Static configuration

### Process types
- `step` — Do something (may include inline `logic:` as Python)
- `gate` — Decision point with `condition:` and `branches:`
- `checkpoint` — Human approval required
- `spawn` — Create sub-agents (supports `recursive: true`)
- `protocol` — Multi-party interaction
- `policy` — Cross-cutting constraint

### Edge types
- `flow` — Sequential control flow
- `invoke` — Call an agent/tool and get a response
- `loop` — Conditional back-edge
- `branch` — Conditional forward-edge (from gates)
- `read` / `write` — Store access
- `modify` / `observe` — Policy interactions

Full type system: `ONTOLOGY.yaml`

## Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `validate.py` | Check spec against ontology | `python3 validate.py spec.yaml` |
| `instantiate.py` | Generate runnable Python agent | `python3 instantiate.py spec.yaml -o agent.py` |
| `specgen.py` | Generate spec from description | `python3 specgen.py desc.md -o spec.yaml --validate --fix` |
| `spec-viewer.html` | Interactive visualization | Serve via HTTP, open in browser |

## Architecture

```
ONTOLOGY.yaml          # Type system (entity types, edge types, constraints)
     |
     v
specs/*.yaml           # Agent specifications (9 agents)
     |
     +---> validate.py       # Validation against ontology
     +---> instantiate.py    # Code generation -> agents/*.py
     +---> spec-viewer.html  # Visualization (4 views)

test_descriptions/*.md # Natural language agent descriptions
     |
     v
specgen.py             # LLM-powered spec generation
     |
     v
specs/*.yaml           # Generated specs (validated + auto-fixed)
```

## Formal Foundation

OpenClaw uses a **reified hypergraph** constrained by an **ontology**:

- **Reified hypergraph**: Any relationship can be a node, any node can participate in any number of relationships. This means templates, dynamic topology, conditional wiring, self-modification — all expressible.
- **Ontology**: Constrains the hypergraph so code generators know what to expect. Defines types, required fields, and validation rules.

See `VISION.md` for the full rationale and `ONTOLOGY.yaml` for the complete type system.

## Requirements

- Python 3.8+
- `pyyaml` (`pip install pyyaml`)
- `openai` (for running generated agents and specgen): `pip install openai`
- Modern browser (for spec-viewer)

## Project Structure

```
specs/                  # Agent specifications (YAML)
agents/                 # Generated runnable agents (Python)
test_descriptions/      # Natural language descriptions for specgen testing
ONTOLOGY.yaml           # The type system
validate.py             # Spec validator
instantiate.py          # Code generator
specgen.py              # Description-to-spec pipeline
spec-viewer.html        # Interactive multi-view visualization
trace.json              # LLM call traces from agent runs
```
