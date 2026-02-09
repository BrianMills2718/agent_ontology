# OpenClaw

**OpenAPI for Agents** — a universal format for describing AI agent architectures.

```
docs  -->  spec  -->  { validate, visualize, instantiate, analyze }
```

Given any agent's documentation, OpenClaw lets you:
1. **Decompose** it into a structured YAML spec
2. **Validate** the spec against a formal ontology (structural + graph analysis)
3. **Visualize** it as an interactive flowchart
4. **Instantiate** it as a runnable Python agent with real LLM calls
5. **Analyze** complexity, traces, and runtime metrics
6. **Test** agents automatically with canned inputs and validators

Nobody does all six today. This is the gap.

## Quick Start

```bash
# 1. Validate a spec
python3 validate.py specs/react.yaml

# 2. Generate a runnable agent
python3 instantiate.py specs/react.yaml

# 3. Run it (requires API keys in .env.local)
export $(cat .env.local | xargs)
python3 agents/react_agent.py

# 4. View specs in the browser
python3 -m http.server 8000
# Open http://localhost:8000/spec-viewer.html

# 5. Analyze the trace
python3 analyze_trace.py trace.json

# 6. Score spec complexity
python3 complexity.py --all specs/

# 7. Run all agent tests
python3 test_agents.py
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

Validates against `ONTOLOGY.yaml` with 20+ rules:
- Entity/process/edge type validation with required fields
- Schema reference resolution
- Graph connectivity (unreachable processes, disconnected chains)
- Fan-out without join detection
- Empty process shells (no logic, invocations, or store access)
- Schema field collision warnings in fan-out patterns

### Step 3: Visualize

Open `spec-viewer.html` in a browser (via HTTP server). Four views:

- **Graph** — Interactive canvas flowchart. Drag nodes, click for details, hover for tooltips.
- **State Machine** — Linear process flow with gates, branches, loops, and agent invocations.
- **Schemas** — All data schemas with field types and cross-references.
- **Compare All** — Side-by-side comparison table across all 9 agent specs.

### Step 4: Instantiate

```bash
python3 instantiate.py specs/react.yaml
```

Generates a complete Python agent with:
- State machine (PROCESSES dict + TRANSITIONS dict)
- **Fan-out support**: Multiple outgoing flow edges run all branches sequentially
- **Namespaced state**: Agent outputs stored under `state.data["schema_name"]` and `state.data["process_id_result"]` to prevent field collisions
- Schema-aware LLM calls (input extraction, output parsing, structured prompts)
- Gate condition evaluation (parsed from human-readable conditions)
- Trace logging with metrics (LLM calls, duration, schema compliance, token estimates)
- Store abstractions (queue, vector, buffer, log)
- Multi-model routing: `claude*` → Anthropic, `gemini*` → Google genai, else → OpenAI

## Agent Catalog

9 agent specs, all validated. 8 are instantiable and runnable with `gemini-3-flash-preview`.

| Spec | Type | Ent | Proc | Sch | Complexity | Status |
|------|------|-----|------|-----|------------|--------|
| `claude-code` | Tool-use agent | 22 | 19 | 40 | 83.2 (very complex) | Description only |
| `crew` | Multi-agent | 6 | 17 | 9 | 61.4 (complex) | Working |
| `react` | Reason+Act | 6 | 8 | 8 | 60.7 (complex) | Working |
| `code_reviewer` | Parallel analysis | 7 | 10 | 9 | 60.6 (complex) | Working |
| `babyagi` | Task-driven autonomous | 5 | 5 | 9 | 59.5 (moderate) | Working |
| `autogpt` | Goal-driven | 6 | 11 | 10 | 59.2 (moderate) | Working |
| `babyagi_autogen` | Task-driven | 5 | 3 | 9 | 58.4 (moderate) | Working |
| `debate` | Multi-agent debate | 6 | 13 | 7 | 57.7 (moderate) | Working |
| `rag` | Retrieval-augmented | 5 | 10 | 10 | 50.4 (moderate) | Working |

Complexity scores computed by `complexity.py` using weighted graph metrics (entities, edges, fan-out, loops, schema count, graph depth, invocation density).

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
| `instantiate.py` | Generate runnable Python agent | `python3 instantiate.py spec.yaml` |
| `specgen.py` | Generate spec from description | `python3 specgen.py desc.md -o spec.yaml --validate --fix` |
| `spec-viewer.html` | Interactive visualization | Serve via HTTP, open in browser |
| `test_agents.py` | Automated agent testing | `python3 test_agents.py --agent react` |
| `analyze_trace.py` | Trace analysis and comparison | `python3 analyze_trace.py trace.json` |
| `complexity.py` | Spec complexity scoring | `python3 complexity.py --all specs/` |
| `mutate.py` | Spec mutation engine | `python3 mutate.py spec.yaml --random -n 5` |

## Architecture

```
ONTOLOGY.yaml          # Type system (entity types, edge types, constraints)
     |
     v
specs/*.yaml           # Agent specifications (9 agents)
     |
     +---> validate.py       # Validation (20+ rules, graph analysis)
     +---> instantiate.py    # Code generation -> agents/*.py
     +---> complexity.py     # Complexity scoring (10 metrics)
     +---> mutate.py         # Spec mutation engine (8 operators)
     +---> spec-viewer.html  # Visualization (4 views)

agents/*.py            # Generated runnable agents
     |
     +---> test_agents.py    # Automated testing with validators
     +---> trace.json        # Runtime traces
              |
              v
         analyze_trace.py    # Trace analysis + comparison

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
- `google-genai` (for running generated agents with Gemini): `pip install google-genai`
- `openai` (for specgen and OpenAI model agents): `pip install openai`
- `anthropic` (optional, for Claude model agents): `pip install anthropic`
- Modern browser (for spec-viewer)

API keys go in `.env.local`:
```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...     # optional, for specgen
ANTHROPIC_API_KEY=...  # optional, for Claude model agents
```

## Project Structure

```
specs/                  # Agent specifications (YAML)
agents/                 # Generated runnable agents (Python)
test_descriptions/      # Natural language descriptions for specgen testing
traces/                 # Per-agent trace files from test runs
ONTOLOGY.yaml           # The type system
validate.py             # Spec validator (20+ rules)
instantiate.py          # Code generator (fan-out, namespacing, metrics)
specgen.py              # Description-to-spec pipeline
spec-viewer.html        # Interactive multi-view visualization
test_agents.py          # Automated test harness
analyze_trace.py        # Trace analysis and comparison
complexity.py           # Spec complexity scoring
mutate.py               # Spec mutation engine (8 operators)
trace.json              # LLM call traces from last agent run
```
