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

# 2. Generate a runnable agent (custom backend)
python3 instantiate.py specs/react.yaml

# 3. Generate a LangGraph agent
python3 instantiate.py specs/react.yaml --backend langgraph

# 4. Batch-generate all agents (both backends)
python3 instantiate.py --all specs/ -o agents/
python3 instantiate.py --all specs/ -o agents_lg/ --backend langgraph

# 5. Run it (requires API keys in .env.local)
export $(cat .env.local | xargs)
python3 agents/react_agent.py

# 6. View specs in the browser
python3 -m http.server 8000
# Open http://localhost:8000/spec-viewer.html

# 7. Analyze the trace
python3 analyze_trace.py trace.json

# 8. Score spec complexity
python3 complexity.py --all specs/

# 9. Run all agent tests
python3 test_agents.py

# 10. Override model for all agents
python3 test_agents.py --model gpt-4o --agent react

# 11. Compare models side-by-side
python3 test_agents.py --compare-models gemini-3-flash-preview gpt-4o-mini gpt-4o

# 12. Mutate and evolve a spec
python3 mutate.py specs/react.yaml --random -n 3
python3 evolve.py specs/react.yaml --generations 2 --population 4

# 13. Evolve with crossover and benchmark fitness
python3 evolve.py specs/react.yaml --generations 3 --population 6 --crossover --benchmark gsm8k

# 14. E2E specgen pipeline test
python3 test_specgen.py --fix

# 15. Analyze spec coverage of ontology features
python3 coverage.py --all specs/

# 16. Lint specs for anti-patterns
python3 lint.py --all specs/ --severity warn

# 17. Classify agent topologies
python3 topology.py --all specs/

# 18. Compare two spec versions
python3 spec_diff.py specs/react.yaml specs/autogpt.yaml

# 19. Project health dashboard
python3 dashboard.py specs/
python3 dashboard.py --brief specs/

# 20. Export Mermaid flowcharts
python3 mermaid.py specs/react.yaml
python3 mermaid.py --all specs/

# 21. Spec similarity and clustering
python3 similarity.py --all specs/ --top 5 --clusters 5

# 22. Migrate specs to new version
python3 migrate.py --all specs/ --to 1.1 --dry-run
python3 migrate.py --list-versions

# 23. Run property-based tests (no API keys needed)
python3 test_properties.py

# 24. Generate comparative analysis report
python3 comparative_report.py --all specs/
python3 comparative_report.py --all specs/ --json

# 25. Compose patterns into new agents
python3 compose.py compose_specs/react_refine.yaml -o specs/react_refine.yaml --validate

# 26. Detect patterns in existing specs
python3 -c "from patterns import detect_patterns; import yaml; print(detect_patterns(yaml.safe_load(open('specs/react.yaml'))))"

# 27. Import a LangGraph agent into an OpenClaw spec
python3 import_langgraph.py agents_lg/react_agent_lg.py -o specs/imported_react.yaml --validate

# 28. Run benchmark suites
python3 benchmark.py --suite gsm8k --agent self_refine --examples 3
python3 benchmark.py --suite hotpotqa --agent react --examples 5
python3 benchmark.py --suite arc --agent react --examples 5
python3 benchmark.py --suite humaneval --agent multi_agent_codegen --examples 3

# 29. OWL bridge: round-trip test (YAML -> OWL -> YAML)
python3 owl_bridge.py --round-trip
python3 owl_bridge.py --round-trip specs/react.yaml

# 30. OWL bridge: pattern classification via OWL structural model
python3 owl_bridge.py --classify

# 31. OWL bridge: export reconstructed YAML from OWL
python3 owl_bridge.py --export specs/react.yaml
```

## The Pipeline

### Step 1: Write or generate a spec

Specs are YAML files following the OpenClaw ontology. You can write one by hand or generate one from documentation:

```bash
# Auto-generate from a natural language description
python3 specgen.py description.md -o specs/my_agent.yaml --validate --fix
```

The `specgen.py` pipeline loads the ontology + example specs as context for an LLM, generates a spec, validates it, and optionally auto-fixes errors. Pattern-aware: scans descriptions for architectural patterns (reasoning loop, critique cycle, debate, retrieval, etc.), selects matching example specs, and includes pattern context in the prompt.

### Step 2: Validate

```bash
python3 validate.py specs/my_agent.yaml
# Output: errors (must fix) and warnings (informational)
```

Validates against `ONTOLOGY.yaml` with 25+ rules:
- Entity/process/edge type validation with required fields
- Schema reference resolution
- Graph connectivity (unreachable processes, disconnected chains)
- Fan-out without join detection
- Empty process shells (no logic, invocations, or store access)
- Schema field collision warnings in fan-out patterns
- Error handler scope and retry configuration validation

### Step 3: Visualize

Open `spec-viewer.html` in a browser (via HTTP server). Five views:

- **Overview** — Simplified architecture diagram with BFS spine layout. Steps and agents are collapsed into single nodes. Entity connections shown as single lines per (process, entity) pair — click for sidebar details showing edge types, labels, and return schemas. Draggable nodes.
- **Graph** — Interactive canvas flowchart. Drag nodes, click for details, hover for tooltips.
- **State Machine** — Linear process flow with gates, branches, loops, and agent invocations.
- **Schemas** — All data schemas with field types and cross-references.
- **Compare All** — Side-by-side comparison table across all 22 agent specs.

Supports trace overlay: load a `trace.json` to see execution counts, durations, and LLM call heat-maps on the State Machine view.

### Step 4: Instantiate

```bash
# Custom backend (state machine)
python3 instantiate.py specs/react.yaml

# LangGraph backend (StateGraph)
python3 instantiate.py specs/react.yaml --backend langgraph
```

Two backends available:

**Custom backend** (default) generates a complete Python agent with:
- State machine (PROCESSES dict + TRANSITIONS dict)
- **Fan-out support**: Multiple outgoing flow edges run all branches sequentially
- **Namespaced state**: Agent outputs stored under `state.data["schema_name"]` and `state.data["process_id_result"]` to prevent field collisions
- **Edge-driven store access**: Store reads generated from `read` edges (before logic blocks), writes from `write` edges (after invocations). Supports `query_key` for parameterized reads.
- Schema-aware LLM calls (input extraction, output parsing, structured prompts)
- Gate condition evaluation (parsed from human-readable conditions)
- Trace logging with metrics (LLM calls, duration, schema compliance, token estimates)
- Store abstractions (queue, vector, buffer, log)
- Multi-model routing: `claude*` → Anthropic, `gemini*` → Google genai, else → OpenAI
- Runtime model override via `OPENCLAW_MODEL` env var
- LLM retry with exponential backoff (3 attempts)
- Runtime schema validation (field presence + type checking)
- Configurable `MAX_ITERATIONS` via `OPENCLAW_MAX_ITER` env var

**LangGraph backend** (`--backend langgraph`) generates agents using LangGraph's `StateGraph`:
- Steps → nodes, gates → routing functions + conditional edges, flow → edges, loops → back-edges
- LangChain ChatModels (`ChatGoogleGenerativeAI`, `ChatAnthropic`, `ChatOpenAI`)
- TypedDict-based state with fields derived from schemas + logic block analysis
- Gate chaining support (gate → gate inserts pass-through node)
- Channel support: publish/subscribe edges generate channel read/write code
- All 22 specs generate valid Python; verified E2E on multiple agents

## Agent Catalog

22 agent specs, all validated. 21 are instantiable and runnable with `gemini-3-flash-preview`.

| Spec | Type | Ent | Proc | Sch | Complexity | Status |
|------|------|-----|------|-----|------------|--------|
| `claude-code` | Tool-use agent | 22 | 19 | 40 | 83.2 (very complex) | Description only |
| `meta_prompting` | Dynamic delegation | 6 | 15 | 11 | 70.4 (complex) | Working |
| `mixture_of_agents` | Parallel proposers | 6 | 11 | 7 | 68.8 (complex) | Working |
| `plan_and_solve` | Decompose+verify | 4 | 9 | 9 | 65.7 (complex) | Working |
| `socratic_tutor` | Human-in-loop tutoring | 7 | 12 | 13 | 65.5 (complex) | Working |
| `reflexion` | Self-reflection loop | 5 | 11 | 10 | 62.1 (complex) | Working |
| `crew` | Multi-agent | 6 | 17 | 9 | 61.4 (complex) | Working |
| `react` | Reason+Act | 6 | 8 | 8 | 60.7 (complex) | Working |
| `code_reviewer` | Parallel analysis | 7 | 10 | 9 | 60.6 (complex) | Working |
| `babyagi` | Task-driven autonomous | 5 | 5 | 9 | 59.5 (moderate) | Working |
| `autogpt` | Goal-driven | 6 | 11 | 10 | 59.2 (moderate) | Working |
| `babyagi_autogen` | Task-driven | 5 | 3 | 9 | 58.4 (moderate) | Working |
| `debate` | Multi-agent debate | 6 | 13 | 7 | 57.7 (moderate) | Working |
| `voyager` | Open-ended exploration | 5 | 12 | 9 | 55.5 (moderate) | Working |
| `tree_of_thought` | Tree search | 3 | 7 | 7 | 55.4 (moderate) | Working |
| `self_refine` | Generate-critique | 2 | 7 | 7 | 55.0 (moderate) | Working |
| `lats` | MCTS tree search | 5 | 8 | 8 | 51.2 (moderate) | Working |
| `rag` | Retrieval-augmented | 5 | 10 | 10 | 50.4 (moderate) | Working |
| `map_reduce` | Parallel chunk processing | 4 | 6 | 11 | 48.6 (moderate) | Working |
| `multi_agent_codegen` | Code generation pipeline | 5 | 5 | 7 | 47.4 (moderate) | Working |
| `customer_support_swarm` | Handoff-based routing | 8 | 7 | 7 | — | Working |
| `software_team` | Pub/sub team pipeline | 10 | 8 | 10 | — | Working |

Complexity scores computed by `complexity.py` using weighted graph metrics (entities, edges, fan-out, loops, schema count, graph depth, invocation density).

## Spec Format

A spec has four sections:

```yaml
name: "My Agent"
version: "1.0"
description: "What this agent does"
entry_point: first_step

entities:       # Things that exist (agents, stores, tools, humans, channels, teams)
processes:      # Things that happen (steps, gates, checkpoints)
edges:          # Connections (flow, invoke, loop, branch, read, write, publish, subscribe, handoff)
schemas:        # Data shapes flowing between components
```

### Entity types
- `agent` — LLM-backed component with system prompt, model, I/O schemas
- `store` — Persistent state (queue, vector, buffer, log)
- `tool` — External capability (API, function, system)
- `human` — Human-in-the-loop participant
- `config` — Static configuration
- `channel` — Named pub/sub communication channel with message schema and reducer
- `team` — Agent group with strategy (sequential, hierarchical, consensus, round-robin)
- `conversation` — Multi-turn dialogue with history and persistence

### Process types
- `step` — Do something (may include inline `logic:` as Python)
- `gate` — Decision point with `condition:` and `branches:`
- `checkpoint` — Human approval required
- `spawn` — Create sub-agents (supports `recursive: true`)
- `protocol` — Multi-party interaction
- `policy` — Cross-cutting constraint
- `error_handler` — Structured error handling with retry, fallback, timeout

### Edge types
- `flow` — Sequential control flow
- `invoke` — Call an agent/tool and get a response
- `loop` — Conditional back-edge
- `branch` — Conditional forward-edge (from gates)
- `read` / `write` — Store access
- `publish` / `subscribe` — Channel-based messaging
- `handoff` — Agent-to-agent control transfer
- `error` — Error flow routing to handler
- `modify` / `observe` — Policy interactions

Full type system: `ONTOLOGY.yaml` | Machine-readable: `spec_schema.json` (JSON Schema)

## Tools

| Tool | Purpose | Usage |
|------|---------|-------|
| `validate.py` | Check spec against ontology | `python3 validate.py spec.yaml` |
| `instantiate.py` | Generate runnable agent (custom or LangGraph) | `python3 instantiate.py spec.yaml [--backend langgraph]` |
| `specgen.py` | Generate spec from description (pattern-aware) | `python3 specgen.py desc.md -o spec.yaml --validate --fix` |
| `spec-viewer.html` | Interactive visualization (5 views) | Serve via HTTP, open in browser |
| `test_agents.py` | Automated agent testing | `python3 test_agents.py --agent react` |
| `analyze_trace.py` | Trace analysis and comparison | `python3 analyze_trace.py trace.json` |
| `complexity.py` | Spec complexity scoring | `python3 complexity.py --all specs/` |
| `mutate.py` | Spec mutation engine (field + pattern-level) | `python3 mutate.py spec.yaml --random -n 5` |
| `evolve.py` | Evolutionary search with crossover | `python3 evolve.py spec.yaml --generations 3 --crossover` |
| `benchmark.py` | Benchmark suite (HotpotQA, GSM8K, ARC, HumanEval) | `python3 benchmark.py --suite gsm8k --agent self_refine` |
| `patterns.py` | Pattern library (7 patterns) | `from patterns import detect_patterns, PATTERNS` |
| `compose.py` | Compose patterns into new specs | `python3 compose.py compose_spec.yaml -o spec.yaml` |
| `test_specgen.py` | Specgen E2E testing | `python3 test_specgen.py --fix` |
| `spec_diff.py` | Structured spec comparison | `python3 spec_diff.py old.yaml new.yaml` |
| `coverage.py` | Ontology feature coverage | `python3 coverage.py --all specs/` |
| `lint.py` | Anti-pattern detection (10 rules) | `python3 lint.py --all specs/` |
| `topology.py` | Control-flow topology classifier | `python3 topology.py --all specs/` |
| `dashboard.py` | Unified project health report | `python3 dashboard.py specs/` |
| `migrate.py` | Spec version migration | `python3 migrate.py --all specs/ --to 2.0 --dry-run` |
| `mermaid.py` | Mermaid flowchart export | `python3 mermaid.py specs/react.yaml` |
| `similarity.py` | Spec similarity & clustering | `python3 similarity.py --all specs/ --clusters 5` |
| `test_properties.py` | Property-based structural tests (174+) | `python3 test_properties.py` |
| `comparative_report.py` | Cross-spec comparative analysis | `python3 comparative_report.py --all specs/` |
| `import_langgraph.py` | Import LangGraph StateGraph → YAML spec | `python3 import_langgraph.py agent.py -o spec.yaml` |
| `owl_bridge.py` | OWL dual representation: YAML<->OWL round-trip + classification | `python3 owl_bridge.py --round-trip` |
| `ontology_owl.py` | OWL/DL pattern classification | `python3 ontology_owl.py` |
| `ontology_rdf.py` | RDF export + semantic pattern queries | `python3 ontology_rdf.py` |

## Architecture

```
ONTOLOGY.yaml          # Type system (entity types, edge types, constraints)
     |
     v
specs/*.yaml           # Agent specifications (22 agents)
     |
     +---> validate.py       # Validation (25+ rules, graph analysis)
     +---> instantiate.py    # Code generation -> agents/*.py or agents_lg/*.py
     +---> complexity.py     # Complexity scoring (10 metrics)
     +---> mutate.py         # Spec mutation engine (field + pattern-level)
     +---> evolve.py         # Evolutionary search (mutate → test → select)
     +---> patterns.py       # Pattern library (7 reusable patterns)
     +---> compose.py        # Compose patterns into new specs
     +---> coverage.py       # Ontology feature coverage report
     +---> lint.py           # Anti-pattern detection (10 rules)
     +---> topology.py       # Control-flow topology classifier
     +---> spec_diff.py      # Structured spec comparison
     +---> dashboard.py      # Unified project health report
     +---> mermaid.py        # Mermaid flowchart export
     +---> similarity.py     # Spec similarity & clustering
     +---> migrate.py        # Spec version migration
     +---> comparative_report.py  # Cross-spec comparative analysis
     +---> test_properties.py     # Property-based structural tests (174+)
     +---> spec-viewer.html  # Visualization (5 views + trace overlay)

agents/*.py            # Generated runnable agents (custom backend)
agents_lg/*.py         # Generated runnable agents (LangGraph backend)
     |
     +---> test_agents.py    # Automated testing + multi-model comparison
     +---> benchmark.py      # Benchmark suite (HotpotQA, GSM8K, ARC, HumanEval)
     +---> trace.json        # Runtime traces
              |
              v
         analyze_trace.py    # Trace analysis + comparison

compose_specs/*.yaml   # Pattern composition recipes
     |
     v
compose.py             # Pattern → spec composition
     |
     v
specs/*.yaml           # Composed specs (validated)

test_descriptions/*.md # Natural language agent descriptions
     |
     v
specgen.py             # LLM-powered spec generation
     |
     v
specs/*.yaml           # Generated specs (validated + auto-fixed)
     |
     v
test_specgen.py        # E2E pipeline testing
```

## Formal Foundation

OpenClaw uses a **reified hypergraph** constrained by an **ontology**:

- **Reified hypergraph**: Any relationship can be a node, any node can participate in any number of relationships. This means templates, dynamic topology, conditional wiring, self-modification — all expressible.
- **Ontology**: Constrains the hypergraph so code generators know what to expect. Defines types, required fields, and validation rules.

See `SPEC.md` for the standalone format specification, `spec_schema.json` for machine-readable JSON Schema validation, `ROADMAP.md` for the long-term vision, and `ONTOLOGY.yaml` for the internal type system definition.

## Requirements

- Python 3.8+
- Modern browser (for spec-viewer)

```bash
pip install -r requirements.txt           # Core + LLM providers + LangGraph
pip install -r requirements-dev.txt       # Adds OWL/RDF experimental tools
```

API keys go in `.env.local`:
```bash
GEMINI_API_KEY=...
OPENAI_API_KEY=...     # optional, for specgen
ANTHROPIC_API_KEY=...  # optional, for Claude model agents
```

## Project Structure

```
specs/                  # Agent specifications (YAML, 22 specs)
agents/                 # Generated runnable agents (custom backend)
agents_lg/              # Generated runnable agents (LangGraph backend)
compose_specs/          # Pattern composition recipes (3 examples)
test_descriptions/      # Natural language descriptions for specgen testing
benchmarks/             # Benchmark datasets (HotpotQA, GSM8K, ARC, HumanEval) and scoring
traces/                 # Per-agent trace files from test runs
ONTOLOGY.yaml           # The type system (internal)
SPEC.md                 # Formal spec format specification (standalone)
spec_schema.json        # JSON Schema for machine-readable spec validation
requirements.txt        # Python dependencies
requirements-dev.txt    # Development/research dependencies
validate.py             # Spec validator (25+ rules)
instantiate.py          # Code generator (custom + LangGraph backends)
specgen.py              # Description-to-spec pipeline (pattern-aware)
spec-viewer.html        # Interactive multi-view visualization (5 views + trace overlay)
test_agents.py          # Automated test harness + multi-model comparison
test_specgen.py         # E2E specgen pipeline testing
analyze_trace.py        # Trace analysis and comparison
complexity.py           # Spec complexity scoring
coverage.py             # Ontology feature coverage report
lint.py                 # Anti-pattern linter (10 rules)
topology.py             # Control-flow topology classifier
spec_diff.py            # Structured spec comparison
dashboard.py            # Unified project health report
mermaid.py              # Mermaid flowchart export
similarity.py           # Spec similarity & clustering
migrate.py              # Spec version migration
comparative_report.py   # Cross-spec comparative analysis
test_properties.py      # Property-based structural tests (174+ tests)
patterns.py             # Pattern library (7 reusable architectural patterns)
compose.py              # Pattern composition operator
import_langgraph.py     # Import LangGraph StateGraph Python files → YAML specs
owl_bridge.py           # OWL dual representation: YAML<->OWL round-trip + classification
ontology_owl.py         # OWL/DL pattern classification
ontology_rdf.py         # RDF export + semantic pattern queries
gaps.md                 # Ontology expressiveness gap analysis
ROADMAP.md              # Strategic roadmap and long-term vision
.github/workflows/      # CI: validate, lint, syntax-check, smoke-test
mutate.py               # Spec mutation engine (field + pattern-level operators)
evolve.py               # Evolutionary search with crossover + benchmark fitness
benchmark.py            # Benchmark suite (HotpotQA, GSM8K, ARC, HumanEval) with multi-run stats
trace.json              # LLM call traces from last agent run
```
