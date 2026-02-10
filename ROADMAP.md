# OpenClaw Roadmap

**Date:** 2026-02-09
**Current state:** v0.2 ontology, 22 specs, 27 tools, 2 code gen backends

---

## Where We Are

### Assets
- **Ontology** (ONTOLOGY.yaml v0.2): 8 entity types, 7 process types, 12 edge types, 25+ validation rules
- **Spec catalog**: 22 agent specs covering major patterns in the literature (ReAct, debate, RAG, plan-and-solve, self-refine, tree-of-thought, LATS, voyager, multi-agent codegen, map-reduce, socratic tutor, reflexion, mixture-of-agents, meta-prompting, customer-support swarm, software team, etc.)
- **Full pipeline**: docs -> spec -> {validate, visualize, instantiate, test, analyze, mutate, evolve, benchmark}
- **Two code gen backends**: custom state machine + LangGraph
- **Pattern library**: 7 reusable patterns with composition, mutation, crossover
- **Benchmarks**: 4 datasets (HotpotQA, GSM8K, ARC, HumanEval)
- **Property tests**: 174/174 pass across all 22 specs

### Core Thesis
Agent architectures are typed graphs. If you make the graph formal, you unlock validation, visualization, code generation, testing, analysis, comparison, mutation, and evolution for free.

---

## Ontology Completeness Assessment

### Confirmed Complete
These areas were evaluated against 10+ frameworks (LangGraph, CrewAI, AutoGen, MetaGPT, CAMEL, OpenAI Agents SDK, Google ADK, DSPy, Magentic-One, MCP) and are fully expressible:

| Pattern | How it's expressed | Frameworks covered |
|---------|-------------------|-------------------|
| Reasoning loops (ReAct) | step + gate + loop edge | All |
| Critique/refine cycles | step + gate + loop edge | Self-refine, DSPy |
| Fan-out/fan-in | Multiple invoke edges from one step | Code reviewer, map-reduce |
| Pub/sub messaging | channel entity + publish/subscribe edges | MetaGPT, AutoGen |
| Agent handoffs | handoff edge + gate routing | OpenAI Swarm, CrewAI |
| Team execution | team entity with strategy field | CrewAI, MetaGPT |
| Conversations | conversation entity | CAMEL, OpenAI threads |
| Nested loops | Multiple loop edges at different scopes | Magentic-One dual-loop |
| Recursive composition | spawn with template: 'self' | Meta-prompting |
| Human-in-the-loop | checkpoint process | Socratic tutor |
| Guardrails/policies | policy process with targets + effect | OpenAI Agents SDK, Google ADK |
| Error handling | error_handler process + error edges | All production frameworks |
| Store-backed memory | store entity (vector, kv, queue, etc.) | RAG, BabyAGI, AutoGPT |
| Composable termination | termination_condition structured type | AutoGen |
| Checkpointing | checkpointing config on spec_shape | LangGraph |

### Mild Gaps (Optional Improvements)

#### 1. Multimodal Schema Types
**What**: Schema field types are string/integer/float/object/list. No image/audio/video types.
**Impact**: Can't distinguish "this agent receives a screenshot" from "this agent receives text" at the spec level. Both would be `type: string` (base64) or `type: object`.
**Fix**: Add `image`, `audio`, `video` as recognized schema field types. Purely additive — no structural change.
**Verdict**: Nice-to-have. Doesn't block any current patterns. Modern LLMs handle multimodal via their API, not via schema types.

#### 2. Memory Consolidation (Episodic -> Semantic)
**What**: No way to express "periodically consolidate raw episodic traces into summarized semantic memory."
**Impact**: Reflexion spec has episodic memory, but the consolidation step is manual logic, not a declarable pattern.
**Fix**: Already expressible as a maintenance flow (step with logic that reads episodic store, summarizes, writes to semantic store). Not a new ontology type — just a spec pattern.
**Verdict**: Not an ontology gap. Could add as a pattern in patterns.py.

#### 3. MCP Tool Discovery
**What**: MCP tools are discovered at runtime, not declared statically.
**Impact**: Our tool entities are statically wired in specs.
**Analysis**: This is a runtime concern, not an architecture concern. `tool_type: mcp` already declares "this agent uses MCP." Which specific tools are available is determined at runtime by the MCP server — same as a database query returning different rows. Adding a "discovery flag" would mix runtime semantics into the structural ontology.
**Verdict**: Not an ontology gap. Already handled by `tool_type: mcp`.

#### 4. DSPy Prompt Optimization
**What**: DSPy compiles/optimizes prompts against metrics. No way to declare optimization targets in specs.
**Impact**: Can't express "optimize this agent's system prompt for exact_match on GSM8K."
**Analysis**: Optimization is a meta-process that operates on specs/agents, not part of the architecture itself. Like compiler optimizations aren't part of the source language.
**Verdict**: Not an ontology gap. Would be a separate tool (`dspy_optimize.py`) in the pipeline, not an ontology extension.

### Not Gaps (Confirmed Out of Scope)
- **Deployment topology** (which machine agents run on) — runtime concern
- **Cost/budget constraints** — runtime concern
- **Authentication/authorization** — runtime concern
- **Streaming/async execution details** — runtime concern
- **Distributed runtime** — runtime concern

---

## Strategic Direction

### Value Proposition
"OpenAPI for Agents" — a universal spec format for describing AI agent architectures, with a complete toolchain for validation, visualization, code generation, and analysis.

### The unique asset is the ontology + spec format, not the code generator.
Code generators are commodity (LangGraph, CrewAI will always run their own frameworks better). But nobody has the interchange format.

### Three audiences (in priority order)

1. **Framework interop** (highest leverage): Import from LangGraph/CrewAI/AutoGen -> YAML spec -> Export to any target. The "Rosetta Stone for agent frameworks."

2. **Architecture documentation/audit**: Point at existing agent code, get a validated architecture diagram + complexity analysis + lint warnings. Useful TODAY for teams with agents.

3. **Rapid prototyping**: Describe agent in English -> get running code. Useful but competing with native frameworks.

---

## Plan: Order of Attack

### Phase 1: Solidify the Ontology (CURRENT)
**Goal**: Declare the ontology complete for v1.0 and write the formal specification.

#### 1.1 Formal Spec Document
Write a clean, standalone specification of the YAML format — independent of this repo's tooling. Like the OpenAPI spec document.

- **Input**: ONTOLOGY.yaml + 22 example specs
- **Output**: `SPEC.md` — a human-readable specification with:
  - Format definition (entities, processes, edges, schemas)
  - Type reference (all entity/process/edge types with fields)
  - Validation rules
  - Examples for each type
  - Versioning policy
- **Satisfaction criteria**: A developer who has never seen this repo can write a valid spec from SPEC.md alone, and `validate.py` accepts it.
- **Uncertainties**: How formal should it be? JSON Schema level? Or prose + examples level?
- **Questions**: Should we also produce a JSON Schema for the spec format (machine-readable validation)?

#### 1.2 Multimodal Schema Types (Optional)
Add `image`, `audio`, `video` as recognized schema field types.

- **Satisfaction criteria**: Can write a spec with `{ name: screenshot, type: image }` and it validates. Generated code handles multimodal inputs.
- **Uncertainties**: Do we need to change the code generator? Multimodal is handled by the LLM API, not by schema types.
- **Decision**: Defer unless a spec needs it.

### Phase 2: Import Bridge (Highest Leverage)
**Goal**: Prove the interchange story by importing real agent code into specs.

#### 2.1 LangGraph Importer
Parse a LangGraph `StateGraph` definition -> YAML spec.

- **Input**: Python file with `StateGraph`, `add_node`, `add_edge`, `add_conditional_edges`
- **Output**: Valid YAML spec
- **Satisfaction criteria**:
  - Round-trip: import a LangGraph agent -> validate -> export back to LangGraph -> same behavior
  - Works on at least 3 real LangGraph examples (from LangGraph docs/tutorials)
- **Uncertainties**:
  - How much can we infer from code? Node function bodies -> logic blocks? Or just the graph structure?
  - Static analysis vs. runtime tracing?
  - What about LangGraph features we can't express? (Should be none, but verify.)
- **Questions**:
  - Do we parse AST or use regex? AST is more robust.
  - Do we need the user to annotate their code, or can we infer everything?

#### 2.2 CrewAI Importer (Optional, after 2.1)
Parse CrewAI crew definitions -> YAML spec.

- **Satisfaction criteria**: Import a CrewAI crew with agents + tasks + process type -> valid spec with team entity.
- **Uncertainties**: CrewAI's API changes frequently. Which version to target?

### Phase 3: Production Code Generation
**Goal**: Generated agents that actually work with real tools and persistent memory.

#### 3.1 Real Store Backends
Generate ChromaDB/SQLite/Redis backends based on `store_type`.

- **Satisfaction criteria**: RAG agent with `store_type: vector` uses ChromaDB. Data persists across runs.
- **Uncertainties**: Dependency management. Do we add chromadb/redis as optional deps?

#### 3.2 MCP Tool Integration
Generate real MCP client code for `tool_type: mcp`.

- **Satisfaction criteria**: Agent with MCP tool entity connects to an MCP server and calls real tools.
- **Uncertainties**: MCP protocol is still evolving. Which version to target?

#### 3.3 DSPy Optimization Backend
Add a `dspy_optimize.py` tool that takes a generated agent + training data -> optimized prompts.

- **Input**: Generated agent file + benchmark dataset + metric
- **Output**: Optimized system prompts written back to the agent file
- **Satisfaction criteria**: ReAct agent on HotpotQA improves EM score by >10% after DSPy optimization.
- **Uncertainties**:
  - Does DSPy work with our generated code structure, or do we need to emit DSPy modules directly?
  - How to map our schemas to DSPy signatures?
- **Questions**: Should this be a new code gen backend (`--backend dspy`) or a post-processing tool?

### Phase 4: Polish & Ecosystem
**Goal**: Make it easy for others to adopt.

#### 4.1 JSON Schema for Spec Format
Machine-readable validation of YAML specs, usable by any language.

#### 4.2 Web-based Spec Editor
Extend spec-viewer.html with editing capabilities.

#### 4.3 VS Code Extension
YAML autocompletion and inline validation for spec files.

#### 4.4 PyPI Package
`pip install openclaw` with CLI tools.

---

## Open Questions

1. **Naming**: "OpenClaw" vs something else? The name should convey "agent architecture standard."
2. **Versioning**: When do we declare v1.0? After the formal spec? After the first importer?
3. **Community**: Should we write a paper? Post on HN? Create a Discord?
4. **Governance**: If this becomes a standard, who maintains it? Just you, or a broader group?
5. **Licensing**: Current repo is public on GitHub. MIT? Apache 2.0?
6. **Ontology additions vs stability**: Every addition to the ontology breaks backward compatibility for importers/exporters. When do we freeze?

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-09 | MCP discovery is NOT an ontology gap | Runtime concern — `tool_type: mcp` already captures the architectural fact |
| 2026-02-09 | DSPy optimization is NOT an ontology concern | Meta-process on specs, not part of agent architecture |
| 2026-02-09 | Multimodal types deferred | No current spec needs them; LLM APIs handle multimodal transparently |
| 2026-02-09 | Memory consolidation is a spec pattern, not ontology gap | Expressible as a maintenance flow with existing types |
| 2026-02-09 | Ontology assessed as substantially complete for v1.0 | Evaluated against 10 frameworks, no structural gaps found |
