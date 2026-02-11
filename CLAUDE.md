# Agent Ontology

## What This Is
A formal representation of agent architecture: YAML specs that are machine-readable, LLM-legible, evolvable, and scale-invariant. The toolchain validates, generates, analyzes, and evolves agent architectures.

## Long-Term Vision: AlphaEvolve for Agent Architectures

The north star is a **self-improving system that designs agents better over time** — an AlphaEvolve-style loop over agent architectures instead of algorithms:

```
Flash generates many spec variants (broad exploration)
    → Automated evaluator scores each (benchmark + verify)
    → Pro analyzes top candidates deeply (diagnosis + refinement)
    → Results feed back as persistent knowledge
    → Next iteration uses accumulated evidence
    → Repeat until convergence
```

This is Phase D in ROADMAP.md. Everything else is infrastructure serving this goal.

### Why Specs Beat Code as a Search Space
AlphaEvolve searches over raw code (hundreds of lines, unstructured). We search over typed graphs (~100 lines of validated YAML). Our advantage:
- **Mutations are semantic**: swap a pattern, add a review step, change a gate — not random text edits
- **Validation is instant**: every candidate is structurally checked before expensive LLM evaluation
- **The search space is navigable**: patterns, topology, similarity all work because specs are constrained
- **Knowledge is structured**: "CritiqueCycle + RetrievalAugmented scores 85% on multi-hop" is a queryable fact, not a number in stdout

### What Exists Today
| Component | Tool | Status |
|-----------|------|--------|
| Broad mutation | `mutate.py` (field + pattern level) | Working |
| LLM-guided mutation | `evolve.py --llm-guided` (Flash generates YAML mutations) | Working |
| Deep analysis | `evolve.py` Mini (gpt-5-mini) analyzes top-K per generation | Working |
| Population evolution | `evolve.py` (selection, crossover, lineage) | Working |
| Knowledge store | `knowledge_store.py` (SQLite, persistent across runs) | Working |
| Benchmark fitness | `benchmark.py` + 4 datasets | Working |
| LLM-guided improvement | `self_improver.yaml` (analyst → mutator → evaluator) | Working |
| Structural verification | `verify.py` (9 checks), `lint.py` (10 rules) | Working |
| OWL reasoning | `owl_bridge.py` (round-trip + pattern classification) | Working |

### What's Proven
- **Level 1**: 26/26 agents pass E2E tests, 214/214 property tests
- **Level 2**: Full evolution loop runs end-to-end (mutate→validate→instantiate→benchmark→select), knowledge store accumulates real data, evidence-backed recommendations work
- **Level 3**: KG RAG agent queries real networkx-backed knowledge graph with fictional data, answer grounded in KG triples

### Known Weaknesses
1. **LLM-guided mutations create degenerate specs** — Flash can insert patterns (e.g., map_reduce) that produce infinite retry loops. Programmatic mutations are reliable; LLM mutations need guardrails (max-iteration checks in inserted patterns)
2. **Evolution can't differentiate on easy benchmarks** — All variants score 100% on simple GSM8K. Need harder benchmarks or weaker starting specs to show fitness improvement
3. **Symbolic types mostly stubs** — `symbolic_inference`, `search`, `world_model` generate placeholder code. Only `knowledge_graph` stores have real runtime (networkx). PDDL planner, RAP, AlphaGeometry specs validate but don't call real engines
4. **compose.py** still uses hardcoded pattern matching, not OWL reasoner

### Model Choices (During Development)
- **Broad exploration**: `gemini-3-flash-preview` — cheap, fast, generates many candidates
- **Deep analysis**: `gpt-5-mini` — higher quality reasoning for top-K refinement
- Upgrade to Pro-tier models once the loop is proven and we're optimizing for quality

### Roadmap to Phase D (Ordered Steps)

**Step 0: Landscape research** — Search for existing work on automated agent architecture search. Know what exists (ADAS, AlphaEvolve, DSPy, AutoAgents, etc.) so we build on prior art rather than reinvent. Document findings.

**Step 1: Knowledge store** — ~~Persistent store (SQLite or JSON)~~ DONE. `knowledge_store.py` with SQLite backend. Records every evolution candidate (genotype, phenotype, lineage, analysis). CLI: `ao-knowledge-store stats|best|patterns|mutations|failures|generations`. Wired into evolve.py — every run persists to `~/.agent_ontology/evolution.db`.

**Step 2: Flash/Mini cascade in evolve.py** — ~~Replace single-model evolution~~ DONE. `--llm-guided` flag enables: Flash (gemini-3-flash-preview) generates mutations informed by spec+errors+analysis+knowledge → validate+benchmark → Mini (gpt-5-mini) analyzes top-K → analysis feeds next generation. YAML retry with error feedback. Env vars: `AGENT_ONTOLOGY_FLASH_MODEL`, `AGENT_ONTOLOGY_ANALYST_MODEL`.

**Step 3: Connect self_improver + benchmarks** — DONE. `benchmark_candidate()` in evolve.py evaluates specs end-to-end.

**Step 4: Evidence-backed recommend.py** — DONE. `recommend.py` queries knowledge store for pattern performance data.

**Step 5: OWL-powered pattern detection** — DONE. `patterns.py` uses `detect_patterns_structural()` with OWL classification.

**Step 6: Close the full loop** — DONE. `ao-design "task description" --benchmark gsm8k --evolve` runs the complete pipeline: recommend → benchmark → evolve → store.

### What NOT to Prioritize
- **More framework exporters** — 5 importers + 2 code gen backends prove the interchange story. More bridges = diminishing returns.
- **Polish/ecosystem** (VS Code extension, web editor) — nice-to-have but doesn't advance the core vision
- **DSPy optimization** — prompt optimization is orthogonal to architecture search

## Project Structure
```
agent_ontology/          Python package (33 modules, pip install -e .)
  ONTOLOGY.yaml          Entity/process/edge type definitions (v0.2+, 9 entity types, 10 process types)
  specs/                 27 agent spec YAMLs (incl. 4 neurosymbolic)
  instantiate.py         Spec → Python agent code (custom + LangGraph backends)
  validate.py            23 structural validation rules
  design.py              End-to-end agent design pipeline (ao-design)
  evolve.py              Evolutionary search over architectures
  mutate.py              Mutation operators (field-level + pattern-level)
  ingest.py              Corpus ingestion for ChromaDB vector stores (ao-ingest)
  knowledge_store.py     SQLite store for evolution results (ao-knowledge-store)
  recommend.py           Evidence-backed architecture recommender (ao-recommend)
  owl_bridge.py          Bidirectional YAML↔OWL with DL pattern classification
  import_*.py            5 framework importers (LangGraph, CrewAI, AutoGen, OpenAI Agents, Google ADK)
  ...                    lint, verify, patterns, compose, benchmark, specgen, etc.
agents/                  27 generated custom-backend agents
agents_lg/               27 generated LangGraph-backend agents
tests/                   test_agents.py (26 E2E), test_properties.py (214), test_roundtrip.py
benchmarks/              Datasets (GSM8K, HotpotQA, ARC, HumanEval) + scoring
```

## Key Commands
```bash
export $(cat .env.local | xargs)                          # Load API keys
python3 agent_ontology/validate.py agent_ontology/specs/*.yaml  # Validate all 27 specs
python3 agent_ontology/instantiate.py SPEC -o agents/X.py       # Generate agent
python3 agent_ontology/instantiate.py --all agent_ontology/specs/ -o agents/  # Batch regen all 27
python3 tests/test_agents.py --agent react --timeout 120        # Test one agent
python3 tests/test_agents.py --timeout 120                      # Test all 26 runnable agents
python3 tests/test_properties.py                                # 214 property tests
python3 -m agent_ontology.design "math solver" --benchmark gsm8k --evolve  # Full design pipeline
python3 -m agent_ontology.evolve agent_ontology/specs/plan_and_solve.yaml -g 3 -p 5 --benchmark gsm8k  # Evolution
python3 -m agent_ontology.knowledge_store stats              # Query knowledge store
python3 -m agent_ontology.knowledge_store best gsm8k         # Best genotypes for benchmark
python3 -m agent_ontology.ingest docs/ --store my_kb         # Ingest docs into ChromaDB
python3 -m agent_ontology.recommend "multi-hop QA" --use-knowledge-store  # Get recommendations
```

## Design Principles
1. **Fail Loud** — No silent fallbacks
2. **Simplest thing that works** — Don't overengineer
3. **Delete > Comment** — Remove unused code
4. **Flat > Nested** — Prefer flat structures
5. **Search > Design** — Prefer evolving architectures over hand-designing them

## Key Technical Details
- State model: `state.data` dict with flat keys
- Gate conditions: parsed by `_generate_gate_check()` in instantiate.py
- Env var prefix: `AGENT_ONTOLOGY_` (MODEL, MAX_ITER)
- LLM routing: `call_llm()` routes by model prefix (claude→Anthropic, gemini→Google, else→OpenAI)
- Generated agents go to `agents/` (custom) and `agents_lg/` (LangGraph)
- All tool source is in the `agent_ontology/` package — use relative imports within
