# Agent Ontology

## What This Is
A formal representation of agent architecture: YAML specs that are machine-readable, LLM-legible, evolvable, and scale-invariant. The toolchain validates, generates, analyzes, and evolves agent architectures.

## Long-Term Vision: AlphaEvolve for Agent Architectures

The north star is a **self-improving system that designs agents better over time** — an AlphaEvolve-style loop over agent architectures instead of algorithms:

```
Flash diagnoses failures and prescribes targeted mutations (progressive disclosure)
    → Automated evaluator scores each (benchmark + verify + early stop)
    → Pro analyzes top candidates deeply (diagnosis + refinement)
    → Results feed back as persistent knowledge
    → Next iteration uses accumulated evidence + failure details
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
| Broad mutation | `mutate.py` (field + pattern level, insert_pattern blacklisted) | Working |
| LLM-guided mutation | `evolve.py --llm-guided` (structured menu selection) | Working |
| Progressive disclosure | `evolve.py` (diagnose → prescribe via edit_prompt/menu) | Working |
| Deep analysis | `evolve.py` Mini (gpt-5-mini) analyzes top-K per generation | Working |
| Population evolution | `evolve.py` (selection, crossover, lineage) | Working |
| Knowledge store | `knowledge_store.py` (SQLite, persistent across runs) | Working |
| Benchmark fitness | `benchmark.py` + 8 datasets (incl. multidoc, kb_tool) | Working |
| Error analysis | Per-example failure tracking, failure_summary in LLM context | Working |
| Early stop | Skip remaining examples after N failures (saves ~40% budget) | Working |
| Multi-run eval | `--eval-runs N` averages fitness across runs | Working |
| Benchmark transforms | Domain-specific prompt fixes (ARC letters, GSM8K traps) | Working |
| Structural verification | `verify.py` (9 checks), `lint.py` (10 rules) | Working |
| OWL reasoning | `owl_bridge.py` (round-trip + pattern classification) | Working |

### What's Proven
- **Level 1**: 26/26 agents pass E2E tests, 246/246 property tests (31 specs)
- **Level 2**: Evolution finds genuinely better architectures. On GSM8K-Tricky (25 trick questions):
  - Baseline (minimal_solver, no CoT): 96% EM, fitness 200.8
  - Best evolved variant (change_model mutation): 100% EM, fitness 216.4
  - Separately verified: add_chain_of_thought prompt mutation also achieves 100% EM
  - On ARC (25 science MC questions): baseline 32% EM → best evolved 36% EM
  - Knowledge store has 75 candidates across 4 benchmarks with mutation effectiveness data
  - `change_model` is highest single-run mutation (216.4), `insert_pattern` consistently worst (avg 37.5, now blacklisted)
  - On MultiDoc (25 cross-reference questions): baseline 72% EM → best evolved 76% EM (modify_prompt)
  - Progressive disclosure pipeline operational: correctly diagnoses failures but over-corrects on rewrites
- **Level 3**: KG RAG agent queries real networkx-backed knowledge graph with fictional data

### Known Weaknesses (and fixes)
1. ~~**Pattern insertion breaks task-specific agents**~~ FIXED: `insert_pattern` blacklisted (Session 24).
2. ~~**LLM-guided mutation selection is too narrow**~~ FIXED: Progressive disclosure pipeline (diagnose → prescribe) with `edit_prompt` operator for surgical find-and-replace on prompts. Menu selection preferred when pre-validated transforms match.
3. ~~**Prompt transforms can contradict**~~ FIXED: `add_chain_of_thought` strips conflicting instructions before appending.
4. ~~**Near-saturated accuracy limits evolution**~~ FIXED: Custom multi-doc benchmark has genuine headroom (baseline 72% EM, structured 76% EM, hard questions consistently wrong).
5. ~~**Freeform prompt rewrite over-corrects**~~ FIXED: Replaced `rewrite_prompt` with `edit_prompt` (surgical find-and-replace). LLM provides `old_text` and `new_text` for targeted edits. Verified working in progressive disclosure pipeline.
6. **Search stagnation in later generations** — Average fitness declines Gen 1→2→3. Greedy top-K selection narrows to local optima. Needs diversity injection or periodic restarts.
5. ~~**No error analysis**~~ FIXED: Per-example failure details + failure_summary fed to LLM mutations (Session 24).
6. **Symbolic types mostly stubs** — `symbolic_inference`, `search`, `world_model` generate placeholder code. Only `knowledge_graph` stores have real runtime (networkx).

### Strategic Direction (Session 24)

**Problem**: Existing benchmarks (GSM8K, ARC) don't differentiate architectures — a single LLM call saturates performance. Evolution works but has little headroom to show improvement.

**Solution — Two parallel tracks**:

**Track 1: Progressive disclosure mutations** — Replace single-shot menu selection with multi-call pipeline:
1. **Diagnose** (Flash): "Here are 5 failed examples. What pattern explains the failures?"
2. **Prescribe** (Flash): "Given that diagnosis, here's the spec. Write a specific fix."
3. **Analyze** (Mini): "Here are the top candidates. Why did the best one work?"

This lets the LLM *generate* fixes from error patterns (e.g., "outputs numbers instead of letters" → add format instruction) instead of picking from a pre-coded menu.

**Track 2: Custom multi-doc reasoning benchmark** — A benchmark where architecture genuinely matters:
- 25 questions requiring cross-referencing 3-5 fact cards
- Trap types: contradictions, misleading details, arithmetic aggregation, negation, temporal reasoning
- A single LLM call can't reliably handle cross-reference + trap combinations
- A retrieve → verify → reason → check architecture should do measurably better
- This creates genuine headroom for evolution to search within

### Model Choices (During Development)
- **Broad exploration**: `gemini-3-flash-preview` — cheap, fast, generates many candidates
- **Deep analysis**: `gpt-5-mini` — higher quality reasoning for top-K refinement
- Upgrade to Pro-tier models once the loop is proven and we're optimizing for quality

### Roadmap to Phase D (Ordered Steps)

**Steps 0-6**: DONE. See git history.

**Step 7: Evolution efficiency** — DONE (Session 24). Blacklisted `insert_pattern`, added early stop (skip after 3 failures), per-example error analysis with failure_summary, multi-run evaluation (`--eval-runs`), benchmark-specific prompt transforms.

**Step 8: Progressive disclosure mutations** — DONE. Multi-call pipeline: diagnose failures → prescribe fix via `edit_prompt` (surgical find-and-replace) or menu selection → apply. Falls back to menu selection when no failure data exists.

**Step 9: Architecture-sensitive benchmark** — DONE. Custom multi-doc reasoning benchmark (25 questions) where multi-step architecture outperforms single-shot. Baseline 72% EM, structured 76% EM.

**Step 10: Multi-tool benchmark** — DONE. KB-Tool benchmark (25 questions over fictional knowledge base, 2-4 chained tool calls). All entities fictional → 0% without tools. Monkey-patches tool functions at runtime. `kb_react.yaml` spec, `kb_tools.py` module, `score_kb_tool` scoring.

**Step 11: Compelling experiment** — Run evolution on kb_tool benchmark. Show: 0% (no tools) < X% (naive ReAct) < Y% (evolved). This is the proof that architecture search matters for tool-using agents.

### What NOT to Prioritize
- **More framework exporters** — 5 importers + 2 code gen backends prove the interchange story
- **Polish/ecosystem** (VS Code extension, web editor) — doesn't advance the core vision
- **DSPy optimization** — prompt optimization is orthogonal to architecture search
- **Richer ontology / DSL for process logic** — tempting but massive effort; lean into LLM-driven mutation instead

## Project Structure
```
agent_ontology/          Python package (33 modules, pip install -e .)
  ONTOLOGY.yaml          Entity/process/edge type definitions (v0.2+, 9 entity types, 10 process types, 13 edge types, verify edge added)
  specs/                 31 agent spec YAMLs (30 runnable + 1 description-only)
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
  benchmarks/            8 datasets (GSM8K, GSM8K-Hard, GSM8K-Tricky, HotpotQA, ARC, HumanEval, MultiDoc, KB-Tool) + scoring
  ...                    lint, verify, patterns, compose, benchmark, specgen, etc.
agents/                  31 generated custom-backend agents
agents_lg/               27 generated LangGraph-backend agents
tests/                   test_agents.py (26 E2E), test_properties.py (246), test_roundtrip.py
```

## Key Commands
```bash
export $(cat .env.local | xargs)                          # Load API keys
python3 agent_ontology/validate.py agent_ontology/specs/*.yaml  # Validate all specs
python3 agent_ontology/instantiate.py SPEC -o agents/X.py       # Generate agent
python3 agent_ontology/instantiate.py --all agent_ontology/specs/ -o agents/  # Batch regen
python3 tests/test_agents.py --agent react --timeout 120        # Test one agent
python3 tests/test_agents.py --timeout 120                      # Test all runnable agents
python3 tests/test_properties.py                                # 246 property tests
python3 -m agent_ontology.evolve specs/X.yaml -g 3 -p 5 --benchmark multidoc --llm-guided  # Evolution
python3 -m agent_ontology.evolve specs/X.yaml --eval-runs 3 --benchmark gsm8k  # Multi-run eval
python3 -m agent_ontology.knowledge_store stats              # Query knowledge store
python3 -m agent_ontology.knowledge_store best gsm8k         # Best genotypes for benchmark
python3 -m agent_ontology.recommend "multi-hop QA" --use-knowledge-store  # Get recommendations
```

## Design Principles
1. **Fail Loud** — No silent fallbacks
2. **Simplest thing that works** — Don't overengineer
3. **Delete > Comment** — Remove unused code
4. **Flat > Nested** — Prefer flat structures
5. **Search > Design** — Prefer evolving architectures over hand-designing them
6. **LLM > DSL** — Let LLMs generate fixes from errors rather than hand-coding every transform

## Key Technical Details
- State model: `state.data` dict with flat keys
- Gate conditions: parsed by `_generate_gate_check()` in instantiate.py
- Env var prefix: `AGENT_ONTOLOGY_` (MODEL, MAX_ITER)
- LLM routing: `call_llm()` routes by model prefix (claude→Anthropic, gemini→Google, else→OpenAI)
- Generated agents go to `agents/` (custom) and `agents_lg/` (LangGraph)
- All tool source is in the `agent_ontology/` package — use relative imports within
- Evolution improvements (Session 24): `insert_pattern` blacklisted, early stop after 3 failures, per-example error analysis, multi-run eval, benchmark-specific prompt transforms
