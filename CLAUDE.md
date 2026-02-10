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
| Population evolution | `evolve.py` (selection, crossover, lineage) | Working |
| Benchmark fitness | `benchmark.py` + 4 datasets | Working |
| LLM-guided improvement | `self_improver.yaml` (analyst → mutator → evaluator) | Working |
| Structural verification | `verify.py` (9 checks), `lint.py` (10 rules) | Working |
| OWL reasoning | `owl_bridge.py` (round-trip + pattern classification) | Working |

### What's Missing (in priority order)
1. **Flash/Pro cascade** — evolve.py uses one model; should use Flash for bulk generation, Pro for deep analysis
2. **Knowledge accumulation** — benchmark results vanish after each run; need persistent store of "pattern X scored Y on benchmark Z"
3. **Evidence-backed reasoning** — self_improver and recommend.py use hardcoded heuristics, not accumulated evidence
4. **Connected loop** — self_improver doesn't benchmark; evolve.py doesn't use LLM reasoning. Wire them together.
5. **OWL-powered tools** — compose.py and detect_patterns() still use hardcoded matching, not the OWL reasoner

### Model Choices (During Development)
- **Broad exploration**: `gemini-3-flash` — cheap, fast, generates many candidates
- **Deep analysis**: `gpt-5-mini` — higher quality reasoning for top-K refinement
- Upgrade to Pro-tier models once the loop is proven and we're optimizing for quality

### Roadmap to Phase D (Ordered Steps)

**Step 0: Landscape research** — Search for existing work on automated agent architecture search. Know what exists (ADAS, AlphaEvolve, DSPy, AutoAgents, etc.) so we build on prior art rather than reinvent. Document findings.

**Step 1: Knowledge store** — Persistent store (SQLite or JSON) accumulating structured facts: `{spec, pattern, benchmark, score, llm_calls, mutation, generation}`. Wire `evolve.py` and `benchmark.py` to write here after every run. This is the memory that makes the system learn.

**Step 2: Flash/Mini cascade in evolve.py** — Replace single-model evolution with: gemini-3-flash generates N mutations cheaply (broad) → automated evaluator scores all (validate + benchmark) → gpt-5-mini analyzes top-K deeply (diagnose why, propose targeted refinements).

**Step 3: Connect self_improver + benchmarks** — self_improver currently evaluates structurally (lint warnings). Add benchmark evaluation: mutate → instantiate → run on benchmark → score. The evaluator agent sees actual performance, not just lint counts.

**Step 4: Evidence-backed recommend.py** — Replace keyword heuristics with queries over the knowledge store. "For multi-hop QA, which patterns scored highest?" makes recommendations data-driven.

**Step 5: OWL-powered pattern detection (Phase C)** — Replace hardcoded `detect_patterns()` with DL classification from `owl_bridge.py`. Eliminates false positives, makes detection label-independent.

**Step 6: Close the full loop (Phase D)** — One command: `ao-design "multi-hop QA agent"` that queries knowledge → proposes architecture (informed by evidence) → validates + benchmarks → iterates with Flash/Mini cascade → stores results back. Each run makes the system smarter.

### What NOT to Prioritize
- **More framework exporters** — 5 importers + 2 code gen backends prove the interchange story. More bridges = diminishing returns.
- **Polish/ecosystem** (VS Code extension, web editor) — nice-to-have but doesn't advance the core vision
- **DSPy optimization** — prompt optimization is orthogonal to architecture search

## Project Structure
```
agent_ontology/          Python package (pip install -e .)
  ONTOLOGY.yaml          Entity/process/edge type definitions (v0.2)
  specs/                 23 agent spec YAMLs
  instantiate.py         Spec → Python agent code (custom + LangGraph backends)
  validate.py            23 structural validation rules
  evolve.py              Evolutionary search over architectures
  mutate.py              Mutation operators (field-level + pattern-level)
  self_improver.yaml     Agent that reasons about + improves specs
  owl_bridge.py          Bidirectional YAML↔OWL with DL pattern classification
  import_*.py            5 framework importers (LangGraph, CrewAI, AutoGen, OpenAI Agents, Google ADK)
  ...                    lint, verify, patterns, compose, benchmark, etc.
agents/                  Generated custom-backend agents (from instantiate.py)
agents_lg/               Generated LangGraph-backend agents
tests/                   test_agents.py, test_properties.py, test_roundtrip.py
benchmarks/              Datasets (GSM8K, HotpotQA, ARC, HumanEval) + scoring
```

## Key Commands
```bash
export $(cat .env.local | xargs)                          # Load API keys
python3 agent_ontology/validate.py agent_ontology/specs/*.yaml  # Validate all
python3 agent_ontology/instantiate.py SPEC -o agents/X.py       # Generate agent
python3 agent_ontology/instantiate.py --all agent_ontology/specs/ -o agents/  # Batch
python3 tests/test_agents.py --agent react --timeout 120        # Test one agent
python3 tests/test_properties.py                                # Property tests
python3 agent_ontology/evolve.py specs/react.yaml -g 3 -p 5    # Evolve an agent
python3 agent_ontology/evolve.py specs/self_refine.yaml --benchmark gsm8k  # Benchmark evolution
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
