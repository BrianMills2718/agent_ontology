# Agent Ontology

## What This Is
A formal representation of agent architecture: YAML specs that are machine-readable, LLM-legible, evolvable, and scale-invariant. The toolchain validates, generates, analyzes, and evolves agent architectures.

## Project Structure
```
agent_ontology/          Python package (pip install -e .)
  ONTOLOGY.yaml          Entity/process/edge type definitions (v0.2)
  specs/                 23 agent spec YAMLs
  instantiate.py         Spec → Python agent code (custom + LangGraph backends)
  validate.py            23 structural validation rules
  evolve.py              Evolutionary search over architectures
  mutate.py              Mutation operators (field-level + pattern-level)
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
```

## Design Principles
1. **Fail Loud** - No silent fallbacks
2. **Simplest thing that works** - Don't overengineer
3. **Delete > Comment** - Remove unused code
4. **Flat > Nested** - Prefer flat structures

## Key Technical Details
- State model: `state.data` dict with flat keys
- Gate conditions: parsed by `_generate_gate_check()` in instantiate.py
- Env var prefix: `AGENT_ONTOLOGY_` (MODEL, MAX_ITER)
- LLM routing: `call_llm()` routes by model prefix (claude→Anthropic, gemini→Google, else→OpenAI)
- Generated agents go to `agents/` (custom) and `agents_lg/` (LangGraph)
- All tool source is in the `agent_ontology/` package — use relative imports within
