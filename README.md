# multiagent-ner

Training-free, span-level multi-agent NER research codebase with dual evidence lines:

- `Expert line`: Expert constraints `E` -> `Y_exp`
- `RE line`: Relation reasoning `R` -> `Y_re`
- Conflict-driven arbitration/debate (only on conflict clusters)
- Verifier-gated memory writeback

## 1. Install

```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
python -m pip install -e .
```

## 2. Environment

Copy `.env.example` to `.env` (or export variables directly):

- `LLM_BASE_URL`
- `LLM_API_KEY`
- `LLM_MODEL`

Default config uses `provider: mock` for local reproducible smoke runs.

## 3. Data Format

No dataset is hard-coded. Reader interface expects generic jsonl lines:

```json
{"id":"s1","text":"John works at Acme Corp.","gold_mentions":[{"start":0,"end":4,"ent_type":"PERSON"}]}
```

- required: `id`, `text`
- optional (for eval): `gold_mentions`

Schema JSON format:

```json
{
  "entity_types": [{"name": "PERSON", "description": "..."}],
  "relation_constraints": [{"name": "works_for", "head_types": ["PERSON"], "tail_types": ["ORG"]}]
}
```

See `schema_example.json`.

## 4. Run Pipeline

```bash
python -m maner.cli.run_pipeline --config configs/default.yaml
```

You can reuse one config for multiple experiments via runtime overrides:

```bash
python -m maner.cli.run_pipeline \
  --config experiments/bc2gm/config.bc2gm.deepseek.yaml \
  --set data.data_path=outputs/bc2gm/bc2gm_test.20.jsonl \
  --set output.predictions_path=outputs/bc2gm/predictions.deepseek.real.20.jsonl
```

Output: `outputs/predictions.jsonl` (configurable)

Each line contains:

- `id`, `text`
- `mentions[]`: span-level offsets + type + confidence + evidence + rationale
- `traces`: per-agent trace, conflict/debate/verifier/memory traces
- `costs`: calls, token usage (if available), latency, debate rounds, trigger rate

## 5. Evaluate

```bash
python -m maner.cli.run_eval --gold_path tests/fixtures/tiny_gold.jsonl --pred_path outputs/predictions.jsonl --schema_path tests/fixtures/schema_example.json
```

Metrics:

- strict span-level exact match F1
- micro / macro (by type)
- error stats: `type_mismatch`, `boundary_mismatch`, `spurious`, `missing`

Evaluation protocol note:

- Do not iteratively tune rules/prompts on the same holdout set.
- Keep a fixed dev split for tuning and a separate untouched test split for final reporting.

## 6. Ablations (all in config)

In `configs/default.yaml` -> `ablations`:

- `w_o_expert`
- `w_o_re`
- `w_o_debate`
- `w_o_verifier`
- `w_o_memory`

## 7. Span Augmentation (optional)

You can allow Expert/RE to propose extra recall spans that are merged into `CandidateSet` with provenance:

- `pipeline.allow_expert_span_augmentation`
- `pipeline.allow_re_span_augmentation`
- `pipeline.rerun_after_span_augmentation`
- `pipeline.max_augmented_spans_per_sample`

Behavior:

- Expert/RE may output `new_spans[]` in strict JSON.
- Pipeline validates offsets/substring and deduplicates by exact offsets.
- Accepted spans are assigned `aug_xxxx` ids and `provenance`.
- If rerun is enabled, pipeline reruns RAG/Expert/RE once on the augmented candidate set before NER.

## 8. Generalization-Safe Runtime Controls

To avoid dataset-specific hardcoding in core runtime:

- `pipeline.inject_high_value_candidates`: default `false`
- `pipeline.lexical_injection_rules`: default `[]` (config-only, no built-in terms)
- `pipeline.mention_blocklist`: default `[]` (config-only, no built-in terms)
- `pipeline.candidate.*`: generic fallback controls (no domain lexicon in code)
- `pipeline.enable_descriptor_expansion`: default `false`; if enabled, descriptor terms can be derived from schema

## 9. Key Modules

- `src/maner/core/`: types, schema, dataset reader, config/prompt loading
- `src/maner/llm/`: provider client + strict JSON parsing/minimal fix
- `src/maner/agents/`: candidate/expert/re/ner/adjudicator/debate/verifier
- `src/maner/orchestrator/`: conflict triage + end-to-end pipeline
- `src/maner/memory/`: sqlite memory store (retrieve/writeback/promote)
- `src/maner/eval/`: strict metrics + error analysis
- `tests/`: unit tests + tiny fixtures

## 10. Provider

Implemented provider:

- `openai_compatible` REST (`base_url + api_key + model`)
- `mock` provider for local deterministic tests/smoke

Provider adapter can be extended in `src/maner/llm/client.py`.
