# Datasets Directory

Place your dataset files here (optional). The code does not hard-code dataset names.

Minimal jsonl format per line:

```json
{"id": "sample-1", "text": "John works at Acme.", "gold_mentions": [{"start":0,"end":4,"ent_type":"PERSON"}]}
```

Use `configs/default.yaml` to point to `data_path` and `schema_path`.
