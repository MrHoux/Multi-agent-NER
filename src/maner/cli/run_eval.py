import json

from maner.core.schema import load_schema
from maner.eval.metrics import evaluate_from_files


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run strict span-level evaluation")
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--pred_path", required=True)
    parser.add_argument("--schema_path", required=True)
    args = parser.parse_args()

    schema = load_schema(args.schema_path)
    result = evaluate_from_files(args.gold_path, args.pred_path, schema)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
