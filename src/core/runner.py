import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from jsonschema import validate

from .contracts import AnnotationModel, AnnotationProvider, AnnotationSample


DEFAULT_SCHEMA_PATH = Path("specs/jsonschema/dqi-annotation-result.schema.json")


def _load_schema(schema_path: Path) -> Dict[str, Any]:
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_pipeline(
    samples: Iterable[AnnotationSample],
    provider: AnnotationProvider,
    model: AnnotationModel,
    run_id: str,
    schema_path: Path = DEFAULT_SCHEMA_PATH,
    timestamp_utc: Optional[str] = None,
) -> List[Dict[str, Any]]:
    schema = _load_schema(schema_path)
    run_timestamp = timestamp_utc or _timestamp_utc()
    output: List[Dict[str, Any]] = []

    for sample in samples:
        labels = provider.annotate(sample)
        result: Dict[str, Any] = {
            "sample_id": sample.sample_id,
            "labels": labels,
            "model": {"provider": model.provider, "name": model.name},
            "run_metadata": {
                "run_id": run_id,
                "timestamp_utc": run_timestamp,
                "repair_applied": False,
            },
        }
        validate(instance=result, schema=schema)
        output.append(result)

    return output


def write_results_jsonl(results: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")
