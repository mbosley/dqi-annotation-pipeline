import json
import tempfile
import unittest
from pathlib import Path

from src.core import (
    AnnotationModel,
    AnnotationSample,
    SyntheticKeywordProvider,
    run_pipeline,
    write_results_jsonl,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_PATH = REPO_ROOT / "examples/synthetic/dqi_samples.jsonl"
SCHEMA_PATH = REPO_ROOT / "specs/jsonschema/dqi-annotation-result.schema.json"


def _load_samples() -> list[AnnotationSample]:
    samples: list[AnnotationSample] = []
    with SAMPLES_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            samples.append(AnnotationSample(sample_id=row["sample_id"], text=row["text"]))
    return samples


class SyntheticPipelineIntegrationTest(unittest.TestCase):
    def test_pipeline_runs_and_writes_valid_jsonl(self) -> None:
        samples = _load_samples()
        provider = SyntheticKeywordProvider()
        model = AnnotationModel(provider="synthetic", name="keyword-v1")

        results = run_pipeline(
            samples=samples,
            provider=provider,
            model=model,
            run_id="itest-run-001",
            schema_path=SCHEMA_PATH,
            timestamp_utc="2026-03-01T12:00:00+00:00",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["sample_id"], "syn-001")
        self.assertEqual(results[0]["labels"]["reason_giving"], 2)
        self.assertEqual(results[1]["labels"]["respect"], 2)
        self.assertEqual(results[0]["model"]["provider"], "synthetic")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.jsonl"
            write_results_jsonl(results, output_path)
            self.assertTrue(output_path.exists())
            written_rows = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(written_rows), 2)


if __name__ == "__main__":
    unittest.main()
