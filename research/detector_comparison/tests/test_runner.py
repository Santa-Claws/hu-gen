import json
import tempfile
import unittest
from pathlib import Path

from research.detector_comparison.runner import run_samples


class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output = Path(self.temp_dir.name) / "scores.jsonl"
        self.samples = [
            {"id": "one", "text": "first", "text_sha256": "hash-one", "label": "human_authored"},
            {"id": "two", "text": "second", "text_sha256": "hash-two", "label": "ai_generated"},
        ]
        self.configuration = {"id": "neo-baseline", "fingerprint": "cfg-123"}

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_writes_one_success_record_per_sample(self) -> None:
        summary = run_samples(
            self.samples,
            scorer=lambda text: {"criterion": len(text), "reported_ai_probability": 0.25, "tokens": 1},
            output_path=self.output,
            run_id="run-001",
            configuration=self.configuration,
            host_metadata={"gpu": "test"},
        )

        records = [json.loads(line) for line in self.output.read_text().splitlines()]
        self.assertEqual(summary, {"completed": 2, "failed": 0, "skipped": 0})
        self.assertEqual([r["sample_id"] for r in records], ["one", "two"])
        self.assertTrue(all(r["status"] == "success" for r in records))
        self.assertTrue(all(r["configuration_fingerprint"] == "cfg-123" for r in records))
        self.assertTrue(all("elapsed_ms" in r for r in records))

    def test_skips_previously_completed_hash_for_same_configuration(self) -> None:
        calls: list[str] = []
        scorer = lambda text: calls.append(text) or {"criterion": 1.0, "reported_ai_probability": 0.5, "tokens": 1}
        run_samples(self.samples, scorer, self.output, "run-001", self.configuration, {})
        second = run_samples(self.samples, scorer, self.output, "run-002", self.configuration, {})

        self.assertEqual(calls, ["first", "second"])
        self.assertEqual(second, {"completed": 0, "failed": 0, "skipped": 2})

    def test_records_a_failure_and_continues_to_later_samples(self) -> None:
        def scorer(text: str) -> dict:
            if text == "first":
                raise ValueError("bad sample")
            return {"criterion": 1.0, "reported_ai_probability": 0.5, "tokens": 1}

        summary = run_samples(self.samples, scorer, self.output, "run-001", self.configuration, {})
        records = [json.loads(line) for line in self.output.read_text().splitlines()]

        self.assertEqual(summary, {"completed": 1, "failed": 1, "skipped": 0})
        self.assertEqual([(r["sample_id"], r["status"]) for r in records], [("one", "failure"), ("two", "success")])
        self.assertEqual(records[0]["error_type"], "ValueError")
