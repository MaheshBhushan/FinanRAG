import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.compare import evaluate_case, load_fixtures


class RetrievalComparisonTests(unittest.TestCase):
    def test_fixture_validation_and_supported_citation_scoring(self):
        fixture = {
            "id": "revenue",
            "question": "What was revenue?",
            "expected_supported": True,
            "expected_sources": ["annual.pdf"],
            "citations": ["annual.pdf", "other.pdf"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixtures.json"
            path.write_text(json.dumps([fixture]), encoding="utf-8")
            self.assertEqual(load_fixtures(path), [fixture])

        result = evaluate_case(fixture, [{"id": "chunk-1", "source": "annual.pdf", "score": 0.8}])
        self.assertTrue(result["retrieval_supported"])
        self.assertEqual(result["citation_precision"], 0.5)
        self.assertEqual(result["citation_recall"], 1.0)
        self.assertEqual(result["citation_failures"], ["other.pdf"])

    def test_unsupported_answer_is_recorded_without_retrieval_support(self):
        fixture = {
            "question": "Unsupported question?",
            "expected_supported": False,
            "answer": "A fabricated answer",
            "citations": [],
        }
        result = evaluate_case(fixture, [{"id": "chunk-1", "source": "unrelated.pdf", "score": 0.2}])
        self.assertFalse(result["retrieval_supported"])
        self.assertTrue(result["unsupported_answer"])
        self.assertIsNone(result["citation_precision"])


if __name__ == "__main__":
    unittest.main()
